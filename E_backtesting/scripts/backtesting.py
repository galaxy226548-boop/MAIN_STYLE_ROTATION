"""Run signal-position backtests from per-factor target-position files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"
SY_BASELINE_DIR = PROJECT_ROOT / "SY_Baseline"
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

try:
    import matplotlib
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    if LOCAL_PYTHON.exists() and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv_mktp").resolve():
        os.execv(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv])
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for path in (PROJECT_ROOT, SY_BASELINE_DIR, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Config import Config  # noqa: E402
from G_engine.sample_window import SampleWindow, resolve_sample_window  # noqa: E402
from engine import calc_pre_trade_actual_weight, run_single_track_backtest, weights_are_different  # noqa: E402
from nav import build_single_track_daily_nav  # noqa: E402
from returns import to_simple_return  # noqa: E402
from summary import build_track_summary_table  # noqa: E402


TRACK_COL = "track_id"
DEFAULT_POSITION_FILE = PROJECT_ROOT / "C_positions" / "output" / "factor_positions" / "ZHAO01_position.xlsx"
DEFAULT_MARKET_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "market_df.parquet"
DEFAULT_DATA_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "data_df.parquet"
DEFAULT_BENCHMARK_POSITION_PATH = PROJECT_ROOT / "C_positions" / "reference" / "benchmark_positions.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"
BENCHMARK_MODE = "rebalance_50"
BENCHMARK_REBALANCE_MONTHS = 2
WEIGHT_TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the style-rotation backtest from a factor position Excel file."
    )
    parser.add_argument("--position-file", type=Path, default=DEFAULT_POSITION_FILE)
    parser.add_argument("--market-path", type=Path, default=DEFAULT_MARKET_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--benchmark-position-path", type=Path, default=DEFAULT_BENCHMARK_POSITION_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample", choices=["all", "ins", "oos", "custom"], default="all")
    parser.add_argument("--start-date", help="Optional sample start date, e.g. 2020-01-01.")
    parser.add_argument("--end-date", help="Optional sample end date, e.g. 2024-12-31.")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_factor_col(position_file: Path) -> str:
    stem = position_file.stem
    suffix = "_position"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def read_position_file(position_file: Path) -> pd.DataFrame:
    if not position_file.exists():
        raise FileNotFoundError(f"Cannot find position file: {position_file}")

    position_df = pd.read_excel(position_file, index_col=0)
    position_df.index = pd.to_datetime(position_df.index)
    position_df = position_df.sort_index()

    rename_map = {
        "target_growth_weight": "target_weight_g",
        "target_value_weight": "target_weight_v",
    }
    position_df = position_df.rename(columns=rename_map)

    required_cols = ["signal_ls", "target_weight_g", "target_weight_v"]
    missing_cols = [col for col in required_cols if col not in position_df.columns]
    if missing_cols:
        raise KeyError(f"position file is missing required columns: {missing_cols}")

    out = position_df[required_cols].copy()
    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    validate_position_df(out)
    return out


def validate_position_df(position_df: pd.DataFrame) -> None:
    non_null_signal = position_df["signal_ls"].notna()
    weight_sum = (
        position_df.loc[non_null_signal, "target_weight_g"]
        + position_df.loc[non_null_signal, "target_weight_v"]
    )
    invalid_weight_mask = (
        position_df.loc[non_null_signal, ["target_weight_g", "target_weight_v"]]
        .isna()
        .any(axis=1)
    )
    if invalid_weight_mask.any():
        bad_dates = invalid_weight_mask[invalid_weight_mask].index[:5].tolist()
        raise ValueError(f"non-null signal rows have missing target weights, examples: {bad_dates}")
    if not weight_sum.empty and not ((weight_sum - 1.0).abs() < 1e-10).all():
        bad_dates = weight_sum[(weight_sum - 1.0).abs() >= 1e-10].index[:5].tolist()
        raise ValueError(f"non-null target weights must sum to 1, examples: {bad_dates}")


def load_inputs(
    position_file: Path,
    market_path: Path,
    data_path: Path,
    benchmark_position_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not market_path.exists():
        raise FileNotFoundError(f"Cannot find market parquet: {market_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data parquet: {data_path}")
    if not benchmark_position_path.exists():
        raise FileNotFoundError(f"Cannot find benchmark position parquet: {benchmark_position_path}")

    position_df = read_position_file(position_file)
    market_df = pd.read_parquet(market_path).sort_index()
    data_df = pd.read_parquet(data_path).sort_index()
    benchmark_position_df = pd.read_parquet(benchmark_position_path).sort_index()

    for df in (market_df, data_df, benchmark_position_df):
        df.index = pd.to_datetime(df.index)

    return position_df, market_df, data_df, benchmark_position_df


def build_work_df(
    market_df: pd.DataFrame,
    position_df: pd.DataFrame,
    factor_col: str,
    sample_window: SampleWindow | None = None,
) -> pd.DataFrame:
    cols_to_keep = [
        TRACK_COL,
        "fwd_ret_g",
        "fwd_ret_v",
        "holding_days",
        "target_return_diff",
        "next_date",
        "target_label",
        "close_g",
        "close_v",
    ]
    missing_cols = [col for col in cols_to_keep if col not in market_df.columns]
    if missing_cols:
        raise KeyError(f"market_df is missing required columns: {missing_cols}")

    if sample_window is None:
        sample_window = resolve_sample_window(Config, sample="all")

    # 样本内外默认日期只在 Config 中维护；这里根据运行参数切分实际回测区间。
    sample_mask = (market_df.index >= sample_window.start) & (market_df.index <= sample_window.end)
    work_df = market_df.loc[sample_mask, cols_to_keep].copy().sort_index()
    missing_position_dates = work_df.index.difference(position_df.index)
    if len(missing_position_dates) > 0:
        examples = missing_position_dates[:5].tolist()
        raise ValueError(f"position file does not cover market dates, examples: {examples}")

    aligned_position = position_df.reindex(work_df.index)
    work_df[factor_col] = aligned_position["signal_ls"]
    work_df["signal_ls"] = aligned_position["signal_ls"]
    work_df["target_weight_g"] = aligned_position["target_weight_g"]
    work_df["target_weight_v"] = aligned_position["target_weight_v"]
    work_df["signal_update_flag"] = work_df["signal_ls"].notna().astype(int)
    work_df["next_date"] = pd.to_datetime(work_df["next_date"])
    work_df["fwd_ret_g_simple"] = to_simple_return(work_df["fwd_ret_g"], return_type=Config.RETURN_TYPE)
    work_df["fwd_ret_v_simple"] = to_simple_return(work_df["fwd_ret_v"], return_type=Config.RETURN_TYPE)
    work_df["fwd_ret_diff_simple"] = work_df["fwd_ret_g_simple"] - work_df["fwd_ret_v_simple"]
    return work_df


def validate_benchmark_position_df(
    benchmark_position_df: pd.DataFrame,
    required_index: pd.Index,
) -> pd.DataFrame:
    required_cols = ["post_trade_weight_g", "post_trade_weight_v", "signal_update_flag"]
    missing_cols = [col for col in required_cols if col not in benchmark_position_df.columns]
    if missing_cols:
        raise KeyError(f"benchmark position file is missing required columns: {missing_cols}")

    missing_dates = required_index.difference(benchmark_position_df.index)
    if len(missing_dates) > 0:
        examples = missing_dates[:5].tolist()
        raise ValueError(f"benchmark positions do not cover market dates, examples: {examples}")

    return benchmark_position_df.reindex(required_index).copy()


def build_strategy_target_df(work_df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    cols = [
        TRACK_COL,
        factor_col,
        "fwd_ret_g",
        "fwd_ret_v",
        "holding_days",
        "target_return_diff",
        "next_date",
        "target_label",
        "close_g",
        "close_v",
        "fwd_ret_g_simple",
        "fwd_ret_v_simple",
        "fwd_ret_diff_simple",
        "target_weight_g",
        "target_weight_v",
        "signal_ls",
        "signal_update_flag",
    ]
    strategy_target_df = work_df[cols].copy().sort_index()
    strategy_target_df[TRACK_COL] = strategy_target_df[TRACK_COL].astype("Int64")
    strategy_target_df["signal_update_flag"] = strategy_target_df["signal_update_flag"].astype(int)
    return strategy_target_df


def build_benchmark_target_df(
    work_df: pd.DataFrame,
    benchmark_position_df: pd.DataFrame,
) -> pd.DataFrame:
    benchmark_ref = validate_benchmark_position_df(benchmark_position_df, work_df.index)
    benchmark_target_df = work_df[
        [
            TRACK_COL,
            "fwd_ret_g_simple",
            "fwd_ret_v_simple",
            "holding_days",
            "next_date",
        ]
    ].copy()
    benchmark_target_df["target_weight_g"] = benchmark_ref["post_trade_weight_g"].astype(float)
    benchmark_target_df["target_weight_v"] = benchmark_ref["post_trade_weight_v"].astype(float)
    benchmark_target_df["signal_ls"] = 0.0
    benchmark_target_df["signal_update_flag"] = benchmark_ref["signal_update_flag"].fillna(0).astype(int)
    benchmark_target_df["post_trade_weight_g"] = benchmark_ref["post_trade_weight_g"].astype(float)
    benchmark_target_df["post_trade_weight_v"] = benchmark_ref["post_trade_weight_v"].astype(float)
    benchmark_target_df[TRACK_COL] = benchmark_target_df[TRACK_COL].astype("Int64")
    return benchmark_target_df


def run_single_track_benchmark_from_positions(
    track_target_df: pd.DataFrame,
    trans_fee: float,
    charge_initial_trade: bool,
) -> pd.DataFrame:
    track_df = track_target_df.copy().sort_index(ascending=True)
    result_rows = []

    prev_post_weight_g = np.nan
    prev_post_weight_v = np.nan
    prev_ret_g_simple = np.nan
    prev_ret_v_simple = np.nan
    has_prev = False

    for dt, row in track_df.iterrows():
        if not has_prev:
            pre_trade_weight_g = 0.0
            pre_trade_weight_v = 0.0
        else:
            pre_trade_weight_g, pre_trade_weight_v = calc_pre_trade_actual_weight(
                prev_post_weight_g=prev_post_weight_g,
                prev_post_weight_v=prev_post_weight_v,
                prev_ret_g_simple=prev_ret_g_simple,
                prev_ret_v_simple=prev_ret_v_simple,
            )

        post_trade_weight_g = float(row["post_trade_weight_g"])
        post_trade_weight_v = float(row["post_trade_weight_v"])
        need_trade = weights_are_different(
            pre_trade_weight_g,
            pre_trade_weight_v,
            post_trade_weight_g,
            post_trade_weight_v,
            WEIGHT_TOL,
        )
        trade_flag = int(need_trade)
        turnover_2way = (
            abs(post_trade_weight_g - pre_trade_weight_g)
            + abs(post_trade_weight_v - pre_trade_weight_v)
            if trade_flag == 1
            else 0.0
        )
        if (not has_prev) and (charge_initial_trade is False):
            cost_rate = 0.0
        else:
            cost_rate = turnover_2way * trans_fee

        fwd_ret_g_simple = float(row["fwd_ret_g_simple"])
        fwd_ret_v_simple = float(row["fwd_ret_v_simple"])
        period_ret_long_gross = (
            post_trade_weight_g * fwd_ret_g_simple
            + post_trade_weight_v * fwd_ret_v_simple
        )
        period_ret_long_net = (1 - cost_rate) * (1 + period_ret_long_gross) - 1

        result_rows.append(
            {
                TRACK_COL: row[TRACK_COL],
                "next_date": row["next_date"],
                "signal_ls_raw": row["signal_ls"],
                "effective_signal_ls": 0.0,
                "signal_update_flag": row["signal_update_flag"],
                "pre_trade_weight_g": pre_trade_weight_g,
                "pre_trade_weight_v": pre_trade_weight_v,
                "post_trade_weight_g": post_trade_weight_g,
                "post_trade_weight_v": post_trade_weight_v,
                "trade_flag": trade_flag,
                "signal_count": 1,
                "turnover_2way": turnover_2way,
                "cost_rate": cost_rate,
                "fwd_ret_g_simple": fwd_ret_g_simple,
                "fwd_ret_v_simple": fwd_ret_v_simple,
                "holding_days": row["holding_days"],
                "period_ret_long_gross": period_ret_long_gross,
                "period_ret_long_net": period_ret_long_net,
                "period_ret_ls": 0.0,
            }
        )

        prev_post_weight_g = post_trade_weight_g
        prev_post_weight_v = post_trade_weight_v
        prev_ret_g_simple = fwd_ret_g_simple
        prev_ret_v_simple = fwd_ret_v_simple
        has_prev = True

    return pd.DataFrame(result_rows, index=track_df.index)


def run_period_backtests(
    strategy_target_df: pd.DataFrame,
    benchmark_target_df: pd.DataFrame,
    track_list: list[int],
    trans_fee: float,
    charge_initial_trade: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_result_list = []
    for track_id in track_list:
        track_target_df = (
            benchmark_target_df[benchmark_target_df[TRACK_COL] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        benchmark_result_list.append(
            run_single_track_benchmark_from_positions(
                track_target_df=track_target_df,
                trans_fee=trans_fee,
                charge_initial_trade=charge_initial_trade,
            )
        )
    benchmark_period_df = pd.concat(benchmark_result_list, axis=0).sort_index(ascending=True)

    strategy_result_list = []
    for track_id in track_list:
        track_target_df = (
            strategy_target_df[strategy_target_df[TRACK_COL] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        benchmark_ref_track_df = (
            benchmark_period_df[benchmark_period_df[TRACK_COL] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        strategy_result_list.append(
            run_single_track_backtest(
                track_target_df=track_target_df,
                benchmark_mode=BENCHMARK_MODE,
                trans_fee=trans_fee,
                charge_initial_trade=charge_initial_trade,
                receiver="strategy",
                benchmark_ref_track_df=benchmark_ref_track_df,
            )
        )
    strategy_period_df = pd.concat(strategy_result_list, axis=0).sort_index(ascending=True)

    assert len(benchmark_period_df) == len(strategy_period_df), (
        "benchmark_period_df 与 strategy_period_df 行数不一致，请检查输入表是否对齐"
    )
    assert benchmark_period_df.index.equals(strategy_period_df.index), (
        "benchmark_period_df 与 strategy_period_df 的日期索引不一致，请检查输入表是否对齐"
    )
    assert (benchmark_period_df[TRACK_COL].values == strategy_period_df[TRACK_COL].values).all(), (
        "benchmark_period_df 与 strategy_period_df 的 track_id 结构不一致，请检查输入表是否对齐"
    )
    return strategy_period_df, benchmark_period_df


def build_daily_asset_ret_df(data_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["close_g", "close_v"]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise KeyError(f"data_df is missing required columns: {missing_cols}")

    daily_asset_ret_df = data_df[required_cols].copy().sort_index()
    daily_asset_ret_df["ret_g_daily_simple"] = daily_asset_ret_df["close_g"].pct_change()
    daily_asset_ret_df["ret_v_daily_simple"] = daily_asset_ret_df["close_v"].pct_change()
    return daily_asset_ret_df[["ret_g_daily_simple", "ret_v_daily_simple"]]


def build_nav_outputs(
    strategy_period_df: pd.DataFrame,
    benchmark_period_df: pd.DataFrame,
    daily_asset_ret_df: pd.DataFrame,
    track_list: list[int],
) -> tuple[dict[int, pd.Series], dict[int, pd.Series], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    strategy_track_nav_dict = {}
    benchmark_track_nav_dict = {}

    for track_id in track_list:
        strategy_track_df = strategy_period_df[strategy_period_df[TRACK_COL] == track_id].copy().sort_index()
        strategy_track_nav = build_single_track_daily_nav(
            track_period_df=strategy_track_df,
            daily_asset_ret_df=daily_asset_ret_df,
        )
        strategy_track_nav.name = f"track_{track_id}"
        strategy_track_nav_dict[track_id] = strategy_track_nav

        benchmark_track_df = benchmark_period_df[benchmark_period_df[TRACK_COL] == track_id].copy().sort_index()
        benchmark_track_nav = build_single_track_daily_nav(
            track_period_df=benchmark_track_df,
            daily_asset_ret_df=daily_asset_ret_df,
        )
        benchmark_track_nav.name = f"track_{track_id}"
        benchmark_track_nav_dict[track_id] = benchmark_track_nav

    strategy_track_nav_df = pd.concat(
        [strategy_track_nav_dict[track_id] for track_id in track_list],
        axis=1,
        join="outer",
    ).sort_index()
    strategy_track_nav_df = strategy_track_nav_df.dropna(how="any")
    strategy_combo_nav = strategy_track_nav_df.mean(axis=1)
    strategy_combo_nav.name = "strategy_combo_nav"

    benchmark_track_nav_df = pd.concat(
        [benchmark_track_nav_dict[track_id] for track_id in track_list],
        axis=1,
        join="outer",
    ).sort_index()
    benchmark_track_nav_df = benchmark_track_nav_df.dropna(how="any")
    benchmark_combo_nav = benchmark_track_nav_df.mean(axis=1)
    benchmark_combo_nav.name = "benchmark_combo_nav"

    return (
        strategy_track_nav_dict,
        benchmark_track_nav_dict,
        strategy_track_nav_df,
        benchmark_track_nav_df,
        strategy_combo_nav,
        benchmark_combo_nav,
    )


def build_summary_outputs(
    strategy_period_df: pd.DataFrame,
    benchmark_period_df: pd.DataFrame,
    strategy_track_nav_dict: dict[int, pd.Series],
    benchmark_track_nav_dict: dict[int, pd.Series],
    track_list: list[int],
    rf: float,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    track_summary_dict = {}
    for track_id in track_list:
        strategy_track_result_df = strategy_period_df[strategy_period_df[TRACK_COL] == track_id].copy().sort_index()
        benchmark_track_result_df = benchmark_period_df[benchmark_period_df[TRACK_COL] == track_id].copy().sort_index()
        track_summary_dict[track_id] = build_track_summary_table(
            strategy_track_result_df=strategy_track_result_df,
            benchmark_track_result_df=benchmark_track_result_df,
            strategy_track_nav=strategy_track_nav_dict[track_id],
            benchmark_track_nav=benchmark_track_nav_dict[track_id],
            rf=rf,
        )

    all_track_summary_df = pd.concat(
        [track_summary_dict[track_id] for track_id in track_list],
        axis=0,
        keys=track_list,
        names=[TRACK_COL, "period"],
    )
    avg_summary_df = all_track_summary_df.groupby(level="period", sort=False).mean()
    avg_summary_df = avg_summary_df.reindex(track_summary_dict[track_list[0]].index)

    full_period_df = pd.DataFrame(
        [track_summary_dict[track_id].loc["全区间"] for track_id in track_list],
        index=track_list,
    )
    std_summary_df = pd.DataFrame(full_period_df.std(axis=0, ddof=1)).T
    std_summary_df.index = ["五轨标准差"]
    return track_summary_dict, all_track_summary_df, avg_summary_df, std_summary_df


def build_screening_outputs(
    avg_summary_df: pd.DataFrame,
    all_track_summary_df: pd.DataFrame,
    factor_col: str,
) -> tuple[pd.DataFrame, int, float, str]:
    last_row = avg_summary_df.iloc[-1]
    screen_result_df = pd.DataFrame(
        {
            "condition": [
                "全区间发生调仓次数 > 15",
                "超额年化收益或绝对年化收益 > 0",
                "超额夏普 > 0",
                "成长/价值波段胜率均 > 50%",
                "Period胜率 > 52%",
                "盈亏比 > 1",
                "期望收益 > 0",
            ],
            "is_pass": [
                last_row["trade_count"] > 15,
                (last_row["ann_ret_ls_abs"] > 0) or (last_row["ann_ret_long_abs"] > 0),
                last_row["sharpe_excess"] > 0,
                (last_row["growth_regime_win_rate"] > 0.5)
                and (last_row["value_regime_win_rate"] > 0.5),
                last_row["period_win_rate"] > 0.52,
                last_row["payoff_ratio"] > 1,
                last_row["expectancy"] > 0,
            ],
        }
    )
    pass_count = int(screen_result_df["is_pass"].sum())
    result_period_label = avg_summary_df.index[-1]
    result_monthly_win_rate = float(
        all_track_summary_df.xs(result_period_label, level="period")["monthly_win_rate"].mean()
    )
    result_period_win_rate = float(last_row["period_win_rate"])
    result_payoff_ratio = float(last_row["payoff_ratio"])
    result_expectancy = float(last_row["expectancy"])
    monthly_win_rate_tag = (
        "MWnan"
        if pd.isna(result_monthly_win_rate)
        else f"mw{result_monthly_win_rate * 100:.2f}"
    )
    result_prefix = f"{pass_count}_{monthly_win_rate_tag}_{factor_col}"
    return (
        screen_result_df,
        pass_count,
        result_monthly_win_rate,
        result_period_win_rate,
        result_payoff_ratio,
        result_expectancy,
        result_prefix,
    )


def save_summary_excel(
    excel_path: Path,
    screen_result_df: pd.DataFrame,
    pass_count: int,
    result_monthly_win_rate: float,
    result_period_win_rate: float,
    result_payoff_ratio: float,
    result_expectancy: float,
    track_summary_dict: dict[int, pd.DataFrame],
    avg_summary_df: pd.DataFrame,
    std_summary_df: pd.DataFrame,
    track_list: list[int],
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        screen_result_df.to_excel(writer, sheet_name="screening", startrow=0, startcol=0, index=False)
        pass_count_df = pd.DataFrame(
            {
                "condition": [
                    "pass_count",
                    "monthly_win_rate",
                    "period_win_rate",
                    "payoff_ratio",
                    "expectancy",
                ],
                "is_pass": [
                    pass_count,
                    result_monthly_win_rate,
                    result_period_win_rate,
                    result_payoff_ratio,
                    result_expectancy,
                ],
            }
        )
        pass_count_df.to_excel(
            writer,
            sheet_name="screening",
            startrow=len(screen_result_df) + 2,
            startcol=0,
            index=False,
        )

        sheet_name = "summary"
        start_row = 0
        for track_id in track_list:
            pd.DataFrame([[f"track_{track_id}_summary"]]).to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=start_row,
                startcol=0,
                index=False,
                header=False,
            )
            track_summary_dict[track_id].to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=start_row + 1,
                startcol=0,
                index=True,
            )
            start_row = start_row + 1 + 1 + len(track_summary_dict[track_id]) + 1

        pd.DataFrame([["avg_summary"]]).to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=start_row,
            startcol=0,
            index=False,
            header=False,
        )
        avg_summary_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1, startcol=0, index=True)
        start_row = start_row + 1 + 1 + len(avg_summary_df) + 1

        pd.DataFrame([["std_summary"]]).to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=start_row,
            startcol=0,
            index=False,
            header=False,
        )
        std_summary_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1, startcol=0, index=True)


def save_combo_nav_plot(
    output_path: Path,
    factor_col: str,
    strategy_combo_nav: pd.Series,
    benchmark_combo_nav: pd.Series,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_combo_nav.index, strategy_combo_nav.values, label="strategy combo", linewidth=2)
    plt.plot(benchmark_combo_nav.index, benchmark_combo_nav.values, label="benchmark combo", linewidth=2)
    plt.title(f"Combo NAV - {factor_col} ({BENCHMARK_MODE})")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_combo_nav_ratio_plot(
    output_path: Path,
    factor_col: str,
    strategy_combo_nav: pd.Series,
    benchmark_combo_nav: pd.Series,
) -> None:
    combo_ratio_df = pd.concat([strategy_combo_nav, benchmark_combo_nav], axis=1, join="inner").sort_index()
    combo_ratio_df.columns = ["strategy_combo_nav", "benchmark_combo_nav"]
    if combo_ratio_df.empty:
        raise ValueError("五轨组合净值对齐后为空，请检查 strategy_combo_nav / benchmark_combo_nav")
    if (combo_ratio_df["benchmark_combo_nav"] == 0).any():
        raise ValueError("benchmark_combo_nav 存在 0，无法计算比值")

    combo_nav_ratio = combo_ratio_df["strategy_combo_nav"] / combo_ratio_df["benchmark_combo_nav"]
    plt.figure(figsize=(12, 6))
    plt.plot(combo_nav_ratio.index, combo_nav_ratio.values, label="strategy combo / benchmark combo", linewidth=2)
    plt.axhline(1.0, linestyle="--", linewidth=1, label="ratio = 1")
    plt.title(f"Combo NAV Ratio - {factor_col} ({BENCHMARK_MODE})")
    plt.xlabel("Date")
    plt.ylabel("NAV Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# 未来如不需要五轨 NAV 图，可将以下函数体和调用处注释掉。
def save_track_nav_plots(
    output_dir: Path,
    strategy_track_nav_dict: dict[int, pd.Series],
    benchmark_track_nav_dict: dict[int, pd.Series],
    track_list: list[int],
) -> None:
    for track_id in track_list:
        strategy_track_nav = strategy_track_nav_dict[track_id]
        benchmark_track_nav = benchmark_track_nav_dict[track_id]
        track_nav_path = output_dir / f"track_{track_id}_nav.png"
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_track_nav.index, strategy_track_nav.values, label=f"strategy track {track_id}", linewidth=2)
        plt.plot(benchmark_track_nav.index, benchmark_track_nav.values, label=f"benchmark track {track_id}", linewidth=2)
        plt.title(f"Track {track_id} NAV")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(track_nav_path, dpi=300, bbox_inches="tight")
        plt.close()


# 未来如不需要五轨 NAV ratio 图，可将以下函数体和调用处注释掉。
def save_track_nav_ratio_plots(
    output_dir: Path,
    strategy_track_nav_dict: dict[int, pd.Series],
    benchmark_track_nav_dict: dict[int, pd.Series],
    track_list: list[int],
) -> None:
    for track_id in track_list:
        track_ratio_df = pd.concat(
            [strategy_track_nav_dict[track_id], benchmark_track_nav_dict[track_id]],
            axis=1,
            join="inner",
        ).sort_index()
        track_ratio_df.columns = ["strategy_nav", "benchmark_nav"]
        if track_ratio_df.empty:
            raise ValueError(f"track {track_id} 对齐后为空，请检查净值序列")
        if (track_ratio_df["benchmark_nav"] == 0).any():
            raise ValueError(f"track {track_id} 的 benchmark_nav 存在 0，无法计算比值")

        track_nav_ratio = track_ratio_df["strategy_nav"] / track_ratio_df["benchmark_nav"]
        track_nav_ratio_path = output_dir / f"track_{track_id}_nav_ratio.png"
        plt.figure(figsize=(12, 6))
        plt.plot(track_nav_ratio.index, track_nav_ratio.values, label=f"track {track_id}: strategy / benchmark", linewidth=2)
        plt.axhline(1.0, linestyle="--", linewidth=1, label="ratio = 1")
        plt.title(f"Track {track_id} NAV Ratio")
        plt.xlabel("Date")
        plt.ylabel("NAV Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(track_nav_ratio_path, dpi=300, bbox_inches="tight")
        plt.close()


# 未来如不需要落盘中间表，可将以下函数体和调用处注释掉。
def save_intermediate_outputs(
    output_dir: Path,
    strategy_target_df: pd.DataFrame,
    benchmark_target_df: pd.DataFrame,
    strategy_period_df: pd.DataFrame,
    benchmark_period_df: pd.DataFrame,
    strategy_combo_nav: pd.Series,
    benchmark_combo_nav: pd.Series,
    avg_summary_df: pd.DataFrame,
) -> None:
    strategy_target_df.to_parquet(output_dir / "strategy_target_df.parquet")
    benchmark_target_df.to_parquet(output_dir / "benchmark_target_df.parquet")
    strategy_period_df.to_parquet(output_dir / "strategy_period_df.parquet")
    benchmark_period_df.to_parquet(output_dir / "benchmark_period_df.parquet")
    strategy_combo_nav.to_frame().to_csv(output_dir / "strategy_combo_nav.csv")
    benchmark_combo_nav.to_frame().to_csv(output_dir / "benchmark_combo_nav.csv")
    avg_summary_df.to_csv(output_dir / "avg_summary_df.csv")


def main() -> None:
    args = parse_args()
    position_file = resolve_project_path(args.position_file)
    market_path = resolve_project_path(args.market_path)
    data_path = resolve_project_path(args.data_path)
    benchmark_position_path = resolve_project_path(args.benchmark_position_path)
    output_root = resolve_project_path(args.output_root)
    sample_window = resolve_sample_window(
        Config,
        sample=args.sample,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    factor_col = infer_factor_col(position_file)
    trans_fee = float(Config.TRANS_FEE)
    charge_initial_trade = bool(Config.CHARGE_INITIAL_TRADE)
    rf = float(Config.RF)

    position_df, market_df, data_df, benchmark_position_df = load_inputs(
        position_file=position_file,
        market_path=market_path,
        data_path=data_path,
        benchmark_position_path=benchmark_position_path,
    )

    work_df = build_work_df(
        market_df=market_df,
        position_df=position_df,
        factor_col=factor_col,
        sample_window=sample_window,
    )
    track_list = sorted(work_df[TRACK_COL].dropna().astype(int).unique())
    if not track_list:
        raise ValueError("No valid track_id values found in work_df")

    strategy_target_df = build_strategy_target_df(work_df=work_df, factor_col=factor_col)
    benchmark_target_df = build_benchmark_target_df(work_df=work_df, benchmark_position_df=benchmark_position_df)
    strategy_period_df, benchmark_period_df = run_period_backtests(
        strategy_target_df=strategy_target_df,
        benchmark_target_df=benchmark_target_df,
        track_list=track_list,
        trans_fee=trans_fee,
        charge_initial_trade=charge_initial_trade,
    )

    zero_signal_mask = strategy_target_df["signal_ls"].eq(0)
    if zero_signal_mask.any():
        zero_signal_weights = strategy_period_df.loc[zero_signal_mask, ["post_trade_weight_g", "post_trade_weight_v"]]
        zero_benchmark_weights = benchmark_period_df.loc[zero_signal_mask, ["post_trade_weight_g", "post_trade_weight_v"]]
        max_zero_diff = (zero_signal_weights - zero_benchmark_weights).abs().to_numpy().max()
        if max_zero_diff > 1e-10:
            raise AssertionError(f"signal_ls=0 rows do not match benchmark weights, max diff={max_zero_diff}")

    daily_asset_ret_df = build_daily_asset_ret_df(data_df=data_df)
    (
        strategy_track_nav_dict,
        benchmark_track_nav_dict,
        strategy_track_nav_df,
        benchmark_track_nav_df,
        strategy_combo_nav,
        benchmark_combo_nav,
    ) = build_nav_outputs(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        daily_asset_ret_df=daily_asset_ret_df,
        track_list=track_list,
    )

    (
        track_summary_dict,
        all_track_summary_df,
        avg_summary_df,
        std_summary_df,
    ) = build_summary_outputs(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        strategy_track_nav_dict=strategy_track_nav_dict,
        benchmark_track_nav_dict=benchmark_track_nav_dict,
        track_list=track_list,
        rf=rf,
    )

    (
        screen_result_df,
        pass_count,
        result_monthly_win_rate,
        result_period_win_rate,
        result_payoff_ratio,
        result_expectancy,
        result_prefix,
    ) = build_screening_outputs(
        avg_summary_df=avg_summary_df,
        all_track_summary_df=all_track_summary_df,
        factor_col=factor_col,
    )

    output_dir = output_root / result_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    save_summary_excel(
        excel_path=output_dir / f"{result_prefix}_{BENCHMARK_MODE}_summary.xlsx",
        screen_result_df=screen_result_df,
        pass_count=pass_count,
        result_monthly_win_rate=result_monthly_win_rate,
        result_period_win_rate=result_period_win_rate,
        result_payoff_ratio=result_payoff_ratio,
        result_expectancy=result_expectancy,
        track_summary_dict=track_summary_dict,
        avg_summary_df=avg_summary_df,
        std_summary_df=std_summary_df,
        track_list=track_list,
    )
    save_combo_nav_plot(
        output_path=output_dir / f"{factor_col}_{BENCHMARK_MODE}_combo_nav.png",
        factor_col=factor_col,
        strategy_combo_nav=strategy_combo_nav,
        benchmark_combo_nav=benchmark_combo_nav,
    )
    save_combo_nav_ratio_plot(
        output_path=output_dir / f"{factor_col}_{BENCHMARK_MODE}_combo_nav_ratio.png",
        factor_col=factor_col,
        strategy_combo_nav=strategy_combo_nav,
        benchmark_combo_nav=benchmark_combo_nav,
    )

    # 未来如不需要五轨 NAV 图，可将以下调用注释掉。
    save_track_nav_plots(
        output_dir=output_dir,
        strategy_track_nav_dict=strategy_track_nav_dict,
        benchmark_track_nav_dict=benchmark_track_nav_dict,
        track_list=track_list,
    )
    # 未来如不需要五轨 NAV ratio 图，可将以下调用注释掉。
    save_track_nav_ratio_plots(
        output_dir=output_dir,
        strategy_track_nav_dict=strategy_track_nav_dict,
        benchmark_track_nav_dict=benchmark_track_nav_dict,
        track_list=track_list,
    )
    # 未来如不需要落盘中间表，可将以下调用注释掉。
    # save_intermediate_outputs(
    #     output_dir=output_dir,
    #     strategy_target_df=strategy_target_df,
    #     benchmark_target_df=benchmark_target_df,
    #     strategy_period_df=strategy_period_df,
    #     benchmark_period_df=benchmark_period_df,
    #     strategy_combo_nav=strategy_combo_nav,
    #     benchmark_combo_nav=benchmark_combo_nav,
    #     avg_summary_df=avg_summary_df,
    # )

    print(f"factor: {factor_col}")
    print(f"position file: {position_file}")
    print(f"sample: {sample_window.name} ({sample_window.start_text} -> {sample_window.end_text})")
    print(f"trans_fee: {trans_fee}")
    print(f"charge_initial_trade: {charge_initial_trade}")
    print(f"pass_count: {pass_count}")
    print(f"monthly_win_rate: {result_monthly_win_rate:.6f}")
    print(f"period_win_rate: {result_period_win_rate:.6f}")
    print(f"payoff_ratio: {result_payoff_ratio:.6f}")
    print(f"expectancy: {result_expectancy:.6f}")
    print(f"outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
