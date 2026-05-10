"""Run the legacy-compatible single-factor backtest chain for OW_F001 only."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"

try:
    import pandas as pd
    import numpy as np
except ModuleNotFoundError:
    if LOCAL_PYTHON.exists() and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv_mktp").resolve():
        os.execv(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv])
    raise


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config import Config
from backtest.returns import to_simple_return
from backtest.engine import run_single_track_backtest
from backtest.nav import build_single_track_daily_nav
from backtest.summary import build_track_summary_table
from config.single_factor_config import OW_F001_CN10Y_Q3Y as FACTOR_CONFIG
from data_io.loaders import load_data_and_market
from factors.raw_loader import ensure_raw_factor
from positions.benchmark import build_benchmark_rebalance_flag
from signals.mount import merge_factor_to_market
from signals.target_position import (
    build_event_target_position,
    build_state_target_position,
)


RAW_FACTOR_COL = FACTOR_CONFIG.raw_factor_col
FACTOR_NAME = FACTOR_CONFIG.factor_name
FACTOR_BAR = FACTOR_CONFIG.bar
SIGNAL_TYPE = FACTOR_CONFIG.signal_type
TRACK_COL = "track_id"

BENCHMARK_MODE = FACTOR_CONFIG.benchmark_mode
BENCHMARK_REBALANCE_MONTHS = FACTOR_CONFIG.benchmark_rebalance_months
TRANS_FEE = FACTOR_CONFIG.trans_fee
CHARGE_INITIAL_TRADE = FACTOR_CONFIG.charge_initial_trade

OUTPUT_DIR = FACTOR_CONFIG.output_dir


def build_work_df(market_df_all: pd.DataFrame, factor_available_date: pd.Timestamp) -> pd.DataFrame:
    cols_to_keep = [
        "track_id",
        FACTOR_NAME,
        "fwd_ret_g",
        "fwd_ret_v",
        "holding_days",
        "target_return_diff",
        "next_date",
        "target_label",
        "close_g",
        "close_v",
    ]
    missing_cols = [col for col in cols_to_keep if col not in market_df_all.columns]
    if missing_cols:
        raise KeyError(f"market_df_all is missing required columns: {missing_cols}")

    work_df = market_df_all[market_df_all.index > factor_available_date].copy().sort_index(ascending=True)
    work_df = work_df[cols_to_keep].copy()
    work_df["next_date"] = pd.to_datetime(work_df["next_date"])
    work_df["fwd_ret_g_simple"] = to_simple_return(work_df["fwd_ret_g"], return_type=Config.RETURN_TYPE)
    work_df["fwd_ret_v_simple"] = to_simple_return(work_df["fwd_ret_v"], return_type=Config.RETURN_TYPE)
    work_df["fwd_ret_diff_simple"] = work_df["fwd_ret_g_simple"] - work_df["fwd_ret_v_simple"]
    return work_df


def build_benchmark_target_df(
    work_df: pd.DataFrame,
    track_list: list[int],
) -> pd.DataFrame:
    cols_to_copy = [
        "track_id",
        FACTOR_NAME,
        "fwd_ret_g_simple",
        "fwd_ret_v_simple",
        "holding_days",
        "next_date",
    ]
    benchmark_target_df = work_df[cols_to_copy].copy()
    benchmark_target_df["target_weight_g"] = 0.5
    benchmark_target_df["target_weight_v"] = 0.5
    benchmark_target_df["signal_ls"] = 0
    benchmark_target_df["signal_update_flag"] = 0

    for track_id in track_list:
        mask = benchmark_target_df["track_id"] == track_id
        track_index = benchmark_target_df.loc[mask].index

        if BENCHMARK_MODE == "buy_and_hold_50":
            rebalance_flag = pd.Series(0, index=track_index, dtype="int64")
            rebalance_flag.iloc[0] = 1
        elif BENCHMARK_MODE == "rebalance_50":
            rebalance_flag = build_benchmark_rebalance_flag(
                track_index=track_index,
                rebalance_months=BENCHMARK_REBALANCE_MONTHS,
            )
        else:
            raise ValueError(
                f"benchmark_mode 只能是 'buy_and_hold_50' 或 'rebalance_50'，当前值为：{BENCHMARK_MODE}"
            )

        benchmark_target_df.loc[mask, "signal_update_flag"] = rebalance_flag.values

    benchmark_target_df["signal_update_flag"] = benchmark_target_df["signal_update_flag"].astype(int)
    return benchmark_target_df


def build_strategy_target_df(work_df: pd.DataFrame, track_list: list[int]) -> pd.DataFrame:
    all_track_results = []
    for track_id in track_list:
        track_df = work_df[work_df[TRACK_COL] == track_id].copy().sort_index(ascending=True)
        if SIGNAL_TYPE == "state":
            track_result = build_state_target_position(track_df, factor_col=FACTOR_NAME, bar=FACTOR_BAR)
        elif SIGNAL_TYPE == "event":
            track_result = build_event_target_position(track_df, event_col=FACTOR_NAME, bar=FACTOR_BAR)
        else:
            raise ValueError(f"signal_type 只能是 'state' 或 'event'，当前值是: {SIGNAL_TYPE}")
        all_track_results.append(track_result)

    strategy_target_df = pd.concat(all_track_results, axis=0)
    strategy_target_df = strategy_target_df.sort_index(ascending=True)
    strategy_target_df["track_id"] = strategy_target_df["track_id"].astype("Int64")
    return strategy_target_df


def run_period_backtests(
    strategy_target_df: pd.DataFrame,
    benchmark_target_df: pd.DataFrame,
    track_list: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_result_list = []
    for track_id in track_list:
        track_target_df = (
            benchmark_target_df[benchmark_target_df["track_id"] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        track_period_df = run_single_track_backtest(
            track_target_df=track_target_df,
            benchmark_mode=BENCHMARK_MODE,
            trans_fee=TRANS_FEE,
            charge_initial_trade=CHARGE_INITIAL_TRADE,
            receiver="benchmark",
            benchmark_ref_track_df=None,
        )
        benchmark_result_list.append(track_period_df)

    benchmark_period_df = pd.concat(benchmark_result_list, axis=0).sort_index(ascending=True)

    strategy_result_list = []
    for track_id in track_list:
        track_target_df = (
            strategy_target_df[strategy_target_df["track_id"] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        benchmark_ref_track_df = (
            benchmark_period_df[benchmark_period_df["track_id"] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        track_period_df = run_single_track_backtest(
            track_target_df=track_target_df,
            benchmark_mode=BENCHMARK_MODE,
            trans_fee=TRANS_FEE,
            charge_initial_trade=CHARGE_INITIAL_TRADE,
            receiver="strategy",
            benchmark_ref_track_df=benchmark_ref_track_df,
        )
        strategy_result_list.append(track_period_df)

    strategy_period_df = pd.concat(strategy_result_list, axis=0).sort_index(ascending=True)

    assert len(benchmark_period_df) == len(strategy_period_df), (
        "benchmark_period_df 与 strategy_period_df 行数不一致，请检查输入表是否对齐"
    )
    assert benchmark_period_df.index.equals(strategy_period_df.index), (
        "benchmark_period_df 与 strategy_period_df 的日期索引不一致，请检查输入表是否对齐"
    )
    assert (benchmark_period_df["track_id"].values == strategy_period_df["track_id"].values).all(), (
        "benchmark_period_df 与 strategy_period_df 的 track_id 结构不一致，请检查输入表是否对齐"
    )

    return strategy_period_df, benchmark_period_df


def build_daily_asset_ret_df(data_df: pd.DataFrame) -> pd.DataFrame:
    daily_asset_ret_df = data_df[["close_g", "close_v"]].copy().sort_index(ascending=True)
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
        strategy_track_df = (
            strategy_period_df[strategy_period_df["track_id"] == track_id]
            .copy()
            .sort_index()
        )
        strategy_track_nav = build_single_track_daily_nav(
            track_period_df=strategy_track_df,
            daily_asset_ret_df=daily_asset_ret_df,
        )
        strategy_track_nav.name = f"track_{track_id}"
        strategy_track_nav_dict[track_id] = strategy_track_nav

        benchmark_track_df = (
            benchmark_period_df[benchmark_period_df["track_id"] == track_id]
            .copy()
            .sort_index()
        )
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


def build_avg_summary_df(
    strategy_period_df: pd.DataFrame,
    benchmark_period_df: pd.DataFrame,
    strategy_track_nav_dict: dict[int, pd.Series],
    benchmark_track_nav_dict: dict[int, pd.Series],
    track_list: list[int],
) -> pd.DataFrame:
    track_summary_dict = {}
    for track_id in track_list:
        strategy_track_result_df = (
            strategy_period_df[strategy_period_df["track_id"] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        benchmark_track_result_df = (
            benchmark_period_df[benchmark_period_df["track_id"] == track_id]
            .copy()
            .sort_index(ascending=True)
        )
        track_summary_dict[track_id] = build_track_summary_table(
            strategy_track_result_df=strategy_track_result_df,
            benchmark_track_result_df=benchmark_track_result_df,
            strategy_track_nav=strategy_track_nav_dict[track_id],
            benchmark_track_nav=benchmark_track_nav_dict[track_id],
        )

    all_track_summary_df = pd.concat(
        [track_summary_dict[track_id] for track_id in track_list],
        axis=0,
        keys=track_list,
        names=["track_id", "period"],
    )
    avg_summary_df = all_track_summary_df.groupby(level="period", sort=False).mean()
    return avg_summary_df.reindex(track_summary_dict[track_list[0]].index)


def print_value_counts(name: str, series: pd.Series) -> None:
    print(f"{name}:")
    print(series.value_counts(dropna=False).sort_index().to_string())


def save_outputs(
    strategy_target_df: pd.DataFrame,
    benchmark_target_df: pd.DataFrame,
    strategy_period_df: pd.DataFrame,
    benchmark_period_df: pd.DataFrame,
    strategy_combo_nav: pd.Series,
    benchmark_combo_nav: pd.Series,
    avg_summary_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_target_df.to_parquet(OUTPUT_DIR / "strategy_target_df.parquet")
    benchmark_target_df.to_parquet(OUTPUT_DIR / "benchmark_target_df.parquet")
    strategy_period_df.to_parquet(OUTPUT_DIR / "strategy_period_df.parquet")
    benchmark_period_df.to_parquet(OUTPUT_DIR / "benchmark_period_df.parquet")
    strategy_combo_nav.to_frame().to_csv(OUTPUT_DIR / "strategy_combo_nav.csv")
    benchmark_combo_nav.to_frame().to_csv(OUTPUT_DIR / "benchmark_combo_nav.csv")
    avg_summary_df.to_csv(OUTPUT_DIR / "avg_summary_df.csv")


def main() -> None:
    data_df, market_df, data_path, market_path = load_data_and_market(PROJECT_ROOT)
    print(f"data path: {data_path}")
    print(f"market path: {market_path}")

    print(f"data_df shape: {data_df.shape}")
    print(f"market_df shape: {market_df.shape}")

    data_df = ensure_raw_factor(data_df=data_df, raw_factor_col=RAW_FACTOR_COL, project_root=PROJECT_ROOT)

    factor_non_null = int(data_df[RAW_FACTOR_COL].notna().sum())
    factor_available_date = data_df[RAW_FACTOR_COL].first_valid_index()
    if factor_available_date is None:
        raise ValueError(f"{RAW_FACTOR_COL} 全部为空，无法运行单因子回测。")

    print(f"{RAW_FACTOR_COL} non_null_count: {factor_non_null}")
    print(f"factor_available_date: {factor_available_date}")

    data_df[RAW_FACTOR_COL] = data_df[RAW_FACTOR_COL].replace([np.inf, -np.inf], np.nan)

    market_df = merge_factor_to_market(
        data_df=data_df,
        market_df=market_df,
        raw_factor_col=RAW_FACTOR_COL,
        factor_type=SIGNAL_TYPE,
    )

    all_mask = (market_df.index >= Config.ALL_START) & (market_df.index <= Config.ALL_END)
    market_df_all = market_df.loc[all_mask].copy()

    track_list = sorted(market_df[TRACK_COL].dropna().astype(int).unique())
    work_df = build_work_df(market_df_all=market_df_all, factor_available_date=factor_available_date)
    print(f"work_df shape: {work_df.shape}")
    print(f"work_df next_date NaT count: {int(work_df['next_date'].isna().sum())}")
    print("work_df rows by track_id:")
    print(work_df["track_id"].value_counts(dropna=False).sort_index().to_string())

    benchmark_target_df = build_benchmark_target_df(work_df=work_df, track_list=track_list)
    strategy_target_df = build_strategy_target_df(work_df=work_df, track_list=track_list)
    print(f"strategy_target_df shape: {strategy_target_df.shape}")
    print(f"benchmark_target_df shape: {benchmark_target_df.shape}")
    print("strategy_target_df rows by track_id:")
    print(strategy_target_df["track_id"].value_counts(dropna=False).sort_index().to_string())
    print(f"strategy_target_df next_date NaT count: {int(strategy_target_df['next_date'].isna().sum())}")
    print_value_counts("strategy_target_df signal_ls distribution", strategy_target_df["signal_ls"])
    print_value_counts("strategy_target_df signal_update_flag distribution", strategy_target_df["signal_update_flag"])
    print_value_counts("benchmark_target_df signal_update_flag distribution", benchmark_target_df["signal_update_flag"])

    strategy_period_df, benchmark_period_df = run_period_backtests(
        strategy_target_df=strategy_target_df,
        benchmark_target_df=benchmark_target_df,
        track_list=track_list,
    )

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

    avg_summary_df = build_avg_summary_df(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        strategy_track_nav_dict=strategy_track_nav_dict,
        benchmark_track_nav_dict=benchmark_track_nav_dict,
        track_list=track_list,
    )

    print(f"strategy_combo_nav start/end: {strategy_combo_nav.index.min()} -> {strategy_combo_nav.index.max()}")
    print(f"benchmark_combo_nav start/end: {benchmark_combo_nav.index.min()} -> {benchmark_combo_nav.index.max()}")

    save_outputs(
        strategy_target_df=strategy_target_df,
        benchmark_target_df=benchmark_target_df,
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        strategy_combo_nav=strategy_combo_nav,
        benchmark_combo_nav=benchmark_combo_nav,
        avg_summary_df=avg_summary_df,
    )
    print(f"outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
