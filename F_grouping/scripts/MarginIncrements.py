"""Scan marginal contribution from adding one candidate factor to the base COMB.

Usage:
    python F_grouping/scripts/MarginIncrements.py

This script does not regenerate underlying factors and does not write to output_COMB.
It builds in-memory grouped binary signals, maps them to target positions, then
reuses the existing backtest and summary functions.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ========== 1. Central config ==========

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

BASE_FACTORS = ["I001","I002","I006","G026","G027","P035","V168","V004"]

INPUT_DIR = PROJECT_ROOT / "F_grouping" / "input_COMB"
USABLE_FACTORS_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "usable_factors.xlsx"
OUTPUT_ROOT = PROJECT_ROOT / "F_grouping" / "marginal_test"
TOTAL_OUTPUT_DIR = OUTPUT_ROOT / "output"

MIN_NON_NULL = 30
MIN_PASS_COUNT_FOR_CANDIDATE_OUTPUT = 3
GROUP_NAME_BASE = "BASE"

MARKET_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "market_df.parquet"
DATA_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "data_df.parquet"
BENCHMARK_POSITION_PATH = PROJECT_ROOT / "C_positions" / "reference" / "benchmark_positions.parquet"

SUMMARY_OUTPUT_NAME = "marginal_contribution_summary.xlsx"
NAV_OUTPUT_NAME = "nav_curves.parquet"

METRIC_SPECS = [
    ("annual_return", "ann_ret_long_abs", "gt"),
    ("sharpe", "sharpe_abs", "gt"),
    ("max_dd", "max_dd_abs", "lt"),
    ("win_rate", "monthly_win_rate", "gt"),
    ("value_win_rate", "value_regime_win_rate", "gt"),
    ("expectancy", "expectancy", "gt"),
]


# ========== 2. Import existing project functions ==========

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "SY_Baseline",
    PROJECT_ROOT / "F_grouping" / "scripts",
    PROJECT_ROOT / "C_positions" / "scripts",
    PROJECT_ROOT / "E_backtesting" / "scripts",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Config import Config  # noqa: E402
from G_engine.sample_window import SampleWindow, resolve_sample_window  # noqa: E402
from build_signal_group_binary_interactive import (  # noqa: E402
    build_factor_matrix,
    build_signal_ls_matrix,
    collect_selected_factors,
    list_input_files,
)
from generate_factor_positions import build_factor_position_df, validate_factor_position_df  # noqa: E402
from backtesting import (  # noqa: E402
    TRACK_COL,
    build_benchmark_target_df,
    build_daily_asset_ret_df,
    build_nav_outputs,
    build_strategy_target_df,
    build_summary_outputs,
    build_work_df,
    run_period_backtests,
)
from postprocess_marginal_summary import enrich_summary, write_summary  # noqa: E402


# ========== 3. Basic helpers ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan marginal contribution of adding each usable factor to the fixed base COMB."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of non-base candidates to process.",
    )
    parser.add_argument(
        "--min-non-null",
        type=int,
        default=MIN_NON_NULL,
        help="Minimum non-null signal count required for a candidate.",
    )
    parser.add_argument("--sample", choices=["all", "ins", "oos", "custom"], default="all")
    parser.add_argument("--start-date", help="Optional sample start date, e.g. 2020-01-01.")
    parser.add_argument("--end-date", help="Optional sample end date, e.g. 2024-12-31.")
    return parser.parse_args()


def load_usable_candidates(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find usable factor file: {path}")

    df = pd.read_excel(path)
    if "factor_id" not in df.columns:
        raise KeyError(f"{path} is missing required column: factor_id")

    candidates = (
        df["factor_id"]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    candidates = [factor for factor in candidates if factor]
    return list(dict.fromkeys(candidates))


def load_backtest_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_paths = [
        path
        for path in (MARKET_PATH, DATA_PATH, BENCHMARK_POSITION_PATH)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing backtest input files: {missing_paths}")

    market_df = pd.read_parquet(MARKET_PATH).sort_index()
    data_df = pd.read_parquet(DATA_PATH).sort_index()
    benchmark_position_df = pd.read_parquet(BENCHMARK_POSITION_PATH).sort_index()

    for df in (market_df, data_df, benchmark_position_df):
        df.index = pd.to_datetime(df.index)

    return market_df, data_df, benchmark_position_df


def build_group_signal(
    factor_names: list[str],
    group_name: str,
    input_files: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_df, _scan_record, scan_warnings = collect_selected_factors(
        factor_names=factor_names,
        input_files=input_files,
    )
    factor_df, _groups, factor_warnings = build_factor_matrix(
        source_df=source_df,
        base_group=group_name,
        signal_group=f"{group_name}_signal",
        binary_group=group_name,
    )
    warnings = scan_warnings + factor_warnings
    signal_ls_df = build_signal_ls_matrix(factor_df=factor_df)
    return source_df, signal_ls_df, warnings


def run_combo_backtest(
    combo_name: str,
    signal_series: pd.Series,
    market_df: pd.DataFrame,
    data_df: pd.DataFrame,
    benchmark_position_df: pd.DataFrame,
    sample_window: SampleWindow,
) -> dict[str, object]:
    position_df = build_factor_position_df(signal_series)
    validate_factor_position_df(combo_name, position_df)
    position_df = position_df.rename(
        columns={
            "target_growth_weight": "target_weight_g",
            "target_value_weight": "target_weight_v",
        }
    )

    work_df = build_work_df(
        market_df=market_df,
        position_df=position_df,
        factor_col=combo_name,
        sample_window=sample_window,
    )
    track_list = sorted(work_df[TRACK_COL].dropna().astype(int).unique())
    if not track_list:
        raise ValueError("No valid track_id values found in work_df")

    strategy_target_df = build_strategy_target_df(work_df=work_df, factor_col=combo_name)
    benchmark_target_df = build_benchmark_target_df(
        work_df=work_df,
        benchmark_position_df=benchmark_position_df,
    )
    strategy_period_df, benchmark_period_df = run_period_backtests(
        strategy_target_df=strategy_target_df,
        benchmark_target_df=benchmark_target_df,
        track_list=track_list,
        trans_fee=float(Config.TRANS_FEE),
        charge_initial_trade=bool(Config.CHARGE_INITIAL_TRADE),
    )

    zero_signal_mask = strategy_target_df["signal_ls"].eq(0)
    if zero_signal_mask.any():
        zero_signal_weights = strategy_period_df.loc[
            zero_signal_mask,
            ["post_trade_weight_g", "post_trade_weight_v"],
        ]
        zero_benchmark_weights = benchmark_period_df.loc[
            zero_signal_mask,
            ["post_trade_weight_g", "post_trade_weight_v"],
        ]
        max_zero_diff = (zero_signal_weights - zero_benchmark_weights).abs().to_numpy().max()
        if max_zero_diff > 1e-10:
            raise AssertionError(f"signal_ls=0 rows do not match benchmark weights, max diff={max_zero_diff}")

    daily_asset_ret_df = build_daily_asset_ret_df(data_df=data_df)
    (
        strategy_track_nav_dict,
        benchmark_track_nav_dict,
        _strategy_track_nav_df,
        _benchmark_track_nav_df,
        strategy_combo_nav,
        benchmark_combo_nav,
    ) = build_nav_outputs(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        daily_asset_ret_df=daily_asset_ret_df,
        track_list=track_list,
    )
    (
        _track_summary_dict,
        all_track_summary_df,
        avg_summary_df,
        _std_summary_df,
    ) = build_summary_outputs(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        strategy_track_nav_dict=strategy_track_nav_dict,
        benchmark_track_nav_dict=benchmark_track_nav_dict,
        track_list=track_list,
        rf=float(Config.RF),
    )

    return {
        "position_df": position_df,
        "work_df": work_df,
        "strategy_period_df": strategy_period_df,
        "benchmark_period_df": benchmark_period_df,
        "strategy_combo_nav": strategy_combo_nav,
        "benchmark_combo_nav": benchmark_combo_nav,
        "avg_summary_df": avg_summary_df,
        "all_track_summary_df": all_track_summary_df,
    }


def get_full_period_metrics(avg_summary_df: pd.DataFrame) -> pd.Series:
    if "全区间" not in avg_summary_df.index:
        raise KeyError("avg_summary_df is missing full-period row: 全区间")

    row = avg_summary_df.loc["全区间"]
    missing_cols = [source_col for _label, source_col, _op in METRIC_SPECS if source_col not in row.index]
    if missing_cols:
        raise KeyError(f"avg_summary_df full-period row is missing columns: {missing_cols}")
    return row


def compare_metrics(base_row: pd.Series, new_row: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {}
    pass_fields: list[str] = []

    for label, source_col, op in METRIC_SPECS:
        base_value = pd.to_numeric(base_row[source_col], errors="coerce")
        new_value = pd.to_numeric(new_row[source_col], errors="coerce")
        delta_value = new_value - base_value
        if pd.isna(base_value) or pd.isna(new_value):
            is_pass = False
        elif op == "gt":
            is_pass = bool(new_value > base_value)
        elif op == "lt":
            is_pass = bool(new_value < base_value)
        else:
            raise ValueError(f"Unsupported metric comparison op: {op}")

        result[f"base_{label}"] = base_value
        result[f"new_{label}"] = new_value
        result[f"delta_{label}"] = delta_value
        result[f"pass_{label}"] = is_pass
        pass_fields.append(f"pass_{label}")

    result["pass_count"] = int(sum(bool(result[field]) for field in pass_fields))
    return result


def calc_signal_diagnostics(
    candidate_factor: str,
    candidate_source_df: pd.DataFrame,
    base_signal: pd.Series,
    new_signal: pd.Series,
    market_df: pd.DataFrame,
) -> dict[str, float]:
    candidate_signal = pd.to_numeric(candidate_source_df[candidate_factor], errors="coerce")
    aligned = pd.concat(
        [
            candidate_signal.rename("candidate_signal"),
            pd.to_numeric(base_signal, errors="coerce").rename("base_signal"),
            pd.to_numeric(new_signal, errors="coerce").rename("new_signal"),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    valid_corr = aligned[["candidate_signal", "base_signal"]].dropna()
    signal_corr = (
        float(valid_corr["candidate_signal"].corr(valid_corr["base_signal"]))
        if len(valid_corr) >= 2
        else np.nan
    )

    direction_df = aligned[["candidate_signal", "base_signal", "new_signal"]].dropna()
    direction_df = direction_df[
        (direction_df["candidate_signal"] != 0)
        & (direction_df["base_signal"] != 0)
    ]
    if direction_df.empty:
        disagreement_ratio = np.nan
        disagreement_win_rate = np.nan
    else:
        disagreement_mask = np.sign(direction_df["candidate_signal"]) != np.sign(direction_df["base_signal"])
        disagreement_ratio = float(disagreement_mask.mean())
        disagreement_dates = direction_df.index[disagreement_mask]
        if len(disagreement_dates) == 0 or "target_return_diff" not in market_df.columns:
            # Retain the column as NaN if there are no disagreement samples, or if
            # the market target needed for hit-rate diagnostics is unavailable.
            disagreement_win_rate = np.nan
        else:
            target = pd.to_numeric(market_df["target_return_diff"], errors="coerce")
            hit_df = pd.concat(
                [
                    np.sign(direction_df.loc[disagreement_dates, "new_signal"]).rename("new_direction"),
                    target.reindex(disagreement_dates).rename("target_return_diff"),
                ],
                axis=1,
            ).dropna()
            hit_df = hit_df[hit_df["target_return_diff"] != 0]
            disagreement_win_rate = (
                float((hit_df["new_direction"] == np.sign(hit_df["target_return_diff"])).mean())
                if not hit_df.empty
                else np.nan
            )

    return {
        "signal_corr_with_base": signal_corr,
        "disagreement_ratio": disagreement_ratio,
        "disagreement_win_rate": disagreement_win_rate,
    }


def empty_summary_row(candidate_factor: str, skip_reason: str) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_factor": candidate_factor,
        "status": "skipped",
        "skip_reason": skip_reason,
        "signal_corr_with_base": np.nan,
        "disagreement_ratio": np.nan,
        "disagreement_win_rate": np.nan,
        "pass_count": 0,
    }
    for label, _source_col, _op in METRIC_SPECS:
        row[f"base_{label}"] = np.nan
        row[f"new_{label}"] = np.nan
        row[f"delta_{label}"] = np.nan
        row[f"pass_{label}"] = False
    return row


def safe_float_tag(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "nan"
    return f"{float(number):.6f}"


def safe_folder_name(candidate_factor: str, pass_count: object, expectancy: object) -> str:
    pass_count_value = pd.to_numeric(pass_count, errors="coerce")
    pass_count_tag = "nan" if pd.isna(pass_count_value) else str(int(pass_count_value))
    raw_name = f"{pass_count_tag}_BASE+{candidate_factor}_{safe_float_tag(expectancy)}"
    return re.sub(r"[^A-Za-z0-9_.+=-]+", "_", raw_name)


def nav_curve_frame(
    candidate_factor: str,
    base_nav: pd.Series,
    new_nav: pd.Series,
) -> pd.DataFrame:
    nav_df = pd.concat(
        [
            base_nav.rename("base_nav"),
            new_nav.rename("new_nav"),
        ],
        axis=1,
        join="inner",
    ).sort_index()
    nav_df.index.name = "date"
    nav_df = nav_df.reset_index()
    nav_df.insert(1, "candidate_factor", candidate_factor)
    return nav_df[["date", "base_nav", "candidate_factor", "new_nav"]]


def write_candidate_outputs(
    candidate_factor: str,
    summary_row: dict[str, object],
    nav_df: pd.DataFrame,
    output_root: Path,
) -> Path | None:
    pass_count = int(summary_row.get("pass_count", 0) or 0)
    if pass_count < MIN_PASS_COUNT_FOR_CANDIDATE_OUTPUT:
        return None

    output_dir = output_root / safe_folder_name(
        candidate_factor=candidate_factor,
        pass_count=pass_count,
        expectancy=summary_row.get("new_expectancy"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary_row]).to_excel(output_dir / SUMMARY_OUTPUT_NAME, index=False)
    nav_df.to_parquet(output_dir / NAV_OUTPUT_NAME, index=False)
    return output_dir


def sort_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["pass_count", "delta_sharpe", "delta_annual_return", "delta_max_dd"]
    for col in sort_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    return summary_df.sort_values(
        by=sort_cols,
        ascending=[False, False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def print_top20(summary_df: pd.DataFrame) -> None:
    tested_df = summary_df[summary_df["status"].eq("tested")].copy()
    top_cols = [
        "candidate_factor",
        "pass_count",
        "delta_sharpe",
        "delta_annual_return",
        "delta_max_dd",
        "new_expectancy",
    ]
    available_cols = [col for col in top_cols if col in tested_df.columns]
    if tested_df.empty:
        print("No successfully tested candidates.")
        return
    print("\n========== pass_count top 20 ==========")
    print(tested_df.loc[:, available_cols].head(20).to_string(index=False))


# ========== 4. Main scan ==========

def main() -> tuple[Path, Path]:
    args = parse_args()
    min_non_null = int(args.min_non_null)
    sample_window = resolve_sample_window(
        Config,
        sample=args.sample,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    sample_output_root = OUTPUT_ROOT / sample_window.name
    total_output_dir = sample_output_root / "output"
    graph_output_dir = sample_output_root / "output_graph"

    input_files = list_input_files(INPUT_DIR)
    candidates = load_usable_candidates(USABLE_FACTORS_PATH)
    candidates = [
        factor
        for factor in candidates
        if factor not in set(BASE_FACTORS) and not factor.startswith("W")
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    print("========== MarginIncrements ==========")
    print("base_factors:", BASE_FACTORS)
    print("candidate_count:", len(candidates))
    print("input_dir:", INPUT_DIR)
    print("output_root:", sample_output_root)
    print(f"sample: {sample_window.name} ({sample_window.start_text} -> {sample_window.end_text})")
    print("min_non_null:", min_non_null)

    print("\n========== Load backtest inputs ==========")
    market_df, data_df, benchmark_position_df = load_backtest_inputs()
    # 样本内外默认日期只在 Config 中维护；边际贡献扫描使用同一个窗口。
    daily_market_index = market_df.loc[
        (market_df.index >= sample_window.start) & (market_df.index <= sample_window.end)
    ].index

    print("\n========== Build and backtest base once ==========")
    base_source_df, base_signal_df, base_warnings = build_group_signal(
        factor_names=BASE_FACTORS,
        group_name=GROUP_NAME_BASE,
        input_files=input_files,
    )
    if base_warnings:
        for warning in base_warnings:
            print("BASE WARNING:", warning)

    base_signal = base_signal_df[GROUP_NAME_BASE]
    if int(base_signal.notna().sum()) < min_non_null:
        raise ValueError(f"Base signal has fewer than {min_non_null} non-null samples.")

    base_result = run_combo_backtest(
        combo_name=GROUP_NAME_BASE,
        signal_series=base_signal,
        market_df=market_df,
        data_df=data_df,
        benchmark_position_df=benchmark_position_df,
        sample_window=sample_window,
    )
    base_full_row = get_full_period_metrics(base_result["avg_summary_df"])
    base_nav = base_result["strategy_combo_nav"]
    print("Base full-period metrics:")
    print(base_full_row[[source_col for _label, source_col, _op in METRIC_SPECS]].to_string())

    summary_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    tested_count = 0
    skipped_count = 0

    for idx, candidate_factor in enumerate(candidates, start=1):
        print(f"\n[{idx}/{len(candidates)}] testing candidate: {candidate_factor}")

        try:
            combo_factors = [*BASE_FACTORS, candidate_factor]
            combo_name = f"BASE+{candidate_factor}"
            candidate_source_df, new_signal_df, group_warnings = build_group_signal(
                factor_names=combo_factors,
                group_name=combo_name,
                input_files=input_files,
            )
            new_signal = new_signal_df[combo_name]

            candidate_non_null = int(candidate_source_df[candidate_factor].notna().sum())
            new_non_null = int(new_signal.notna().sum())
            if candidate_non_null < min_non_null:
                raise ValueError(f"candidate signal non-null count {candidate_non_null} < {min_non_null}")
            if new_non_null < min_non_null:
                raise ValueError(f"combo signal non-null count {new_non_null} < {min_non_null}")

            missing_dates = daily_market_index.difference(new_signal.index)
            if len(missing_dates) > 0:
                examples = [str(x.date()) for x in missing_dates[:5]]
                raise ValueError(f"combo signal does not cover market dates, examples: {examples}")

            new_result = run_combo_backtest(
                combo_name=combo_name,
                signal_series=new_signal,
                market_df=market_df,
                data_df=data_df,
                benchmark_position_df=benchmark_position_df,
                sample_window=sample_window,
            )
            new_full_row = get_full_period_metrics(new_result["avg_summary_df"])

            row = {
                "candidate_factor": candidate_factor,
                "status": "tested",
                "skip_reason": "",
                "candidate_non_null": candidate_non_null,
                "combo_non_null": new_non_null,
                "warnings": "; ".join(group_warnings),
            }
            row.update(compare_metrics(base_full_row, new_full_row))
            row.update(
                calc_signal_diagnostics(
                    candidate_factor=candidate_factor,
                    candidate_source_df=candidate_source_df,
                    base_signal=base_signal,
                    new_signal=new_signal,
                    market_df=market_df,
                )
            )

            nav_df = nav_curve_frame(
                candidate_factor=candidate_factor,
                base_nav=base_nav,
                new_nav=new_result["strategy_combo_nav"],
            )
            output_dir = write_candidate_outputs(
                candidate_factor=candidate_factor,
                summary_row=row,
                nav_df=nav_df,
                output_root=sample_output_root,
            )
            summary_rows.append(row)
            nav_frames.append(nav_df)
            tested_count += 1
            print(
                "tested:",
                candidate_factor,
                f"pass_count={row['pass_count']}",
                f"new_expectancy={row['new_expectancy']:.6g}",
            )
            if output_dir is None:
                print(
                    "candidate outputs: skipped because "
                    f"pass_count<{MIN_PASS_COUNT_FOR_CANDIDATE_OUTPUT}"
                )
            else:
                print("candidate outputs:", output_dir)

        except Exception as exc:  # noqa: BLE001
            skipped_count += 1
            skip_reason = str(exc)
            print(f"SKIP {candidate_factor}: {skip_reason}")
            summary_rows.append(empty_summary_row(candidate_factor, skip_reason))

    summary_df = sort_summary(pd.DataFrame(summary_rows))
    total_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = total_output_dir / SUMMARY_OUTPUT_NAME
    nav_path = total_output_dir / NAV_OUTPUT_NAME
    summary_df.to_excel(summary_path, index=False)
    enriched_summary_df = enrich_summary(
        summary_path=summary_path,
        usable_path=USABLE_FACTORS_PATH,
    )
    write_summary(enriched_summary_df, summary_path)

    if nav_frames:
        all_nav_df = pd.concat(nav_frames, axis=0, ignore_index=True)
    else:
        all_nav_df = pd.DataFrame(columns=["date", "base_nav", "candidate_factor", "new_nav"])
    all_nav_df.to_parquet(nav_path, index=False)

    print("\n========== Scan complete ==========")
    print(f"successfully tested candidates: {tested_count}")
    print(f"skipped candidates: {skipped_count}")
    print_top20(summary_df)
    print("\n========== Output files ==========")
    print(f"summary: {summary_path}")
    print(f"nav_curves: {nav_path}")
    return nav_path, graph_output_dir


# ========== 5. Plot marginal NAV curves ==========

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_NAV_CURVES_PATH = OUTPUT_ROOT / "all" / "output" / NAV_OUTPUT_NAME
_GRAPH_OUTPUT_DIR = OUTPUT_ROOT / "all" / "output_graph"


def plot_marginal_nav_curves(
    nav_curves_path: Path = _NAV_CURVES_PATH,
    output_dir: Path = _GRAPH_OUTPUT_DIR,
) -> None:
    if not nav_curves_path.exists():
        print(f"[plot] nav_curves not found: {nav_curves_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_nav_df = pd.read_parquet(nav_curves_path)
    all_nav_df["date"] = pd.to_datetime(all_nav_df["date"])

    candidates = sorted(all_nav_df["candidate_factor"].unique())
    print(f"[plot] plotting {len(candidates)} candidate factors → {output_dir}")

    for candidate_factor in candidates:
        df = (
            all_nav_df[all_nav_df["candidate_factor"] == candidate_factor]
            .copy()
            .sort_values("date")
        )
        base_first = pd.to_numeric(df["base_nav"].iloc[0], errors="coerce")
        if pd.isna(base_first) or base_first == 0:
            print(f"[plot] skip {candidate_factor}: invalid base_nav at first row")
            continue

        relative_nav = (
            pd.to_numeric(df["new_nav"], errors="coerce")
            / pd.to_numeric(df["base_nav"], errors="coerce")
        )
        start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["date"], relative_nav, linewidth=1.2, color="#1f77b4")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(
            f"Relative NAV: BASE + {candidate_factor}  ({start_date} → {end_date})",
            fontsize=11,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Relative NAV  (new / base)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"{candidate_factor}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved: {out_path}")

    print(f"[plot] done — {len(candidates)} charts saved to {output_dir}")


if __name__ == "__main__":
    _nav_path, _graph_output_dir = main()
    plot_marginal_nav_curves(_nav_path, _graph_output_dir)
