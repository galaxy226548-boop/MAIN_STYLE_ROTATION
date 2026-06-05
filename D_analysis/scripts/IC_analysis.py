"""Batch IC analysis for mounted normalized style-rotation factors.

Inputs:
    B_factors/output/zhao_mounted_normalized_factors.parquet
    A_data/prepared_data/market_df.parquet

Outputs:
    D_analysis/IC_output/{factor}_IC_analysis.xlsx
    D_analysis/IC_output/{factor}_rolling_IC.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SY_BASELINE_DIR = PROJECT_ROOT / "SY_Baseline"

for import_dir in [PROJECT_ROOT, SY_BASELINE_DIR]:
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from Config import Config  # noqa: E402
from G_engine.sample_window import SampleWindow, resolve_sample_window  # noqa: E402


FACTOR_PATH = PROJECT_ROOT / "B_factors" / "output" / "zhao_mounted_normalized_factors.parquet"
FACTOR_METADATA_PATH = PROJECT_ROOT / "B_factors" / "reference" / "factor_done.json"
MARKET_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "market_df.parquet"
OUTPUT_DIR = PROJECT_ROOT / "D_analysis" / "IC_output"

BASE_TARGET_COL = "target_return_diff"
FACTOR_VALUE_COL = "factor_value"
TRACK_COL = "track_id"
STATE_ROLLING_WINDOW = 12
STATE_ROLLING_MIN_N = 12
EVENT_ROLLING_WINDOW = 6
EVENT_ROLLING_MIN_N = 6
MIN_VALID_SAMPLE = getattr(Config, "MIN_VALID_SAMPLE", 10)
DEFAULT_SIGNAL_TYPE = "state"
SUPPORTED_SIGNAL_TYPES = {"event", "state"}
EVENT_NAN_RATIO_THRESHOLD = 0.30


def calc_future_horizon_return(
    ret_series: pd.Series,
    horizon: int,
    return_type: str = Config.RETURN_TYPE,
) -> pd.Series:
    """Calculate same-track forward cumulative returns."""
    if return_type == "log":
        return (
            ret_series.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .sum()
            .iloc[::-1]
        )
    if return_type == "simple":
        return (
            (1 + ret_series)
            .iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .apply(np.prod, raw=True)
            .iloc[::-1]
            - 1
        )
    raise ValueError(f"Unsupported return_type: {return_type}")


def calc_ic_stats(x: pd.Series, y: pd.Series, nw_lag: int = 1, min_n: int = MIN_VALID_SAMPLE) -> pd.Series:
    """Calculate Pearson IC, Rank IC, sample coverage, and HAC t/p."""
    df = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)})
    n_total = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    coverage = n / n_total if n_total > 0 else np.nan

    empty_result = pd.Series(
        {
            "pearson_ic": np.nan,
            "rank_ic": np.nan,
            "N_valid": n,
            "N_total": n_total,
            "coverage": coverage,
            "nw_t": np.nan,
            "nw_p": np.nan,
        }
    )
    if n < min_n or df["x"].nunique() <= 1 or df["y"].nunique() <= 1:
        return empty_result

    try:
        pearson_ic = pearsonr(df["x"], df["y"])[0]
        rank_ic = spearmanr(df["x"], df["y"])[0]
        model = OLS(df["y"], add_constant(df["x"])).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(nw_lag)},
        )
        nw_t = model.tvalues.get("x", np.nan)
        nw_p = model.pvalues.get("x", np.nan)
    except Exception:
        return empty_result

    return pd.Series(
        {
            "pearson_ic": pearson_ic,
            "rank_ic": rank_ic,
            "N_valid": n,
            "N_total": n_total,
            "coverage": coverage,
            "nw_t": nw_t,
            "nw_p": nw_p,
        }
    )


def calc_rolling_ic(
    x: pd.Series,
    y: pd.Series,
    window: int = STATE_ROLLING_WINDOW,
    min_n: int = STATE_ROLLING_MIN_N,
) -> pd.Series:
    """Rolling Pearson IC with a fixed number of valid observations required."""
    df = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)})
    values: list[float] = []

    for i in range(len(df)):
        start_idx = max(0, i - window + 1)
        sub_df = df.iloc[start_idx : i + 1].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub_df) < min_n or sub_df["x"].nunique() <= 1 or sub_df["y"].nunique() <= 1:
            values.append(np.nan)
            continue
        try:
            values.append(pearsonr(sub_df["x"], sub_df["y"])[0])
        except Exception:
            values.append(np.nan)

    return pd.Series(values, index=df.index, name="ic")


def infer_signal_type_from_nan_ratio(factor_series: pd.Series) -> tuple[str, float]:
    nan_ratio = float(factor_series.isna().mean()) if len(factor_series) else 0.0
    signal_type = "event" if nan_ratio >= EVENT_NAN_RATIO_THRESHOLD else "state"
    return signal_type, nan_ratio


def load_factor_signal_types(factor_df: pd.DataFrame) -> dict[str, str]:
    factor_columns = factor_df.columns.tolist()

    if not FACTOR_METADATA_PATH.exists():
        print(f"Warning: missing factor metadata json: {FACTOR_METADATA_PATH}; inferring signal_type from NaN ratios")
        factor_signal_types = {}
        for factor_name in factor_columns:
            signal_type, nan_ratio = infer_signal_type_from_nan_ratio(factor_df[factor_name])
            print(f"Warning: signal_type missing for {factor_name}; inferred {signal_type} from nan_ratio={nan_ratio:.2%}")
            factor_signal_types[factor_name] = signal_type
        return factor_signal_types

    with FACTOR_METADATA_PATH.open(encoding="utf-8") as f:
        metadata = json.load(f)

    signal_types_by_factor_id: dict[str, str] = {}
    for sheet_payload in metadata.get("sheets", {}).values():
        for record in sheet_payload.get("records", []):
            signal_type = str(record.get("signal_type") or "").strip().lower()
            if signal_type not in SUPPORTED_SIGNAL_TYPES:
                continue
            for factor_id in [record.get("factor_id"), record.get("code")]:
                if factor_id:
                    signal_types_by_factor_id.setdefault(str(factor_id), signal_type)

    factor_signal_types: dict[str, str] = {}
    for factor_name in factor_columns:
        signal_type = signal_types_by_factor_id.get(factor_name)
        if signal_type is None:
            signal_type, nan_ratio = infer_signal_type_from_nan_ratio(factor_df[factor_name])
            print(f"Warning: signal_type missing for {factor_name}; inferred {signal_type} from nan_ratio={nan_ratio:.2%}")
        factor_signal_types[factor_name] = signal_type

    return factor_signal_types


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not FACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing factor parquet: {FACTOR_PATH}")
    if not MARKET_PATH.exists():
        raise FileNotFoundError(f"Missing market parquet: {MARKET_PATH}")

    factor_df = pd.read_parquet(FACTOR_PATH)
    market_df = pd.read_parquet(MARKET_PATH)
    factor_df.index = pd.to_datetime(factor_df.index)
    market_df.index = pd.to_datetime(market_df.index)
    factor_df = factor_df.sort_index()
    market_df = market_df.sort_index()

    required_cols = {TRACK_COL, BASE_TARGET_COL, "fwd_ret_g", "fwd_ret_v"}
    missing_cols = sorted(required_cols - set(market_df.columns))
    if missing_cols:
        raise KeyError(f"market_df is missing required columns: {missing_cols}")

    factor_signal_types = load_factor_signal_types(factor_df)

    return factor_df, market_df, factor_signal_types


def prepare_market_targets(market_df: pd.DataFrame, sample_window: SampleWindow | None = None) -> pd.DataFrame:
    out = market_df.copy()
    for k in Config.BUCKET_WEEKS:
        out[f"{Config.BUCKET_PREFIX}{k}"] = out.groupby(TRACK_COL)[BASE_TARGET_COL].shift(-(k - 1))

    for h in Config.HORIZON_WEEKS:
        g_col = f"horizon_ret_g_{h}"
        v_col = f"horizon_ret_v_{h}"
        target_col = f"{Config.HORIZON_PREFIX}{h}"
        out[g_col] = out.groupby(TRACK_COL)["fwd_ret_g"].transform(
            lambda s: calc_future_horizon_return(s, horizon=h, return_type=Config.RETURN_TYPE)
        )
        out[v_col] = out.groupby(TRACK_COL)["fwd_ret_v"].transform(
            lambda s: calc_future_horizon_return(s, horizon=h, return_type=Config.RETURN_TYPE)
        )
        out[target_col] = out[g_col] - out[v_col]

    if sample_window is None:
        sample_window = resolve_sample_window(Config, sample="all")

    # 样本内外默认日期只在 Config 中维护；这里根据运行参数切分 IC 统计区间。
    sample_mask = (out.index >= sample_window.start) & (out.index <= sample_window.end)
    return out.loc[sample_mask].copy()


def build_factor_frame(market_df: pd.DataFrame, factor_series: pd.Series) -> pd.DataFrame:
    factor_value = factor_series.rename(FACTOR_VALUE_COL)
    work_df = market_df.join(factor_value, how="inner")
    work_df[FACTOR_VALUE_COL] = pd.to_numeric(work_df[FACTOR_VALUE_COL], errors="coerce")
    return work_df.sort_index()


def rolling_ic_detail(
    work_df: pd.DataFrame,
    track_list: list[int],
    window: int,
    min_n: int,
) -> pd.DataFrame:
    pieces = []
    for track_id in track_list:
        sub_df = work_df[work_df[TRACK_COL].astype(int) == track_id].sort_index().copy()
        sub_df["ic"] = calc_rolling_ic(
            sub_df[FACTOR_VALUE_COL],
            sub_df[BASE_TARGET_COL],
            window=window,
            min_n=min_n,
        )
        sub_df["ic_ma"] = sub_df["ic"].rolling(window=window, min_periods=min_n).mean()
        sub_df["track_id"] = track_id
        pieces.append(sub_df[[FACTOR_VALUE_COL, BASE_TARGET_COL, "track_id", "ic", "ic_ma"]])

    if not pieces:
        return pd.DataFrame(columns=[FACTOR_VALUE_COL, BASE_TARGET_COL, "track_id", "ic", "ic_ma"])
    return pd.concat(pieces, axis=0).sort_index()


def summarize_rolling_ic_by_year(work_df: pd.DataFrame, rolling_df: pd.DataFrame, track_list: list[int]) -> pd.DataFrame:
    rows = []
    years = sorted(pd.Index(work_df.index.year).dropna().unique())

    for year in years:
        year_mask = work_df.index.year == year
        rolling_year_mask = rolling_df.index.year == year
        for track_id in track_list:
            track_mask = work_df[TRACK_COL].astype(int) == track_id
            raw_sub = work_df.loc[year_mask & track_mask, [FACTOR_VALUE_COL, BASE_TARGET_COL]]

            rolling_sub = rolling_df.loc[
                rolling_year_mask & (rolling_df["track_id"].astype(int) == track_id),
                "ic",
            ].dropna()

            raw_stats = calc_ic_stats(
                raw_sub[FACTOR_VALUE_COL],
                raw_sub[BASE_TARGET_COL],
                nw_lag=1,
                min_n=MIN_VALID_SAMPLE,
            )

            rows.append(
                {
                    "year": int(year),
                    "track_id": int(track_id),
                    "ic_mean": rolling_sub.mean() if len(rolling_sub) else np.nan,
                    "ic_positive_ratio": (rolling_sub > 0).mean() if len(rolling_sub) else np.nan,
                    "rolling_ic_count": int(len(rolling_sub)),
                    "nw_p_value": raw_stats["nw_p"],
                }
            )

    return pd.DataFrame(rows)


def calc_pos_ic_probability(
    sub_df: pd.DataFrame,
    target_col: str,
    window: int,
    min_n: int,
) -> float:
    rolling_ic = calc_rolling_ic(
        sub_df[FACTOR_VALUE_COL],
        sub_df[target_col],
        window=window,
        min_n=min_n,
    ).dropna()
    return (rolling_ic > 0).mean() if len(rolling_ic) else np.nan


def target_specs(factor_name: str) -> list[tuple[str, str, int]]:
    specs = [(f"f_{factor_name}", BASE_TARGET_COL, 1)]
    specs += [
        (f"{factor_name}_Bucket_{k}", f"{Config.BUCKET_PREFIX}{k}", 1)
        for k in Config.BUCKET_WEEKS
    ]
    specs += [
        (f"{factor_name}_Horizon_{h}", f"{Config.HORIZON_PREFIX}{h}", h)
        for h in Config.HORIZON_WEEKS
    ]
    return specs


def build_ic_summary(
    factor_name: str,
    work_df: pd.DataFrame,
    track_list: list[int],
    base_rolling_df: pd.DataFrame,
    rolling_window: int,
    rolling_min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    internal_rows = []
    base_pos_prob_by_track = {}

    for track_id in track_list:
        track_rolling = base_rolling_df.loc[
            base_rolling_df["track_id"].astype(int) == track_id,
            "ic",
        ].dropna()
        base_pos_prob_by_track[track_id] = (track_rolling > 0).mean() if len(track_rolling) else np.nan

    for row_name, target_col, nw_lag in target_specs(factor_name):
        row = {}
        internal_row = {}

        for track_id in track_list:
            sub_df = work_df[work_df[TRACK_COL].astype(int) == track_id]
            stats = calc_ic_stats(
                sub_df[FACTOR_VALUE_COL],
                sub_df[target_col],
                nw_lag=nw_lag,
                min_n=MIN_VALID_SAMPLE,
            )

            row[f"pearson_ic_track{track_id}"] = stats["pearson_ic"]
            row[f"rank_ic_track{track_id}"] = stats["rank_ic"]
            row[f"coverage_track{track_id}"] = stats["coverage"]
            row[f"nw_t_track{track_id}"] = stats["nw_t"]
            row[f"nw_p_track{track_id}"] = stats["nw_p"]
            if target_col == BASE_TARGET_COL:
                pos_ic_prob = base_pos_prob_by_track[track_id]
            else:
                pos_ic_prob = calc_pos_ic_probability(
                    sub_df,
                    target_col=target_col,
                    window=rolling_window,
                    min_n=rolling_min_n,
                )
            row[f"pos_ic_prob_track{track_id}"] = pos_ic_prob

            internal_row[f"n_track{track_id}"] = stats["N_valid"]
            internal_row[f"n_total_track{track_id}"] = stats["N_total"]
            internal_row[f"coverage_track{track_id}"] = stats["coverage"]

        rows.append(pd.Series(row, name=row_name))
        internal_rows.append(pd.Series(internal_row, name=row_name))

    summary_df = pd.DataFrame(rows)
    internal_df = pd.DataFrame(internal_rows)

    ordered_cols = []
    for track_id in track_list:
        ordered_cols.extend(
            [
                f"pearson_ic_track{track_id}",
                f"rank_ic_track{track_id}",
                f"coverage_track{track_id}",
                f"nw_t_track{track_id}",
                f"nw_p_track{track_id}",
                f"pos_ic_prob_track{track_id}",
            ]
        )

    return summary_df.reindex(columns=ordered_cols), internal_df


def make_coverage_text(internal_df: pd.DataFrame, rows: list[str], track_list: list[int]) -> str:
    lines = []
    for row in rows:
        if row not in internal_df.index:
            continue
        n_vals = pd.to_numeric(
            pd.Series([internal_df.loc[row, f"n_track{t}"] for t in track_list]),
            errors="coerce",
        )
        cov_vals = pd.to_numeric(
            pd.Series([internal_df.loc[row, f"coverage_track{t}"] for t in track_list]),
            errors="coerce",
        )
        n_mean = n_vals.mean(skipna=True)
        cov_mean = cov_vals.mean(skipna=True)
        label = row.replace("f_", "Raw ")
        n_text = "NaN" if pd.isna(n_mean) else f"{n_mean:.0f}"
        cov_text = "NaN" if pd.isna(cov_mean) else f"{cov_mean:.1%}"
        lines.append(f"{label}: N={n_text}, coverage={cov_text}")
    return "\n".join(lines)


def save_rolling_plot(
    factor_name: str,
    rolling_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    track_list: list[int],
    output_path: Path,
    rolling_window: int,
) -> None:
    fig, axes = plt.subplots(len(track_list), 1, figsize=(14, max(3, len(track_list) * 2.4)), sharex=True)
    if len(track_list) == 1:
        axes = [axes]

    any_valid = False
    for ax, track_id in zip(axes, track_list):
        plot_df = rolling_df[rolling_df["track_id"].astype(int) == track_id].sort_index()
        valid_count = int(plot_df["ic"].notna().sum()) if "ic" in plot_df.columns else 0
        any_valid = any_valid or valid_count > 0

        ax.plot(plot_df.index, plot_df["ic"], label="IC", alpha=0.4, linewidth=1.1)
        ax.plot(plot_df.index, plot_df["ic_ma"], label=f"{rolling_window}-period moving avg", linewidth=1.8)
        ax.axhline(0, linewidth=0.9, color="black", alpha=0.6)
        ax.set_title(f"Track {track_id}", fontsize=11)
        ax.set_ylabel("IC")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

        if valid_count > 0:
            ic_mean = plot_df["ic"].mean()
            ic_std = plot_df["ic"].std()
            ax.text(
                0.01,
                0.92,
                f"Mean {ic_mean:.3f} | Std {ic_std:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
            )

    if not any_valid:
        axes[0].text(
            0.5,
            0.5,
            "No valid rolling IC values",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )

    footnote = make_coverage_text(internal_df, [f"f_{factor_name}"], track_list)
    fig.suptitle(f"{factor_name} Rolling IC vs {BASE_TARGET_COL}", fontsize=15)
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_excel(output_path: Path, ic_df: pd.DataFrame, rolling_summary_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        ic_df.to_excel(writer, sheet_name="IC analysis", index=True)
        rolling_summary_df.to_excel(writer, sheet_name="rolling_ic", index=False)


def get_rolling_params(signal_type: str) -> tuple[int, int]:
    if signal_type == "event":
        return EVENT_ROLLING_WINDOW, EVENT_ROLLING_MIN_N
    return STATE_ROLLING_WINDOW, STATE_ROLLING_MIN_N


def build_ic_sample_frame(work_df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    if signal_type == "event":
        return work_df[work_df[FACTOR_VALUE_COL].notna()].copy()
    return work_df


def analyze_factor(
    factor_name: str,
    factor_series: pd.Series,
    market_df: pd.DataFrame,
    track_list: list[int],
    signal_type: str,
) -> None:
    work_df = build_factor_frame(market_df, factor_series)
    ic_sample_df = build_ic_sample_frame(work_df, signal_type)
    rolling_window, rolling_min_n = get_rolling_params(signal_type)
    rolling_df = rolling_ic_detail(
        ic_sample_df,
        track_list,
        window=rolling_window,
        min_n=rolling_min_n,
    )
    rolling_summary_df = summarize_rolling_ic_by_year(ic_sample_df, rolling_df, track_list)
    ic_df, internal_df = build_ic_summary(
        factor_name,
        ic_sample_df,
        track_list,
        rolling_df,
        rolling_window,
        rolling_min_n,
    )

    excel_path = OUTPUT_DIR / f"{factor_name}_IC_analysis.xlsx"
    figure_path = OUTPUT_DIR / f"{factor_name}_rolling_IC.png"
    write_excel(excel_path, ic_df, rolling_summary_df)
    save_rolling_plot(factor_name, rolling_df, internal_df, track_list, figure_path, rolling_window)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch IC analysis for zhao mounted normalized factors.")
    parser.add_argument(
        "--factor",
        action="append",
        help="Factor column to analyze. Can be passed multiple times. Defaults to all columns.",
    )
    parser.add_argument("--sample", choices=["all", "ins", "oos", "custom"], default="all")
    parser.add_argument("--start-date", help="Optional sample start date, e.g. 2020-01-01.")
    parser.add_argument("--end-date", help="Optional sample end date, e.g. 2024-12-31.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_window = resolve_sample_window(
        Config,
        sample=args.sample,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    factor_df, raw_market_df, factor_signal_types = load_inputs()
    market_df = prepare_market_targets(raw_market_df, sample_window=sample_window)
    track_list = sorted(market_df[TRACK_COL].dropna().astype(int).unique().tolist())
    print(f"sample: {sample_window.name} ({sample_window.start_text} -> {sample_window.end_text})")

    factors = args.factor or factor_df.columns.tolist()
    missing_factors = sorted(set(factors) - set(factor_df.columns))
    if missing_factors:
        raise KeyError(f"Unknown factor columns in mounted factor parquet: {missing_factors}")

    for factor_name in factors:
        signal_type = factor_signal_types.get(factor_name, DEFAULT_SIGNAL_TYPE)
        analyze_factor(factor_name, factor_df[factor_name], market_df, track_list, signal_type)
        print(f"Saved IC analysis for {factor_name} ({signal_type})")

    print(f"Done. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
