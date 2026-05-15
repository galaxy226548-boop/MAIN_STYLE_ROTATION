"""Build growth/value conditional odds curves from valuation quantiles.

This script is a research-only diagnostic builder. It estimates full-sample
odds curves for manual inspection. If these curves are later converted into a
daily live factor, the odds estimation must be changed to use only samples
available on or before each holding_date.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/main_style_rotation_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
A_DATA_DIR = PROJECT_ROOT / "A_data"
OUTPUT_DIR = PROJECT_ROOT / "B_factors" / "output" / "conditional_odds"

STYLE_POOL_PATHS = {
    "growth": {
        "top": A_DATA_DIR / "output" / "Astockdaily_Fri_growth_top20.parquet",
        "bottom": A_DATA_DIR / "output" / "Astockdaily_Fri_growth_bottom20.parquet",
    },
    "value": {
        "top": A_DATA_DIR / "output" / "Astockdaily_Fri_value_top20.parquet",
        "bottom": A_DATA_DIR / "output" / "Astockdaily_Fri_value_bottom20.parquet",
    },
}

PB_DATA_PATH = A_DATA_DIR / "prepared_data" / "Astockdaily.parquet"
STYLE_INDEX_PATHS = {
    "growth": A_DATA_DIR / "prepared_data" / "growth_index.xlsx",
    "value": A_DATA_DIR / "prepared_data" / "value_index.xlsx",
}

MIN_QUANTILE_PERIODS = 52
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
GRID_BANDWIDTH = 0.10
LOWESS_FRAC = 0.5


class DateFilter(NamedTuple):
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None


class PoolDiagnostics(NamedTuple):
    total_count: int
    valid_count: int
    date_count: int
    renormalized_date_count: int
    simple_mean_fallback_date_count: int


def normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def ensure_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build growth/value conditional odds curves from valuation quantiles."
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive source-data start date, e.g. 2010-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive source-data end date, e.g. 2012-12-31.",
    )
    parser.add_argument(
        "--quantile-mode",
        choices=["full-sample", "expanding"],
        default="full-sample",
        help=(
            "Quantile mode. Use full-sample for research charts on the selected date range. "
            "Use expanding to compute historical quantiles with no future data."
        ),
    )
    parser.add_argument(
        "--return-mode",
        choices=["relative", "absolute"],
        default="relative",
        help=(
            "Return mode for odds. Use relative for style-vs-opposite-style returns "
            "(growth_ret - value_ret, value_ret - growth_ret). Use absolute to switch "
            "back to each style index's own future return."
        ),
    )
    parser.add_argument(
        "--valuation-mode",
        choices=["relative-spread", "internal-spread"],
        default="relative-spread",
        help=(
            "Valuation mode. Use relative-spread for growth/value top20 BP spread. "
            "Use internal-spread to switch back to each style's top20-bottom20 BP spread."
        ),
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional tag inserted into plot filenames, e.g. 2010_2012.",
    )
    return parser.parse_args()


def parse_date_filter(args: argparse.Namespace) -> DateFilter:
    start_date = pd.to_datetime(args.start_date, errors="raise").normalize() if args.start_date else None
    end_date = pd.to_datetime(args.end_date, errors="raise").normalize() if args.end_date else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError(f"start-date must be <= end-date: {start_date.date()} > {end_date.date()}")
    return DateFilter(start_date=start_date, end_date=end_date)


def apply_date_filter(df: pd.DataFrame, date_filter: DateFilter) -> pd.DataFrame:
    out = df
    if date_filter.start_date is not None:
        out = out[out["holding_date"] >= date_filter.start_date]
    if date_filter.end_date is not None:
        out = out[out["holding_date"] <= date_filter.end_date]
    return out.copy()


def load_pb_data(path: Path, date_filter: DateFilter) -> pd.DataFrame:
    """Load market PB data and compute market bp_proxy diagnostics by date."""
    ensure_path(path)
    pb_df = pd.read_parquet(path, columns=["TradingDate", "PB"])
    pb_df["holding_date"] = normalize_date(pb_df["TradingDate"])
    pb_df["PB"] = pd.to_numeric(pb_df["PB"], errors="coerce")
    pb_df = pb_df[pb_df["holding_date"].notna()].copy()
    pb_df = apply_date_filter(pb_df, date_filter)
    pb_df = pb_df[pb_df["PB"] > 0].copy()
    if pb_df.empty:
        raise ValueError(f"No valid PB rows found in {path}")

    pb_df["bp_proxy"] = 1.0 / pb_df["PB"]
    pb_df["bp_proxy"] = pb_df["bp_proxy"].replace([np.inf, -np.inf], np.nan)
    market_df = (
        pb_df.dropna(subset=["bp_proxy"])
        .groupby("holding_date", as_index=False)
        .agg(market_bp_mean=("bp_proxy", "mean"), market_count=("bp_proxy", "count"))
        .sort_values("holding_date")
    )
    if market_df.empty:
        raise ValueError(f"No valid market bp_proxy rows found in {path}")
    return market_df


def load_style_pool(path: Path, date_filter: DateFilter) -> pd.DataFrame:
    ensure_path(path)
    pool_df = pd.read_parquet(path)
    required_cols = {"signal_date", "PB"}
    missing_cols = required_cols - set(pool_df.columns)
    if missing_cols:
        raise KeyError(f"{path} missing required columns: {sorted(missing_cols)}")

    pool_df = pool_df.copy()
    pool_df["holding_date"] = normalize_date(pool_df["signal_date"])
    pool_df["PB"] = pd.to_numeric(pool_df["PB"], errors="coerce")
    if "weight" in pool_df.columns:
        pool_df["weight"] = pd.to_numeric(pool_df["weight"], errors="coerce")
    else:
        pool_df["weight"] = np.nan
    pool_df = pool_df[pool_df["holding_date"].notna()].copy()
    pool_df = apply_date_filter(pool_df, date_filter)
    if pool_df.empty:
        raise ValueError(f"No valid holding_date rows found in {path}")
    return pool_df


def _weighted_bp_mean_by_date(pool_df: pd.DataFrame) -> tuple[pd.DataFrame, PoolDiagnostics]:
    rows: list[dict[str, float | pd.Timestamp | int]] = []
    total_count = len(pool_df)
    valid_count = int((pool_df["PB"] > 0).sum())
    renormalized_date_count = 0
    simple_mean_fallback_date_count = 0

    for holding_date, group in pool_df.groupby("holding_date", sort=True):
        valid = group[group["PB"] > 0].copy()
        valid["bp_proxy"] = (1.0 / valid["PB"]).replace([np.inf, -np.inf], np.nan)
        valid = valid.dropna(subset=["bp_proxy"])

        bp_mean = np.nan
        if not valid.empty:
            weight = valid["weight"].replace([np.inf, -np.inf], np.nan)
            weight_sum = weight.sum(skipna=True)
            if weight.notna().any() and np.isfinite(weight_sum) and weight_sum > 0:
                normalized_weight = weight.fillna(0.0) / weight_sum
                bp_mean = float((valid["bp_proxy"] * normalized_weight).sum())
                if not np.isclose(weight_sum, 1.0):
                    renormalized_date_count += 1
            else:
                bp_mean = float(valid["bp_proxy"].mean())
                simple_mean_fallback_date_count += 1

        rows.append(
            {
                "holding_date": holding_date,
                "bp_mean": bp_mean,
                "count": int(len(group)),
                "valid_count": int(len(valid)),
            }
        )

    out = pd.DataFrame(rows).sort_values("holding_date")
    diagnostics = PoolDiagnostics(
        total_count=total_count,
        valid_count=valid_count,
        date_count=int(out["holding_date"].nunique()),
        renormalized_date_count=renormalized_date_count,
        simple_mean_fallback_date_count=simple_mean_fallback_date_count,
    )
    return out, diagnostics


def compute_style_valuation_state(
    style: str,
    top_pool_df: pd.DataFrame,
    bottom_pool_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, PoolDiagnostics]]:
    top_df, top_diag = _weighted_bp_mean_by_date(top_pool_df)
    bottom_df, bottom_diag = _weighted_bp_mean_by_date(bottom_pool_df)

    merged = top_df.rename(
        columns={"bp_mean": "top20_bp_mean", "count": "top_count", "valid_count": "top_valid_count"}
    ).merge(
        bottom_df.rename(
            columns={
                "bp_mean": "bottom20_bp_mean",
                "count": "bottom_count",
                "valid_count": "bottom_valid_count",
            }
        ),
        on="holding_date",
        how="outer",
    )
    merged = merged.merge(market_df, on="holding_date", how="left")
    merged["style"] = style
    merged["valuation_diff_adj"] = (
        (merged["top20_bp_mean"] - merged["bottom20_bp_mean"]) / merged["market_bp_mean"]
    )
    merged = merged[
        [
            "holding_date",
            "style",
            "top20_bp_mean",
            "bottom20_bp_mean",
            "market_bp_mean",
            "valuation_diff_adj",
            "top_count",
            "bottom_count",
            "top_valid_count",
            "bottom_valid_count",
            "market_count",
        ]
    ].sort_values("holding_date")
    return merged, {"top": top_diag, "bottom": bottom_diag}


def build_relative_valuation_state(internal_valuation_df: pd.DataFrame) -> pd.DataFrame:
    required_styles = {"growth", "value"}
    styles = set(internal_valuation_df["style"].dropna().unique())
    missing_styles = required_styles - styles
    if missing_styles:
        raise ValueError(f"Cannot compute relative valuation spread; missing styles: {sorted(missing_styles)}")

    base_cols = [
        "holding_date",
        "style",
        "top20_bp_mean",
        "bottom20_bp_mean",
        "market_bp_mean",
        "top_count",
        "bottom_count",
        "top_valid_count",
        "bottom_valid_count",
        "market_count",
    ]
    internal_df = internal_valuation_df[base_cols].copy()
    growth_top = (
        internal_df[internal_df["style"].eq("growth")]
        .set_index("holding_date")["top20_bp_mean"]
        .rename("growth_top20_bp_mean")
    )
    value_top = (
        internal_df[internal_df["style"].eq("value")]
        .set_index("holding_date")["top20_bp_mean"]
        .rename("value_top20_bp_mean")
    )
    relative_lookup = pd.concat([growth_top, value_top], axis=1)
    relative_lookup["relative_bp_spread"] = (
        relative_lookup["growth_top20_bp_mean"] - relative_lookup["value_top20_bp_mean"]
    )

    out = internal_df.merge(relative_lookup.reset_index(), on="holding_date", how="left")
    out["valuation_diff_adj"] = out["relative_bp_spread"] / out["market_bp_mean"]
    value_mask = out["style"].eq("value")
    out.loc[value_mask, "valuation_diff_adj"] = -out.loc[value_mask, "valuation_diff_adj"]
    out["valuation_mode"] = "relative-spread"
    return out.sort_values(["style", "holding_date"])


def build_valuation_state(valuation_parts: list[pd.DataFrame], valuation_mode: str) -> pd.DataFrame:
    internal_df = pd.concat(valuation_parts, ignore_index=True).sort_values(["style", "holding_date"])
    internal_df["growth_top20_bp_mean"] = np.nan
    internal_df["value_top20_bp_mean"] = np.nan
    internal_df["relative_bp_spread"] = np.nan
    internal_df["valuation_mode"] = "internal-spread"

    if valuation_mode == "internal-spread":
        return internal_df
    if valuation_mode == "relative-spread":
        return build_relative_valuation_state(internal_df)
    raise ValueError(f"Unsupported valuation_mode: {valuation_mode}")


def compute_expanding_quantile(series: pd.Series, min_periods: int = MIN_QUANTILE_PERIODS) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    history: list[float] = []

    for idx, value in values.items():
        if pd.isna(value):
            continue
        history.append(float(value))
        if len(history) < min_periods:
            continue
        hist = pd.Series(history, dtype="float64")
        out.at[idx] = float(hist.rank(pct=True, method="average").iloc[-1])

    return out


def compute_full_sample_quantile(series: pd.Series, min_periods: int = MIN_QUANTILE_PERIODS) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna()
    if len(valid) < min_periods:
        return out
    out.loc[valid.index] = valid.rank(pct=True, method="average")
    return out


def compute_valuation_quantile(series: pd.Series, quantile_mode: str) -> pd.Series:
    if quantile_mode == "full-sample":
        return compute_full_sample_quantile(series, min_periods=MIN_QUANTILE_PERIODS)
    if quantile_mode == "expanding":
        return compute_expanding_quantile(series, min_periods=MIN_QUANTILE_PERIODS)
    raise ValueError(f"Unsupported quantile_mode: {quantile_mode}")


def compute_future_returns(path: Path, style: str, date_filter: DateFilter) -> pd.DataFrame:
    ensure_path(path)
    index_df = pd.read_excel(path, sheet_name="历史价格")
    required_cols = {"date", "close"}
    missing_cols = required_cols - set(index_df.columns)
    if missing_cols:
        raise KeyError(f"{path} missing required columns: {sorted(missing_cols)}")

    index_df = index_df[["date", "close"]].copy()
    index_df["holding_date"] = normalize_date(index_df["date"])
    index_df["close"] = pd.to_numeric(index_df["close"], errors="coerce")
    index_df = index_df[index_df["holding_date"].notna()].copy()
    index_df = apply_date_filter(index_df, date_filter)
    index_df = (
        index_df.sort_values("holding_date")
        .drop_duplicates(subset=["holding_date"], keep="last")
        .set_index("holding_date")
    )
    if index_df["close"].dropna().empty:
        raise ValueError(f"No valid close rows found in {path}")

    index_df["future_ret_5d"] = index_df["close"].shift(-5) / index_df["close"] - 1.0
    index_df["future_ret_20d"] = index_df["close"].shift(-20) / index_df["close"] - 1.0
    out = index_df[["future_ret_5d", "future_ret_20d"]].reset_index()
    out["style"] = style
    return out


def build_return_samples(style_return_parts: list[pd.DataFrame], return_mode: str) -> pd.DataFrame:
    absolute_returns = pd.concat(style_return_parts, ignore_index=True)
    if return_mode == "absolute":
        return absolute_returns
    if return_mode != "relative":
        raise ValueError(f"Unsupported return_mode: {return_mode}")

    wide = absolute_returns.pivot(
        index="holding_date",
        columns="style",
        values=["future_ret_5d", "future_ret_20d"],
    )
    required = [
        ("future_ret_5d", "growth"),
        ("future_ret_5d", "value"),
        ("future_ret_20d", "growth"),
        ("future_ret_20d", "value"),
    ]
    missing_cols = [col for col in required if col not in wide.columns]
    if missing_cols:
        raise KeyError(f"Cannot compute relative returns; missing columns: {missing_cols}")

    rows: list[pd.DataFrame] = []
    for style, sign in [("growth", 1.0), ("value", -1.0)]:
        rel_df = pd.DataFrame(index=wide.index)
        rel_df["future_ret_5d"] = sign * (
            wide[("future_ret_5d", "growth")] - wide[("future_ret_5d", "value")]
        )
        rel_df["future_ret_20d"] = sign * (
            wide[("future_ret_20d", "growth")] - wide[("future_ret_20d", "value")]
        )
        rel_df["style"] = style
        rows.append(rel_df.reset_index())

    return pd.concat(rows, ignore_index=True)


def compute_odds_grid(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    horizons = {"5d": "future_ret_5d", "20d": "future_ret_20d"}

    for style in ["growth", "value"]:
        style_df = samples_df[samples_df["style"].eq(style)].copy()
        for horizon, ret_col in horizons.items():
            for q in GRID:
                q_low = max(0.0, float(q) - GRID_BANDWIDTH)
                q_high = min(1.0, float(q) + GRID_BANDWIDTH)
                local = style_df[
                    style_df["valuation_quantile"].between(q_low, q_high, inclusive="both")
                ][ret_col].dropna()
                pos_ret = local[local > 0]
                neg_ret = local[local < 0]
                pos_mean = float(pos_ret.mean()) if len(pos_ret) else np.nan
                neg_mean = float(neg_ret.mean()) if len(neg_ret) else np.nan
                odds_value = np.nan
                if len(pos_ret) >= 3 and len(neg_ret) >= 3 and np.isfinite(neg_mean) and neg_mean != 0:
                    odds_value = pos_mean / abs(neg_mean)

                rows.append(
                    {
                        "style": style,
                        "horizon": horizon,
                        "q": float(q),
                        "q_low": q_low,
                        "q_high": q_high,
                        "sample_count": int(len(local)),
                        "pos_count": int(len(pos_ret)),
                        "neg_count": int(len(neg_ret)),
                        "pos_mean": pos_mean,
                        "neg_mean": neg_mean,
                        "odds": odds_value,
                    }
                )

    return pd.DataFrame(rows)


def fit_lowess_curve(odds_grid_df: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []

    for (style, horizon), group in odds_grid_df.groupby(["style", "horizon"], sort=True):
        fitted = group.copy().sort_values("q")
        fitted["lowess_odds"] = np.nan
        valid = fitted[fitted["odds"].notna()].copy()
        if len(valid) < 5:
            print(f"[LOWESS] Skip {style} {horizon}: valid odds points < 5 ({len(valid)})")
            out_parts.append(fitted)
            continue

        smoothed = lowess(
            endog=valid["odds"].to_numpy(dtype="float64"),
            exog=valid["q"].to_numpy(dtype="float64"),
            frac=LOWESS_FRAC,
            return_sorted=False,
        )
        fitted.loc[valid.index, "lowess_odds"] = smoothed
        out_parts.append(fitted)

    return pd.concat(out_parts, ignore_index=True)


def compute_quantile_decile_summary(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    horizons = {"5d": "future_ret_5d", "20d": "future_ret_20d"}

    for style in ["growth", "value"]:
        style_df = samples_df[samples_df["style"].eq(style)].copy()
        for decile in range(1, 11):
            q_low = (decile - 1) / 10
            q_high = decile / 10
            if decile == 1:
                mask = style_df["valuation_quantile"].between(q_low, q_high, inclusive="both")
            else:
                mask = (style_df["valuation_quantile"] > q_low) & (style_df["valuation_quantile"] <= q_high)
            bucket = style_df.loc[mask].copy()
            for horizon, ret_col in horizons.items():
                ret = bucket[ret_col].dropna()
                rows.append(
                    {
                        "style": style,
                        "horizon": horizon,
                        "decile": decile,
                        "q_low": q_low,
                        "q_high": q_high,
                        "sample_count": int(len(ret)),
                        "ret_mean": float(ret.mean()) if len(ret) else np.nan,
                        "ret_median": float(ret.median()) if len(ret) else np.nan,
                        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                        "pos_count": int((ret > 0).sum()),
                        "neg_count": int((ret < 0).sum()),
                    }
                )

    return pd.DataFrame(rows)


def plot_lowess_curves(lowess_df: pd.DataFrame, horizon: str, output_path: Path, output_tag: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = lowess_df[(lowess_df["style"].eq("growth")) & (lowess_df["horizon"].eq(horizon))].sort_values("q")
    if sub.empty:
        raise ValueError(f"No growth odds rows available for horizon={horizon}")
    ax.scatter(
        sub["q"],
        sub["odds"],
        s=24,
        alpha=0.35,
        color="tab:blue",
        label="grid odds",
    )
    smooth = sub.dropna(subset=["lowess_odds"])
    if not smooth.empty:
        ax.plot(
            smooth["q"],
            smooth["lowess_odds"],
            linewidth=2.2,
            color="tab:blue",
            label="LOWESS",
        )

    title_suffix = "5 Trading Days" if horizon == "5d" else "20 Trading Days"
    tag_text = f" ({output_tag.replace('_', '-')})" if output_tag else ""
    ax.set_title(f"Growth vs Value Conditional Odds{tag_text} - {title_suffix}")
    ax.set_xlabel("growth_relative_cheapness_quantile")
    ax.set_ylabel("odds of growth_ret - value_ret")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.01,
        0.98,
        f"LOWESS frac={LOWESS_FRAC}, bandwidth=+/-{GRID_BANDWIDTH:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] Saved {output_path}")


def _print_path_log() -> None:
    print("[PATHS] Data paths")
    print(f"  PB file: {PB_DATA_PATH}")
    for style, paths in STYLE_POOL_PATHS.items():
        print(f"  {style} top file: {paths['top']}")
        print(f"  {style} bottom file: {paths['bottom']}")
    for style, path in STYLE_INDEX_PATHS.items():
        print(f"  {style} future close file: {path}")


def _print_date_filter(date_filter: DateFilter) -> None:
    start_text = date_filter.start_date.date() if date_filter.start_date is not None else "unbounded"
    end_text = date_filter.end_date.date() if date_filter.end_date is not None else "unbounded"
    print(f"[DATE FILTER] source data holding_date range: {start_text} to {end_text}")


def _print_pool_diagnostics(style: str, diags: dict[str, PoolDiagnostics]) -> None:
    for side, diag in diags.items():
        match_rate = diag.valid_count / diag.total_count if diag.total_count else np.nan
        print(
            f"[PB] {style} {side}: valid PB match rate={match_rate:.2%}, "
            f"rows={diag.total_count}, valid_rows={diag.valid_count}, "
            f"holding_dates={diag.date_count}, renormalized_dates={diag.renormalized_date_count}, "
            f"simple_mean_fallback_dates={diag.simple_mean_fallback_date_count}"
        )


def _print_return_quality(samples_df: pd.DataFrame) -> None:
    for style, group in samples_df.groupby("style", sort=True):
        print(f"[DATES] {style}: holding_date count={group['holding_date'].nunique()}")
        for ret_col in ["future_ret_5d", "future_ret_20d"]:
            valid_count = int(group[ret_col].notna().sum())
            missing_ratio = float(group[ret_col].isna().mean())
            print(
                f"[RET] {style} {ret_col}: valid_count={valid_count}, "
                f"missing_ratio={missing_ratio:.2%}"
            )


def _print_odds_quality(odds_grid_df: pd.DataFrame) -> None:
    for (style, horizon), group in odds_grid_df.groupby(["style", "horizon"], sort=True):
        valid_points = int(group["odds"].notna().sum())
        print(f"[ODDS] {style} {horizon}: valid q points={valid_points}/{len(group)}")


def main() -> None:
    args = parse_args()
    date_filter = parse_date_filter(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _print_path_log()
    _print_date_filter(date_filter)
    print(f"[QUANTILE] mode={args.quantile_mode}")
    print("[QUANTILE] To compute historical no-lookahead quantiles, rerun with --quantile-mode expanding")
    print(f"[RETURN] mode={args.return_mode}")
    print("[RETURN] To switch back to each style index's own future return, rerun with --return-mode absolute")
    print(f"[VALUATION] mode={args.valuation_mode}")
    print("[VALUATION] To switch back to style internal top-bottom spread, rerun with --valuation-mode internal-spread")
    if args.output_tag:
        print(f"[OUTPUT TAG] plot filenames will include: {args.output_tag}")

    market_df = load_pb_data(PB_DATA_PATH, date_filter)
    print("[MARKET] market_count stats")
    print(market_df["market_count"].describe().to_string())

    valuation_parts: list[pd.DataFrame] = []
    for style, paths in STYLE_POOL_PATHS.items():
        top_pool_df = load_style_pool(paths["top"], date_filter)
        bottom_pool_df = load_style_pool(paths["bottom"], date_filter)
        style_state_df, diagnostics = compute_style_valuation_state(style, top_pool_df, bottom_pool_df, market_df)
        valuation_parts.append(style_state_df)
        _print_pool_diagnostics(style, diagnostics)

    valuation_df = build_valuation_state(valuation_parts, args.valuation_mode)
    # full-sample is intended for research charts over a fixed sample window.
    # For historical/no-lookahead quantiles, pass --quantile-mode expanding.
    valuation_df["valuation_quantile"] = (
        valuation_df.groupby("style", group_keys=False)["valuation_diff_adj"]
        .apply(lambda s: compute_valuation_quantile(s, args.quantile_mode))
        .astype("float64")
    )

    return_parts = [
        compute_future_returns(path, style, date_filter)
        for style, path in STYLE_INDEX_PATHS.items()
    ]
    future_ret_df = build_return_samples(return_parts, args.return_mode)
    samples_df = valuation_df.merge(future_ret_df, on=["holding_date", "style"], how="left")
    samples_df = samples_df.sort_values(["style", "holding_date"])

    sample_cols = [
        "holding_date",
        "style",
        "valuation_diff_adj",
        "valuation_quantile",
        "future_ret_5d",
        "future_ret_20d",
        "top20_bp_mean",
        "bottom20_bp_mean",
        "market_bp_mean",
        "growth_top20_bp_mean",
        "value_top20_bp_mean",
        "relative_bp_spread",
        "valuation_mode",
        "top_count",
        "bottom_count",
        "top_valid_count",
        "bottom_valid_count",
        "market_count",
    ]
    samples_df = samples_df[sample_cols]
    samples_parquet_path = OUTPUT_DIR / "style_valuation_return_samples.parquet"
    samples_csv_path = OUTPUT_DIR / "style_valuation_return_samples.csv"
    samples_xlsx_path = OUTPUT_DIR / "style_valuation_return_samples.xlsx"
    samples_df.to_parquet(samples_parquet_path, index=False)
    samples_df.to_csv(samples_csv_path, index=False)
    samples_df.to_excel(samples_xlsx_path, index=False)
    print(f"[OUTPUT] Saved {samples_parquet_path}")
    print(f"[OUTPUT] Saved {samples_csv_path}")
    print(f"[OUTPUT] Saved {samples_xlsx_path}")
    _print_return_quality(samples_df)

    odds_grid_df = compute_odds_grid(samples_df)
    odds_grid_parquet_path = OUTPUT_DIR / "style_odds_grid.parquet"
    odds_grid_csv_path = OUTPUT_DIR / "style_odds_grid.csv"
    odds_grid_preview_path = OUTPUT_DIR / "style_odds_grid_preview.xlsx"
    odds_grid_df.to_parquet(odds_grid_parquet_path, index=False)
    odds_grid_df.to_csv(odds_grid_csv_path, index=False)
    odds_grid_df.head(5).to_excel(odds_grid_preview_path, index=False)
    print(f"[OUTPUT] Saved {odds_grid_parquet_path}")
    print(f"[OUTPUT] Saved {odds_grid_csv_path}")
    print(f"[OUTPUT] Saved {odds_grid_preview_path}")
    _print_odds_quality(odds_grid_df)

    lowess_df = fit_lowess_curve(odds_grid_df)
    lowess_parquet_path = OUTPUT_DIR / "style_odds_lowess.parquet"
    lowess_csv_path = OUTPUT_DIR / "style_odds_lowess.csv"
    lowess_df[
        ["style", "horizon", "q", "odds", "lowess_odds", "sample_count", "pos_count", "neg_count"]
    ].to_parquet(lowess_parquet_path, index=False)
    lowess_df[
        ["style", "horizon", "q", "odds", "lowess_odds", "sample_count", "pos_count", "neg_count"]
    ].to_csv(lowess_csv_path, index=False)
    print(f"[OUTPUT] Saved {lowess_parquet_path}")
    print(f"[OUTPUT] Saved {lowess_csv_path}")

    decile_summary_df = compute_quantile_decile_summary(samples_df)
    decile_summary_parquet_path = OUTPUT_DIR / "style_quantile_decile_summary.parquet"
    decile_summary_csv_path = OUTPUT_DIR / "style_quantile_decile_summary.csv"
    decile_summary_xlsx_path = OUTPUT_DIR / "style_quantile_decile_summary.xlsx"
    decile_summary_df.to_parquet(decile_summary_parquet_path, index=False)
    decile_summary_df.to_csv(decile_summary_csv_path, index=False)
    decile_summary_df.to_excel(decile_summary_xlsx_path, index=False)
    print(f"[OUTPUT] Saved {decile_summary_parquet_path}")
    print(f"[OUTPUT] Saved {decile_summary_csv_path}")
    print(f"[OUTPUT] Saved {decile_summary_xlsx_path}")

    plot_name_prefix = "conditional_odds"
    if args.output_tag:
        plot_name_prefix = f"{plot_name_prefix}_{args.output_tag}"
    plot_lowess_curves(lowess_df, "5d", OUTPUT_DIR / f"{plot_name_prefix}_5d.png", args.output_tag)
    plot_lowess_curves(lowess_df, "20d", OUTPUT_DIR / f"{plot_name_prefix}_20d.png", args.output_tag)


if __name__ == "__main__":
    main()
