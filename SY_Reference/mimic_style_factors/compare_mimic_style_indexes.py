"""Compare mimic style-index closes with official style-index closes.

Inputs:
    A_data/prepared_data/growth_factor_Fri.parquet
    A_data/prepared_data/value_factor_Fri.parquet
    A_data/prepared_data/growth_index.xlsx
    A_data/prepared_data/value_index.xlsx

Outputs:
    SY_Reference/mimic_style_factors/output/*_close_dual_axis.png
    SY_Reference/mimic_style_factors/output/*_close_rebased.png
    SY_Reference/mimic_style_factors/output/*_close_compare.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_GROWTH_FACTOR_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "growth_factor_Fri.parquet"
DEFAULT_VALUE_FACTOR_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "value_factor_Fri.parquet"
DEFAULT_GROWTH_INDEX_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "growth_index.xlsx"
DEFAULT_VALUE_INDEX_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "value_index.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "output"

REQUIRED_FACTOR_COLUMNS = {"holding_date", "weight", "close"}
REQUIRED_INDEX_COLUMNS = {"date", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare self-built growth/value index closes with official style-index closes.",
    )
    parser.add_argument("--growth-factor-path", type=Path, default=DEFAULT_GROWTH_FACTOR_PATH)
    parser.add_argument("--value-factor-path", type=Path, default=DEFAULT_VALUE_FACTOR_PATH)
    parser.add_argument("--growth-index-path", type=Path, default=DEFAULT_GROWTH_INDEX_PATH)
    parser.add_argument("--value-index-path", type=Path, default=DEFAULT_VALUE_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_mimic_close(factor_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(factor_path)
    missing_columns = REQUIRED_FACTOR_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"{factor_path} missing columns: {sorted(missing_columns)}")

    df = df.loc[:, ["holding_date", "weight", "close"]].copy()
    df["date"] = pd.to_datetime(df["holding_date"]).dt.normalize()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    valid_close = df["close"].notna() & df["weight"].notna()
    df["valid_weight"] = df["weight"].where(valid_close, 0.0)
    df["weighted_close"] = (df["weight"] * df["close"]).where(valid_close, 0.0)

    grouped = (
        df.groupby("date", as_index=False)
        .agg(
            raw_weighted_close=("weighted_close", "sum"),
            weight_sum=("weight", "sum"),
            valid_weight_sum=("valid_weight", "sum"),
            component_count=("close", "size"),
            valid_close_count=("close", "count"),
        )
        .sort_values("date")
    )
    grouped["mimic_close"] = grouped["raw_weighted_close"] / grouped["valid_weight_sum"]
    grouped.loc[grouped["valid_weight_sum"] <= 0, "mimic_close"] = pd.NA
    grouped["valid_weight_coverage"] = grouped["valid_weight_sum"] / grouped["weight_sum"]

    return grouped.loc[
        :,
        [
            "date",
            "mimic_close",
            "raw_weighted_close",
            "weight_sum",
            "valid_weight_sum",
            "valid_weight_coverage",
            "component_count",
            "valid_close_count",
        ],
    ]


def load_benchmark_close(index_path: Path) -> pd.DataFrame:
    df = pd.read_excel(index_path, sheet_name=0)
    missing_columns = REQUIRED_INDEX_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"{index_path} missing columns: {sorted(missing_columns)}")

    out = df.loc[:, ["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["benchmark_close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.drop(columns="close").dropna(subset=["date", "benchmark_close"])
    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def build_compare_frame(factor_path: Path, index_path: Path) -> pd.DataFrame:
    mimic_df = load_mimic_close(factor_path)
    benchmark_df = load_benchmark_close(index_path)
    compare_df = benchmark_df.merge(mimic_df, on="date", how="inner").sort_values("date")
    compare_df = compare_df.dropna(subset=["benchmark_close", "mimic_close"])
    return compare_df


def save_dual_axis_plot(compare_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, left_ax = plt.subplots(figsize=(12, 6))
    right_ax = left_ax.twinx()

    left_line = left_ax.plot(
        compare_df["date"],
        compare_df["benchmark_close"],
        color="#1f77b4",
        linewidth=1.5,
        label="Official close",
    )
    right_line = right_ax.plot(
        compare_df["date"],
        compare_df["mimic_close"],
        color="#d62728",
        linewidth=1.5,
        label="Mimic close",
    )

    left_ax.set_title(title)
    left_ax.set_xlabel("Date")
    left_ax.set_ylabel("Official close")
    right_ax.set_ylabel("Mimic close")
    left_ax.grid(True, alpha=0.25)

    lines = left_line + right_line
    labels = [line.get_label() for line in lines]
    left_ax.legend(lines, labels, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rebased_plot(compare_df: pd.DataFrame, title: str, output_path: Path) -> None:
    plot_df = compare_df.copy()
    official_base = plot_df["benchmark_close"].iloc[0]
    mimic_base = plot_df["mimic_close"].iloc[0]
    if official_base == 0 or mimic_base == 0:
        raise ValueError(f"Cannot rebase {title}: first close value is zero.")

    plot_df["official_rebased"] = plot_df["benchmark_close"] / official_base * 100.0
    plot_df["mimic_rebased"] = plot_df["mimic_close"] / mimic_base * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plot_df["date"],
        plot_df["official_rebased"],
        color="#1f77b4",
        linewidth=1.5,
        label="Official close rebased",
    )
    ax.plot(
        plot_df["date"],
        plot_df["mimic_rebased"],
        color="#d62728",
        linewidth=1.5,
        label="Mimic close rebased",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased close, first overlap date = 100")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_style_outputs(
    style_name: str,
    compare_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    csv_path = output_dir / f"{style_name}_close_compare.csv"
    dual_axis_path = output_dir / f"{style_name}_close_dual_axis.png"
    rebased_path = output_dir / f"{style_name}_close_rebased.png"

    compare_df.to_csv(csv_path, index=False)
    save_dual_axis_plot(
        compare_df,
        f"{style_name.title()} Style Close Comparison",
        dual_axis_path,
    )
    save_rebased_plot(
        compare_df,
        f"{style_name.title()} Style Close Comparison, Rebased",
        rebased_path,
    )

    print(
        f"{style_name}: {len(compare_df)} rows, "
        f"{compare_df['date'].min().date()} to {compare_df['date'].max().date()}"
    )
    print(f"  csv: {csv_path}")
    print(f"  dual-axis plot: {dual_axis_path}")
    print(f"  rebased plot: {rebased_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    growth_compare_df = build_compare_frame(args.growth_factor_path, args.growth_index_path)
    value_compare_df = build_compare_frame(args.value_factor_path, args.value_index_path)

    save_style_outputs("growth", growth_compare_df, output_dir)
    save_style_outputs("value", value_compare_df, output_dir)


if __name__ == "__main__":
    main()
