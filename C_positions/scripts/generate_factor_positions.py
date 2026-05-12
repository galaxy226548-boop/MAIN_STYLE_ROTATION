"""Generate one target-position Excel file per factor signal column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SIGNAL_PATH = PROJECT_ROOT / "B_factors" / "output" / "zhao_signal_ls.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "C_positions" / "output" / "factor_positions"

OUTPUT_COLUMNS = [
    "signal_ls",
    "target_growth_weight",
    "target_value_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-factor target-position Excel files from a signal_ls matrix."
    )
    parser.add_argument(
        "--signal-path",
        type=Path,
        default=DEFAULT_SIGNAL_PATH,
        help="Path to a wide parquet file whose rows are dates and columns are factor ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for {factor_id}_position.xlsx output files.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_factor_position_df(signal_series: pd.Series) -> pd.DataFrame:
    signal_ls = pd.to_numeric(signal_series, errors="coerce")
    position_df = pd.DataFrame(index=signal_series.index)
    position_df["signal_ls"] = signal_ls
    position_df["target_growth_weight"] = 0.5 + 0.5 * signal_ls
    position_df["target_value_weight"] = 0.5 - 0.5 * signal_ls
    return position_df[OUTPUT_COLUMNS]


def validate_factor_position_df(factor_id: str, position_df: pd.DataFrame) -> None:
    signal_ls = position_df["signal_ls"]
    target_growth_weight = position_df["target_growth_weight"]
    target_value_weight = position_df["target_value_weight"]

    signal_nan_mask = signal_ls.isna()
    target_nan_mask = target_growth_weight.isna() & target_value_weight.isna()
    if not signal_nan_mask.equals(target_nan_mask):
        raise ValueError(
            f"{factor_id}: target weights must both be NaN exactly where signal_ls is NaN"
        )

    non_null_mask = signal_ls.notna()
    weight_sum = target_growth_weight.loc[non_null_mask] + target_value_weight.loc[non_null_mask]
    if not weight_sum.empty and not ((weight_sum - 1.0).abs() < 1e-12).all():
        raise ValueError(f"{factor_id}: non-null target weights do not sum to 1")


def format_signal_summary(signal_series: pd.Series) -> str:
    total_count = len(signal_series)
    non_null_count = int(signal_series.notna().sum())
    nan_ratio = signal_series.isna().mean() if total_count else 0.0
    valid_signal = signal_series.dropna()

    if valid_signal.empty:
        value_summary = "min=NaN, median=NaN, max=NaN"
    else:
        value_summary = (
            f"min={valid_signal.min():.6g}, "
            f"median={valid_signal.median():.6g}, "
            f"max={valid_signal.max():.6g}"
        )

    return (
        f"rows={total_count}, non_null={non_null_count}, "
        f"nan_ratio={nan_ratio:.2%}, {value_summary}"
    )


def main() -> None:
    args = parse_args()

    signal_path = resolve_project_path(args.signal_path)
    output_dir = resolve_project_path(args.output_dir)

    if not signal_path.exists():
        raise FileNotFoundError(f"Cannot find signal parquet: {signal_path}")

    signal_df = pd.read_parquet(signal_path).sort_index()
    signal_df.index = pd.to_datetime(signal_df.index)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"signal path: {signal_path}")
    print(f"signal shape: {signal_df.shape}")
    print(f"date range: {signal_df.index.min()} -> {signal_df.index.max()}")
    print(f"output dir: {output_dir}")

    for factor_id in signal_df.columns:
        position_df = build_factor_position_df(signal_df[factor_id])
        validate_factor_position_df(str(factor_id), position_df)

        output_path = output_dir / f"{factor_id}_position.xlsx"
        position_df.to_excel(output_path, index=True)

        summary = format_signal_summary(position_df["signal_ls"])
        print(f"{factor_id}: {summary}")
        print(f"written: {output_path}")


if __name__ == "__main__":
    main()
