"""Generate the standalone daily benchmark position file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SY_BASELINE_DIR = PROJECT_ROOT / "SY_Baseline"

for path in (PROJECT_ROOT, SY_BASELINE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Config import Config  # noqa: E402
from C_positions.benchmark_utils import build_daily_benchmark_position  # noqa: E402


DEFAULT_DATA_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "data_df.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "C_positions" / "output" / "benchmark_positions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate daily 50/50 benchmark positions on data_df.index."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to data_df.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Parquet output path for the benchmark position file.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["buy_and_hold_50", "rebalance_50"],
        default="rebalance_50",
        help="Benchmark rebalance rule.",
    )
    parser.add_argument(
        "--rebalance-months",
        type=int,
        default=2,
        help="Month interval for rebalance_50 mode.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Only write parquet; skip the sidecar CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = args.data_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data_df parquet: {data_path}")

    output_path = args.output_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    data_df = pd.read_parquet(data_path)
    benchmark_position_df = build_daily_benchmark_position(
        data_df=data_df,
        start_date=Config.ALL_START,
        end_date=Config.ALL_END,
        benchmark_mode=args.benchmark_mode,
        rebalance_months=args.rebalance_months,
        weight_tol=1e-10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_position_df.to_parquet(output_path)

    csv_path = output_path.with_suffix(".csv")
    if not args.no_csv:
        benchmark_position_df.to_csv(csv_path, index=True)

    print(f"benchmark positions rows: {len(benchmark_position_df)}")
    print(f"date range: {benchmark_position_df.index.min()} -> {benchmark_position_df.index.max()}")
    print(f"parquet written: {output_path}")
    if not args.no_csv:
        print(f"csv written: {csv_path}")


if __name__ == "__main__":
    main()
