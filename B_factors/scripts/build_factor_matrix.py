"""Build normalized factor matrix, mounted factors, and long/short signals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from factor_utils import (
    build_threshold_signal_ls_df,
    load_benchmark_index,
    load_default_data,
    load_record_all_factor_metadata_with_records,
    mount_factor_source_frame,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)
from paper_odds_win_style_rotation import (
    PAPER_ID,
    generate_paper_odds_win_style_rotation_factor_source_frame,
)


def generate_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    factor_frames = [
        generate_paper_odds_win_style_rotation_factor_source_frame(data_df),
    ]
    factor_source_df = pd.concat(factor_frames, axis=1)
    duplicated_cols = factor_source_df.columns[factor_source_df.columns.duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"factor source columns duplicated: {duplicated_cols}")
    return factor_source_df


def _print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
    print(f"{label} mounted_normalized_factor_df shape:", mounted_factor_df.shape)
    print(f"{label} signal_ls_df shape:", signal_ls_df.shape)
    print(f"{label} factor columns:", list(mounted_factor_df.columns))
    print(f"{label} factor non-null summary:")
    for factor_col in mounted_factor_df.columns:
        series = mounted_factor_df[factor_col]
        print(
            factor_col,
            "non_na=", int(series.notna().sum()),
            "first=", series.first_valid_index(),
            "last=", series.last_valid_index(),
        )


def main() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df = generate_factor_source_frame(data_df)
    factor_metadata, missing_bar_defaults, selected_records = load_record_all_factor_metadata_with_records(
        PAPER_ID,
        list(factor_source_df.columns),
    )
    mounted_normalized_factor_df = mount_factor_source_frame(
        factor_source_df=factor_source_df,
        market_df=market_df,
        benchmark_index=benchmark_index,
        metadata=factor_metadata,
    )
    signal_ls_df = build_threshold_signal_ls_df(mounted_normalized_factor_df, factor_metadata)
    output_paths = save_factor_outputs(
        mounted_normalized_factor_df=mounted_normalized_factor_df,
        signal_ls_df=signal_ls_df,
        missing_bar_defaults=missing_bar_defaults,
        output_prefix=PAPER_ID,
        write_empty_missing_bar_file=False,
    )

    for label, path in output_paths.items():
        print(f"{label} saved to:", path)
    generated_path = save_generated_factor_records(selected_records, PAPER_ID)
    print("generated records saved to:", generated_path)
    _print_factor_output_summary(PAPER_ID, mounted_normalized_factor_df, signal_ls_df)


if __name__ == "__main__":
    main()
