"""Post-process MarginIncrements summary with usable factor metadata.

Usage:
    python F_grouping/scripts/postprocess_marginal_summary.py

The script reads the latest marginal_contribution_summary.xlsx, inserts the
usable factor name as the first column, appends the rest of usable_factors.xlsx
metadata at the end, and writes the summary directly under:

    F_grouping/marginal_test/output/marginal_contribution_summary.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

OUTPUT_DIR = PROJECT_ROOT / "F_grouping" / "marginal_test" / "output"
SUMMARY_PATH = OUTPUT_DIR / "marginal_contribution_summary.xlsx"
LEGACY_SUMMARY_PATH = OUTPUT_DIR / "marginal_contribution" / "marginal_contribution_summary.xlsx"
USABLE_FACTORS_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "usable_factors.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich marginal contribution summary with usable factor metadata."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Summary workbook to read and overwrite.",
    )
    parser.add_argument(
        "--usable-factors-path",
        type=Path,
        default=USABLE_FACTORS_PATH,
        help="usable_factors.xlsx path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Output workbook path.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_summary_input(path: Path) -> Path:
    if path.exists():
        return path
    if LEGACY_SUMMARY_PATH.exists():
        print(f"summary not found at {path}; using legacy summary: {LEGACY_SUMMARY_PATH}")
        return LEGACY_SUMMARY_PATH
    raise FileNotFoundError(f"Cannot find summary workbook: {path}")


def clean_factor_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def build_usable_metadata(usable_path: Path) -> pd.DataFrame:
    if not usable_path.exists():
        raise FileNotFoundError(f"Cannot find usable factor workbook: {usable_path}")

    usable_df = pd.read_excel(usable_path)
    required_cols = ["factor_name", "factor_id"]
    missing_cols = [col for col in required_cols if col not in usable_df.columns]
    if missing_cols:
        raise KeyError(f"usable_factors.xlsx is missing required columns: {missing_cols}")

    usable_df = usable_df.copy()
    usable_df["factor_id"] = clean_factor_id(usable_df["factor_id"])
    usable_df = usable_df.drop_duplicates(subset=["factor_id"], keep="first")

    metadata_cols = [col for col in usable_df.columns if col not in {"factor_name", "factor_id"}]
    metadata_df = usable_df[["factor_id", "factor_name", *metadata_cols]].copy()
    metadata_df = metadata_df.rename(
        columns={col: f"usable_{col}" for col in metadata_cols}
    )
    return metadata_df


def enrich_summary(summary_path: Path, usable_path: Path) -> pd.DataFrame:
    summary_df = pd.read_excel(summary_path)
    if "candidate_factor" not in summary_df.columns:
        raise KeyError("summary workbook is missing required column: candidate_factor")

    summary_df = summary_df.copy()
    summary_df["candidate_factor"] = clean_factor_id(summary_df["candidate_factor"])
    summary_df = summary_df.drop(columns=["factor_name"], errors="ignore")

    usable_metadata_df = build_usable_metadata(usable_path)
    enriched_df = summary_df.merge(
        usable_metadata_df,
        how="left",
        left_on="candidate_factor",
        right_on="factor_id",
        sort=False,
    )

    factor_name = enriched_df.pop("factor_name")
    enriched_df.insert(0, "factor_name", factor_name)
    enriched_df = enriched_df.drop(columns=["factor_id"])
    return enriched_df


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_input(resolve_project_path(args.summary_path))
    usable_path = resolve_project_path(args.usable_factors_path)
    output_path = resolve_project_path(args.output_path)

    enriched_df = enrich_summary(summary_path=summary_path, usable_path=usable_path)
    write_summary(enriched_df, output_path)

    print(f"summary input: {summary_path}")
    print(f"usable factors: {usable_path}")
    print(f"rows: {len(enriched_df)}")
    print(f"columns: {len(enriched_df.columns)}")
    print(f"output saved to: {output_path}")


if __name__ == "__main__":
    main()
