"""Check IC analysis outputs for factors with likely negative IC behavior.

Inputs:
    D_analysis/IC_output/**/*_IC_analysis.xlsx

Outputs:
    D_analysis/check_output/nega_doubt.md
    D_analysis/check_output/nega_checked.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "D_analysis" / "IC_output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "D_analysis" / "check_output"
DOUBT_FILENAME = "nega_doubt.md"
CHECKED_FILENAME = "nega_checked.md"
ANALYSIS_SUFFIX = "_IC_analysis.xlsx"
PROB_PREFIX = "pos_ic_prob_track"
THRESHOLD = 0.5
MIN_NEGATIVE_RATIO = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check IC analysis positive-IC probability columns.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing IC analysis xlsx files. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for nega_doubt.md and nega_checked.md. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-check every IC analysis file and rewrite both output markdown files.",
    )
    return parser.parse_args()


def factor_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(ANALYSIS_SUFFIX):
        raise ValueError(f"Unexpected IC analysis filename: {path}")
    return name[: -len(ANALYSIS_SUFFIX)]


def read_recorded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def write_recorded_ids(path: Path, factor_ids: set[str]) -> None:
    path.write_text(
        "".join(f"{factor_id}\n" for factor_id in sorted(factor_ids)),
        encoding="utf-8",
    )


def track_has_negative_doubt(values: pd.Series) -> bool:
    first_value = values.iloc[0]
    if pd.isna(first_value) or first_value >= THRESHOLD:
        return False

    later_values = values.iloc[1:].dropna()
    if later_values.empty:
        return True

    negative_ratio = (later_values < THRESHOLD).mean()
    return bool(negative_ratio >= MIN_NEGATIVE_RATIO)


def has_negative_doubt(path: Path) -> bool:
    df = pd.read_excel(path, sheet_name="IC analysis", index_col=0)
    prob_cols = [col for col in df.columns if str(col).startswith(PROB_PREFIX)]
    if df.empty or not prob_cols:
        return False

    for col in prob_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if track_has_negative_doubt(values):
            return True

    return False


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    doubt_path = output_dir / DOUBT_FILENAME
    checked_path = output_dir / CHECKED_FILENAME

    doubt_ids = set() if args.rebuild else read_recorded_ids(doubt_path)
    checked_ids = set() if args.rebuild else read_recorded_ids(checked_path)
    already_seen = doubt_ids | checked_ids

    newly_doubted: set[str] = set()
    newly_checked: set[str] = set()
    errors: list[str] = []

    for path in sorted(input_dir.rglob(f"*{ANALYSIS_SUFFIX}")):
        factor_id = factor_id_from_path(path)
        if not args.rebuild and factor_id in already_seen:
            continue

        try:
            if has_negative_doubt(path):
                newly_doubted.add(factor_id)
            else:
                newly_checked.add(factor_id)
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    doubt_ids.update(newly_doubted)
    checked_ids.update(newly_checked)
    checked_ids.difference_update(doubt_ids)

    write_recorded_ids(doubt_path, doubt_ids)
    write_recorded_ids(checked_path, checked_ids)

    run_mode = "rebuild" if args.rebuild else "incremental"
    print(f"Mode: {run_mode}")
    print(f"Scanned files: {len(newly_doubted) + len(newly_checked)}")
    print(f"Doubt factors found: {len(newly_doubted)}")
    print(f"Checked factors found: {len(newly_checked)}")
    print(f"Doubt output: {doubt_path}")
    print(f"Checked output: {checked_path}")

    if errors:
        print("Files skipped due to errors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
