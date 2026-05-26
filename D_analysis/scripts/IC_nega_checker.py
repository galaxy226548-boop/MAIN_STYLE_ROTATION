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

    cleanup_removed_duplicate_outputs()
    export_nega_doubt_factor_records(doubt_path)


def read_removed_duplicate_factor_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    factor_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3 or cells[1] == "Removed factor":
            continue

        removed_factor = cells[1]
        if removed_factor:
            factor_ids.add(removed_factor)

    return factor_ids


def factor_group_from_id(factor_id: str) -> str | None:
    for char in factor_id:
        if char.isalpha():
            return char.upper()
    return None


def remove_path(path: Path) -> bool:
    import shutil

    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def cleanup_removed_duplicate_outputs() -> None:
    duplicates_path = PROJECT_ROOT / "B_factors" / "output" / "grouping_document" / "RemoveDuplicates.md"
    if not duplicates_path.exists():
        print(f"Duplicate cleanup skipped: {duplicates_path} does not exist")
        return

    removed_factor_ids = read_removed_duplicate_factor_ids(duplicates_path)
    ic_deleted = 0
    backtest_deleted = 0

    ic_root = PROJECT_ROOT / "D_analysis" / "IC_output"
    backtest_root = PROJECT_ROOT / "E_backtesting" / "Result"

    for factor_id in sorted(removed_factor_ids):
        group = factor_group_from_id(factor_id)
        if group is None:
            continue

        ic_dir = ic_root / f"factor_{group}"
        for ic_path in (
            ic_dir / f"{factor_id}_IC_analysis.xlsx",
            ic_dir / f"{factor_id}_rolling_IC.png",
        ):
            if remove_path(ic_path):
                ic_deleted += 1

        backtest_dir = backtest_root / f"factor_{group}"
        if backtest_dir.exists():
            for backtest_path in backtest_dir.glob(f"*_*_{factor_id}"):
                if remove_path(backtest_path):
                    backtest_deleted += 1

    print(f"Duplicate cleanup factors loaded: {len(removed_factor_ids)}")
    print(f"Duplicate cleanup IC files deleted: {ic_deleted}")
    print(f"Duplicate cleanup backtest outputs deleted: {backtest_deleted}")


def export_nega_doubt_factor_records(doubt_path: Path) -> None:
    import json
    from datetime import datetime, timezone

    generated_path = PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json"
    output_path = PROJECT_ROOT / "D_analysis" / "check_output" / "nega_doubt_factors.json"

    if not doubt_path.exists():
        print(f"Nega doubt factor export skipped: {doubt_path} does not exist")
        return
    if not generated_path.exists():
        print(f"Nega doubt factor export skipped: {generated_path} does not exist")
        return

    doubt_ids = read_recorded_ids(doubt_path)
    generated_data = json.loads(generated_path.read_text(encoding="utf-8"))
    generated_records = generated_data.get("records", [])

    matched_records = [
        record
        for record in generated_records
        if str(record.get("factor_id", "")).strip() in doubt_ids
    ]
    matched_ids = {
        str(record.get("factor_id", "")).strip()
        for record in matched_records
        if str(record.get("factor_id", "")).strip()
    }

    output_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(generated_path.relative_to(PROJECT_ROOT)),
        "doubt_file": str(doubt_path.relative_to(PROJECT_ROOT)),
        "doubt_factor_count": len(doubt_ids),
        "matched_record_count": len(matched_records),
        "missing_factor_ids": sorted(doubt_ids - matched_ids),
        "records": matched_records,
    }

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Nega doubt factor records exported: {output_path}")
    print(f"Nega doubt factor records matched: {len(matched_records)}")
    print(f"Nega doubt factor ids missing: {len(doubt_ids - matched_ids)}")


if __name__ == "__main__":
    main()
