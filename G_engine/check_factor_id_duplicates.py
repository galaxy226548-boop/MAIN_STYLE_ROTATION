"""Check duplicate factor ids in generated score/reference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    (PROJECT_ROOT / "D_analysis" / "check_output" / "IC_score.xlsx", "factor_id"),
    (PROJECT_ROOT / "F_grouping" / "reference" / "backtesting_score.xlsx", "factor_name"),
    (PROJECT_ROOT / "F_grouping" / "reference" / "usable_factors.xlsx", "factor_id"),
]
FACTOR_GENERATED_PATH = PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check duplicate factor ids in final outputs.")
    parser.add_argument(
        "--allow-generated-prefix-duplicates",
        action="store_true",
        help="Allow factor_generated duplicates when they belong to different generated output prefixes.",
    )
    return parser.parse_args()


def duplicate_counts(values: pd.Series) -> pd.Series:
    counts = values.dropna().astype(str).value_counts()
    return counts[counts > 1]


def check_xlsx(path: Path, column: str) -> list[str]:
    if not path.exists():
        return [f"missing file: {path}"]
    df = pd.read_excel(path)
    if column not in df.columns:
        return [f"{path} missing column {column!r}"]
    duplicates = duplicate_counts(df[column])
    if duplicates.empty:
        print(f"OK {path.relative_to(PROJECT_ROOT)}: no duplicate {column}")
        return []
    return [f"{path.relative_to(PROJECT_ROOT)} duplicate {column}: {duplicates.to_dict()}"]


def check_factor_generated(allow_prefix_duplicates: bool) -> list[str]:
    if not FACTOR_GENERATED_PATH.exists():
        return [f"missing file: {FACTOR_GENERATED_PATH}"]

    payload = json.loads(FACTOR_GENERATED_PATH.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    rows = [
        {
            "factor_id": str(record.get("factor_id")),
            "prefix": str(record.get("_generated_output_prefix") or ""),
        }
        for record in records
        if isinstance(record, dict) and record.get("factor_id")
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"OK {FACTOR_GENERATED_PATH.relative_to(PROJECT_ROOT)}: no factor_id records")
        return []

    if allow_prefix_duplicates:
        duplicates = df.groupby(["factor_id", "prefix"]).size()
        duplicates = duplicates[duplicates > 1]
        label = "factor_id/prefix"
    else:
        duplicates = duplicate_counts(df["factor_id"])
        label = "factor_id"

    if len(duplicates) == 0:
        print(f"OK {FACTOR_GENERATED_PATH.relative_to(PROJECT_ROOT)}: no duplicate {label}")
        return []
    return [f"{FACTOR_GENERATED_PATH.relative_to(PROJECT_ROOT)} duplicate {label}: {duplicates.to_dict()}"]


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    for path, column in CHECKS:
        errors.extend(check_xlsx(path, column))
    errors.extend(check_factor_generated(args.allow_generated_prefix_duplicates))

    if errors:
        print("Duplicate check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Duplicate check passed.")


if __name__ == "__main__":
    main()
