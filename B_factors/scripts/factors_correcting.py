"""Flip factor and signal directions in grouped parquet outputs.

This script fixes factors whose value direction was previously emitted with
the wrong sign. It reads grouped output files under
``B_factors/output/grouping_document`` and multiplies the matching factor
columns by ``-1`` in both mounted factor and signal parquet files.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_GROUPING_DIR = PROJECT_ROOT / "B_factors" / "output" / "grouping_document"
DEFAULT_FACTORS_TO_FLIP = (
    "D001",
    "L028",
    "F006",
    "F008",
    "F009",
    "F012",
    "P012",
    "P041",
    "I005",
    "I006",
    "I009",
    "C033",
    "C034",
    "I074",
    "C024",
    "G044",
    "G049",
    "G059",
    "G066",
    "V022",
)

MOUNTED_TEMPLATE = "factor_{category}_mounted_normalized_factors.parquet"
SIGNAL_TEMPLATE = "factor_{category}_signal_ls.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flip selected factor columns in grouped mounted-factor and "
            "signal parquet files."
        )
    )
    parser.add_argument(
        "--grouping-dir",
        type=Path,
        default=DEFAULT_GROUPING_DIR,
        help="Directory containing factor_{category}_*.parquet grouped outputs.",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=list(DEFAULT_FACTORS_TO_FLIP),
        help="Factor IDs to flip. Defaults to the known correction list.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing files.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn instead of failing when a file or factor column is missing.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy original parquet files before writing corrections.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help=(
            "Directory for original parquet backups. Defaults to a timestamped "
            "folder under the grouping directory."
        ),
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normalize_factor_id(factor_id: str) -> str:
    normalized = factor_id.strip().upper()
    if len(normalized) < 2 or not normalized[0].isalpha() or not normalized[1:].isdigit():
        raise ValueError(
            f"Unsupported factor_id {factor_id!r}; expected one letter followed by digits."
        )
    return normalized


def _group_by_category(factor_ids: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for factor_id in factor_ids:
        normalized = _normalize_factor_id(factor_id)
        grouped.setdefault(normalized[0], [])
        if normalized not in grouped[normalized[0]]:
            grouped[normalized[0]].append(normalized)
    return grouped


def _category_paths(grouping_dir: Path, category: str) -> tuple[Path, Path]:
    mounted_path = grouping_dir / MOUNTED_TEMPLATE.format(category=category)
    signal_path = grouping_dir / SIGNAL_TEMPLATE.format(category=category)
    return mounted_path, signal_path


def _check_targets(
    grouping_dir: Path,
    factors_by_category: dict[str, list[str]],
    allow_missing: bool,
) -> dict[Path, list[str]]:
    targets: dict[Path, list[str]] = {}
    missing: list[str] = []

    if not grouping_dir.exists():
        raise FileNotFoundError(f"Grouping directory does not exist: {grouping_dir}")

    for category, factor_ids in sorted(factors_by_category.items()):
        for path in _category_paths(grouping_dir, category):
            if not path.exists():
                missing.append(f"missing file: {_relative(path)}")
                continue

            columns = set(pd.read_parquet(path).columns)
            missing_columns = [factor_id for factor_id in factor_ids if factor_id not in columns]
            if missing_columns:
                missing.append(
                    f"{_relative(path)} missing columns: {', '.join(missing_columns)}"
                )
                continue

            targets[path] = factor_ids

    if missing:
        message = "Correction target check found missing items:\n" + "\n".join(
            f"- {item}" for item in missing
        )
        if not allow_missing:
            raise FileNotFoundError(message)
        print(message)

    return targets


def _default_backup_dir(grouping_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return grouping_dir / f"factors_correcting_backup_{timestamp}"


def _backup_files(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, backup_dir / path.name)


def _flip_columns(path: Path, factor_ids: list[str]) -> None:
    df = pd.read_parquet(path)
    for factor_id in factor_ids:
        df[factor_id] = df[factor_id] * -1
    df.to_parquet(path)


def main() -> None:
    args = parse_args()
    grouping_dir = args.grouping_dir.resolve()
    factors_by_category = _group_by_category(args.factors)
    targets = _check_targets(
        grouping_dir=grouping_dir,
        factors_by_category=factors_by_category,
        allow_missing=args.allow_missing,
    )

    if not targets:
        print("No correction targets found.")
        return

    print("Correction targets:")
    for path, factor_ids in sorted(targets.items()):
        print(f"- {_relative(path)}: {', '.join(factor_ids)}")

    if args.dry_run:
        print("Dry run only; no files were changed.")
        return

    if not args.no_backup:
        backup_dir = (args.backup_dir or _default_backup_dir(grouping_dir)).resolve()
        _backup_files(sorted(targets), backup_dir)
        print(f"Backed up originals to {_relative(backup_dir)}")

    for path, factor_ids in sorted(targets.items()):
        _flip_columns(path, factor_ids)
        print(f"Updated {_relative(path)}")


if __name__ == "__main__":
    main()
