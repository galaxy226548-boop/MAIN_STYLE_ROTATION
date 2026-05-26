"""Move mis-grouped factor columns and generated outputs to a new category.

The default correction moves selected factors into the configured target group.
It appends them after the current maximum target-category factor id, updates
grouped parquet columns, updates ``factor_generated.json`` factor/signal ids,
and moves matching backtest and IC output filenames between category folders.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_GROUPING_DIR = PROJECT_ROOT / "B_factors" / "output" / "grouping_document"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json"
DEFAULT_BACKTEST_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"
DEFAULT_IC_ROOT = PROJECT_ROOT / "D_analysis" / "IC_output"

DEFAULT_OLD_FACTORS = ("V179", "V180")
DEFAULT_TARGET_CATEGORY = "P"

MOUNTED_TEMPLATE = "factor_{category}_mounted_normalized_factors.parquet"
SIGNAL_TEMPLATE = "factor_{category}_signal_ls.parquet"
FACTOR_ID_PATTERN = re.compile(r"^(?P<category>[A-Za-z])(?P<number>\d+)$")
METADATA_ID_FIELDS = ("factor_id", "signal_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regroup selected factor ids from one grouped output category to another."
    )
    parser.add_argument("--grouping-dir", type=Path, default=DEFAULT_GROUPING_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--backtest-root", type=Path, default=DEFAULT_BACKTEST_ROOT)
    parser.add_argument("--ic-root", type=Path, default=DEFAULT_IC_ROOT)
    parser.add_argument(
        "--old-factor",
        action="append",
        dest="old_factors",
        help="Old factor id to regroup. Can be passed multiple times.",
    )
    parser.add_argument(
        "--target-category",
        default=DEFAULT_TARGET_CATEGORY,
        help=f"New factor category letter. Defaults to {DEFAULT_TARGET_CATEGORY}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files or moving outputs.",
    )
    parser.add_argument(
        "--allow-missing-outputs",
        action="store_true",
        help="Warn instead of failing when matching backtest or IC outputs are missing.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy parquet/json inputs before writing corrections.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory. Defaults to a timestamped folder under grouping_document.",
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _parse_factor_id(factor_id: str) -> tuple[str, int, int]:
    normalized = factor_id.strip().upper()
    match = FACTOR_ID_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Unsupported factor_id {factor_id!r}; expected one letter followed by digits."
        )
    number_text = match.group("number")
    return match.group("category").upper(), int(number_text), len(number_text)


def _category_paths(grouping_dir: Path, category: str) -> tuple[Path, Path]:
    return (
        grouping_dir / MOUNTED_TEMPLATE.format(category=category),
        grouping_dir / SIGNAL_TEMPLATE.format(category=category),
    )


def _read_group_pair(grouping_dir: Path, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mounted_path, signal_path = _category_paths(grouping_dir, category)
    if not mounted_path.exists():
        raise FileNotFoundError(f"Missing mounted parquet: {_relative(mounted_path)}")
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing signal parquet: {_relative(signal_path)}")

    mounted_df = pd.read_parquet(mounted_path)
    signal_df = pd.read_parquet(signal_path)
    if list(map(str, mounted_df.columns)) != list(map(str, signal_df.columns)):
        raise ValueError(
            "Mounted and signal columns do not match for "
            f"factor_{category}: {_relative(mounted_path)} / {_relative(signal_path)}"
        )
    return mounted_df, signal_df


def _current_max_id(columns: list[str], category: str) -> tuple[int, int]:
    max_number = 0
    max_width = 3
    for column in columns:
        try:
            col_category, col_number, col_width = _parse_factor_id(str(column))
        except ValueError:
            continue
        if col_category == category:
            max_number = max(max_number, col_number)
            max_width = max(max_width, col_width)
    return max_number, max_width


def build_mapping(
    old_factors: list[str],
    target_category: str,
    target_columns: list[str],
) -> dict[str, str]:
    target_category = target_category.strip().upper()
    if len(target_category) != 1 or not target_category.isalpha():
        raise ValueError(f"Unsupported target category: {target_category!r}")

    max_number, width = _current_max_id(target_columns, target_category)
    target_existing = set(map(str, target_columns))
    mapping: dict[str, str] = {}
    for old_factor in old_factors:
        old_category, _, old_width = _parse_factor_id(old_factor)
        old_factor = old_factor.strip().upper()
        if old_category == target_category:
            raise ValueError(f"{old_factor} is already in target category {target_category}.")
        if old_factor in mapping:
            raise ValueError(f"Duplicate old factor id in request: {old_factor}")

        width = max(width, old_width)
        while True:
            max_number += 1
            new_factor = f"{target_category}{max_number:0{width}d}"
            if new_factor not in target_existing and new_factor not in mapping.values():
                break
        mapping[old_factor] = new_factor
    return mapping


def validate_source_columns(
    source_mounted_df: pd.DataFrame,
    source_signal_df: pd.DataFrame,
    mapping: dict[str, str],
) -> None:
    source_columns = set(map(str, source_mounted_df.columns))
    signal_columns = set(map(str, source_signal_df.columns))
    missing_source = sorted(set(mapping) - source_columns)
    missing_signal = sorted(set(mapping) - signal_columns)
    if missing_source or missing_signal:
        raise KeyError(
            "Missing regrouping columns. "
            f"Mounted missing: {missing_source}; signal missing: {missing_signal}"
        )


def update_group_parquets(
    grouping_dir: Path,
    source_category: str,
    target_category: str,
    mapping: dict[str, str],
) -> None:
    source_mounted_df, source_signal_df = _read_group_pair(grouping_dir, source_category)
    target_mounted_df, target_signal_df = _read_group_pair(grouping_dir, target_category)
    validate_source_columns(source_mounted_df, source_signal_df, mapping)

    for old_factor, new_factor in mapping.items():
        target_mounted_df[new_factor] = source_mounted_df[old_factor]
        target_signal_df[new_factor] = source_signal_df[old_factor]

    source_mounted_df = source_mounted_df.drop(columns=list(mapping))
    source_signal_df = source_signal_df.drop(columns=list(mapping))

    source_mounted_path, source_signal_path = _category_paths(grouping_dir, source_category)
    target_mounted_path, target_signal_path = _category_paths(grouping_dir, target_category)
    source_mounted_df.to_parquet(source_mounted_path)
    source_signal_df.to_parquet(source_signal_path)
    target_mounted_df.to_parquet(target_mounted_path)
    target_signal_df.to_parquet(target_signal_path)


def update_metadata(metadata_path: Path, mapping: dict[str, str]) -> int:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata json: {_relative(metadata_path)}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"Unsupported metadata shape in {_relative(metadata_path)}")

    update_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in METADATA_ID_FIELDS:
            old_value = str(record.get(field, "")).strip().upper()
            if old_value in mapping:
                record[field] = mapping[old_value]
                update_count += 1

    metadata_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return update_count


def _paths_containing_id(root: Path, old_factor: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(f"*{old_factor}*") if path.name != ".DS_Store")


def plan_output_moves(
    output_root: Path,
    source_category: str,
    target_category: str,
    mapping: dict[str, str],
) -> list[tuple[Path, Path]]:
    source_dir = output_root / f"factor_{source_category}"
    target_dir = output_root / f"factor_{target_category}"
    moves: list[tuple[Path, Path]] = []

    for old_factor, new_factor in mapping.items():
        for old_path in _paths_containing_id(source_dir, old_factor):
            relative = old_path.relative_to(source_dir)
            new_relative = Path(*(part.replace(old_factor, new_factor) for part in relative.parts))
            moves.append((old_path, target_dir / new_relative))

    # Move parents before children; after a directory move, rename files inside it.
    return sorted(moves, key=lambda item: (len(item[0].parts), str(item[0])))


def validate_output_moves(
    moves: list[tuple[Path, Path]],
    mapping: dict[str, str],
    output_label: str,
    allow_missing_outputs: bool,
) -> None:
    found_old_ids = {old for old, _ in mapping.items() if any(old in src.name for src, _ in moves)}
    missing = sorted(set(mapping) - found_old_ids)
    if missing:
        message = f"{output_label} outputs missing for: {', '.join(missing)}"
        if allow_missing_outputs:
            print(f"Warning: {message}")
        else:
            raise FileNotFoundError(message)

    for source_path, target_path in moves:
        if target_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {_relative(target_path)}")


def apply_output_moves(moves: list[tuple[Path, Path]], mapping: dict[str, str]) -> None:
    moved_dirs: list[tuple[Path, Path]] = []
    for source_path, target_path in moves:
        if any(source_path.is_relative_to(old_dir) for old_dir, _ in moved_dirs):
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        if target_path.is_dir():
            moved_dirs.append((source_path, target_path))

    for _, moved_dir in moved_dirs:
        nested_paths = sorted(
            [path for path in moved_dir.rglob("*") if path.is_file()],
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in nested_paths:
            new_name = path.name
            for old_factor, new_factor in mapping.items():
                new_name = new_name.replace(old_factor, new_factor)
            if new_name != path.name:
                target_path = path.with_name(new_name)
                if target_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing output: {_relative(target_path)}")
                path.rename(target_path)


def _default_backup_dir(grouping_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return grouping_dir / f"factors_regrouping_backup_{timestamp}"


def backup_inputs(grouping_dir: Path, metadata_path: Path, backup_dir: Path, categories: set[str]) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for category in sorted(categories):
        for path in _category_paths(grouping_dir, category):
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
    if metadata_path.exists():
        shutil.copy2(metadata_path, backup_dir / metadata_path.name)


def print_mapping(mapping: dict[str, str]) -> None:
    print("Regrouping mapping:")
    for old_factor, new_factor in mapping.items():
        print(f"- {old_factor} -> {new_factor}")


def print_moves(label: str, moves: list[tuple[Path, Path]]) -> None:
    print(f"{label} output moves: {len(moves)}")
    for source_path, target_path in moves:
        print(f"- {_relative(source_path)} -> {_relative(target_path)}")


def main() -> None:
    args = parse_args()
    grouping_dir = args.grouping_dir.resolve()
    metadata_path = args.metadata_path.resolve()
    backtest_root = args.backtest_root.resolve()
    ic_root = args.ic_root.resolve()
    old_factors = args.old_factors or list(DEFAULT_OLD_FACTORS)

    normalized_old_factors = [factor.strip().upper() for factor in old_factors]
    source_categories = {_parse_factor_id(factor)[0] for factor in normalized_old_factors}
    if len(source_categories) != 1:
        raise ValueError(f"Expected one source category, got: {sorted(source_categories)}")
    source_category = next(iter(source_categories))
    target_category = args.target_category.strip().upper()

    target_mounted_df, _ = _read_group_pair(grouping_dir, target_category)
    mapping = build_mapping(
        old_factors=normalized_old_factors,
        target_category=target_category,
        target_columns=list(map(str, target_mounted_df.columns)),
    )

    source_mounted_df, source_signal_df = _read_group_pair(grouping_dir, source_category)
    validate_source_columns(source_mounted_df, source_signal_df, mapping)

    backtest_moves = plan_output_moves(backtest_root, source_category, target_category, mapping)
    ic_moves = plan_output_moves(ic_root, source_category, target_category, mapping)
    validate_output_moves(backtest_moves, mapping, "Backtest", args.allow_missing_outputs)
    validate_output_moves(ic_moves, mapping, "IC", args.allow_missing_outputs)

    print_mapping(mapping)
    print_moves("Backtest", backtest_moves)
    print_moves("IC", ic_moves)

    if args.dry_run:
        print("Dry run only; no files were changed.")
        return

    if not args.no_backup:
        backup_dir = (args.backup_dir or _default_backup_dir(grouping_dir)).resolve()
        backup_inputs(
            grouping_dir=grouping_dir,
            metadata_path=metadata_path,
            backup_dir=backup_dir,
            categories={source_category, target_category},
        )
        print(f"Backed up grouped parquets and metadata to {_relative(backup_dir)}")

    update_group_parquets(grouping_dir, source_category, target_category, mapping)
    metadata_updates = update_metadata(metadata_path, mapping)

    apply_output_moves(backtest_moves, mapping)
    apply_output_moves(ic_moves, mapping)

    print(f"Updated metadata id fields: {metadata_updates}")
    print("Regrouping completed.")


if __name__ == "__main__":
    main()
