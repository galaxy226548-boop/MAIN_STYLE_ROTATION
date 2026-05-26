from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "B_factors" / "output" / "grouping_document"
DEFAULT_REPORT_PATH = DEFAULT_INPUT_DIR / "RemoveDuplicates.md"
FILE_PATTERN = re.compile(
    r"^factor_(?P<group>[A-Za-z]+)_(?P<kind>mounted_normalized_factors|signal_ls)\.parquet$"
)
FACTOR_ID_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<number>\d+)$")


@dataclass(frozen=True)
class DuplicateRecord:
    group: str
    removed_factor: str
    kept_factor: str


@dataclass(frozen=True)
class GroupFiles:
    group: str
    factor_path: Path
    signal_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate factor columns from grouping_document parquet files. "
            "A duplicate is removed only when both mounted_normalized_factors and "
            "signal_ls columns are exactly identical."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing factor_* parquet files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and write the report without modifying parquet files.",
    )
    return parser.parse_args()


def factor_sort_key(factor_id: str) -> tuple[str, int, str]:
    match = FACTOR_ID_PATTERN.match(factor_id)
    if not match:
        return factor_id, 10**12, factor_id
    return match.group("prefix"), int(match.group("number")), factor_id


def discover_group_files(input_dir: Path) -> tuple[list[GroupFiles], list[Path], list[Path]]:
    grouped_paths: dict[str, dict[str, Path]] = {}
    parquet_paths = sorted(input_dir.glob("*.parquet"))
    ignored_paths: list[Path] = []

    for path in parquet_paths:
        match = FILE_PATTERN.match(path.name)
        if not match:
            ignored_paths.append(path)
            continue
        grouped_paths.setdefault(match.group("group"), {})[match.group("kind")] = path

    group_files: list[GroupFiles] = []
    for group, paths_by_kind in sorted(grouped_paths.items()):
        factor_path = paths_by_kind.get("mounted_normalized_factors")
        signal_path = paths_by_kind.get("signal_ls")
        if factor_path is None or signal_path is None:
            ignored_paths.extend(paths_by_kind.values())
            continue
        group_files.append(GroupFiles(group=group, factor_path=factor_path, signal_path=signal_path))

    return group_files, sorted(ignored_paths), parquet_paths


def values_equal(left: pd.Series, right: pd.Series) -> bool:
    return left.equals(right)


def find_duplicate_records(group_files: GroupFiles) -> list[DuplicateRecord]:
    factor_df = pd.read_parquet(group_files.factor_path)
    signal_df = pd.read_parquet(group_files.signal_path)

    common_columns = sorted(set(factor_df.columns).intersection(signal_df.columns), key=factor_sort_key)
    duplicate_records: list[DuplicateRecord] = []
    kept_factors: list[str] = []

    for factor_id in common_columns:
        matched_kept_factor = None
        for kept_factor in kept_factors:
            same_factor_values = values_equal(factor_df[factor_id], factor_df[kept_factor])
            same_signal_values = values_equal(signal_df[factor_id], signal_df[kept_factor])
            if same_factor_values and same_signal_values:
                matched_kept_factor = kept_factor
                break

        if matched_kept_factor is None:
            kept_factors.append(factor_id)
            continue

        duplicate_records.append(
            DuplicateRecord(
                group=group_files.group,
                removed_factor=factor_id,
                kept_factor=matched_kept_factor,
            )
        )

    return duplicate_records


def drop_duplicate_columns(parquet_paths: list[Path], duplicate_records: list[DuplicateRecord]) -> None:
    removed_columns = {record.removed_factor for record in duplicate_records}
    if not removed_columns:
        return

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        columns_to_drop = [column for column in df.columns if column in removed_columns]
        if not columns_to_drop:
            continue
        df = df.drop(columns=columns_to_drop)
        df.to_parquet(parquet_path)


def build_report(
    group_files: list[GroupFiles],
    duplicate_records: list[DuplicateRecord],
    ignored_paths: list[Path],
    dry_run: bool,
) -> str:
    lines = [
        "# RemoveDuplicates",
        "",
        f"- Mode: {'dry-run, parquet files not modified' if dry_run else 'applied, parquet files modified'}",
        f"- Scanned groups: {', '.join(files.group for files in group_files) if group_files else 'none'}",
        f"- Removed duplicate factors: {len(duplicate_records)}",
        "",
    ]

    if duplicate_records:
        lines.extend(
            [
                "## Removed Factors",
                "",
                "| Group | Removed factor | Kept factor | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for record in duplicate_records:
            lines.append(
                f"| {record.group} | {record.removed_factor} | {record.kept_factor} | "
                "mounted_normalized_factors and signal_ls are both identical |"
            )
        lines.append("")
    else:
        lines.extend(["## Removed Factors", "", "No duplicate factors found.", ""])

    if ignored_paths:
        lines.extend(["## Ignored Parquet Files", ""])
        for path in ignored_paths:
            lines.append(f"- {path.name}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    report_path = args.report_path.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    group_files, ignored_paths, parquet_paths = discover_group_files(input_dir)
    duplicate_records: list[DuplicateRecord] = []
    for files in group_files:
        duplicate_records.extend(find_duplicate_records(files))

    if not args.dry_run:
        drop_duplicate_columns(parquet_paths, duplicate_records)

    report = build_report(
        group_files=group_files,
        duplicate_records=duplicate_records,
        ignored_paths=ignored_paths,
        dry_run=args.dry_run,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(duplicate_records)} duplicate factor columns.")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
