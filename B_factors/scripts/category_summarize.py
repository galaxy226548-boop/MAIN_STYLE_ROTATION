"""Group output parquet files by signal_id category.

This script scans paired ``*_signal_ls.parquet`` and
``*_mounted_normalized_factors.parquet`` files under ``B_factors/output``.
Columns are treated as signal_id values, grouped by their first letter, and
written to ``B_factors/output/grouping_document``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "B_factors" / "output"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "grouping_document"
DEFAULT_GENERATED_PATH = DEFAULT_INPUT_DIR / "factor_generated.json"

SIGNAL_SUFFIX = "_signal_ls.parquet"
MOUNTED_SUFFIX = "_mounted_normalized_factors.parquet"
SIGNAL_ID_PATTERN = re.compile(r"^(?P<category>[A-Za-z])(?P<number>\d+)$")
SCRIPT_NAME = "category_summarize.py"


@dataclass(frozen=True)
class ParquetPair:
    prefix: str
    signal_path: Path
    mounted_path: Path


@dataclass
class ColumnEntry:
    pair: ParquetPair
    position: int
    old_signal_id: str
    category: str
    number: int
    width: int
    new_signal_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group B_factors/output parquet columns by signal_id category and "
            "renumber duplicate signal_id values."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-path", type=Path, default=DEFAULT_GENERATED_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned renames and outputs without writing files.",
    )
    parser.add_argument(
        "--strict-generated-records",
        action="store_true",
        help=(
            "Abort when a renamed signal_id has no matching record in "
            "factor_generated.json. By default the script continues and records "
            "the unmatched signal_id values in the report."
        ),
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _parse_signal_id(signal_id: str) -> tuple[str, int, int]:
    match = SIGNAL_ID_PATTERN.fullmatch(signal_id)
    if match is None:
        raise ValueError(
            f"Unsupported signal_id {signal_id!r}; expected one letter followed by digits."
        )
    number_text = match.group("number")
    return match.group("category").upper(), int(number_text), len(number_text)


def discover_pairs(input_dir: Path) -> list[ParquetPair]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pairs: list[ParquetPair] = []
    for signal_path in sorted(input_dir.glob(f"*{SIGNAL_SUFFIX}")):
        if signal_path.parent != input_dir:
            continue
        prefix = signal_path.name[: -len(SIGNAL_SUFFIX)]
        mounted_path = input_dir / f"{prefix}{MOUNTED_SUFFIX}"
        if not mounted_path.exists():
            print(f"Skip {_relative(signal_path)}: missing paired mounted parquet.")
            continue
        pairs.append(ParquetPair(prefix=prefix, signal_path=signal_path, mounted_path=mounted_path))
    return pairs


def collect_column_entries(pairs: list[ParquetPair]) -> list[ColumnEntry]:
    entries: list[ColumnEntry] = []
    for pair in pairs:
        signal_df = pd.read_parquet(pair.signal_path)
        mounted_df = pd.read_parquet(pair.mounted_path)
        signal_columns = [str(column) for column in signal_df.columns]
        mounted_columns = [str(column) for column in mounted_df.columns]
        if signal_columns != mounted_columns:
            raise ValueError(
                "Column mismatch between paired files: "
                f"{_relative(pair.signal_path)} and {_relative(pair.mounted_path)}"
            )

        for position, signal_id in enumerate(signal_columns):
            category, number, width = _parse_signal_id(signal_id)
            entries.append(
                ColumnEntry(
                    pair=pair,
                    position=position,
                    old_signal_id=signal_id,
                    category=category,
                    number=number,
                    width=width,
                )
            )
    return entries


def assign_duplicate_signal_ids(entries: list[ColumnEntry]) -> list[ColumnEntry]:
    category_state: dict[str, dict[str, Any]] = {}
    for entry in entries:
        state = category_state.setdefault(
            entry.category,
            {"used": set(), "max_number": 0, "width": entry.width},
        )
        state["used"].add(entry.old_signal_id)
        state["max_number"] = max(state["max_number"], entry.number)
        state["width"] = max(state["width"], entry.width)

    seen_by_category: dict[str, set[str]] = {}
    renamed_entries: list[ColumnEntry] = []
    for entry in entries:
        seen = seen_by_category.setdefault(entry.category, set())
        if entry.old_signal_id not in seen:
            entry.new_signal_id = entry.old_signal_id
            seen.add(entry.old_signal_id)
            continue

        state = category_state[entry.category]
        used: set[str] = state["used"]
        width = max(int(state["width"]), entry.width)
        while True:
            state["max_number"] += 1
            candidate = f"{entry.category}{int(state['max_number']):0{width}d}"
            if candidate not in used:
                break
        used.add(candidate)
        seen.add(candidate)
        entry.new_signal_id = candidate
        renamed_entries.append(entry)
    return renamed_entries


def entries_by_pair(entries: list[ColumnEntry]) -> dict[ParquetPair, list[ColumnEntry]]:
    grouped: dict[ParquetPair, list[ColumnEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.pair, []).append(entry)
    return grouped


def _renamed_columns(columns: list[str], entries: list[ColumnEntry]) -> list[str]:
    renamed = list(columns)
    for entry in entries:
        new_signal_id = entry.new_signal_id or entry.old_signal_id
        renamed[entry.position] = new_signal_id
    return renamed


def _read_pair_with_final_columns(
    pair: ParquetPair,
    entries: list[ColumnEntry],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = pd.read_parquet(pair.signal_path)
    mounted_df = pd.read_parquet(pair.mounted_path)
    final_columns = _renamed_columns([str(column) for column in signal_df.columns], entries)
    signal_df.columns = final_columns
    mounted_df.columns = final_columns
    return signal_df, mounted_df


def write_source_parquets(
    grouped_entries: dict[ParquetPair, list[ColumnEntry]],
    dry_run: bool,
) -> None:
    for pair, entries in grouped_entries.items():
        if not any(entry.new_signal_id != entry.old_signal_id for entry in entries):
            continue
        if dry_run:
            continue
        signal_df, mounted_df = _read_pair_with_final_columns(pair, entries)
        signal_df.to_parquet(pair.signal_path)
        mounted_df.to_parquet(pair.mounted_path)


def build_group_frames(
    grouped_entries: dict[ParquetPair, list[ColumnEntry]],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    signal_frames: dict[str, list[pd.DataFrame]] = {}
    mounted_frames: dict[str, list[pd.DataFrame]] = {}

    for pair, entries in grouped_entries.items():
        signal_df, mounted_df = _read_pair_with_final_columns(pair, entries)
        columns_by_category: dict[str, list[str]] = {}
        for entry in entries:
            final_signal_id = entry.new_signal_id or entry.old_signal_id
            columns_by_category.setdefault(entry.category, []).append(final_signal_id)

        for category, columns in columns_by_category.items():
            signal_frames.setdefault(category, []).append(signal_df.loc[:, columns])
            mounted_frames.setdefault(category, []).append(mounted_df.loc[:, columns])

    grouped_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for category in sorted(signal_frames):
        signal_group = pd.concat(signal_frames[category], axis=1)
        mounted_group = pd.concat(mounted_frames[category], axis=1)
        grouped_frames[category] = (signal_group, mounted_group)
    return grouped_frames


def write_grouped_parquets(
    grouped_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, (signal_df, mounted_df) in grouped_frames.items():
        signal_df.to_parquet(output_dir / f"factor_{category}_signal_ls.parquet")
        mounted_df.to_parquet(output_dir / f"factor_{category}_mounted_normalized_factors.parquet")


def _record_signal_id(record: dict[str, Any]) -> str | None:
    signal_id = record.get("signal_id")
    if signal_id is not None:
        return str(signal_id)
    legacy_signal_id = record.get("factor_id")
    if legacy_signal_id is not None:
        return str(legacy_signal_id)
    return None


def _record_matches_entry(record: dict[str, Any], entry: ColumnEntry) -> bool:
    prefix = str(record.get("_generated_output_prefix") or "")
    return prefix == entry.pair.prefix and _record_signal_id(record) == entry.old_signal_id


def _record_already_renamed(record: dict[str, Any], entry: ColumnEntry) -> bool:
    prefix = str(record.get("_generated_output_prefix") or "")
    return (
        prefix == entry.pair.prefix
        and str(record.get("_renamed_from_signal_id") or "") == entry.old_signal_id
        and _record_signal_id(record) == entry.new_signal_id
    )


def update_generated_records(
    generated_path: Path,
    renamed_entries: list[ColumnEntry],
    dry_run: bool,
) -> tuple[int, list[ColumnEntry]]:
    if not renamed_entries or not generated_path.exists():
        return 0, list(renamed_entries)

    payload = json.loads(generated_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unsupported generated records payload: {generated_path}")

    renamed_at = datetime.now(timezone.utc).isoformat()
    updated = 0
    unmatched_entries: list[ColumnEntry] = []
    used_record_indexes: set[int] = set()
    for entry in renamed_entries:
        new_signal_id = entry.new_signal_id
        if new_signal_id is None:
            continue
        matched = False
        for index, record in enumerate(records):
            if index in used_record_indexes or not isinstance(record, dict):
                continue
            if _record_already_renamed(record, entry):
                used_record_indexes.add(index)
                updated += 1
                matched = True
                break
            if not _record_matches_entry(record, entry):
                continue
            if "signal_id" in record:
                record["signal_id"] = new_signal_id
            if "factor_id" in record:
                record["factor_id"] = new_signal_id
            record["_renamed_from_signal_id"] = entry.old_signal_id
            record["_renamed_by"] = SCRIPT_NAME
            record["_renamed_at"] = renamed_at
            used_record_indexes.add(index)
            updated += 1
            matched = True
            break
        if not matched:
            unmatched_entries.append(entry)

    if dry_run or updated == 0:
        return updated, unmatched_entries

    if isinstance(payload, dict):
        payload["generated_at"] = renamed_at
        payload["records"] = records
        generated_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        generated_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return updated, unmatched_entries


def build_report_lines(
    pairs: list[ParquetPair],
    grouped_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    renamed_entries: list[ColumnEntry],
    generated_updates: int,
    unmatched_generated_entries: list[ColumnEntry],
    dry_run: bool,
) -> list[str]:
    lines = [
        "# Category summarize report",
        "",
        f"- mode: {'dry-run' if dry_run else 'write'}",
        f"- scanned pairs: {len(pairs)}",
        f"- categories: {', '.join(sorted(grouped_frames)) if grouped_frames else 'none'}",
        f"- renamed duplicate signal_id columns: {len(renamed_entries)}",
        f"- factor_generated.json records updated: {generated_updates}",
        f"- renamed columns without generated records: {len(unmatched_generated_entries)}",
        "",
        "## Outputs",
        "",
    ]
    if grouped_frames:
        lines.append("| category | signal columns | mounted columns |")
        lines.append("| --- | ---: | ---: |")
        for category, (signal_df, mounted_df) in grouped_frames.items():
            lines.append(f"| {category} | {signal_df.shape[1]} | {mounted_df.shape[1]} |")
    else:
        lines.append("No grouped outputs.")

    lines.extend(["", "## Renames", ""])
    if renamed_entries:
        lines.append("| source prefix | old signal_id | new signal_id |")
        lines.append("| --- | --- | --- |")
        for entry in renamed_entries:
            lines.append(f"| {entry.pair.prefix} | {entry.old_signal_id} | {entry.new_signal_id} |")
    else:
        lines.append("No duplicate signal_id values found.")

    lines.extend(["", "## Missing generated records", ""])
    if unmatched_generated_entries:
        lines.append("| source prefix | old signal_id | new signal_id |")
        lines.append("| --- | --- | --- |")
        for entry in unmatched_generated_entries:
            lines.append(f"| {entry.pair.prefix} | {entry.old_signal_id} | {entry.new_signal_id} |")
    else:
        lines.append("All renamed signal_id values matched factor_generated.json records.")
    return lines


def write_report(lines: list[str], output_dir: Path, dry_run: bool) -> None:
    print("\n".join(lines))
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "category_summarize_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    generated_path = args.generated_path.resolve()

    pairs = discover_pairs(input_dir)
    entries = collect_column_entries(pairs)
    renamed_entries = assign_duplicate_signal_ids(entries)
    grouped_entries = entries_by_pair(entries)
    grouped_frames = build_group_frames(grouped_entries)

    generated_updates, unmatched_generated_entries = update_generated_records(
        generated_path=generated_path,
        renamed_entries=renamed_entries,
        dry_run=args.dry_run,
    )
    if unmatched_generated_entries and not args.dry_run and args.strict_generated_records:
        missing = ", ".join(
            f"{entry.pair.prefix}:{entry.old_signal_id}->{entry.new_signal_id}"
            for entry in unmatched_generated_entries
        )
        raise ValueError(
            "Some renamed signal_id values have no matching factor_generated.json "
            "record. Re-run without --strict-generated-records to continue and "
            f"write them to the report. Missing: {missing}"
        )
    write_source_parquets(grouped_entries, dry_run=args.dry_run)
    write_grouped_parquets(grouped_frames, output_dir=output_dir, dry_run=args.dry_run)

    report_lines = build_report_lines(
        pairs=pairs,
        grouped_frames=grouped_frames,
        renamed_entries=renamed_entries,
        generated_updates=generated_updates,
        unmatched_generated_entries=unmatched_generated_entries,
        dry_run=args.dry_run,
    )
    write_report(report_lines, output_dir=output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
