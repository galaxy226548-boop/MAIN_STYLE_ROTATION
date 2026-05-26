"""Split negative-IC doubt factor records into signal-type batches.

Inputs:
    D_analysis/check_output/nega_doubt_factors.json

Outputs:
    D_analysis/check_output/nega_check/batch_{signal_type}_{batch_index:02d}.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "D_analysis" / "check_output" / "nega_doubt_factors.json"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent
KEEP_FIELDS = [
    "factor_id",
    "factor",
    "signal_type",
    "docu",
    "condition",
    "bar",
    "ret_bar",
    "result",
]
BATCH_SIZES = {
    "event": 30,
    "state": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split nega doubt factor records into review batches.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for batch JSON files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def load_input(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Input JSON must contain a list field named 'records': {path}")

    return data


def simplify_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in KEEP_FIELDS}


def chunk_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def write_batch(
    output_dir: Path,
    signal_type: str,
    batch_index: int,
    total_batches: int,
    records: list[dict[str, Any]],
) -> Path:
    output_path = output_dir / f"batch_{signal_type}_{batch_index:02d}.json"
    payload = {
        "signal_type": signal_type,
        "batch_index": batch_index,
        "total_batches": total_batches,
        "record_count": len(records),
        "records": records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_input(input_path)
    source_records = data["records"]
    matched_record_count = data.get("matched_record_count")

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unsupported_records: list[dict[str, Any]] = []

    for record in source_records:
        signal_type = record.get("signal_type")
        simplified = simplify_record(record)
        if signal_type in BATCH_SIZES:
            grouped_records[signal_type].append(simplified)
        else:
            unsupported_records.append(simplified)

    total_written_records = 0
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")

    for signal_type in sorted(grouped_records):
        batch_size = BATCH_SIZES[signal_type]
        batches = chunk_records(grouped_records[signal_type], batch_size)
        batch_counts: list[int] = []

        for index, batch in enumerate(batches, start=1):
            write_batch(output_dir, signal_type, index, len(batches), batch)
            batch_counts.append(len(batch))

        signal_total = sum(batch_counts)
        total_written_records += signal_total
        print(
            f"{signal_type}: batches={len(batches)}, "
            f"batch_size={batch_size}, batch_counts={batch_counts}, total={signal_total}"
        )

    if unsupported_records:
        print(f"Unsupported signal_type records skipped: {len(unsupported_records)}")

    source_count = len(source_records)
    matched_ok = matched_record_count == total_written_records
    source_ok = source_count == total_written_records
    print(f"Source records: {source_count}")
    print(f"Matched record count: {matched_record_count}")
    print(f"Written records: {total_written_records}")
    print(f"Written equals source records: {source_ok}")
    print(f"Written equals matched_record_count: {matched_ok}")


if __name__ == "__main__":
    main()
