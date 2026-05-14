"""Batch rerun grouped factor/signal combinations from existing record JSONs.

Reads *factor_grouping_record.json files produced by build_factor_group_interactive.py
or build_signal_group_interactive.py and reruns the corresponding grouping logic
without any interactive input.

Usage:
    # Rerun all records found in the default output_COMB directory
    python F_grouping/scripts/group_batch_rerun.py

    # Rerun specific groups only
    python F_grouping/scripts/group_batch_rerun.py --group W003 W004

    # Rerun specific record JSON files
    python F_grouping/scripts/group_batch_rerun.py --records F_grouping/output_COMB/W004_factor_grouping_record.json

    # Preview what would be rerun without writing any files
    python F_grouping/scripts/group_batch_rerun.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_COMB_DIR = PROJECT_ROOT / "F_grouping" / "output_COMB"


def _load_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIGNAL_SUFFIX = "_signal"


def _detect_mode(record: dict) -> str:
    """Detect whether a record is 'factor' or 'signal' mode.

    New format: signal records have 'input_factor_group' key.
    Old format (W003): signal records have factor_group ending with '_signal'.
    """
    if "input_factor_group" in record:
        return "signal"
    fg = record.get("factor_group", "")
    if isinstance(fg, str) and fg.endswith(_SIGNAL_SUFFIX):
        return "signal"
    return "factor"


def _extract_factor_names(record: dict) -> list[str]:
    """Extract the source factor name list, supporting both old and new record formats."""
    names = record.get("requested_factor_names")
    if names:
        return list(names)
    # Old format (W003): source_data.columns holds the bottom-level factor names.
    names = record.get("source_data", {}).get("columns")
    if names:
        return list(names)
    names = record.get("source_data", {}).get("selected_columns")
    if names:
        return list(names)
    return []


def _filter_by_group(record_files: list[Path], groups: list[str]) -> list[Path]:
    groups_set = set(groups)
    filtered: list[Path] = []
    for rec_path in record_files:
        try:
            record = json.loads(rec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fg = record.get("factor_group", "")
        bg = record.get("base_group", "")
        sg = record.get("signal_group", "")
        if groups_set & {fg, bg, sg}:
            filtered.append(rec_path)
    return filtered


def run_factor_group(fg_mod: ModuleType, record: dict) -> list[str]:
    factor_group: str = record["factor_group"]
    factor_names: list[str] = _extract_factor_names(record)
    if not factor_names:
        raise ValueError(f"No source factor names found in record for {factor_group!r}")

    input_files = fg_mod.list_input_files(fg_mod.INPUT_DIR)
    source_df, scan_record, scan_warnings = fg_mod.collect_selected_factors(
        factor_names=factor_names,
        input_files=input_files,
    )
    factor_df, groups, factor_warnings = fg_mod.build_factor_matrix(
        source_df=source_df,
        factor_group=factor_group,
    )
    warnings: list[str] = scan_warnings + factor_warnings
    signal_ls_df = fg_mod.build_signal_ls_matrix(factor_df)
    output_paths = fg_mod.build_output_paths(factor_group)

    fg_mod.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record_out = fg_mod.build_record(
        factor_group=factor_group,
        requested_factor_names=factor_names,
        source_df=source_df,
        factor_df=factor_df,
        groups=groups,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )

    factor_df.to_parquet(output_paths["factor_parquet"])
    signal_ls_df.to_parquet(output_paths["signal_parquet"])
    fg_mod.write_factor_frame_xlsx(factor_df, output_paths["factor_xlsx"])
    fg_mod.write_factor_frame_xlsx(signal_ls_df, output_paths["signal_xlsx"])
    output_paths["record_json"].write_text(
        json.dumps(record_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"  source_df shape: {source_df.shape}  |  factor columns: {list(factor_df.columns)}")
    return warnings


def run_signal_group(sg_mod: ModuleType, record: dict) -> list[str]:
    # New format has explicit base_group/signal_group; old format only has factor_group.
    signal_group: str = record.get("signal_group") or record["factor_group"]
    base_group, signal_group = sg_mod.normalize_factor_group(signal_group)
    input_factor_group: str = record.get("input_factor_group") or base_group
    factor_names: list[str] = _extract_factor_names(record)
    if not factor_names:
        raise ValueError(f"No source factor names found in record for {signal_group!r}")

    input_files = sg_mod.list_input_files(sg_mod.INPUT_DIR)
    source_df, scan_record, scan_warnings = sg_mod.collect_selected_factors(
        factor_names=factor_names,
        input_files=input_files,
    )
    factor_df, groups, factor_warnings = sg_mod.build_factor_matrix(
        source_df=source_df,
        base_group=base_group,
        signal_group=signal_group,
    )
    warnings: list[str] = scan_warnings + factor_warnings
    signal_ls_df = sg_mod.build_signal_ls_matrix(factor_df)
    output_paths = sg_mod.build_output_paths(signal_group)

    sg_mod.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record_out = sg_mod.build_record(
        input_factor_group=input_factor_group,
        base_group=base_group,
        signal_group=signal_group,
        requested_factor_names=factor_names,
        source_df=source_df,
        factor_df=factor_df,
        groups=groups,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )

    factor_df.to_parquet(output_paths["factor_parquet"])
    signal_ls_df.to_parquet(output_paths["signal_parquet"])
    sg_mod.write_factor_frame_xlsx(factor_df, output_paths["factor_xlsx"])
    sg_mod.write_factor_frame_xlsx(signal_ls_df, output_paths["signal_xlsx"])
    output_paths["record_json"].write_text(
        json.dumps(record_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"  source_df shape: {source_df.shape}  |  factor columns: {list(factor_df.columns)}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch rerun grouped factor/signal combinations from existing record JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--record-dir",
        type=Path,
        default=OUTPUT_COMB_DIR,
        metavar="DIR",
        help="Directory to scan for *factor_grouping_record.json files "
             "(default: F_grouping/output_COMB)",
    )
    source_group.add_argument(
        "--records",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="Specific record JSON file paths to rerun",
    )
    parser.add_argument(
        "--group",
        nargs="+",
        metavar="NAME",
        help="Only rerun records matching these group names (e.g. W003 W004); "
             "matches factor_group, base_group, or signal_group",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be rerun without writing any files",
    )
    args = parser.parse_args()

    # Step 1: collect candidate record files
    if args.records:
        record_files: list[Path] = [Path(p).resolve() for p in args.records]
    else:
        record_dir = args.record_dir.resolve()
        if not record_dir.exists():
            print(f"ERROR: --record-dir does not exist: {record_dir}", file=sys.stderr)
            sys.exit(1)
        record_files = sorted(record_dir.glob("*factor_grouping_record.json"))

    # Step 2: apply --group filter if requested
    if args.group:
        record_files = _filter_by_group(record_files, args.group)

    if not record_files:
        print("No matching record files found.")
        return

    print(f"Found {len(record_files)} record file(s) to process.")

    # Step 3: dry-run preview
    if args.dry_run:
        print("\n[DRY-RUN] Records that would be rerun:")
        for rec_path in record_files:
            try:
                record = json.loads(rec_path.read_text(encoding="utf-8"))
                mode = _detect_mode(record)
                fg = record.get("factor_group", "?")
                factor_names = _extract_factor_names(record)
                print(f"  {rec_path.name}")
                print(f"    mode={mode}  factor_group={fg}  factors({len(factor_names)})={factor_names}")
            except Exception as e:
                print(f"  {rec_path.name}  →  [PARSE ERROR] {e}")
        return

    # Step 4: load modules once
    fg_mod = _load_module(SCRIPTS_DIR / "build_factor_group_interactive.py")
    sg_mod = _load_module(SCRIPTS_DIR / "build_signal_group_interactive.py")

    results: list[tuple[str, str]] = []

    for rec_path in record_files:
        try:
            record = json.loads(rec_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"\n[SKIP] {rec_path.name}: failed to parse JSON: {e}", file=sys.stderr)
            results.append((rec_path.name, f"PARSE ERROR: {e}"))
            continue

        mode = _detect_mode(record)
        fg_label = record.get("factor_group", rec_path.name)
        print(f"\n========== {fg_label} ({mode}) ==========")

        try:
            if mode == "factor":
                warnings = run_factor_group(fg_mod, record)
            else:
                warnings = run_signal_group(sg_mod, record)
            if warnings:
                for w in warnings:
                    print(f"  [WARN] {w}")
            results.append((rec_path.name, "OK"))
            print("  → Done")
        except Exception as e:
            results.append((rec_path.name, f"FAIL: {e}"))
            print(f"  → FAIL: {e}", file=sys.stderr)

    # Step 5: summary
    print("\n========== Summary ==========")
    ok_count = sum(1 for _, status in results if status == "OK")
    failed = [(name, status) for name, status in results if status != "OK"]
    print(f"OK: {ok_count} / {len(results)}")
    if failed:
        print("Failed:")
        for name, status in failed:
            print(f"  {name}: {status}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
