"""Migrate ZHAO-derived outputs to collision-free factor ids.

Dry-run is the default. Pass --apply to update JSON/matrix outputs and remove
stale top-level ZHAO-derived artifacts before rerunning the ZHAO pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTOR_DONE_PATH = PROJECT_ROOT / "B_factors" / "reference" / "factor_done.json"
FACTOR_GENERATED_PATH = PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json"
B_FACTOR_OUTPUT = PROJECT_ROOT / "B_factors" / "output"
POSITION_DIR = PROJECT_ROOT / "C_positions" / "output" / "factor_positions"
IC_DIR = PROJECT_ROOT / "D_analysis" / "IC_output"
BACKTEST_DIR = PROJECT_ROOT / "E_backtesting" / "Result"


ZHAO_NEW_IDS = {
    "ZHAO01": "V015",
    "ZHAO02": "V016",
    "ZHAO03": "V017",
    "ZHAO04": "V018",
    "ZHAO05": "G020",
    "ZHAO06": "G021",
    "ZHAO07": "I024",
    "ZHAO08": "I025",
    "ZHAO09": "P015",
    "ZHAO10": "P016",
    "ZHAO11": "G022",
    "ZHAO12": "G023",
    "ZHAO13": "G024",
    "ZHAO14": "O010",
    "ZHAO15": "L025",
    "ZHAO16": "L026",
    "ZHAO17": "L027",
    "ZHAO18": "L028",
    "ZHAO19": "G025",
    "ZHAO20": "F006",
    "ZHAO21": "F007",
    "ZHAO22": "P017",
    "ZHAO23": "D012",
    "ZHAO24": "I026",
    "ZHAO25": "V019",
    "ZHAO26": "V020",
    "ZHAO27": "V021",
    "ZHAO28": "V022",
}


@dataclass
class MigrationPlan:
    old_to_new: dict[str, str]
    stale_ids: set[str]
    paths_to_remove: list[Path]
    factor_done_updates: list[tuple[str, str, str]]
    factor_generated_updates: list[tuple[str, str, str]]
    matrix_updates: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate ZHAO-derived factor ids.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_current_zhao_ids(factor_done: dict[str, Any]) -> dict[str, str]:
    current: dict[str, str] = {}
    for sheet in factor_done.get("sheets", {}).values():
        for record in sheet.get("records", []):
            code = str(record.get("code") or "")
            if code in ZHAO_NEW_IDS and record.get("factor_id"):
                current[code] = str(record["factor_id"])
    return current


def build_old_to_new(current_zhao_ids: dict[str, str]) -> dict[str, str]:
    old_to_new: dict[str, str] = {}
    for code, new_id in ZHAO_NEW_IDS.items():
        old_id = current_zhao_ids.get(code, code)
        old_to_new[old_id] = new_id
        old_to_new[code] = new_id
    return old_to_new


def rename_columns(columns: list[object], old_to_new: dict[str, str]) -> list[object]:
    return [old_to_new.get(str(column), str(column)) for column in columns]


def zhao_matrix_paths() -> list[Path]:
    names = [
        "zhao_signal_ls.parquet",
        "zhao_mounted_normalized_factors.parquet",
        "zhao_signal_ls.xlsx",
        "zhao_mounted_normalized_factors.xlsx",
        "zhao_normalized_factors.xlsx",
    ]
    return [B_FACTOR_OUTPUT / name for name in names if (B_FACTOR_OUTPUT / name).exists()]


def collect_matrix_updates(old_to_new: dict[str, str]) -> list[Path]:
    updates: list[Path] = []
    for path in zhao_matrix_paths():
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            new_cols = rename_columns(list(df.columns), old_to_new)
        else:
            df = pd.read_excel(path, nrows=0)
            columns = list(df.columns)
            data_columns = columns[1:] if columns and str(columns[0]).lower() in {"date", "trade_dt"} else columns
            new_cols = [columns[0], *rename_columns(data_columns, old_to_new)] if data_columns != columns else rename_columns(columns, old_to_new)
        if list(map(str, new_cols)) != list(map(str, df.columns)):
            updates.append(path)
    return updates


def update_matrix(path: Path, old_to_new: dict[str, str]) -> None:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df.columns = rename_columns(list(df.columns), old_to_new)
        df.to_parquet(path)
        return

    df = pd.read_excel(path, index_col=0)
    df.columns = rename_columns(list(df.columns), old_to_new)
    df.to_excel(path)


def parsed_backtest_factor_id(path: Path) -> str | None:
    match = re.search(r"mw\d+(?:\.\d+)?_(.+)$", path.name)
    if not match:
        return None
    return match.group(1)


def collect_paths_to_remove(stale_ids: set[str]) -> list[Path]:
    paths: list[Path] = []

    for factor_id in sorted(stale_ids):
        for path in [
            POSITION_DIR / f"{factor_id}_position.xlsx",
            IC_DIR / f"{factor_id}_IC_analysis.xlsx",
            IC_DIR / f"{factor_id}_rolling_IC.png",
        ]:
            if path.exists():
                paths.append(path)

    if BACKTEST_DIR.exists():
        for child in BACKTEST_DIR.iterdir():
            if not child.is_dir():
                continue
            factor_id = parsed_backtest_factor_id(child)
            if factor_id in stale_ids:
                paths.append(child)

    return sorted(paths)


def build_plan() -> MigrationPlan:
    factor_done = load_json(FACTOR_DONE_PATH)
    current_zhao_ids = collect_current_zhao_ids(factor_done)
    old_to_new = build_old_to_new(current_zhao_ids)
    new_ids = set(ZHAO_NEW_IDS.values())
    stale_ids = set(old_to_new) | new_ids

    factor_done_updates: list[tuple[str, str, str]] = []
    for sheet in factor_done.get("sheets", {}).values():
        for record in sheet.get("records", []):
            code = str(record.get("code") or "")
            if code in ZHAO_NEW_IDS:
                old_id = str(record.get("factor_id") or "")
                new_id = ZHAO_NEW_IDS[code]
                if old_id != new_id:
                    factor_done_updates.append((code, old_id, new_id))

    factor_generated_updates: list[tuple[str, str, str]] = []
    if FACTOR_GENERATED_PATH.exists():
        payload = load_json(FACTOR_GENERATED_PATH)
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for record in records:
            if not isinstance(record, dict):
                continue
            code = str(record.get("code") or "")
            output_prefix = str(record.get("_generated_output_prefix") or "")
            if code in ZHAO_NEW_IDS and output_prefix == "zhao":
                old_id = str(record.get("factor_id") or "")
                new_id = ZHAO_NEW_IDS[code]
                if old_id != new_id:
                    factor_generated_updates.append((code, old_id, new_id))

    return MigrationPlan(
        old_to_new=old_to_new,
        stale_ids=stale_ids,
        paths_to_remove=collect_paths_to_remove(stale_ids),
        factor_done_updates=factor_done_updates,
        factor_generated_updates=factor_generated_updates,
        matrix_updates=collect_matrix_updates(old_to_new),
    )


def apply_json_updates() -> None:
    factor_done = load_json(FACTOR_DONE_PATH)
    for sheet in factor_done.get("sheets", {}).values():
        for record in sheet.get("records", []):
            code = str(record.get("code") or "")
            if code in ZHAO_NEW_IDS:
                record["factor_id"] = ZHAO_NEW_IDS[code]
    dump_json(FACTOR_DONE_PATH, factor_done)

    if FACTOR_GENERATED_PATH.exists():
        payload = load_json(FACTOR_GENERATED_PATH)
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for record in records:
            if not isinstance(record, dict):
                continue
            code = str(record.get("code") or "")
            output_prefix = str(record.get("_generated_output_prefix") or "")
            if code in ZHAO_NEW_IDS and output_prefix == "zhao":
                record["_previous_factor_id"] = record.get("factor_id")
                record["factor_id"] = ZHAO_NEW_IDS[code]
        dump_json(FACTOR_GENERATED_PATH, payload)


def remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def print_plan(plan: MigrationPlan, apply: bool) -> None:
    print("Mode:", "APPLY" if apply else "DRY-RUN")
    print("ID mapping:")
    for code, new_id in sorted(ZHAO_NEW_IDS.items()):
        print(f"- {code} -> {new_id}")
    print("factor_done updates:", len(plan.factor_done_updates))
    for code, old_id, new_id in plan.factor_done_updates:
        print(f"- factor_done {code}: {old_id} -> {new_id}")
    print("factor_generated updates:", len(plan.factor_generated_updates))
    for code, old_id, new_id in plan.factor_generated_updates:
        print(f"- factor_generated {code}: {old_id} -> {new_id}")
    print("matrix files to rewrite:", len(plan.matrix_updates))
    for path in plan.matrix_updates:
        print(f"- matrix {path}")
    print("top-level generated paths to remove:", len(plan.paths_to_remove))
    for path in plan.paths_to_remove:
        print(f"- remove {path}")


def main() -> None:
    args = parse_args()
    plan = build_plan()
    print_plan(plan, args.apply)

    if not args.apply:
        return

    apply_json_updates()
    for path in plan.matrix_updates:
        update_matrix(path, plan.old_to_new)
    remove_paths(plan.paths_to_remove)
    print("Applied ZHAO id migration.")


if __name__ == "__main__":
    main()
