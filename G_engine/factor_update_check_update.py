"""Run factor check scripts and report updated files.

Usage:
    python G_engine/factor_update_check_update.py

Execution order:
    1. IC_nega_checker.py --rebuild  →  nega_doubt.md, nega_checked.md
    2. IC_score.py                    →  IC_score.xlsx
    3. backtesting_score.py           →  backtesting_score.xlsx
    4. factor_exclusion.py            →  usable_factors.xlsx

Updated files:
    - D_analysis/check_output/nega_doubt.md
    - D_analysis/check_output/nega_checked.md
    - D_analysis/check_output/IC_score.xlsx
    - F_grouping/reference/backtesting_score.xlsx
    - F_grouping/reference/usable_factors.xlsx
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

SCRIPTS = [
    ("D_analysis/scripts/IC_nega_checker.py", ["--rebuild"]),
    ("D_analysis/scripts/IC_score.py", []),
    ("F_grouping/scripts/backtesting_score.py", []),
    ("F_grouping/scripts/factor_exclusion.py", []),
]

OUTPUT_FILES = [
    "D_analysis/check_output/nega_doubt.md",
    "D_analysis/check_output/nega_checked.md",
    "D_analysis/check_output/IC_score.xlsx",
    "F_grouping/reference/backtesting_score.xlsx",
    "F_grouping/reference/usable_factors.xlsx",
]


def main() -> None:
    for script_rel, args in SCRIPTS:
        script_path = PROJECT_ROOT / script_rel
        print(f"\n{'='*60}")
        print(f"Running: {script_path.name}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(script_path), *args],
            check=False,
        )
        if result.returncode != 0:
            print(f"Error: {script_path.name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("已完成因子检查数据更新，更新的文件为：")
    for f in OUTPUT_FILES:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
