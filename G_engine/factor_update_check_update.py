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

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor check scripts and report updated files.")
    parser.add_argument("--sample", choices=["all", "ins", "oos"], default="all")
    return parser.parse_args()


def build_script_plan(sample: str) -> tuple[list[tuple[str, list[str]]], list[Path]]:
    ic_output_root = PROJECT_ROOT / "D_analysis" / "IC_output"
    backtest_output_root = PROJECT_ROOT / "E_backtesting" / "Result"
    check_output_dir = PROJECT_ROOT / "D_analysis" / "check_output" / sample
    grouping_reference_dir = PROJECT_ROOT / "F_grouping" / "reference" / sample

    ic_score_path = check_output_dir / "IC_score.xlsx"
    backtesting_score_path = grouping_reference_dir / "backtesting_score.xlsx"
    usable_factors_path = grouping_reference_dir / "usable_factors.xlsx"

    scripts = [
        (
            "D_analysis/scripts/IC_nega_checker.py",
            [
                "--rebuild",
                "--input-dir",
                str(ic_output_root),
                "--output-dir",
                str(check_output_dir),
                "--sample",
                sample,
            ],
        ),
        (
            "D_analysis/scripts/IC_score.py",
            [
                "--input-dir",
                str(ic_output_root),
                "--output-path",
                str(ic_score_path),
                "--sample",
                sample,
            ],
        ),
        (
            "F_grouping/scripts/backtesting_score.py",
            [
                "--input-dir",
                str(backtest_output_root),
                "--output-path",
                str(backtesting_score_path),
                "--sample",
                sample,
            ],
        ),
        (
            "F_grouping/scripts/factor_exclusion.py",
            [
                "--ic-score-path",
                str(ic_score_path),
                "--backtesting-score-path",
                str(backtesting_score_path),
                "--output-path",
                str(usable_factors_path),
            ],
        ),
    ]
    output_files = [
        check_output_dir / "nega_doubt.md",
        check_output_dir / "nega_checked.md",
        ic_score_path,
        backtesting_score_path,
        usable_factors_path,
    ]
    return scripts, output_files


def main() -> None:
    args = parse_args()
    scripts, output_files = build_script_plan(args.sample)

    print(f"sample: {args.sample}")
    for script_rel, script_args in scripts:
        script_path = PROJECT_ROOT / script_rel
        print(f"\n{'='*60}")
        print(f"Running: {script_path.name}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(script_path), *script_args],
            check=False,
        )
        if result.returncode != 0:
            print(f"Error: {script_path.name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("已完成因子检查数据更新，更新的文件为：")
    for path in output_files:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
