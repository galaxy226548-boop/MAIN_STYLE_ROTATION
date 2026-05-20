"""Run the factor position, backtest, and IC-analysis pipeline.

This wrapper intentionally leaves the framework scripts unchanged. It passes
paths to scripts that already expose CLI arguments, and imports IC_analysis.py
with temporary module-level path overrides for factor-specific outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"

if LOCAL_PYTHON.exists() and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv_mktp").resolve():
    os.execv(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv])

import pandas as pd

DEFAULT_SIGNAL_PATH = (
    PROJECT_ROOT
    / "B_factors"
    / "output"
    / "如何从赔率和胜率看成长价值轮动——市场风格轮动系列_signal_ls.parquet"
)
DEFAULT_FACTOR_PATH = (
    PROJECT_ROOT
    / "B_factors"
    / "output"
    / "如何从赔率和胜率看成长价值轮动——市场风格轮动系列_mounted_normalized_factors.parquet"
)

POSITION_SCRIPT = PROJECT_ROOT / "C_positions" / "scripts" / "generate_factor_positions.py"
BACKTEST_SCRIPT = PROJECT_ROOT / "E_backtesting" / "scripts" / "backtesting.py"
IC_SCRIPT = PROJECT_ROOT / "D_analysis" / "scripts" / "IC_analysis.py"

POSITION_OUTPUT_ROOT = PROJECT_ROOT / "C_positions" / "output" / "factor_positions"
BACKTEST_OUTPUT_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"
IC_OUTPUT_ROOT = PROJECT_ROOT / "D_analysis" / "IC_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factor position generation, backtesting, and IC analysis for a factor file pair."
    )
    parser.add_argument(
        "--signal-path",
        type=Path,
        default=DEFAULT_SIGNAL_PATH,
        help="Wide signal_ls parquet. Rows are dates and columns are factor ids.",
    )
    parser.add_argument(
        "--factor-path",
        type=Path,
        default=DEFAULT_FACTOR_PATH,
        help="Wide mounted normalized factor-value parquet. Rows are dates and columns are factor ids.",
    )
    parser.add_argument(
        "--run-name",
        help="Output grouping name. Defaults to the signal filename without _signal_ls.",
    )
    parser.add_argument(
        "--factor",
        action="append",
        help="Factor column to run. Can be passed multiple times. Defaults to all columns.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_run_name(signal_path: Path) -> str:
    stem = signal_path.stem
    for suffix in ("_signal_ls", "-signal_ls", ".signal_ls"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def python_executable() -> str:
    if LOCAL_PYTHON.exists():
        return str(LOCAL_PYTHON)
    return sys.executable


def read_parquet_header(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find parquet: {path}")
    return pd.read_parquet(path)


def is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def is_grouping_input_pair(signal_path: Path, factor_path: Path) -> bool:
    grouping_root = PROJECT_ROOT / "F_grouping"
    return is_under_directory(signal_path, grouping_root) and is_under_directory(factor_path, grouping_root)


def validate_inputs(
    signal_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    requested_factors: list[str] | None,
    grouping_pair: bool = False,
) -> list[str]:
    signal_cols = [str(col) for col in signal_df.columns]
    factor_cols = [str(col) for col in factor_df.columns]
    signal_set = set(signal_cols)
    factor_set = set(factor_cols)

    if signal_set != factor_set:
        only_signal = sorted(signal_set - factor_set)
        only_factor = sorted(factor_set - signal_set)
        raise ValueError(
            "Signal and factor parquet columns do not match. "
            f"Only in signal: {only_signal}; only in factor: {only_factor}"
        )

    if requested_factors:
        missing = sorted(set(requested_factors) - signal_set)
        if missing:
            raise KeyError(f"Requested factors are missing from inputs: {missing}")
        return requested_factors

    if grouping_pair:
        if not signal_cols:
            raise ValueError("F_grouping signal parquet has no factor columns.")

        final_factor = signal_cols[-1]
        if final_factor not in factor_set:
            raise KeyError(f"Final F_grouping signal factor is missing from factor parquet: {final_factor}")
        if not final_factor.startswith("W"):
            raise ValueError(
                "F_grouping inputs default to the final combined W factor, "
                f"but the final signal column is {final_factor!r}."
            )
        return [final_factor]

    return signal_cols


def maybe_write_filtered_signal(
    signal_df: pd.DataFrame,
    factors: list[str],
    signal_path: Path,
    temp_dir: Path,
) -> Path:
    if list(map(str, signal_df.columns)) == factors:
        return signal_path

    filtered_signal_path = temp_dir / "filtered_signal_ls.parquet"
    signal_df.loc[:, factors].to_parquet(filtered_signal_path)
    return filtered_signal_path


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}]")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def generate_positions(signal_path: Path, output_dir: Path) -> None:
    run_command(
        [
            python_executable(),
            str(POSITION_SCRIPT),
            "--signal-path",
            str(signal_path),
            "--output-dir",
            str(output_dir),
        ],
        label="generate positions",
    )


def expected_position_files(output_dir: Path, factors: list[str]) -> list[Path]:
    position_files = [output_dir / f"{factor}_position.xlsx" for factor in factors]
    missing = [path for path in position_files if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing generated position files: {missing_text}")
    return position_files


def run_backtests(position_files: list[Path], output_root: Path) -> None:
    for position_file in position_files:
        run_command(
            [
                python_executable(),
                str(BACKTEST_SCRIPT),
                "--position-file",
                str(position_file),
                "--output-root",
                str(output_root),
            ],
            label=f"backtest {position_file.stem.removesuffix('_position')}",
        )


def load_ic_module():
    spec = importlib.util.spec_from_file_location("pipeline_ic_analysis", IC_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import IC analysis script: {IC_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ic_analysis(factor_path: Path, output_dir: Path, factors: list[str]) -> None:
    print("\n[IC analysis]")
    print(f"factor path: {factor_path}")
    print(f"output dir: {output_dir}")

    ic_module = load_ic_module()
    ic_module.FACTOR_PATH = factor_path
    ic_module.OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_df, raw_market_df, factor_signal_types = ic_module.load_inputs()
    market_df = ic_module.prepare_market_targets(raw_market_df)
    track_list = sorted(market_df[ic_module.TRACK_COL].dropna().astype(int).unique().tolist())

    missing_factors = sorted(set(factors) - set(map(str, factor_df.columns)))
    if missing_factors:
        raise KeyError(f"Unknown factor columns in mounted factor parquet: {missing_factors}")

    for factor_name in factors:
        signal_type = factor_signal_types.get(factor_name, ic_module.DEFAULT_SIGNAL_TYPE)
        ic_module.analyze_factor(
            factor_name=factor_name,
            factor_series=factor_df[factor_name],
            market_df=market_df,
            track_list=track_list,
            signal_type=signal_type,
        )
        print(f"Saved IC analysis for {factor_name} ({signal_type})")

    print(f"Done. Output directory: {output_dir}")


def main() -> None:
    args = parse_args()
    signal_path = resolve_project_path(args.signal_path)
    factor_path = resolve_project_path(args.factor_path)
    run_name = args.run_name or infer_run_name(signal_path)

    print("factor pipeline")
    print(f"signal path: {signal_path}")
    print(f"factor path: {factor_path}")
    print(f"run name: {run_name}")

    signal_df = read_parquet_header(signal_path)
    factor_df = read_parquet_header(factor_path)
    grouping_pair = is_grouping_input_pair(signal_path, factor_path)
    factors = validate_inputs(signal_df, factor_df, args.factor, grouping_pair=grouping_pair)

    position_output_dir = POSITION_OUTPUT_ROOT / run_name
    backtest_output_root = BACKTEST_OUTPUT_ROOT / run_name
    ic_output_dir = IC_OUTPUT_ROOT / run_name

    if grouping_pair and not args.factor:
        print("F_grouping input pair detected; defaulting to final W factor only.")
    print(f"factors: {', '.join(factors)}")
    print(f"position output dir: {position_output_dir}")
    print(f"backtest output root: {backtest_output_root}")
    print(f"IC output dir: {ic_output_dir}")

    with tempfile.TemporaryDirectory(prefix="factor_pipeline_") as temp_name:
        temp_dir = Path(temp_name)
        effective_signal_path = maybe_write_filtered_signal(
            signal_df=signal_df,
            factors=factors,
            signal_path=signal_path,
            temp_dir=temp_dir,
        )

        generate_positions(signal_path=effective_signal_path, output_dir=position_output_dir)
        position_files = expected_position_files(position_output_dir, factors)
        print(f"generated position files: {len(position_files)}")

        run_backtests(position_files=position_files, output_root=backtest_output_root)
        run_ic_analysis(factor_path=factor_path, output_dir=ic_output_dir, factors=factors)

    print("\npipeline completed")


if __name__ == "__main__":
    main()
