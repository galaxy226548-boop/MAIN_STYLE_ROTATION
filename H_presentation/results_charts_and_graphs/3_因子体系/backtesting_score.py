"""Collect rebalance_50 screening scores from backtesting summary files.

Usage:
    python F_grouping/scripts/backtesting_score.py

Input:
    E_backtesting/Result/**/*rebalance_50_summary.xlsx

Output:
    F_grouping/reference/backtesting_score.xlsx
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

INPUT_DIR = PROJECT_ROOT / "E_backtesting" / "Result"
OUTPUT_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "backtesting_score.xlsx"

INPUT_FILE_PATTERN = "*rebalance_50_summary.xlsx"
SUMMARY_SUFFIX = "_rebalance_50_summary.xlsx"
FACTOR_NAME_PATTERN = re.compile(r"mw\d+(?:\.\d+)?_(.+)_rebalance_50_summary\.xlsx$")
SCREENING_SHEET = "screening"
SCREENING_ROW_COUNT = 7
MONTHLY_WIN_RATE_LABEL = "monthly_win_rate"
PERIOD_WIN_RATE_LABEL = "period_win_rate"
PAYOFF_RATIO_LABEL = "payoff_ratio"
EXPECTANCY_LABEL = "expectancy"


def parse_factor_name(path: Path) -> str:
    match = FACTOR_NAME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse factor name from file name: {path.name}")
    return match.group(1)


def clean_condition(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_summary_file(path: Path) -> dict[str, object]:
    df = pd.read_excel(path, sheet_name=SCREENING_SHEET)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {SCREENING_SHEET!r}: {path}")
    if len(df) < SCREENING_ROW_COUNT:
        raise ValueError(f"Expected at least {SCREENING_ROW_COUNT} screening rows: {path}")

    condition_col = df.columns[0]
    value_col = df.columns[1]
    screening_df = df.iloc[:SCREENING_ROW_COUNT, :2].copy()

    row: dict[str, object] = {"factor_name": parse_factor_name(path)}
    condition_columns: list[str] = []
    for _, screening_row in screening_df.iterrows():
        condition = clean_condition(screening_row[condition_col])
        if not condition:
            raise ValueError(f"Empty condition in first {SCREENING_ROW_COUNT} rows: {path}")
        row[condition] = screening_row[value_col]
        condition_columns.append(condition)

    row["pass_sum"] = int(sum(bool(row[condition]) for condition in condition_columns))

    condition_series = df[condition_col].astype(str).str.strip()

    monthly_rows = df.loc[condition_series == MONTHLY_WIN_RATE_LABEL, value_col]
    if monthly_rows.empty:
        raise ValueError(f"Missing {MONTHLY_WIN_RATE_LABEL!r}: {path}")
    row["monthly_win_rate"] = pd.to_numeric(monthly_rows.iloc[-1], errors="coerce")

    period_win_rate_rows = df.loc[condition_series == PERIOD_WIN_RATE_LABEL, value_col]
    if period_win_rate_rows.empty:
        raise ValueError(f"Missing {PERIOD_WIN_RATE_LABEL!r}: {path}")
    row["period_win_rate"] = pd.to_numeric(period_win_rate_rows.iloc[-1], errors="coerce")

    payoff_ratio_rows = df.loc[condition_series == PAYOFF_RATIO_LABEL, value_col]
    if payoff_ratio_rows.empty:
        raise ValueError(f"Missing {PAYOFF_RATIO_LABEL!r}: {path}")
    row["payoff_ratio"] = pd.to_numeric(payoff_ratio_rows.iloc[-1], errors="coerce")

    expectancy_rows = df.loc[condition_series == EXPECTANCY_LABEL, value_col]
    if expectancy_rows.empty:
        raise ValueError(f"Missing {EXPECTANCY_LABEL!r}: {path}")
    row["expectancy"] = pd.to_numeric(expectancy_rows.iloc[-1], errors="coerce")

    return row


def collect_scores(input_dir: Path) -> tuple[pd.DataFrame, list[str], int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths = sorted(input_dir.rglob(INPUT_FILE_PATTERN))
    records: list[dict[str, object]] = []
    warnings: list[str] = []

    for path in paths:
        if not path.name.endswith(SUMMARY_SUFFIX):
            continue
        try:
            records.append(read_summary_file(path))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path}: {exc}")

    if not records:
        raise FileNotFoundError(f"No readable summary files found under {input_dir}")

    result_df = pd.DataFrame(records)
    _non_condition = {"factor_name", "pass_sum", "monthly_win_rate", "period_win_rate", "payoff_ratio", "expectancy"}
    condition_columns = [
        column
        for column in result_df.columns
        if column not in _non_condition
    ]
    output_columns = [
        "factor_name",
        *condition_columns,
        "pass_sum",
        "period_win_rate",
        "payoff_ratio",
        "expectancy",
        "monthly_win_rate",
    ]
    result_df = result_df.loc[:, output_columns]
    result_df = result_df.sort_values(
        by=["pass_sum", "expectancy", "factor_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return result_df, warnings, len(paths)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def main() -> None:
    score_df, warnings, scanned_count = collect_scores(INPUT_DIR)
    write_output(score_df, OUTPUT_PATH)

    print(f"scanned files: {scanned_count}")
    print(f"summary rows: {len(score_df)}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
