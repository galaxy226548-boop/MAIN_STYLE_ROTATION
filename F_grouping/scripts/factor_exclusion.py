"""Merge IC scores with backtesting scores and keep usable factors.

Usage:
    python F_grouping/scripts/factor_exclusion.py

Input:
    D_analysis/check_output/IC_score.xlsx
    F_grouping/reference/backtesting_score.xlsx

Output:
    F_grouping/reference/usable_factors.xlsx
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

IC_SCORE_PATH = PROJECT_ROOT / "D_analysis" / "check_output" / "IC_score.xlsx"
BACKTESTING_SCORE_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "backtesting_score.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "usable_factors.xlsx"

IC_FACTOR_COL = "factor_id"
BACKTESTING_FACTOR_COL = "factor_name"
IC_SCORE_COL = "总分"
BACKTESTING_PASS_COL = "pass_sum"
PERIOD_WIN_RATE_COL = "period_win_rate"
PAYOFF_RATIO_COL = "payoff_ratio"
EXPECTANCY_COL = "expectancy"
MONTHLY_WIN_RATE_COL = "monthly_win_rate"
TOTAL_SCORE_COL = "total_score"
PERCENTAGE_FORMAT_COLS = [
    MONTHLY_WIN_RATE_COL,
    PERIOD_WIN_RATE_COL,
    PAYOFF_RATIO_COL,
    EXPECTANCY_COL,
]

IC_SCORE_THRESHOLD = 1.5
BACKTESTING_PASS_THRESHOLD = 3


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")


def build_usable_factors(ic_score_path: Path, backtesting_score_path: Path) -> pd.DataFrame:
    ic_score_df = pd.read_excel(ic_score_path)
    backtesting_score_df = pd.read_excel(backtesting_score_path)

    require_columns(ic_score_df, [IC_FACTOR_COL, IC_SCORE_COL], ic_score_path)
    require_columns(
        backtesting_score_df,
        [BACKTESTING_FACTOR_COL, BACKTESTING_PASS_COL, PERIOD_WIN_RATE_COL, PAYOFF_RATIO_COL, EXPECTANCY_COL],
        backtesting_score_path,
    )

    merged_df = ic_score_df.merge(
        backtesting_score_df,
        how="left",
        left_on=IC_FACTOR_COL,
        right_on=BACKTESTING_FACTOR_COL,
        sort=False,
    )
    merged_df = merged_df.drop(columns=[BACKTESTING_FACTOR_COL])

    ic_score = pd.to_numeric(merged_df[IC_SCORE_COL], errors="coerce")
    pass_sum = pd.to_numeric(merged_df[BACKTESTING_PASS_COL], errors="coerce")
    expectancy = pd.to_numeric(merged_df[EXPECTANCY_COL], errors="coerce")
    period_win_rate_pass = pd.to_numeric(merged_df[PERIOD_WIN_RATE_COL], errors="coerce").gt(0.52).astype(int)
    payoff_ratio_pass = pd.to_numeric(merged_df[PAYOFF_RATIO_COL], errors="coerce").gt(1).astype(int)
    # 当 expectancy <= 0 时，将 period_win_rate 和 payoff_ratio 对 pass_sum 的贡献从 total_score 中扣除
    penalty = (period_win_rate_pass + payoff_ratio_pass) * expectancy.le(0).fillna(False).astype(int)
    merged_df[TOTAL_SCORE_COL] = ic_score.add(pass_sum, fill_value=0) - penalty
    missing_backtesting_mask = pass_sum.isna()
    weak_score_mask = (ic_score < IC_SCORE_THRESHOLD) & (pass_sum < BACKTESTING_PASS_THRESHOLD)
    exclusion_mask = missing_backtesting_mask | weak_score_mask

    output_df = (
        merged_df.loc[~exclusion_mask]
        .sort_values([TOTAL_SCORE_COL, EXPECTANCY_COL, IC_FACTOR_COL], ascending=[False, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    if MONTHLY_WIN_RATE_COL in output_df.columns:
        columns = [column for column in output_df.columns if column != TOTAL_SCORE_COL]
        insert_at = columns.index(MONTHLY_WIN_RATE_COL)
        columns.insert(insert_at, TOTAL_SCORE_COL)
        output_df = output_df.loc[:, columns]

    return output_df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        percentage_format = workbook.add_format({"num_format": "0.00%"})
        for column_name in PERCENTAGE_FORMAT_COLS:
            if column_name in df.columns:
                column_index = df.columns.get_loc(column_name)
                worksheet.set_column(column_index, column_index, None, percentage_format)


def main() -> None:
    usable_factors_df = build_usable_factors(IC_SCORE_PATH, BACKTESTING_SCORE_PATH)
    write_output(usable_factors_df, OUTPUT_PATH)

    print(f"IC score rows: {len(pd.read_excel(IC_SCORE_PATH))}")
    print(f"usable factor rows: {len(usable_factors_df)}")
    print(f"output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
