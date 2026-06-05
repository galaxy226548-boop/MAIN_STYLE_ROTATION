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

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

IC_SCORE_PATH = PROJECT_ROOT / "D_analysis" / "check_output" / "IC_score.xlsx"
BACKTESTING_SCORE_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "backtesting_score.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "usable_factors.xlsx"
FACTOR_GENERATED_PATH = PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json"
FACTOR_SUMMARY_PATH = PROJECT_ROOT / "B_factors" / "reference" / "因子汇总.json"
BACKTEST_RESULT_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"

IC_FACTOR_COL = "factor_id"
SIGNAL_ID_COL = "signal_id"
SORTING_FACTOR_COL = "编号"
BACKTESTING_FACTOR_COL = "factor_name"
FACTOR_NAME_COL = "factor_name"
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
LATEST_NAME_ALLOWED_CONFLICT_IDS = {"V001", "V002"}
SORTING_SAMPLE_NAMES = {"all", "ins", "oos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge IC scores with backtesting scores and keep usable factors.")
    parser.add_argument("--ic-score-path", type=Path, default=IC_SCORE_PATH)
    parser.add_argument("--backtesting-score-path", type=Path, default=BACKTESTING_SCORE_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--factor-generated-path", type=Path, default=FACTOR_GENERATED_PATH)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")


def load_factor_summary_name_map(factor_summary_path: Path = FACTOR_SUMMARY_PATH) -> dict[str, str]:
    with factor_summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = payload.get("sheets", {}).get("factors", {}).get("records")
    if not isinstance(records, list):
        raise ValueError(f"Expected sheets.factors.records list in {factor_summary_path}")

    names_by_factor_id: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            continue
        factor_id = record.get("编号")
        factor_name = record.get("原数据")
        if factor_id in (None, "") or factor_name in (None, ""):
            continue
        names_by_factor_id[str(factor_id)].add(str(factor_name))

    factor_name_map: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    for factor_id, names in names_by_factor_id.items():
        sorted_names = sorted(names)
        if len(sorted_names) == 1:
            factor_name_map[factor_id] = sorted_names[0]
            continue
        conflicts[factor_id] = sorted_names

    if conflicts:
        conflict_text = "; ".join(
            f"{factor_id}: {', '.join(names)}"
            for factor_id, names in sorted(conflicts.items())
        )
        raise ValueError(f"Conflicting factor names in {factor_summary_path}: {conflict_text}")

    return factor_name_map


def load_generated_factor_name_map(factor_generated_path: Path) -> dict[str, str]:
    with factor_generated_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Expected list field 'records' in {factor_generated_path}")

    records_by_factor_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        factor_id = record.get(IC_FACTOR_COL)
        factor_name = record.get("原数据") or record.get("factor")
        if factor_id is None or factor_name in (None, ""):
            continue
        records_by_factor_id[str(factor_id)].append(record)

    factor_name_map: dict[str, str] = {}
    for factor_id, factor_records in records_by_factor_id.items():
        names = sorted({str(record.get("原数据") or record["factor"]) for record in factor_records})
        if len(names) == 1:
            factor_name_map[factor_id] = names[0]
            continue

        if factor_id in LATEST_NAME_ALLOWED_CONFLICT_IDS:
            latest_record = max(factor_records, key=lambda record: str(record.get("_generated_at") or ""))
            factor_name_map[factor_id] = str(latest_record.get("原数据") or latest_record["factor"])
            continue

        # factor_generated.json 是生成登记日志；同一编号可能有历史口径冲突。
        # 冲突编号不作为兜底名称来源，避免阻断 usable_factors 生成。
        continue

    return factor_name_map


def add_factor_names(df: pd.DataFrame, factor_generated_path: Path) -> pd.DataFrame:
    generated_name_map = load_generated_factor_name_map(factor_generated_path)
    factor_name_map = {
        **generated_name_map,
        **load_factor_summary_name_map(),
    }
    output_df = df.copy()
    factor_names = output_df[IC_FACTOR_COL].astype(str).map(factor_name_map).fillna("")
    output_df.insert(0, FACTOR_NAME_COL, factor_names)
    return output_df


def resolve_sorting_sample(output_path: Path) -> str:
    output_parent_name = output_path.parent.name
    if output_parent_name in SORTING_SAMPLE_NAMES:
        return output_parent_name
    return "ins"


def find_single_factors_sorting_files(sample: str) -> list[Path]:
    candidate_dirs = [
        BACKTEST_RESULT_ROOT / "I_laboratory" / sample,
        *sorted(BACKTEST_RESULT_ROOT.glob(f"factor_*/{sample}")),
    ]
    sorting_files: list[Path] = []
    for candidate_dir in candidate_dirs:
        if not candidate_dir.is_dir():
            continue
        sorting_files.extend(sorted(candidate_dir.glob("*single_factors_sorting.xlsx")))
    return sorting_files


def load_single_factors_sorting_df(sample: str) -> pd.DataFrame:
    sorting_frames: list[pd.DataFrame] = []
    for sorting_path in find_single_factors_sorting_files(sample):
        sorting_df = pd.read_excel(sorting_path)
        require_columns(sorting_df, [SORTING_FACTOR_COL], sorting_path)
        sorting_df = sorting_df.copy()
        sorting_df[SORTING_FACTOR_COL] = sorting_df[SORTING_FACTOR_COL].astype(str)
        sorting_frames.append(sorting_df)

    if not sorting_frames:
        return pd.DataFrame(columns=[SORTING_FACTOR_COL])

    combined_df = pd.concat(sorting_frames, ignore_index=True)
    return combined_df.drop_duplicates(subset=[SORTING_FACTOR_COL], keep="first")


def append_single_factors_sorting_columns(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    sample = resolve_sorting_sample(output_path)
    sorting_df = load_single_factors_sorting_df(sample)
    if sorting_df.empty:
        return df

    output_df = df.copy()
    merge_key = SIGNAL_ID_COL if SIGNAL_ID_COL in output_df.columns else IC_FACTOR_COL
    output_df[merge_key] = output_df[merge_key].astype(str)

    sorting_columns = [column for column in sorting_df.columns if column != SORTING_FACTOR_COL]
    rename_map = {
        column: f"{column}_sorting"
        for column in sorting_columns
        if column in output_df.columns
    }
    sorting_df = sorting_df.rename(columns=rename_map)

    return output_df.merge(
        sorting_df,
        how="left",
        left_on=merge_key,
        right_on=SORTING_FACTOR_COL,
        sort=False,
    ).drop(columns=[SORTING_FACTOR_COL])


def keep_best_ic_rows(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df["_ic_score_sort"] = pd.to_numeric(output_df[IC_SCORE_COL], errors="coerce")
    output_df = (
        output_df.sort_values(["_ic_score_sort", IC_FACTOR_COL], ascending=[False, True], kind="mergesort")
        .drop_duplicates(subset=[IC_FACTOR_COL], keep="first")
        .drop(columns=["_ic_score_sort"])
    )
    return output_df


def keep_best_backtesting_rows(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    sort_cols = [
        BACKTESTING_PASS_COL,
        EXPECTANCY_COL,
        PERIOD_WIN_RATE_COL,
        PAYOFF_RATIO_COL,
    ]
    temp_cols = [f"_{column}_sort" for column in sort_cols]
    for column, temp_col in zip(sort_cols, temp_cols):
        output_df[temp_col] = pd.to_numeric(output_df[column], errors="coerce")

    output_df = (
        output_df.sort_values([*temp_cols, BACKTESTING_FACTOR_COL], ascending=[False, False, False, False, True], kind="mergesort")
        .drop_duplicates(subset=[BACKTESTING_FACTOR_COL], keep="first")
        .drop(columns=temp_cols)
    )
    return output_df


def build_usable_factors(
    ic_score_path: Path,
    backtesting_score_path: Path,
    factor_generated_path: Path,
) -> pd.DataFrame:
    ic_score_df = pd.read_excel(ic_score_path)
    backtesting_score_df = pd.read_excel(backtesting_score_path)

    require_columns(ic_score_df, [IC_FACTOR_COL, IC_SCORE_COL], ic_score_path)
    require_columns(
        backtesting_score_df,
        [BACKTESTING_FACTOR_COL, BACKTESTING_PASS_COL, PERIOD_WIN_RATE_COL, PAYOFF_RATIO_COL, EXPECTANCY_COL],
        backtesting_score_path,
    )

    ic_score_df = keep_best_ic_rows(ic_score_df)
    backtesting_score_df = keep_best_backtesting_rows(backtesting_score_df)

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

    output_df = add_factor_names(output_df, factor_generated_path)

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
    args = parse_args()
    ic_score_path = args.ic_score_path.resolve()
    backtesting_score_path = args.backtesting_score_path.resolve()
    output_path = args.output_path.resolve()
    factor_generated_path = args.factor_generated_path.resolve()

    usable_factors_df = build_usable_factors(ic_score_path, backtesting_score_path, factor_generated_path)
    usable_factors_df = append_single_factors_sorting_columns(usable_factors_df, output_path)
    write_output(usable_factors_df, output_path)

    print(f"IC score rows: {len(pd.read_excel(ic_score_path))}")
    print(f"usable factor rows: {len(usable_factors_df)}")
    print(f"output saved to: {output_path}")


if __name__ == "__main__":
    main()
