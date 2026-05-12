"""Score IC analysis outputs by original-factor IC criteria.

Inputs:
    D_analysis/IC_output/**/*_IC_analysis.xlsx

Outputs:
    D_analysis/check_output/IC_score.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "D_analysis" / "IC_output"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "D_analysis" / "check_output" / "IC_score.xlsx"
ANALYSIS_SUFFIX = "_IC_analysis.xlsx"
SHEET_NAME = "IC analysis"
TRACK_IDS = range(5)

NW_P_SCORE_COL = "nw_p_track<0.2"
POS_IC_PROB_SCORE_COL = "pos_ic_prob>50%"
PEARSON_IC_SCORE_COL = "pearson_ic>0"
RANK_IC_SCORE_COL = "rank_ic>0"
TOTAL_SCORE_COL = "总分"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score IC analysis xlsx files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing IC analysis xlsx files. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output xlsx path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def factor_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(ANALYSIS_SUFFIX):
        raise ValueError(f"Unexpected IC analysis filename: {path}")
    return name[: -len(ANALYSIS_SUFFIX)]


def track_columns(prefix: str) -> list[str]:
    return [f"{prefix}{track_id}" for track_id in TRACK_IDS]


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{path} is missing required columns: {missing_cols}")


def count_condition(row: pd.Series, columns: list[str], threshold: float, op: str) -> int:
    values = pd.to_numeric(row[columns], errors="coerce")
    if op == "lt":
        return int((values < threshold).sum())
    if op == "gt":
        return int((values > threshold).sum())
    raise ValueError(f"Unsupported comparison op: {op}")


def score_file(path: Path) -> dict[str, float | int | str]:
    df = pd.read_excel(path, sheet_name=SHEET_NAME, index_col=0)
    if df.empty:
        raise ValueError(f"{path} has no rows")

    nw_p_cols = track_columns("nw_p_track")
    pos_ic_prob_cols = track_columns("pos_ic_prob_track")
    pearson_ic_cols = track_columns("pearson_ic_track")
    rank_ic_cols = track_columns("rank_ic_track")
    required_cols = nw_p_cols + pos_ic_prob_cols + pearson_ic_cols + rank_ic_cols
    require_columns(df, required_cols, path)

    original_row = df.iloc[0]
    nw_p_score = 1 if count_condition(original_row, nw_p_cols, 0.2, "lt") >= 3 else 0
    pos_ic_prob_score = 1 if count_condition(original_row, pos_ic_prob_cols, 0.5, "gt") >= 3 else 0
    pearson_ic_score = 0.5 if count_condition(original_row, pearson_ic_cols, 0, "gt") >= 3 else 0
    rank_ic_score = 0.5 if count_condition(original_row, rank_ic_cols, 0, "gt") >= 3 else 0
    total_score = nw_p_score + pos_ic_prob_score + pearson_ic_score + rank_ic_score

    return {
        "factor_id": factor_id_from_path(path),
        NW_P_SCORE_COL: nw_p_score,
        POS_IC_PROB_SCORE_COL: pos_ic_prob_score,
        PEARSON_IC_SCORE_COL: pearson_ic_score,
        RANK_IC_SCORE_COL: rank_ic_score,
        TOTAL_SCORE_COL: total_score,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    errors: list[str] = []

    for path in sorted(input_dir.rglob(f"*{ANALYSIS_SUFFIX}")):
        try:
            rows.append(score_file(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    output_df = pd.DataFrame(
        rows,
        columns=[
            "factor_id",
            NW_P_SCORE_COL,
            POS_IC_PROB_SCORE_COL,
            PEARSON_IC_SCORE_COL,
            RANK_IC_SCORE_COL,
            TOTAL_SCORE_COL,
        ],
    )
    if not output_df.empty:
        output_df = output_df.sort_values("factor_id").reset_index(drop=True)
    output_df.to_excel(output_path, index=False)

    print(f"Scanned files: {len(rows) + len(errors)}")
    print(f"Scored factors: {len(rows)}")
    print(f"Output: {output_path}")

    if errors:
        print("Files skipped due to errors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
