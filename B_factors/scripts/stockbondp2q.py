"""Stock-bond pricing factors from working_multiple_factors_plan.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from factor_utils import (
    PROJECT_ROOT,
    _month_aggregate,
    _register_factor,
    _rolling_quantile_rank_year,
    build_threshold_signal_ls_df,
    load_benchmark_index,
    load_default_data,
    load_prepared_table,
    mount_factor_source_frame,
    read_prepared_series,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "stockbondp2q"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

FACTOR_IDS = ["I057", "I058", "I059", "I060", "I061", "I072"]

VALUATION_FILE = "881001.WI-历史PE-PB-20260518.xlsx"
BOND_INDEX_FILE = "行情统计2026-05-18(H11006.CSI).xlsx"
GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
RATE_TABLE = "rate_daily.parquet"

CN_TREASURY_10Y_COL = "中债国债到期收益率:10年"
CN_AAA_10Y_COL = "中债中短期票据到期收益率(AAA):10年"


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _is_unknown_or_todo(value: object) -> bool:
    return _normalize_plan_text(value).lower() in {"", "unknown", "todo"}


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in FACTOR_IDS:
                continue
            if factor_id != "I072" and (
                _is_unknown_or_todo(record.get("docu")) or _is_unknown_or_todo(record.get("data_field"))
            ):
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            records.append(_record_with_actual_fields(item))

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"working_multiple_factors_plan.json missing implemented records: {missing}")
    return records


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _record_with_actual_fields(record: dict[str, object]) -> dict[str, object]:
    factor_id = str(record["factor_id"])
    if not _normalize_plan_text(record.get("paper_id")):
        record["paper_id"] = "DIY"

    if factor_id == "I057":
        record["data_field"] = (
            "index_eod.parquet[交易所指数代码 in {000985,985,985.0}, 收盘指数]; "
            "行情统计2026-05-18(H11006.CSI).xlsx[交易日期, 收盘价]"
        )
    elif factor_id == "I058":
        record["data_field"] = (
            "881001.WI-历史PE-PB-20260518.xlsx[交易日期, 市盈率TTM加权]; "
            "rate_daily.parquet[中债中短期票据到期收益率(AAA):10年]"
        )
    elif factor_id in {"I059", "I060", "I061"}:
        record["data_field"] = (
            "881001.WI-历史PE-PB-20260518.xlsx[交易日期, 股息率]; "
            "rate_daily.parquet[中债国债到期收益率:10年]"
        )
        if factor_id == "I061":
            record["data_field"] += "; growth_index.xlsx[date, close]; value_index.xlsx[date, close]"
        if factor_id == "I059":
            _append_note(record, "计划表 data_field 写作市盈率TTM加权，但本脚本按因子名称和 calc_method 修正为股息率 - 10年国债收益率。")
    elif factor_id == "I072":
        record["docu"] = VALUATION_FILE
        record["signal_type"] = "state"
        record["data_field"] = "881001.WI-历史PE-PB-20260518.xlsx[交易日期, 股息率]"
        _append_note(record, "计划表 condition 含 sub_2 但该因子为股票分红率，本脚本按万得全A股息率自身的3年滚动分位数实现。")

    return record


def metadata_from_stockbondp2q_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _clean_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


def _percent_points_to_decimal(series: pd.Series) -> pd.Series:
    s = series.astype("float64")
    sample = s.dropna()
    if not sample.empty and sample.abs().median() > 0.5:
        return s / 100.0
    return s


def _load_valuation_series(column: str, *, decimal_percent: bool = False) -> pd.Series:
    df = load_prepared_table(VALUATION_FILE)
    if "交易日期" not in df.columns:
        raise KeyError(f"{VALUATION_FILE} missing 交易日期")
    if column not in df.columns:
        raise KeyError(f"{VALUATION_FILE} missing {column!r}; available={list(df.columns)}")
    series = _as_float_series(df[column], _clean_date_series(df["交易日期"]), column)
    return _percent_points_to_decimal(series) if decimal_percent else series


def _load_csi_all_index_close() -> pd.Series:
    df = load_prepared_table("index_eod.parquet")
    if "交易所指数代码" not in df.columns or "收盘指数" not in df.columns:
        raise KeyError("index_eod.parquet must contain 交易所指数代码 and 收盘指数")
    index_code = df["交易所指数代码"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    index_code = index_code.str.zfill(6)
    out = df.loc[index_code.eq("000985")].copy()
    if out.empty:
        raise ValueError("index_eod.parquet 中找不到中证全指代码 000985/985/985.0")
    if isinstance(out.index, pd.DatetimeIndex):
        dates = pd.Series(out.index, index=out.index)
    else:
        date_col = next((col for col in ["date", "日期", "Idxtrd01", "交易日期"] if col in out.columns), None)
        if date_col is None:
            raise KeyError("index_eod.parquet 没有 datetime index，也找不到日期列")
        dates = _clean_date_series(out[date_col])
    return _as_float_series(out["收盘指数"], dates, "中证全指")


def _load_price_file_close(file_name: str, date_col: str, close_col: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if date_col not in df.columns or close_col not in df.columns:
        raise KeyError(f"{file_name} must contain {date_col!r} and {close_col!r}; available={list(df.columns)}")
    return _as_float_series(df[close_col], _clean_date_series(df[date_col]), name)


def _monthly_return(close: pd.Series, how: str = "last") -> pd.Series:
    monthly_close = _month_aggregate(close.dropna(), how=how)
    return (monthly_close / monthly_close.shift(1) - 1).dropna()


def _tail_signal_from_quantile(quantile: pd.Series, high: float = 0.9, low: float = 0.1) -> pd.Series:
    signal = pd.Series(0.0, index=quantile.index, dtype="float64")
    signal.loc[quantile > high] = 1.0
    signal.loc[quantile < low] = -1.0
    signal.loc[quantile.isna()] = np.nan
    return signal


def _turning_point_signal(
    quantile: pd.Series,
    growth_close: pd.Series,
    value_close: pd.Series,
    window: int = 42,
) -> pd.Series:
    growth_ret = growth_close / growth_close.shift(window) - 1
    value_ret = value_close / value_close.shift(window) - 1
    aligned = pd.concat(
        [quantile.rename("quantile"), growth_ret.rename("growth_ret"), value_ret.rename("value_ret")],
        axis=1,
        sort=False,
    ).dropna(subset=["quantile"])

    signal = pd.Series(0.0, index=aligned.index, dtype="float64")
    high_mask = aligned["quantile"] > 0.9
    low_mask = aligned["quantile"] < 0.1
    value_condition = (aligned["value_ret"] < 0) & (aligned["growth_ret"] > aligned["value_ret"])
    signal.loc[high_mask & value_condition] = -1.0
    signal.loc[high_mask & ~value_condition] = 1.0
    signal.loc[low_mask] = -1.0
    signal.loc[aligned["quantile"].isna()] = np.nan
    return signal


def _calc_stockbondp2q_factor(factor_id: str) -> pd.Series:
    if factor_id == "I057":
        stock_return = _monthly_return(_load_csi_all_index_close())
        bond_return = _monthly_return(_load_price_file_close(BOND_INDEX_FILE, "交易日期", "收盘价", "中证国债指数"))
        spread = (stock_return - bond_return).dropna()
        return 0.5 - _rolling_quantile_rank_year(spread, year=3)

    pe_ttm_weighted = _load_valuation_series("市盈率TTM加权")
    dividend_yield = _load_valuation_series("股息率", decimal_percent=True)
    treasury_10y = read_prepared_series(RATE_TABLE, CN_TREASURY_10Y_COL)

    if factor_id == "I058":
        aaa_10y = read_prepared_series(RATE_TABLE, CN_AAA_10Y_COL)
        erp = ((1.0 / pe_ttm_weighted) / aaa_10y).dropna()
        return erp - erp.rolling(window=63, min_periods=63).mean()

    div_minus_treasury = (dividend_yield - treasury_10y).dropna()
    if factor_id == "I059":
        return _rolling_quantile_rank_year(div_minus_treasury, year=3) - 0.5
    if factor_id == "I060":
        quantile = _rolling_quantile_rank_year(div_minus_treasury, year=4)
        return _tail_signal_from_quantile(_month_aggregate(quantile, how="last"))
    if factor_id == "I061":
        quantile = _rolling_quantile_rank_year(div_minus_treasury, year=4)
        growth_close = _load_price_file_close(GROWTH_INDEX_FILE, "date", "close", "growth_close")
        value_close = _load_price_file_close(VALUE_INDEX_FILE, "date", "close", "value_close")
        return _turning_point_signal(quantile, growth_close, value_close)
    if factor_id == "I072":
        return _rolling_quantile_rank_year(dividend_yield.dropna(), year=3) - 0.5

    raise KeyError(f"Unsupported factor_id: {factor_id}")


def generate_stockbondp2q_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    by_factor_id = {str(record["factor_id"]): record for record in records}
    for factor_id in FACTOR_IDS:
        if factor_id not in by_factor_id:
            continue
        factor_series = _calc_stockbondp2q_factor(factor_id)
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"stockbondp2q factor columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], records


def generate_stockbondp2q_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    factor_source_df, _records = generate_stockbondp2q_factors(data_df)
    return factor_source_df


def _print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
    print(f"{label} mounted_normalized_factor_df shape:", mounted_factor_df.shape)
    print(f"{label} signal_ls_df shape:", signal_ls_df.shape)
    print(f"{label} factor columns:", list(mounted_factor_df.columns))
    print(f"{label} factor non-null summary:")
    for factor_col in mounted_factor_df.columns:
        series = mounted_factor_df[factor_col]
        print(
            factor_col,
            "non_na=", int(series.notna().sum()),
            "first=", series.first_valid_index(),
            "last=", series.last_valid_index(),
        )


def main() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_stockbondp2q_factors(data_df)
    metadata = metadata_from_stockbondp2q_records(selected_records)
    mounted_normalized_factor_df = mount_factor_source_frame(
        factor_source_df=factor_source_df,
        market_df=market_df,
        benchmark_index=benchmark_index,
        metadata=metadata,
    )
    signal_ls_df = build_threshold_signal_ls_df(mounted_normalized_factor_df, metadata)
    output_paths = save_factor_outputs(
        mounted_normalized_factor_df=mounted_normalized_factor_df,
        signal_ls_df=signal_ls_df,
        missing_bar_defaults=[],
        output_prefix=OUTPUT_PREFIX,
        write_empty_missing_bar_file=False,
    )

    for label, path in output_paths.items():
        print(f"{label} saved to:", path)
    generated_path = save_generated_factor_records(selected_records, OUTPUT_PREFIX)
    print("generated records saved to:", generated_path)
    _print_factor_output_summary(OUTPUT_PREFIX, mounted_normalized_factor_df, signal_ls_df)


if __name__ == "__main__":
    main()
