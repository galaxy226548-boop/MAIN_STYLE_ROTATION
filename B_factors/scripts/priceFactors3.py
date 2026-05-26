"""Price style factors from completed V071-V142 plan file 3."""

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
    _as_numeric,
    _register_factor,
    build_threshold_signal_ls_df,
    calc_rolling_zscore,
    load_benchmark_index,
    load_default_data,
    load_prepared_table,
    mount_factor_source_frame,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "priceFactors3"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 3.json"

FACTOR_IDS = [
    "V131",
    "V132",
    "V133",
    "V134",
    "V135",
]

GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
MKT_PRICE_TABLE = "mktP.parquet"


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    wanted = set(FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id in {"V132", "V133"}:
                _append_note(item, "本脚本按计划记录使用60日滚动波动率和504日滚动标准化。")
            if factor_id == "V134":
                _append_note(item, "本脚本按计划记录使用国证成长/国证价值 RSI_5D 差值代理全市场超买超卖。")
            if factor_id == "V135":
                _append_note(item, "本脚本使用 mktP.parquet 复权收盘价计算全市场个股位于20日均线上方占比。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 3 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors3_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


def _read_excel_series(file_name: str, date_col: str, value_col: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if date_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"{file_name} must contain {date_col!r} and {value_col!r}; available={list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    values = _as_numeric(df[value_col])
    return _as_float_series(values, dates, name)


def _load_growth_close() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "close", "growth_close")


def _load_value_close() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "close", "value_close")


def _load_growth_amount() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "amount", "growth_amount")


def _load_value_amount() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "amount", "value_amount")


def _load_growth_rsi_5d() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "RSI_5D", "growth_rsi_5d")


def _load_value_rsi_5d() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "RSI_5D", "value_rsi_5d")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _calc_v131(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    raw = _safe_ratio(growth_close, value_close)
    return calc_rolling_zscore(raw, window=504)


def _calc_v132(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_ret = growth_close.pct_change(fill_method=None)
    value_ret = value_close.pct_change(fill_method=None)
    vol_growth = growth_ret.rolling(60, min_periods=60).std()
    vol_value = value_ret.rolling(60, min_periods=60).std()
    return calc_rolling_zscore(_safe_ratio(vol_growth, vol_value), window=504)


def _calc_v133() -> pd.Series:
    growth_amount = _load_growth_amount()
    value_amount = _load_value_amount()
    vol_amt_growth = growth_amount.rolling(60, min_periods=60).std()
    vol_amt_value = value_amount.rolling(60, min_periods=60).std()
    return calc_rolling_zscore(_safe_ratio(vol_amt_growth, vol_amt_value), window=504)


def _calc_v134() -> pd.Series:
    raw_diff = _load_growth_rsi_5d() - _load_value_rsi_5d()
    short_ma = raw_diff.rolling(5, min_periods=5).mean()
    long_ma = raw_diff.rolling(250, min_periods=250).mean()
    return short_ma - long_ma


def _calc_v135() -> pd.Series:
    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / MKT_PRICE_TABLE,
        columns=["Stkcd", "Trddt", "Adjprcwd"],
    )
    required_cols = {"Stkcd", "Trddt", "Adjprcwd"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"{MKT_PRICE_TABLE} must contain {sorted(required_cols)}; available={list(df.columns)}")

    work = df.dropna(subset=["Stkcd", "Trddt", "Adjprcwd"]).copy()
    work["Trddt"] = pd.to_datetime(work["Trddt"], errors="coerce").dt.normalize()
    work["Adjprcwd"] = pd.to_numeric(work["Adjprcwd"], errors="coerce")
    work = work.dropna(subset=["Trddt", "Adjprcwd"]).sort_values(["Stkcd", "Trddt"])

    ma20 = (
        work.groupby("Stkcd", sort=False)["Adjprcwd"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ratio_above_ma20 = work["Adjprcwd"].gt(ma20).groupby(work["Trddt"]).mean().sort_index()
    raw = ratio_above_ma20.rolling(5, min_periods=5).mean() - ratio_above_ma20.rolling(250, min_periods=250).mean()
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors3_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()

    _register_factor(raw_factor_df, factor_source_df, "V131_raw", _calc_v131(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V132_raw", _calc_v132(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V133_raw", _calc_v133())
    _register_factor(raw_factor_df, factor_source_df, "V134_raw", _calc_v134())
    _register_factor(raw_factor_df, factor_source_df, "V135_raw", _calc_v135())

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors3 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], records


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

    factor_source_df, selected_records = generate_priceFactors3_factors(data_df)
    metadata = metadata_from_priceFactors3_records(selected_records)
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


if __name__ == "__main__" and False:
    main()


# Extended factors from working_multiple_factors_plan_completed_V071_V142 4.json.
PLAN4_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 4.json"
PLAN4_FACTOR_IDS = [
    "V126",
    "V127",
    "V128",
    "V129",
    "V130",
]
EXTENDED_FACTOR_IDS = FACTOR_IDS + PLAN4_FACTOR_IDS

HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"
CSI1000_PRICE_FILE = "中证1000(000852.SH)-历史价格.xlsx"
CSI_ALL_PRICE_FILE = "中证全指(000985.CSI)-历史价格.xlsx"
CHINEXT_PRICE_FILE = "创业板指(399006.SZ)-历史价格.xlsx"
BIAS_FUND_PRICE_FILE = "偏股基金(930950.CSI)-历史价格.xlsx"
WIND_ALL_A_FILE = "8841388.WI_windallA.xlsx"


def _load_plan4_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN4_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN4_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN4_FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id == "V126":
                _append_note(item, "本脚本用 mktP.parquet 的 Dretwd 计算月度 DI，用 Dnvaltrd 作为成交量代理。")
            if factor_id == "V129":
                _append_note(item, "分化度本身方向不确定，本脚本沿计划记录保留正向 zscore 代理。")
            if factor_id == "V130":
                _append_note(item, "头部拥挤度方向不确定，本脚本沿计划记录使用成长/价值周成交额分位数差。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN4_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 4 missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN4_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors3_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _load_index_close(file_name: str, name: str) -> pd.Series:
    return _read_excel_series(file_name, "交易日期", "收盘价", name)


def _rolling_rank_pct(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    s = series.astype("float64").sort_index()

    def rank_last(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) < min_periods:
            return np.nan
        return float((valid <= valid[-1]).sum() / len(valid))

    return s.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _load_monthly_market_di_and_amount() -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / MKT_PRICE_TABLE,
        columns=["Stkcd", "Trddt", "Dretwd", "Dnvaltrd"],
    )
    required_cols = {"Stkcd", "Trddt", "Dretwd", "Dnvaltrd"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"{MKT_PRICE_TABLE} must contain {sorted(required_cols)}; available={list(df.columns)}")

    work = df.dropna(subset=["Stkcd", "Trddt"]).copy()
    work["Trddt"] = pd.to_datetime(work["Trddt"], errors="coerce")
    work["Dretwd"] = pd.to_numeric(work["Dretwd"], errors="coerce")
    work["Dnvaltrd"] = pd.to_numeric(work["Dnvaltrd"], errors="coerce")
    work = work.dropna(subset=["Trddt"])
    work["month"] = work["Trddt"].dt.to_period("M")

    monthly_ret = work.groupby(["month", "Stkcd"], sort=True)["Dretwd"].sum(min_count=1)

    def trimmed_std(values: pd.Series) -> float:
        valid = values.dropna()
        if valid.empty:
            return np.nan
        lower = valid.quantile(0.1)
        upper = valid.quantile(0.9)
        trimmed = valid[(valid >= lower) & (valid <= upper)]
        return float(trimmed.std()) if len(trimmed) > 1 else np.nan

    di = monthly_ret.groupby(level="month").apply(trimmed_std).astype("float64")
    monthly_amount = work.groupby("month", sort=True)["Dnvaltrd"].sum(min_count=1).astype("float64")
    month_dates = pd.PeriodIndex(di.index, freq="M").to_timestamp("M")
    di.index = month_dates
    monthly_amount = monthly_amount.reindex(di.index.to_period("M"))
    monthly_amount.index = month_dates
    return di.sort_index(), monthly_amount.sort_index()


def _calc_v126() -> pd.Series:
    di, monthly_amount = _load_monthly_market_di_and_amount()
    di_raw = di - di.rolling(12, min_periods=12).mean()
    vol_ratio = 12.0 * monthly_amount / monthly_amount.rolling(12, min_periods=12).mean()
    return calc_rolling_zscore(di_raw * vol_ratio, window=120)


def _calc_v127() -> pd.Series:
    hs300_close = _load_index_close(HS300_PRICE_FILE, "hs300_close")
    csi1000_close = _load_index_close(CSI1000_PRICE_FILE, "csi1000_close")
    hs300_ret = hs300_close.pct_change(fill_method=None)
    csi1000_ret = csi1000_close.pct_change(fill_method=None)
    vol_hs300 = hs300_ret.rolling(21, min_periods=21).std()
    vol_csi1000 = csi1000_ret.rolling(21, min_periods=21).std()
    ratio = _safe_ratio(vol_hs300, vol_csi1000)
    q_mean_cur = ratio.rolling(63, min_periods=63).mean()
    q_mean_prev = q_mean_cur.shift(63)
    return calc_rolling_zscore(q_mean_cur - q_mean_prev, window=504)


def _calc_v128() -> pd.Series:
    bias_fund_close = _load_index_close(BIAS_FUND_PRICE_FILE, "bias_fund_close")
    wind_all_a_close = _load_index_close(WIND_ALL_A_FILE, "wind_all_a_close")
    bias_ret = bias_fund_close.pct_change(fill_method=None)
    wind_all_a_ret = wind_all_a_close.pct_change(fill_method=None)
    vol_bias = bias_ret.rolling(63, min_periods=63).std()
    vol_wind_all_a = wind_all_a_ret.rolling(63, min_periods=63).std()
    crowd = _safe_ratio(vol_bias, vol_wind_all_a)
    crowd_eom = crowd.dropna().resample("ME").last()
    q1_mean = crowd_eom.rolling(3, min_periods=3).mean()
    q3_mean = crowd_eom.rolling(9, min_periods=9).mean()
    return calc_rolling_zscore(q1_mean - q3_mean, window=36)


def _calc_v129() -> pd.Series:
    price_df = pd.concat(
        [
            _load_index_close(HS300_PRICE_FILE, "000300"),
            _load_index_close(CSI1000_PRICE_FILE, "000852"),
            _load_index_close(CSI_ALL_PRICE_FILE, "000985"),
            _load_index_close(CHINEXT_PRICE_FILE, "399006"),
        ],
        axis=1,
        sort=True,
    )
    weekly_ret = price_df.resample("W-FRI").last().pct_change(fill_method=None)
    divergence = weekly_ret.std(axis=1, skipna=True)
    return calc_rolling_zscore(divergence, window=104)


def _calc_v130() -> pd.Series:
    growth_weekly_amount = _load_growth_amount().resample("W-FRI").sum(min_count=1)
    value_weekly_amount = _load_value_amount().resample("W-FRI").sum(min_count=1)
    growth_qrank = _rolling_rank_pct(growth_weekly_amount, window=52)
    value_qrank = _rolling_rank_pct(value_weekly_amount, window=52)
    return calc_rolling_zscore(growth_qrank - value_qrank, window=104)


def generate_plan4_priceFactors3_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan4_records()

    _register_factor(raw_factor_df, factor_source_df, "V126_raw", _calc_v126())
    _register_factor(raw_factor_df, factor_source_df, "V127_raw", _calc_v127())
    _register_factor(raw_factor_df, factor_source_df, "V128_raw", _calc_v128())
    _register_factor(raw_factor_df, factor_source_df, "V129_raw", _calc_v129())
    _register_factor(raw_factor_df, factor_source_df, "V130_raw", _calc_v130())

    missing_cols = [factor_id for factor_id in PLAN4_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors3 plan 4 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN4_FACTOR_IDS], records


def generate_extended_priceFactors3_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan3_factor_df, plan3_records = generate_priceFactors3_factors(data_df)
    plan4_factor_df, plan4_records = generate_plan4_priceFactors3_factors(data_df)
    factor_source_df = pd.concat([plan3_factor_df, plan4_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan3_records + plan4_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors3_factors(data_df)
    metadata = metadata_from_extended_priceFactors3_records(selected_records)
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
    main_extended()
