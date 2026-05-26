"""Price style factors from completed V071-V142 plan file 9."""

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


OUTPUT_PREFIX = "priceFactors6"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 9.json"

FACTOR_IDS = [
    "V101",
    "V102",
    "V103",
    "V104",
    "V105",
]

HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"
CSI1000_PRICE_FILE = "中证1000(000852.SH)-历史价格.xlsx"
GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
INDEX_EOD_TABLE = "index_eod.parquet"

CSI500_INDEX = "000905"


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
            if factor_id in {"V101", "V102"}:
                _append_note(item, "本脚本使用 prepared_data 中的沪深300和中证1000历史价格文件，按月末收盘价计算月收益。")
            elif factor_id == "V103":
                _append_note(item, "本脚本使用 growth_index.xlsx/value_index.xlsx 的 turnover_rate 字段直接代理成交额/流通市值。")
            elif factor_id in {"V104", "V105"}:
                _append_note(item, "本脚本使用 index_eod.parquet 的中证500收盘指数，并使用沪深300历史价格文件作为大盘/价值代理。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 9 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors6_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _index_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].replace(".0", "").zfill(6)


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


def _load_hs300_close() -> pd.Series:
    return _read_excel_series(HS300_PRICE_FILE, "交易日期", "收盘价", "hs300_close")


def _load_csi1000_close() -> pd.Series:
    return _read_excel_series(CSI1000_PRICE_FILE, "交易日期", "收盘价", "csi1000_close")


def _load_growth_turnover() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "turnover_rate", "growth_turnover")


def _load_value_turnover() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "turnover_rate", "value_turnover")


def _load_index_eod_series(index_code: str, value_col: str, name: str) -> pd.Series:
    df = load_prepared_table(INDEX_EOD_TABLE)
    if "交易所指数代码" not in df.columns or value_col not in df.columns:
        raise KeyError(f"{INDEX_EOD_TABLE} must contain '交易所指数代码' and {value_col!r}; available={list(df.columns)}")

    if isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index, errors="coerce").normalize()
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        raise KeyError(f"{INDEX_EOD_TABLE} must have a DatetimeIndex or date column")

    work = df.copy()
    work["date"] = dates
    work["index_code"] = work["交易所指数代码"].map(_index_code_key)
    out = work[work["index_code"].eq(_index_code_key(index_code))].copy()
    if out.empty:
        raise ValueError(f"{INDEX_EOD_TABLE} missing index_code={index_code!r}")
    return _as_float_series(out[value_col], out["date"], name)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _month_end_last(series: pd.Series) -> pd.Series:
    return series.astype("float64").dropna().sort_index().resample("ME").last()


def _calc_v101(hs300_close: pd.Series, csi1000_close: pd.Series) -> pd.Series:
    close_large = _month_end_last(hs300_close)
    close_growth = _month_end_last(csi1000_close)
    ret_large = close_large.pct_change(fill_method=None)
    ret_growth = close_growth.pct_change(fill_method=None)
    raw = (ret_large - ret_growth).shift(1) * -1.0
    return calc_rolling_zscore(raw, window=24)


def _calc_v102(hs300_close: pd.Series, csi1000_close: pd.Series) -> pd.Series:
    close_large = _month_end_last(hs300_close)
    close_growth = _month_end_last(csi1000_close)
    ret_large = close_large.pct_change(fill_method=None)
    ret_growth = close_growth.pct_change(fill_method=None)
    raw = (ret_large - ret_growth).shift(3) * -1.0
    return calc_rolling_zscore(raw, window=24)


def _calc_v103() -> pd.Series:
    growth_turnover = _load_growth_turnover()
    value_turnover = _load_value_turnover()
    raw = _safe_ratio(growth_turnover, value_turnover) - 1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v104(hs300_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    aligned = pd.concat([hs300_close.rename("hs300"), csi500_close.rename("csi500")], axis=1).dropna()
    aligned = aligned[(aligned["hs300"] > 0) & (aligned["csi500"] > 0)]
    log_ratio = np.log(_safe_ratio(aligned["hs300"], aligned["csi500"]))
    ma20 = log_ratio.rolling(20, min_periods=20).mean()
    raw = ma20 - log_ratio
    return calc_rolling_zscore(raw, window=504)


def _calc_v105(hs300_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    ret_hs300 = hs300_close.pct_change(20, fill_method=None)
    ret_csi500 = csi500_close.pct_change(20, fill_method=None)
    raw = ret_csi500 - ret_hs300
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors6_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    hs300_close = _load_hs300_close()
    csi1000_close = _load_csi1000_close()
    csi500_close = _load_index_eod_series(CSI500_INDEX, "收盘指数", "csi500_close")

    _register_factor(raw_factor_df, factor_source_df, "V101_raw", _calc_v101(hs300_close, csi1000_close))
    _register_factor(raw_factor_df, factor_source_df, "V102_raw", _calc_v102(hs300_close, csi1000_close))
    _register_factor(raw_factor_df, factor_source_df, "V103_raw", _calc_v103())
    _register_factor(raw_factor_df, factor_source_df, "V104_raw", _calc_v104(hs300_close, csi500_close))
    _register_factor(raw_factor_df, factor_source_df, "V105_raw", _calc_v105(hs300_close, csi500_close))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors6 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors6_factors(data_df)
    metadata = metadata_from_priceFactors6_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V071_V142 10.json.
PLAN10_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 10.json"
PLAN10_FACTOR_IDS = [
    "V096",
    "V097",
    "V098",
    "V099",
    "V100",
]
EXTENDED_FACTOR_IDS = PLAN10_FACTOR_IDS + FACTOR_IDS

CSI100_INDEX = "000903"
CHINEXT_PRICE_FILE = "创业板指(399006.SZ)-历史价格.xlsx"


def _load_plan10_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN10_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN10_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN10_FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id == "V096":
                _append_note(item, "本脚本使用 index_eod.parquet 的 000905 和 000903 成份证券成交金额，按 W-FRI 汇总为周频。")
            elif factor_id == "V097":
                _append_note(item, "本脚本使用 prepared_data 中的创业板指历史价格，并沿计划用沪深300作为主板代理。")
            elif factor_id == "V098":
                item["signal_type"] = "state"
                _append_note(item, "原计划 signal_type=unknown；本脚本按 notes 输出连续 zscore，并按 state 挂载/生成阈值信号。")
            elif factor_id == "V099":
                _append_note(item, "本脚本复用中证500/沪深300相对强弱比值，计算10日均线减30日均线。")
            elif factor_id == "V100":
                _append_note(item, "本脚本使用 growth_index.xlsx/value_index.xlsx 的 close 计算成长-价值对数收益差。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN10_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 10 missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN10_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors6_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _load_index_eod_amount(index_code: str, name: str) -> pd.Series:
    return _load_index_eod_series(index_code, "成份证券成交金额", name)


def _load_chinext_close() -> pd.Series:
    return _read_excel_series(CHINEXT_PRICE_FILE, "交易日期", "收盘价", "chinext_close")


def _load_growth_close() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "close", "growth_close")


def _load_value_close() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "close", "value_close")


def _relative_strength_500_300(hs300_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    return _safe_ratio(csi500_close, hs300_close)


def _calc_v096() -> pd.Series:
    amount_500 = _load_index_eod_amount(CSI500_INDEX, "csi500_amount")
    amount_100 = _load_index_eod_amount(CSI100_INDEX, "csi100_amount")
    amount_500_w = amount_500.resample("W-FRI").sum(min_count=1)
    amount_100_w = amount_100.resample("W-FRI").sum(min_count=1)
    ratio_amt = _safe_ratio(amount_500_w, amount_100_w)
    ma_m = ratio_amt.rolling(window=8, min_periods=8).mean()
    raw = ratio_amt - ma_m
    return calc_rolling_zscore(raw, window=104)


def _calc_v097(hs300_close: pd.Series) -> pd.Series:
    chinext_close = _load_chinext_close()
    ratio = _safe_ratio(chinext_close, hs300_close)
    raw = ratio - ratio.shift(1)
    return calc_rolling_zscore(raw, window=504)


def _calc_v098(hs300_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    relative_strength = _relative_strength_500_300(hs300_close, csi500_close)
    raw = relative_strength - relative_strength.rolling(252, min_periods=252).mean()
    return calc_rolling_zscore(raw, window=504)


def _calc_v099(hs300_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    relative_strength = _relative_strength_500_300(hs300_close, csi500_close)
    raw = relative_strength.rolling(window=10, min_periods=10).mean() - relative_strength.rolling(window=30, min_periods=30).mean()
    return calc_rolling_zscore(raw, window=504)


def _calc_v100() -> pd.Series:
    value_close = _load_value_close()
    growth_close = _load_growth_close()
    aligned = pd.concat([value_close.rename("value"), growth_close.rename("growth")], axis=1).dropna()
    aligned = aligned[(aligned["value"] > 0) & (aligned["growth"] > 0)]
    log_ret_value = np.log(aligned["value"]).diff()
    log_ret_growth = np.log(aligned["growth"]).diff()
    raw = (log_ret_value - log_ret_growth) * -1.0
    return calc_rolling_zscore(raw, window=504)


def generate_plan10_priceFactors6_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan10_records()

    hs300_close = _load_hs300_close()
    csi500_close = _load_index_eod_series(CSI500_INDEX, "收盘指数", "csi500_close")

    _register_factor(raw_factor_df, factor_source_df, "V096_raw", _calc_v096())
    _register_factor(raw_factor_df, factor_source_df, "V097_raw", _calc_v097(hs300_close))
    _register_factor(raw_factor_df, factor_source_df, "V098_raw", _calc_v098(hs300_close, csi500_close))
    _register_factor(raw_factor_df, factor_source_df, "V099_raw", _calc_v099(hs300_close, csi500_close))
    _register_factor(raw_factor_df, factor_source_df, "V100_raw", _calc_v100())

    missing_cols = [factor_id for factor_id in PLAN10_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors6 plan 10 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN10_FACTOR_IDS], records


def generate_extended_priceFactors6_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan10_factor_df, plan10_records = generate_plan10_priceFactors6_factors(data_df)
    plan9_factor_df, plan9_records = generate_priceFactors6_factors(data_df)
    factor_source_df = pd.concat([plan10_factor_df, plan9_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan10_records + plan9_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors6_factors(data_df)
    metadata = metadata_from_extended_priceFactors6_records(selected_records)
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
