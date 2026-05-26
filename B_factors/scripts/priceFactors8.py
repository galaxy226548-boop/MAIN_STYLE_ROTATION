"""Price style factors from completed V071-V142 plan file 13."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


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


OUTPUT_PREFIX = "priceFactors8"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 13.json"

FACTOR_IDS = [
    "V081",
    "V082",
    "V083",
    "V084",
    "V085",
]

VIX_FILE = "VIX.GI-行情统计-20260509.xlsx"
MACRO_DAILY_TABLE = "macro_daily.parquet"
INDEX_EOD_TABLE = "index_eod.parquet"

GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"


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
            if factor_id == "V081":
                _append_note(item, "本脚本使用 VIX.GI-行情统计-20260509.xlsx 的收盘价，并按计划显式 ffill 后计算75日变化率。")
            elif factor_id == "V082":
                _append_note(item, "本脚本使用 macro_daily.parquet 的 CRB现货指数:综合，并按计划执行 expanding 经验CDF逆正态变换。")
            elif factor_id == "V083":
                _append_note(item, "本脚本使用 index_eod.parquet 的国证成长/国证价值收盘指数，向量化计算12日滚动斜率差分。")
            elif factor_id in {"V084", "V085"}:
                _append_note(item, "本脚本使用 index_eod.parquet 的国证成长/国证价值收盘指数，按计划计算相对收益/收益差分。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 13 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors8_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _load_vix_close() -> pd.Series:
    return _read_excel_series(VIX_FILE, "交易日期", "收盘价", "vix_close")


def _load_crb_spot() -> pd.Series:
    df = load_prepared_table(MACRO_DAILY_TABLE)
    if "CRB现货指数:综合" not in df.columns:
        raise KeyError(f"{MACRO_DAILY_TABLE} must contain 'CRB现货指数:综合'; available={list(df.columns)}")
    return _as_float_series(
        df["CRB现货指数:综合"],
        df.index,
        "crb_spot",
    )


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


def _expanding_cdf_to_normal(series: pd.Series) -> pd.Series:
    s = series.astype("float64").sort_index()

    def rank_last(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= valid[-1]).sum() / len(valid))

    probability = s.expanding(min_periods=1).apply(rank_last, raw=True).clip(0.001, 0.999)
    return pd.Series(norm.ppf(probability), index=s.index, dtype="float64")


def _rolling_slope(series: pd.Series, window: int = 12) -> pd.Series:
    s = series.astype("float64").sort_index()
    x = np.arange(window, dtype="float64")
    x = x - x.mean()
    denominator = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values.astype("float64")
        return float(np.dot(x, y - y.mean()) / denominator)

    return s.rolling(window=window, min_periods=window).apply(slope, raw=True)


def _load_style_closes() -> tuple[pd.Series, pd.Series]:
    growth_close = _load_index_eod_series(GROWTH_STYLE_INDEX, "收盘指数", "growth_close").ffill()
    value_close = _load_index_eod_series(VALUE_STYLE_INDEX, "收盘指数", "value_close").ffill()
    return growth_close, value_close


def _calc_v081() -> pd.Series:
    vix = _load_vix_close().ffill()
    raw = _safe_ratio(vix, vix.rolling(75, min_periods=75).mean()) - 1.0
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v082() -> pd.Series:
    crb = _load_crb_spot().ffill()
    log_crb = np.log(crb.where(crb > 0))
    raw = _expanding_cdf_to_normal(log_crb)
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v083(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    slope_g = _rolling_slope(growth_close, 12)
    slope_v = _rolling_slope(value_close, 12)
    raw = (slope_g - slope_g.shift(1)) - (slope_v - slope_v.shift(1))
    return calc_rolling_zscore(raw, window=504)


def _calc_v084(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    ret5_g = growth_close.pct_change(5, fill_method=None)
    ret5_v = value_close.pct_change(5, fill_method=None)
    mean_ret5 = (ret5_g + ret5_v) / 2.0
    raw = (ret5_g - mean_ret5) - (ret5_v - mean_ret5)
    return calc_rolling_zscore(raw, window=504)


def _calc_v085(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    ret20_g = growth_close.pct_change(20, fill_method=None)
    ret20_v = value_close.pct_change(20, fill_method=None)
    raw = (ret20_g - ret20_g.shift(1)) - (ret20_v - ret20_v.shift(1))
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors8_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_close, value_close = _load_style_closes()

    _register_factor(raw_factor_df, factor_source_df, "V081_raw", _calc_v081())
    _register_factor(raw_factor_df, factor_source_df, "V082_raw", _calc_v082())
    _register_factor(raw_factor_df, factor_source_df, "V083_raw", _calc_v083(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V084_raw", _calc_v084(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V085_raw", _calc_v085(growth_close, value_close))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors8 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors8_factors(data_df)
    metadata = metadata_from_priceFactors8_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V071_V142 14/15.json.
from factor_utils import _rolling_quantile_rank_year


PLAN14_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 14.json"
PLAN15_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 15.json"

PLAN14_FACTOR_IDS = [
    "V076",
    "V077",
    "V078",
    "V079",
    "V080",
]
PLAN15_FACTOR_IDS = [
    "V071",
    "V072",
    "V073",
    "V074",
    "V075",
]
EXTENDED_FACTOR_IDS = PLAN15_FACTOR_IDS + PLAN14_FACTOR_IDS + FACTOR_IDS

WIND_ALL_A_FILE = "8841388.WI_windallA.xlsx"
CSI_ALL_PRICE_FILE = "中证全指(000985.CSI)-历史价格.xlsx"
HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"

CSI_ALL_INDEX = "000985"
SSE50_INDEX = "000016"
CSI500_INDEX = "000905"


def _load_plan_records_from_path(
    plan_path: Path,
    factor_ids: list[str],
    file_label: str,
) -> list[dict[str, object]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(plan_path.relative_to(PROJECT_ROOT))
    wanted = set(factor_ids)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id in {"V071", "V076"}:
                _append_note(item, "本脚本使用本地成长/价值或中证全指 prepared 指数文件中的换手率字段。")
            elif factor_id in {"V072", "V073", "V074", "V077", "V078"}:
                _append_note(item, "本脚本使用 8841388.WI_windallA.xlsx 的换手率代理全市场换手率。")
            elif factor_id in {"V075", "V079", "V080"}:
                _append_note(item, "本脚本使用 index_eod.parquet 的指数换手率/收盘指数，并在需要时搭配沪深300历史价格。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in factor_ids if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file {file_label} missing implemented records: {missing}")
    return sorted(records, key=lambda record: factor_ids.index(str(record["factor_id"])))


def _load_plan14_records() -> list[dict[str, object]]:
    return _load_plan_records_from_path(PLAN14_PATH, PLAN14_FACTOR_IDS, "14")


def _load_plan15_records() -> list[dict[str, object]]:
    return _load_plan_records_from_path(PLAN15_PATH, PLAN15_FACTOR_IDS, "15")


def metadata_from_extended_priceFactors8_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return metadata_from_priceFactors8_records(records)


def _load_index_excel_field(file_name: str, value_col: str, name: str) -> pd.Series:
    return _read_excel_series(file_name, "交易日期", value_col, name)


def _load_growth_index_field(value_col: str, name: str) -> pd.Series:
    return _read_excel_series("growth_index.xlsx", "date", value_col, name)


def _load_value_index_field(value_col: str, name: str) -> pd.Series:
    return _read_excel_series("value_index.xlsx", "date", value_col, name)


def _calc_v071() -> pd.Series:
    growth_turnover = _load_growth_index_field("turnover_rate", "growth_turnover")
    value_turnover = _load_value_index_field("turnover_rate", "value_turnover")
    growth_close = _load_growth_index_field("close", "growth_close")
    value_close = _load_value_index_field("close", "value_close")

    ratio = _safe_ratio(growth_turnover, value_turnover)
    ratio_ma90 = ratio.rolling(window=90, min_periods=90).mean()
    quantile_rank = _rolling_quantile_rank_year(ratio_ma90, year=2)
    growth_momentum_15d = growth_close.pct_change(15, fill_method=None) - value_close.pct_change(15, fill_method=None)

    quantile_dev_low = calc_rolling_zscore(0.05 - quantile_rank, window=504)
    quantile_dev_high = calc_rolling_zscore(quantile_rank - 0.95, window=504)
    momentum_z = calc_rolling_zscore(growth_momentum_15d, window=504)

    factor = pd.Series(0.0, index=quantile_rank.index, dtype="float64")
    growth_mask = (quantile_rank < 0.05) & (growth_momentum_15d > 0)
    value_mask = (quantile_rank > 0.95) & (growth_momentum_15d < 0)
    factor.loc[growth_mask] = quantile_dev_low.loc[growth_mask] * momentum_z.loc[growth_mask]
    factor.loc[value_mask] = -quantile_dev_high.loc[value_mask] * momentum_z.loc[value_mask].abs()
    factor.loc[quantile_rank.isna()] = np.nan
    return factor


def _calc_v072(wind_all_a_turnover: pd.Series) -> pd.Series:
    ma60 = wind_all_a_turnover.rolling(window=60, min_periods=60).mean()
    raw = (_safe_ratio(ma60, ma60.shift(1)) - 1.0) * -1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v073(wind_all_a_turnover: pd.Series) -> pd.Series:
    half_year_max = wind_all_a_turnover.rolling(window=126, min_periods=126).max().shift(1)
    half_year_min = wind_all_a_turnover.rolling(window=126, min_periods=126).min().shift(1)
    exceed = _safe_ratio(wind_all_a_turnover, half_year_max) - 1.0
    below = _safe_ratio(wind_all_a_turnover, half_year_min) - 1.0
    exceed_z = calc_rolling_zscore(exceed, window=504)
    below_z = calc_rolling_zscore(below, window=504)

    factor = pd.Series(np.nan, index=wind_all_a_turnover.index, dtype="float64")
    high_mask = wind_all_a_turnover > half_year_max
    low_mask = wind_all_a_turnover < half_year_min
    factor.loc[high_mask] = exceed_z.loc[high_mask]
    factor.loc[low_mask] = below_z.loc[low_mask]
    return factor


def _calc_v074(wind_all_a_turnover: pd.Series) -> pd.Series:
    ma60 = wind_all_a_turnover.rolling(window=60, min_periods=60).mean()
    raw = ma60 - 1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v075() -> pd.Series:
    csi500_turnover = _load_index_eod_series(CSI500_INDEX, "换手率", "csi500_turnover")
    hs300_turnover = _load_index_excel_field(HS300_PRICE_FILE, "换手率", "hs300_turnover")
    ma500 = csi500_turnover.rolling(window=21, min_periods=21).mean()
    ma300 = hs300_turnover.rolling(window=21, min_periods=21).mean()
    raw = ma500 - ma300
    return calc_rolling_zscore(raw, window=504)


def _calc_v076() -> pd.Series:
    csi_all_turnover = _load_index_excel_field(CSI_ALL_PRICE_FILE, "换手率", "csi_all_turnover")
    ma20 = csi_all_turnover.rolling(window=20, min_periods=20).mean()
    raw = ma20 - 1.1
    return calc_rolling_zscore(raw, window=504)


def _calc_v077(wind_all_a_turnover: pd.Series) -> pd.Series:
    deviation = wind_all_a_turnover - wind_all_a_turnover.rolling(window=15, min_periods=15).mean()
    return calc_rolling_zscore(deviation * -1.0, window=504)


def _calc_v078(wind_all_a_turnover: pd.Series) -> pd.Series:
    ma21 = wind_all_a_turnover.rolling(window=21, min_periods=21).mean()
    quantile_rank = _rolling_quantile_rank_year(ma21, year=2)
    dev_high = calc_rolling_zscore(quantile_rank - 0.6, window=504)
    dev_low = calc_rolling_zscore(0.4 - quantile_rank, window=504)

    factor = pd.Series(0.0, index=quantile_rank.index, dtype="float64")
    high_mask = quantile_rank > 0.6
    low_mask = quantile_rank < 0.4
    factor.loc[high_mask] = dev_high.loc[high_mask]
    factor.loc[low_mask] = -dev_low.loc[low_mask]
    factor.loc[quantile_rank.isna()] = np.nan
    return factor


def _calc_v079() -> pd.Series:
    csi_all_turnover = _load_index_eod_series(CSI_ALL_INDEX, "换手率", "csi_all_turnover_index_eod").ffill()
    short_ma = csi_all_turnover.rolling(5, min_periods=5).mean()
    long_ma = csi_all_turnover.rolling(250, min_periods=250).mean()
    raw = short_ma - long_ma
    return calc_rolling_zscore(raw, window=504)


def _calc_v080() -> pd.Series:
    close_50 = _load_index_eod_series(SSE50_INDEX, "收盘指数", "sse50_close").ffill()
    ret = close_50.pct_change(fill_method=None)
    var_20 = ret.rolling(20, min_periods=20).var()
    raw = _safe_ratio(var_20, var_20.shift(20)) - 1.0
    return calc_rolling_zscore(raw, window=504) * -1.0


def generate_plan14_priceFactors8_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan14_records()
    wind_all_a_turnover = _load_index_excel_field(WIND_ALL_A_FILE, "换手率", "wind_all_a_turnover")

    _register_factor(raw_factor_df, factor_source_df, "V076_raw", _calc_v076())
    _register_factor(raw_factor_df, factor_source_df, "V077_raw", _calc_v077(wind_all_a_turnover))
    _register_factor(raw_factor_df, factor_source_df, "V078_raw", _calc_v078(wind_all_a_turnover))
    _register_factor(raw_factor_df, factor_source_df, "V079_raw", _calc_v079())
    _register_factor(raw_factor_df, factor_source_df, "V080_raw", _calc_v080())

    missing_cols = [factor_id for factor_id in PLAN14_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors8 plan 14 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN14_FACTOR_IDS], records


def generate_plan15_priceFactors8_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan15_records()
    wind_all_a_turnover = _load_index_excel_field(WIND_ALL_A_FILE, "换手率", "wind_all_a_turnover")

    _register_factor(raw_factor_df, factor_source_df, "V071_raw", _calc_v071())
    _register_factor(raw_factor_df, factor_source_df, "V072_raw", _calc_v072(wind_all_a_turnover))
    _register_factor(raw_factor_df, factor_source_df, "V073_raw", _calc_v073(wind_all_a_turnover))
    _register_factor(raw_factor_df, factor_source_df, "V074_raw", _calc_v074(wind_all_a_turnover))
    _register_factor(raw_factor_df, factor_source_df, "V075_raw", _calc_v075())

    missing_cols = [factor_id for factor_id in PLAN15_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors8 plan 15 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN15_FACTOR_IDS], records


def generate_extended_priceFactors8_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan15_factor_df, plan15_records = generate_plan15_priceFactors8_factors(data_df)
    plan14_factor_df, plan14_records = generate_plan14_priceFactors8_factors(data_df)
    plan13_factor_df, plan13_records = generate_priceFactors8_factors(data_df)
    factor_source_df = pd.concat([plan15_factor_df, plan14_factor_df, plan13_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan15_records + plan14_records + plan13_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors8_factors(data_df)
    metadata = metadata_from_extended_priceFactors8_records(selected_records)
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
