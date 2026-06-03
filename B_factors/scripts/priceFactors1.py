"""Price and momentum style-rotation factors from working_multiple_factors_plan.json."""

from __future__ import annotations

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
    calc_llt,
    calc_rolling_zscore,
    load_prepared_table,
    normalize_trade_dt,
    prepared_data_dir,
)
from factor_metadata import (
    append_record_note as _append_note,
    build_metadata_from_records,
    load_plan_records,
    normalize_plan_text as _normalize_plan_text,
)
from factor_pipeline_runner import run_factor_module_pipeline
from factor_transforms import as_float_series as _as_float_series


OUTPUT_PREFIX = "priceFactors1"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

LEGACY_FACTOR_IDS = [
    "V019",
    "V020",
    "V022",
    "V228",
    "V229",
    "V025",
    "V026",
    "V027",
    "V039",
    "V041",
    "V043",
    "V051",
    "V059",
    "V060",
    "V061",
    "V062",
    "V064",
]

PLAN_FACTOR_IDS = [
    "V021",
    "V035",
    "V037",
    "V038",
    "V044",
    "V052",
    "V053",
    "V054",
    "V055",
    "V057",
    "V058",
    "V065",
]

FACTOR_IDS = LEGACY_FACTOR_IDS + PLAN_FACTOR_IDS

LEGACY_SIGNAL_TYPES = {
    "V041": "event",
}

UNIMPLEMENTED_FACTORS = {
    "V028": "计划记录为 unknown，且上游指数价格口径不明确。",
    "V029": "计划记录为 unknown，且下游指数价格口径不明确。",
    "V030": "计划记录为 todo，消费指数口径和独立信号阈值不明确。",
    "V031": "计划记录为 todo，非消费指数口径和独立信号阈值不明确。",
    "V032": "计划记录为 todo，消费指数为海通自定义且单因子方向未知。",
    "V033": "计划记录为 todo，非消费指数为海通自定义且单因子方向未知。",
    "V034": "行业 ETF 动量未明确对应成长/价值方向。",
    "V036": "本地 mktP.parquet 缺少 open/high/low，无法计算日内动量。",
    "V040": "需自行构建市值因子多空净值，未使用代理口径。",
    "V042": "计划记录为 unknown，且行业动量方向不对应固定成长/价值。",
    "V056": "计划记录为 unknown，需自建消费/非消费指数。",
    "V063": "需构建微盘和大市值组合，未使用代理口径。",
}

HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"
WIND_ALL_A_FILE = "8841388.WI_windallA.xlsx"
GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
INDEX_EOD_TABLE = "index_eod.parquet"
A_INDEX_EOD_TABLE = "AIndexEODPrices.parquet"
INDEX_WEIGHT_TABLE = "AIndexHS300FreeWeight.parquet"
MKT_PRICE_TABLE = "mktP.parquet"

GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"
HS300_INDEX = "000300.SH"
SSE_INDEX = "000001.SH"
CSI1000_INDEX = "000852.SH"
CHINEXT_INDEX = "399006.SZ"

_A_INDEX_CLOSE_CACHE: dict[str, pd.Series] = {}
_INDEX_EOD_CACHE: pd.DataFrame | None = None
_STYLE_COMPONENT_CACHE: dict[str, dict[pd.Timestamp, list[str]]] | None = None


def _load_plan_records() -> list[dict[str, object]]:
    return load_plan_records(
        plan_path=PLAN_PATH,
        project_root=PROJECT_ROOT,
        factor_ids=PLAN_FACTOR_IDS,
        record_adjuster=_record_with_actual_fields,
    )


def _record_with_actual_fields(record: dict[str, object]) -> dict[str, object]:
    factor_id = str(record["factor_id"])
    record["signal_type"] = _normalize_plan_text(record.get("signal_type")) or "state"
    if factor_id == "V044":
        _append_note(record, "condition 仅给出成长/价值20日收益，本脚本按头部风格20日收益转负构造反转信号。")
    elif factor_id in {"V057", "V058"}:
        _append_note(record, "本脚本按 condition 使用本地 growth_index.xlsx/value_index.xlsx，不额外调整方向。")
    elif factor_id == "V065":
        _append_note(record, "本脚本按 condition 使用国证成长399370的MACD计算5日均值减20日均值。")
    return record


def metadata_from_priceFactors1_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    metadata = {
        factor_id: {
            "signal_type": LEGACY_SIGNAL_TYPES.get(factor_id, "state"),
            "bar": 0.0,
            "factor": None,
            "progress": None,
        }
        for factor_id in LEGACY_FACTOR_IDS
    }
    metadata.update(build_metadata_from_records(records))
    return metadata


def _index_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].replace(".0", "").zfill(6)


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].zfill(6)


def _load_excel_close(file_name: str, date_col: str, close_col: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if date_col not in df.columns or close_col not in df.columns:
        raise KeyError(f"{file_name} must contain {date_col!r} and {close_col!r}; available={list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    close = _as_numeric(df[close_col])
    return _as_float_series(close, dates, name)


def _load_growth_close() -> pd.Series:
    return _load_excel_close(GROWTH_INDEX_FILE, "date", "close", "growth_index")


def _load_value_close() -> pd.Series:
    return _load_excel_close(VALUE_INDEX_FILE, "date", "close", "value_index")


def _load_hs300_close() -> pd.Series:
    return _load_excel_close(HS300_PRICE_FILE, "交易日期", "收盘价", "HS300")


def _load_wind_all_a_close() -> pd.Series:
    return _load_excel_close(WIND_ALL_A_FILE, "交易日期", "收盘价", "WindAllA")


def _load_a_index_close(index_code: str) -> pd.Series:
    if index_code not in _A_INDEX_CLOSE_CACHE:
        df = pd.read_parquet(
            prepared_data_dir / A_INDEX_EOD_TABLE,
            columns=["S_INFO_WINDCODE", "TRADE_DT", "S_DQ_CLOSE"],
            filters=[("S_INFO_WINDCODE", "=", index_code)],
        )
        if df.empty:
            raise ValueError(f"{A_INDEX_EOD_TABLE} missing index_code={index_code!r}")
        dates = normalize_trade_dt(df["TRADE_DT"])
        _A_INDEX_CLOSE_CACHE[index_code] = _as_float_series(df["S_DQ_CLOSE"], dates, index_code)
    return _A_INDEX_CLOSE_CACHE[index_code].copy()


def _load_index_eod() -> pd.DataFrame:
    global _INDEX_EOD_CACHE
    if _INDEX_EOD_CACHE is None:
        df = load_prepared_table(INDEX_EOD_TABLE)
        if "交易所指数代码" not in df.columns:
            raise KeyError(f"{INDEX_EOD_TABLE} missing 交易所指数代码")
        dates = pd.to_datetime(df.index, errors="coerce").normalize()
        out = df.reset_index(drop=True).copy()
        out["date"] = dates
        out["index_code"] = out["交易所指数代码"].map(_index_code_key)
        _INDEX_EOD_CACHE = out[out["date"].notna()].sort_values(["index_code", "date"])
    return _INDEX_EOD_CACHE.copy()


def _load_index_eod_series(index_code: str, value_col: str, name: str) -> pd.Series:
    df = _load_index_eod()
    if value_col not in df.columns:
        raise KeyError(f"{INDEX_EOD_TABLE} missing {value_col!r}")
    out = df[df["index_code"].eq(_index_code_key(index_code))].copy()
    if out.empty:
        raise ValueError(f"{INDEX_EOD_TABLE} missing index_code={index_code!r}")
    return _as_float_series(out[value_col], out["date"], name)


def _rolling_return(close: pd.Series, window: int) -> pd.Series:
    return close.astype("float64").sort_index().pct_change(window, fill_method=None)


def _empirical_cdf_to_normal(series: pd.Series, min_periods: int = 30) -> pd.Series:
    s = series.astype("float64").sort_index()

    def rank_last(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) < min_periods:
            return np.nan
        return float((valid <= valid[-1]).sum() / len(valid))

    probability = s.expanding(min_periods=min_periods).apply(rank_last, raw=True).clip(0.001, 0.999)
    return pd.Series(norm.ppf(probability), index=s.index, dtype="float64")


def _ewm_volatility(
    returns: pd.Series,
    halflife: int = 60,
    window: int = 120,
    annualize: bool = True,
) -> pd.Series:
    ret = returns.astype("float64").sort_index()
    vol = ret.ewm(halflife=halflife, min_periods=window, adjust=False).std()
    valid_count = ret.rolling(window, min_periods=window).count()
    vol = vol.where(valid_count >= window)
    return vol * np.sqrt(252) if annualize else vol


def _load_style_components() -> dict[str, dict[pd.Timestamp, list[str]]]:
    global _STYLE_COMPONENT_CACHE
    if _STYLE_COMPONENT_CACHE is not None:
        return _STYLE_COMPONENT_CACHE

    out: dict[str, dict[pd.Timestamp, list[str]]] = {}
    for index_code in [GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX]:
        comp = pd.read_parquet(
            prepared_data_dir / INDEX_WEIGHT_TABLE,
            columns=["S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT"],
            filters=[("S_INFO_WINDCODE", "=", index_code)],
        )
        if comp.empty:
            raise ValueError(f"{INDEX_WEIGHT_TABLE} missing components for {index_code}")
        comp["TRADE_DT"] = normalize_trade_dt(comp["TRADE_DT"])
        comp["stock_code"] = comp["S_CON_WINDCODE"].map(_stock_code_key)
        comp = comp[comp["TRADE_DT"].notna() & comp["stock_code"].notna()].copy()
        out[index_code] = {
            pd.Timestamp(dt): sorted(group["stock_code"].dropna().unique().tolist())
            for dt, group in comp.groupby("TRADE_DT", sort=True)
        }
    _STYLE_COMPONENT_CACHE = out
    return out


def _style_component_amount_ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    components_by_index: dict[str, dict[pd.Timestamp, list[str]]],
    index_code: str,
) -> pd.Series:
    components = components_by_index[index_code]
    component_dates = pd.DatetimeIndex(sorted(components.keys()))
    ratio = pd.Series(np.nan, index=numerator.index, dtype="float64")
    for dt in numerator.index:
        loc = component_dates.searchsorted(dt, side="right") - 1
        if loc < 0:
            continue
        tickers = [ticker for ticker in components[component_dates[loc]] if ticker in numerator.columns]
        if tickers:
            total = denominator.loc[dt, tickers].sum(skipna=True)
            if total > 0:
                ratio.at[dt] = numerator.loc[dt, tickers].sum(skipna=True) / total
    return ratio


def _calc_v051() -> pd.Series:
    components = _load_style_components()
    tickers = sorted(
        {
            ticker
            for by_date in components.values()
            for component_tickers in by_date.values()
            for ticker in component_tickers
        }
    )
    ticker_set = set(tickers)

    mkt = load_prepared_table(MKT_PRICE_TABLE)
    required = ["Stkcd", "Trddt", "Dnvaltrd", "Adjprcwd"]
    missing = [col for col in required if col not in mkt.columns]
    if missing:
        raise KeyError(f"{MKT_PRICE_TABLE} missing columns: {missing}")
    mkt = mkt.loc[:, required].copy()
    mkt["stock_code"] = mkt["Stkcd"].map(_stock_code_key)
    mkt = mkt[mkt["stock_code"].isin(ticker_set)].copy()
    mkt["date"] = pd.to_datetime(mkt["Trddt"], errors="coerce").dt.normalize()
    mkt["close"] = pd.to_numeric(mkt["Adjprcwd"], errors="coerce")
    mkt["amount"] = pd.to_numeric(mkt["Dnvaltrd"], errors="coerce")
    mkt = mkt[mkt["date"].notna() & mkt["stock_code"].notna()].copy()

    close = mkt.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last").sort_index()
    amount = mkt.pivot_table(index="date", columns="stock_code", values="amount", aggfunc="sum").sort_index()
    close = close.reindex(columns=tickers)
    amount = amount.reindex(close.index).reindex(columns=tickers)

    is_high = close >= close.rolling(63, min_periods=63).max()
    is_low = close <= close.rolling(63, min_periods=63).min()
    valid_amount = amount.where(amount > 0)
    high_amount = valid_amount.where(is_high, 0.0)
    low_amount = valid_amount.where(is_low, 0.0)

    growth_factor = _style_component_amount_ratio(
        high_amount,
        valid_amount,
        components,
        GROWTH_STYLE_INDEX,
    ) - _style_component_amount_ratio(
        low_amount,
        valid_amount,
        components,
        GROWTH_STYLE_INDEX,
    )
    value_factor = _style_component_amount_ratio(
        high_amount,
        valid_amount,
        components,
        VALUE_STYLE_INDEX,
    ) - _style_component_amount_ratio(
        low_amount,
        valid_amount,
        components,
        VALUE_STYLE_INDEX,
    )
    diff = growth_factor - value_factor
    return diff.rolling(5, min_periods=5).mean() - diff.rolling(250, min_periods=250).mean()


def _calc_v064() -> pd.Series:
    macd = _load_index_eod_series(GROWTH_STYLE_INDEX, "MACD", "growth_macd")
    return macd - macd.rolling(3, min_periods=3).mean()


def _calc_v065() -> pd.Series:
    macd = _load_index_eod_series(GROWTH_STYLE_INDEX, "MACD", "growth_macd")
    return macd.rolling(5, min_periods=5).mean() - macd.rolling(20, min_periods=20).mean()


def generate_priceFactors1_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    hs300 = _load_hs300_close()
    growth = _load_growth_close()
    value = _load_value_close()
    growth_ret = growth.pct_change(fill_method=None)
    value_ret = value.pct_change(fill_method=None)

    hs300_ret = hs300.pct_change(fill_method=None)
    vol_30 = hs300_ret.rolling(30, min_periods=30).std()
    v019 = (vol_30 - vol_30.expanding(min_periods=30).median()) * -1.0
    _register_factor(raw_factor_df, factor_source_df, "V019_raw", v019)

    vol_diff_60 = value_ret.rolling(60, min_periods=60).std() - growth_ret.rolling(60, min_periods=60).std()
    _register_factor(raw_factor_df, factor_source_df, "V020_raw", calc_rolling_zscore(vol_diff_60, 180) * -1.0)

    vol_growth_180 = growth_ret.rolling(180, min_periods=180).std()
    vol_value_180 = value_ret.rolling(180, min_periods=180).std()
    _register_factor(raw_factor_df, factor_source_df, "V022_raw", (vol_growth_180 - vol_value_180) * -1.0)

    vol_3m = hs300_ret.rolling(63, min_periods=63).std()
    _register_factor(raw_factor_df, factor_source_df, "V228_raw", _empirical_cdf_to_normal(vol_3m) * -1.0)

    wind_all_a_ret = _load_wind_all_a_close().pct_change(fill_method=None)
    vol_ewm = _ewm_volatility(wind_all_a_ret, halflife=60, window=120, annualize=True)
    _register_factor(raw_factor_df, factor_source_df, "V229_raw", (vol_ewm - 0.20) * -1.0)

    vol_growth_63 = growth_ret.rolling(63, min_periods=63).std()
    vol_value_63 = value_ret.rolling(63, min_periods=63).std()
    _register_factor(raw_factor_df, factor_source_df, "V025_raw", (vol_growth_63 - vol_value_63) * -1.0)

    _register_factor(raw_factor_df, factor_source_df, "V026_raw", _rolling_return(_load_a_index_close(SSE_INDEX), 60))

    value_growth_ret_diff_255 = _rolling_return(value, 255) - _rolling_return(growth, 255)
    _register_factor(raw_factor_df, factor_source_df, "V027_raw", calc_rolling_zscore(value_growth_ret_diff_255, 255) * -1.0)

    rel_price = growth / value
    _register_factor(raw_factor_df, factor_source_df, "V039_raw", rel_price.pct_change(1, fill_method=None))

    rel_nav = (1.0 + growth_ret).cumprod() / (1.0 + value_ret).cumprod()
    _register_factor(raw_factor_df, factor_source_df, "V041_raw", rel_nav - rel_nav.rolling(20, min_periods=20).mean())

    hs300_a = _load_a_index_close(HS300_INDEX)
    csi1000 = _load_a_index_close(CSI1000_INDEX)
    _register_factor(raw_factor_df, factor_source_df, "V043_raw", (_rolling_return(hs300_a, 63) - _rolling_return(csi1000, 63)) * -1.0)

    _register_factor(raw_factor_df, factor_source_df, "V051_raw", _calc_v051())

    growth_minus_value_40 = _rolling_return(growth, 40) - _rolling_return(value, 40)
    _register_factor(raw_factor_df, factor_source_df, "V059_raw", calc_rolling_zscore(growth_minus_value_40, 180))

    chinext = _load_a_index_close(CHINEXT_INDEX)
    ret_dev = hs300_a.pct_change(fill_method=None) - chinext.pct_change(fill_method=None)
    _register_factor(raw_factor_df, factor_source_df, "V060_raw", _empirical_cdf_to_normal(ret_dev) * -1.0)

    _register_factor(raw_factor_df, factor_source_df, "V061_raw", (_rolling_return(hs300_a, 255) - _rolling_return(csi1000, 255)) * -1.0)

    gz_growth = _load_index_eod_series(GROWTH_STYLE_INDEX, "收盘指数", "gz_growth")
    gz_value = _load_index_eod_series(VALUE_STYLE_INDEX, "收盘指数", "gz_value")
    _register_factor(raw_factor_df, factor_source_df, "V062_raw", calc_rolling_zscore(gz_growth / gz_value, 6))

    _register_factor(raw_factor_df, factor_source_df, "V064_raw", _calc_v064())

    vol_growth_60 = growth_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    vol_value_60 = value_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    vol_ratio = vol_growth_60 / vol_value_60.replace(0.0, np.nan)
    _register_factor(raw_factor_df, factor_source_df, "V021_raw", (vol_ratio.rolling(20, min_periods=20).mean() - 1.0) * -1.0)

    ret_growth_20 = _rolling_return(growth, 20)
    ret_value_20 = _rolling_return(value, 20)
    relative_strength = ret_growth_20 - ret_value_20
    _register_factor(raw_factor_df, factor_source_df, "V035_raw", relative_strength - relative_strength.shift(5))

    _register_factor(raw_factor_df, factor_source_df, "V037_raw", _rolling_return(hs300, 63))

    value_growth_rsi = (1.0 + value_ret).cumprod() / (1.0 + growth_ret).cumprod()
    rsi_upper = value_growth_rsi.rolling(60, min_periods=60).max()
    rsi_lower = value_growth_rsi.rolling(60, min_periods=60).min()
    rsi_half = (rsi_upper - rsi_lower) / 2.0
    rsi_mid = (rsi_upper + rsi_lower) / 2.0
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "V038_raw",
        ((value_growth_rsi - rsi_mid) / rsi_half.replace(0.0, np.nan)) * -1.0,
    )

    v044 = pd.Series(0.0, index=ret_growth_20.index, dtype="float64")
    v044.loc[(ret_value_20 > ret_growth_20) & (ret_value_20 < 0)] = 1.0
    v044.loc[(ret_growth_20 > ret_value_20) & (ret_growth_20 < 0)] = -1.0
    v044.loc[ret_growth_20.isna() | ret_value_20.isna()] = np.nan
    _register_factor(raw_factor_df, factor_source_df, "V044_raw", v044)

    rel_nv = growth / value
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "V052_raw",
        rel_nv.rolling(20, min_periods=20).mean() - rel_nv.rolling(180, min_periods=180).mean(),
    )

    ret_diff_60 = _rolling_return(value, 60) - _rolling_return(growth, 60)
    _register_factor(raw_factor_df, factor_source_df, "V053_raw", calc_rolling_zscore(ret_diff_60, 180) * -1.0)

    lagged_value_growth_ret_diff = (value_ret - growth_ret).shift(1) * -1.0
    _register_factor(raw_factor_df, factor_source_df, "V054_raw", lagged_value_growth_ret_diff)
    _register_factor(raw_factor_df, factor_source_df, "V055_raw", lagged_value_growth_ret_diff)

    ret_diff_20 = _rolling_return(value, 20) - _rolling_return(growth, 20)
    _register_factor(raw_factor_df, factor_source_df, "V057_raw", calc_rolling_zscore(ret_diff_20, 180))

    ret_diff_40 = _rolling_return(value, 40) - _rolling_return(growth, 40)
    _register_factor(raw_factor_df, factor_source_df, "V058_raw", calc_rolling_zscore(ret_diff_40, 180))

    _register_factor(raw_factor_df, factor_source_df, "V065_raw", _calc_v065())

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"price factor columns missing after generation: {missing_cols}")

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
    # 完整挂载、信号生成和保存流程交给公共 runner；本模块只保留价格因子公式与 metadata 口径。
    run_factor_module_pipeline(
        output_prefix=OUTPUT_PREFIX,
        generate_factors=generate_priceFactors1_factors,
        metadata_builder=metadata_from_priceFactors1_records,
        print_summary=_print_factor_output_summary,
    )


if __name__ == "__main__":
    main()
