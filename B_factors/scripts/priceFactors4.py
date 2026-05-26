"""Price style factors from completed V071-V142 plan file 5."""

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


OUTPUT_PREFIX = "priceFactors4"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 5.json"

FACTOR_IDS = [
    "V121",
    "V122",
    "V123",
    "V124",
    "V125",
]

HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"
CSI_ALL_PRICE_FILE = "中证全指(000985.CSI)-历史价格.xlsx"
GROWTH_INDEX_FILE = "growth_index.xlsx"
INDEX_EOD_TABLE = "index_eod.parquet"
MKT_PRICE_TABLE = "mktP.parquet"

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
            if factor_id == "V121":
                _append_note(item, "本脚本按计划代理口径使用中证500与沪深300的20日收益差。")
            elif factor_id == "V122":
                _append_note(
                    item,
                    "本脚本使用 mktP.parquet 的月度个股收益截面偏度、峰度和等权市场收益12个月波动率代理综合拥挤度。",
                )
            elif factor_id == "V124":
                _append_note(item, "本脚本按计划代理口径使用沪深300与中证全指成交额之和。")
            elif factor_id == "V125":
                _append_note(item, "本脚本使用 mktP.parquet 的 Dretwd 计算月度截面 DI。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 5 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors4_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _load_growth_ohlc() -> pd.DataFrame:
    df = load_prepared_table(GROWTH_INDEX_FILE)
    required_cols = ["date", "open", "high", "low"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{GROWTH_INDEX_FILE} missing columns: {missing}")
    out = df.loc[:, required_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low"]:
        out[col] = _as_numeric(out[col])
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").set_index("date")
    return out.astype("float64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _load_monthly_market_return_stats() -> pd.DataFrame:
    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / MKT_PRICE_TABLE,
        columns=["Stkcd", "Trddt", "Dretwd"],
    )
    required_cols = {"Stkcd", "Trddt", "Dretwd"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"{MKT_PRICE_TABLE} must contain {sorted(required_cols)}; available={list(df.columns)}")

    work = df.dropna(subset=["Stkcd", "Trddt"]).copy()
    work["Trddt"] = pd.to_datetime(work["Trddt"], errors="coerce")
    work["Dretwd"] = pd.to_numeric(work["Dretwd"], errors="coerce")
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

    by_month = monthly_ret.groupby(level="month", sort=True)
    stats = pd.DataFrame(
        {
            "di": by_month.apply(trimmed_std),
            "skew": by_month.skew(),
            "kurt": by_month.apply(lambda values: values.dropna().kurt() if values.dropna().size > 3 else np.nan),
            "equal_ret": by_month.mean(),
        }
    ).astype("float64")
    stats.index = pd.PeriodIndex(stats.index, freq="M").to_timestamp("M")
    return stats.sort_index()


def _calc_v121() -> pd.Series:
    csi500_close = _load_index_eod_series(CSI500_INDEX, "收盘指数", "csi500_close")
    hs300_close = _read_excel_series(HS300_PRICE_FILE, "交易日期", "收盘价", "hs300_close")
    raw = csi500_close.pct_change(20, fill_method=None) - hs300_close.pct_change(20, fill_method=None)
    return calc_rolling_zscore(raw, window=504)


def _calc_v122(stats: pd.DataFrame) -> pd.Series:
    sub1_z = calc_rolling_zscore(stats["skew"], window=120)
    sub2_z = calc_rolling_zscore(stats["kurt"], window=120)
    sub3 = stats["equal_ret"].rolling(12, min_periods=12).std()
    sub3_z = calc_rolling_zscore(sub3, window=120)
    raw = (sub1_z + sub2_z + sub3_z) / 3.0
    return raw * -1.0


def _calc_v123() -> pd.Series:
    growth = _load_growth_ohlc()
    ar = _safe_ratio(
        (growth["high"] - growth["open"]).rolling(26, min_periods=26).sum(),
        (growth["open"] - growth["low"]).rolling(26, min_periods=26).sum(),
    ) * 100.0
    raw = ar - ar.rolling(60, min_periods=60).mean()
    return calc_rolling_zscore(raw, window=504)


def _calc_v124() -> pd.Series:
    hs300_amount = _read_excel_series(HS300_PRICE_FILE, "交易日期", "成交额(元,CNY)", "hs300_amount")
    csi_all_amount = _read_excel_series(CSI_ALL_PRICE_FILE, "交易日期", "成交额(元,CNY)", "csi_all_amount")
    total_amount = hs300_amount.add(csi_all_amount, fill_value=np.nan)
    raw = (total_amount - total_amount.rolling(23, min_periods=23).mean()) * -1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v125(stats: pd.DataFrame) -> pd.Series:
    raw = stats["di"] - stats["di"].rolling(12, min_periods=12).mean()
    return calc_rolling_zscore(raw, window=120)


def generate_priceFactors4_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    monthly_stats = _load_monthly_market_return_stats()

    _register_factor(raw_factor_df, factor_source_df, "V121_raw", _calc_v121())
    _register_factor(raw_factor_df, factor_source_df, "V122_raw", _calc_v122(monthly_stats))
    _register_factor(raw_factor_df, factor_source_df, "V123_raw", _calc_v123())
    _register_factor(raw_factor_df, factor_source_df, "V124_raw", _calc_v124())
    _register_factor(raw_factor_df, factor_source_df, "V125_raw", _calc_v125(monthly_stats))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors4 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors4_factors(data_df)
    metadata = metadata_from_priceFactors4_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V071_V142 6.json.
PLAN6_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 6.json"
PLAN6_FACTOR_IDS = [
    "V116",
    "V117",
    "V118",
    "V119",
    "V120",
]
EXTENDED_FACTOR_IDS = PLAN6_FACTOR_IDS + FACTOR_IDS

ADJ_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"
INDEX_WEIGHT_TABLE = "AIndexHS300FreeWeight.parquet"
GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"

_STYLE_WEIGHT_CACHE: dict[str, dict[pd.Timestamp, pd.Series]] | None = None


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].zfill(6)


def _stock_windcode_from_stkcd(value: object) -> str:
    code = _stock_code_key(value)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


def _load_plan6_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN6_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN6_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN6_FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id in {"V116", "V117", "V118"}:
                _append_note(item, "本脚本使用 S_DQ_ADJCLOSE.parquet 宽表复权收盘价计算个股收益率。")
            if factor_id in {"V119", "V120"}:
                _append_note(item, "本脚本沿计划记录将 mktP.parquet 的 Dnshrtrd 作为换手率代理，并按成分权重合成风格侧指标。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN6_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 6 missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN6_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors4_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _load_style_weights() -> dict[str, dict[pd.Timestamp, pd.Series]]:
    global _STYLE_WEIGHT_CACHE
    if _STYLE_WEIGHT_CACHE is not None:
        return _STYLE_WEIGHT_CACHE

    frames: list[pd.DataFrame] = []
    for index_code in [GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX]:
        df = pd.read_parquet(
            PROJECT_ROOT / "A_data" / "prepared_data" / INDEX_WEIGHT_TABLE,
            columns=["S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT", "I_WEIGHT"],
            filters=[("S_INFO_WINDCODE", "=", index_code)],
        )
        if df.empty:
            raise ValueError(f"{INDEX_WEIGHT_TABLE} missing components for {index_code}")
        frames.append(df)

    comp = pd.concat(frames, ignore_index=True)
    comp["TRADE_DT"] = pd.to_datetime(comp["TRADE_DT"].astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    comp["TRADE_DT"] = comp["TRADE_DT"].dt.normalize()
    comp["S_CON_WINDCODE"] = comp["S_CON_WINDCODE"].astype(str).str.upper().str.strip()
    comp["I_WEIGHT"] = pd.to_numeric(comp["I_WEIGHT"], errors="coerce")
    comp = comp.dropna(subset=["TRADE_DT", "S_CON_WINDCODE", "I_WEIGHT"]).copy()

    out: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    for index_code, index_group in comp.groupby("S_INFO_WINDCODE", sort=False):
        by_date: dict[pd.Timestamp, pd.Series] = {}
        for dt, date_group in index_group.groupby("TRADE_DT", sort=True):
            weights = date_group.groupby("S_CON_WINDCODE")["I_WEIGHT"].sum()
            weights = weights[weights > 0].astype("float64")
            if not weights.empty:
                by_date[pd.Timestamp(dt)] = weights / weights.sum()
        out[index_code] = by_date

    _STYLE_WEIGHT_CACHE = out
    return out


def _all_style_tickers() -> list[str]:
    weights = _load_style_weights()
    return sorted(
        {
            ticker
            for by_date in weights.values()
            for series in by_date.values()
            for ticker in series.index
        }
    )


def _style_composite(
    metric_df: pd.DataFrame,
    index_code: str,
    weighted: bool = False,
) -> pd.Series:
    by_date = _load_style_weights()[index_code]
    component_dates = pd.DatetimeIndex(sorted(by_date.keys()))
    out = pd.Series(np.nan, index=metric_df.index, dtype="float64")

    for pos, comp_date in enumerate(component_dates):
        next_date = component_dates[pos + 1] if pos + 1 < len(component_dates) else pd.Timestamp.max
        mask = (metric_df.index >= comp_date) & (metric_df.index < next_date)
        if not mask.any():
            continue
        weights = by_date[pd.Timestamp(comp_date)]
        tickers = [ticker for ticker in weights.index if ticker in metric_df.columns]
        if not tickers:
            continue
        block = metric_df.loc[mask, tickers]
        if weighted:
            aligned_weights = weights.reindex(tickers).astype("float64")
            weighted_sum = block.mul(aligned_weights, axis=1).sum(axis=1, min_count=1)
            valid_weight = block.notna().mul(aligned_weights, axis=1).sum(axis=1)
            out.loc[mask] = weighted_sum / valid_weight.replace(0.0, np.nan)
        else:
            out.loc[mask] = block.mean(axis=1, skipna=True)

    return out


def _load_style_return_wide() -> pd.DataFrame:
    tickers = _all_style_tickers()
    columns = ["TRADE_DT"] + tickers
    close = pd.read_parquet(PROJECT_ROOT / "A_data" / "prepared_data" / ADJ_CLOSE_TABLE, columns=columns)
    close["TRADE_DT"] = pd.to_datetime(close["TRADE_DT"], errors="coerce").dt.normalize()
    close = close.dropna(subset=["TRADE_DT"]).sort_values("TRADE_DT")
    close = close.drop_duplicates(subset=["TRADE_DT"], keep="last").set_index("TRADE_DT")
    close = close.apply(pd.to_numeric, errors="coerce")
    return close.pct_change(fill_method=None)


def _load_style_turnover_and_return_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = _all_style_tickers()
    ticker_codes = {_stock_code_key(ticker) for ticker in tickers}
    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / MKT_PRICE_TABLE,
        columns=["Stkcd", "Trddt", "Dnshrtrd", "Dretwd"],
    )
    work = df.dropna(subset=["Stkcd", "Trddt"]).copy()
    work["ticker"] = work["Stkcd"].map(_stock_windcode_from_stkcd)
    work = work[work["Stkcd"].map(_stock_code_key).isin(ticker_codes)].copy()
    work["date"] = pd.to_datetime(work["Trddt"], errors="coerce").dt.normalize()
    work["turnover"] = pd.to_numeric(work["Dnshrtrd"], errors="coerce")
    work["ret"] = pd.to_numeric(work["Dretwd"], errors="coerce")
    work = work.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])

    turnover = work.pivot_table(index="date", columns="ticker", values="turnover", aggfunc="last").sort_index()
    returns = work.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").sort_index()
    turnover = turnover.reindex(columns=tickers)
    returns = returns.reindex(index=turnover.index, columns=tickers)
    return turnover, returns


def _semivol_down(series: pd.Series, window: int = 60) -> pd.Series:
    def calc(values: np.ndarray) -> float:
        neg = values[values < 0]
        return float(np.std(neg, ddof=1)) if len(neg) > 1 else np.nan

    return series.rolling(window, min_periods=window).apply(calc, raw=True)


def _calc_v116(ret_wide: pd.DataFrame) -> pd.Series:
    semivol_dn = ret_wide.apply(_semivol_down, window=60)
    growth = _style_composite(semivol_dn, GROWTH_STYLE_INDEX, weighted=False)
    value = _style_composite(semivol_dn, VALUE_STYLE_INDEX, weighted=False)
    raw = _safe_ratio(growth, value) - 1.0
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v117(ret_wide: pd.DataFrame) -> pd.Series:
    pvolt = ret_wide.rolling(20, min_periods=20).std().rolling(60, min_periods=60).mean()
    growth = _style_composite(pvolt, GROWTH_STYLE_INDEX, weighted=False)
    value = _style_composite(pvolt, VALUE_STYLE_INDEX, weighted=False)
    return calc_rolling_zscore(_safe_ratio(growth, value) - 1.0, window=504)


def _calc_v118(ret_wide: pd.DataFrame) -> pd.Series:
    growth = _style_composite(ret_wide, GROWTH_STYLE_INDEX, weighted=False)
    value = _style_composite(ret_wide, VALUE_STYLE_INDEX, weighted=False)
    growth_cs = pd.Series(np.nan, index=ret_wide.index, dtype="float64")
    value_cs = pd.Series(np.nan, index=ret_wide.index, dtype="float64")

    weights = _load_style_weights()
    for index_code, target in [(GROWTH_STYLE_INDEX, growth_cs), (VALUE_STYLE_INDEX, value_cs)]:
        component_dates = pd.DatetimeIndex(sorted(weights[index_code].keys()))
        for pos, comp_date in enumerate(component_dates):
            next_date = component_dates[pos + 1] if pos + 1 < len(component_dates) else pd.Timestamp.max
            mask = (ret_wide.index >= comp_date) & (ret_wide.index < next_date)
            if not mask.any():
                continue
            tickers = [ticker for ticker in weights[index_code][pd.Timestamp(comp_date)].index if ticker in ret_wide.columns]
            if tickers:
                target.loc[mask] = ret_wide.loc[mask, tickers].std(axis=1, skipna=True)

    _ = growth, value
    return calc_rolling_zscore(_safe_ratio(growth_cs, value_cs) - 1.0, window=504)


def _calc_v119(turnover: pd.DataFrame) -> pd.Series:
    tvr_120d = turnover.rolling(120, min_periods=120).mean()
    growth = _style_composite(tvr_120d, GROWTH_STYLE_INDEX, weighted=True)
    value = _style_composite(tvr_120d, VALUE_STYLE_INDEX, weighted=True)
    return calc_rolling_zscore(_safe_ratio(growth, value) - 1.0, window=504)


def _calc_v120(turnover: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    up_turnover = turnover.where(returns > 0)
    down_turnover = turnover.where(returns < 0)
    tvr_20d_dn = _safe_ratio(
        up_turnover.rolling(20, min_periods=1).sum(),
        down_turnover.rolling(20, min_periods=1).sum(),
    )
    growth = _style_composite(tvr_20d_dn, GROWTH_STYLE_INDEX, weighted=True)
    value = _style_composite(tvr_20d_dn, VALUE_STYLE_INDEX, weighted=True)
    return calc_rolling_zscore(_safe_ratio(growth, value) - 1.0, window=504)


def generate_plan6_priceFactors4_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan6_records()

    ret_wide = _load_style_return_wide()
    turnover, turnover_returns = _load_style_turnover_and_return_wide()

    _register_factor(raw_factor_df, factor_source_df, "V116_raw", _calc_v116(ret_wide))
    _register_factor(raw_factor_df, factor_source_df, "V117_raw", _calc_v117(ret_wide))
    _register_factor(raw_factor_df, factor_source_df, "V118_raw", _calc_v118(ret_wide))
    _register_factor(raw_factor_df, factor_source_df, "V119_raw", _calc_v119(turnover))
    _register_factor(raw_factor_df, factor_source_df, "V120_raw", _calc_v120(turnover, turnover_returns))

    missing_cols = [factor_id for factor_id in PLAN6_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors4 plan 6 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN6_FACTOR_IDS], records


def generate_extended_priceFactors4_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan6_factor_df, plan6_records = generate_plan6_priceFactors4_factors(data_df)
    plan5_factor_df, plan5_records = generate_priceFactors4_factors(data_df)
    factor_source_df = pd.concat([plan6_factor_df, plan5_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan6_records + plan5_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors4_factors(data_df)
    metadata = metadata_from_extended_priceFactors4_records(selected_records)
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
