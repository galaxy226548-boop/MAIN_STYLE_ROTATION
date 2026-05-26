"""Price style factors from completed V071-V142 plan file 7."""

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


OUTPUT_PREFIX = "priceFactors5"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 7.json"

FACTOR_IDS = [
    "V111",
    "V112",
    "V113",
    "V114",
    "V115",
]

GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
ASTOCK_DAILY_TABLE = "Astockdaily.parquet"
ADJ_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"
INDEX_WEIGHT_TABLE = "AIndexHS300FreeWeight.parquet"

GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"
SH50_INDEX = "000016.SH"

_COMPONENT_WEIGHT_CACHE: dict[str, dict[pd.Timestamp, pd.Series]] = {}


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
            if factor_id == "V111":
                _append_note(item, "本脚本使用 w1=40、w2=20 作为研报未明确回归窗口的中间参数。")
            elif factor_id == "V112":
                _append_note(
                    item,
                    "Astockdaily.parquet 的 Ret 字段为股息率说明，本脚本使用 ChangeRatio 作为个股日收益率代理。",
                )
            elif factor_id == "V113":
                _append_note(item, "本脚本按日历月历史成长超额均值构造事件强度，并对同月历史均值使用一期滞后避免使用当月未来收益。")
            elif factor_id == "V114":
                _append_note(
                    item,
                    "本地无个股高低价宽表，本脚本用当日/前日复权收盘价构造近似高低价后计算 RSRS R²。",
                )
            elif factor_id == "V115":
                _append_note(item, "本脚本使用 AIndexHS300FreeWeight.parquet 中的 000016.SH 成分和 S_DQ_ADJCLOSE.parquet 收盘价宽表。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 7 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors5_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _safe_ratio(numerator: pd.Series | pd.DataFrame, denominator: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return numerator / denominator.replace(0.0, np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype="float64")
    x = x - x.mean()
    denominator = float(np.square(x).sum())

    def calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values.astype("float64") - float(np.mean(values))
        return float(np.dot(x, y) / denominator)

    return series.astype("float64").rolling(window, min_periods=window).apply(calc, raw=True)


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].zfill(6)


def _load_component_weights(index_code: str) -> dict[pd.Timestamp, pd.Series]:
    if index_code in _COMPONENT_WEIGHT_CACHE:
        return _COMPONENT_WEIGHT_CACHE[index_code]

    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / INDEX_WEIGHT_TABLE,
        columns=["S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT", "I_WEIGHT"],
        filters=[("S_INFO_WINDCODE", "=", index_code)],
    )
    if df.empty:
        raise ValueError(f"{INDEX_WEIGHT_TABLE} missing components for {index_code}")

    work = df.copy()
    work["TRADE_DT"] = pd.to_datetime(work["TRADE_DT"].astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    work["TRADE_DT"] = work["TRADE_DT"].dt.normalize()
    work["S_CON_WINDCODE"] = work["S_CON_WINDCODE"].astype(str).str.upper().str.strip()
    work["I_WEIGHT"] = pd.to_numeric(work["I_WEIGHT"], errors="coerce")
    work = work.dropna(subset=["TRADE_DT", "S_CON_WINDCODE", "I_WEIGHT"]).copy()

    by_date: dict[pd.Timestamp, pd.Series] = {}
    for dt, date_group in work.groupby("TRADE_DT", sort=True):
        weights = date_group.groupby("S_CON_WINDCODE")["I_WEIGHT"].sum()
        weights = weights[weights > 0].astype("float64")
        if not weights.empty:
            by_date[pd.Timestamp(dt)] = weights / weights.sum()

    if not by_date:
        raise ValueError(f"{INDEX_WEIGHT_TABLE} has no valid component weights for {index_code}")
    _COMPONENT_WEIGHT_CACHE[index_code] = by_date
    return by_date


def _all_tickers_for_indices(index_codes: list[str]) -> list[str]:
    return sorted(
        {
            ticker
            for index_code in index_codes
            for weights in _load_component_weights(index_code).values()
            for ticker in weights.index
        }
    )


def _load_adj_close_wide(tickers: list[str]) -> pd.DataFrame:
    columns = ["TRADE_DT"] + [ticker for ticker in tickers]
    close = pd.read_parquet(PROJECT_ROOT / "A_data" / "prepared_data" / ADJ_CLOSE_TABLE, columns=columns)
    close["TRADE_DT"] = pd.to_datetime(close["TRADE_DT"], errors="coerce").dt.normalize()
    close = close.dropna(subset=["TRADE_DT"]).sort_values("TRADE_DT")
    close = close.drop_duplicates(subset=["TRADE_DT"], keep="last").set_index("TRADE_DT")
    return close.apply(pd.to_numeric, errors="coerce").astype("float64")


def _component_mean(metric_df: pd.DataFrame, index_code: str, weighted: bool = False) -> pd.Series:
    by_date = _load_component_weights(index_code)
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


def _calc_v111(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    ratio = _safe_ratio(growth_close, value_close)
    slope = _rolling_slope(ratio, window=40)
    raw = _rolling_slope(slope, window=20)
    return calc_rolling_zscore(raw, window=504)


def _calc_v112() -> pd.Series:
    df = pd.read_parquet(
        PROJECT_ROOT / "A_data" / "prepared_data" / ASTOCK_DAILY_TABLE,
        columns=["TradingDate", "Symbol", "ChangeRatio", "Amount"],
    )
    required_cols = {"TradingDate", "Symbol", "ChangeRatio", "Amount"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"{ASTOCK_DAILY_TABLE} must contain {sorted(required_cols)}; available={list(df.columns)}")

    work = df.dropna(subset=["TradingDate", "Symbol"]).copy()
    work["date"] = pd.to_datetime(work["TradingDate"], errors="coerce")
    work["ret"] = pd.to_numeric(work["ChangeRatio"], errors="coerce")
    work["amount"] = pd.to_numeric(work["Amount"], errors="coerce")
    work = work.dropna(subset=["date"]).copy()
    work["month"] = work["date"].dt.to_period("M")

    monthly = work.groupby(["month", "Symbol"], sort=True).agg(
        ret=("ret", "sum"),
        amount=("amount", "sum"),
    )
    monthly_ret = monthly["ret"].unstack("Symbol")
    monthly_amount = monthly["amount"].unstack("Symbol")

    lower = monthly_ret.quantile(0.1, axis=1)
    upper = monthly_ret.quantile(0.9, axis=1)
    trimmed_ret = monthly_ret.where(monthly_ret.ge(lower, axis=0) & monthly_ret.le(upper, axis=0))
    trimmed_di = trimmed_ret.std(axis=1, skipna=True)

    rel_amount = _safe_ratio(monthly_amount, monthly_amount.rolling(12, min_periods=12).mean())
    avg_rel_amount = rel_amount.mean(axis=1, skipna=True)

    madi = trimmed_di * avg_rel_amount
    madi.index = pd.PeriodIndex(madi.index, freq="M").to_timestamp("M")
    return calc_rolling_zscore(madi.sort_index(), window=60) * -1.0


def _calc_v113(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_ret = growth_close.pct_change(fill_method=None)
    value_ret = value_close.pct_change(fill_method=None)
    ret_diff = (growth_ret - value_ret).dropna().sort_index()
    grouped = ret_diff.groupby(ret_diff.index.to_period("M"))
    monthly_diff = grouped.sum()
    monthly_diff.index = grouped.apply(lambda group: group.index[-1]).to_numpy()
    monthly_diff = monthly_diff.sort_index().astype("float64")

    out = pd.Series(np.nan, index=monthly_diff.index, dtype="float64")
    for month in range(1, 13):
        mask = monthly_diff.index.month == month
        same_month = monthly_diff.loc[mask]
        out.loc[same_month.index] = same_month.expanding(min_periods=1).mean().shift(1)
    return out


def _calc_v114() -> pd.Series:
    tickers = _all_tickers_for_indices([GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX])
    close = _load_adj_close_wide(tickers)
    prev_close = close.shift(1)
    approx_high = pd.DataFrame(np.maximum(close.to_numpy(), prev_close.to_numpy()), index=close.index, columns=close.columns)
    approx_low = pd.DataFrame(np.minimum(close.to_numpy(), prev_close.to_numpy()), index=close.index, columns=close.columns)
    rsrs_rsq = approx_high.rolling(18, min_periods=18).corr(approx_low).pow(2)
    growth = _component_mean(rsrs_rsq, GROWTH_STYLE_INDEX, weighted=False)
    value = _component_mean(rsrs_rsq, VALUE_STYLE_INDEX, weighted=False)
    raw = _safe_ratio(growth, value) - 1.0
    return calc_rolling_zscore(raw, window=504)


def _mean_cross_corr(ret_df: pd.DataFrame, window: int = 20) -> pd.Series:
    out = pd.Series(np.nan, index=ret_df.index, dtype="float64")
    for pos in range(window, len(ret_df) + 1):
        block = ret_df.iloc[pos - window:pos].dropna(axis=1, how="any")
        if block.shape[1] < 2:
            continue
        corr = block.corr().to_numpy(dtype="float64")
        upper = corr[np.triu_indices_from(corr, k=1)]
        if upper.size:
            out.iloc[pos - 1] = float(np.nanmean(upper))
    return out


def _calc_v115() -> pd.Series:
    tickers = _all_tickers_for_indices([SH50_INDEX])
    close = _load_adj_close_wide(tickers)
    ret = close.pct_change(fill_method=None)

    by_date = _load_component_weights(SH50_INDEX)
    component_dates = pd.DatetimeIndex(sorted(by_date.keys()))
    raw = pd.Series(np.nan, index=ret.index, dtype="float64")
    for pos, comp_date in enumerate(component_dates):
        next_date = component_dates[pos + 1] if pos + 1 < len(component_dates) else pd.Timestamp.max
        mask = (ret.index >= comp_date) & (ret.index < next_date)
        if not mask.any():
            continue
        tickers_at_date = [ticker for ticker in by_date[pd.Timestamp(comp_date)].index if ticker in ret.columns]
        if len(tickers_at_date) < 2:
            continue
        start = max(0, ret.index.get_indexer([ret.index[mask][0]])[0] - 19)
        block = ret.loc[ret.index[start]:ret.index[mask][-1], tickers_at_date]
        raw.loc[mask] = _mean_cross_corr(block, window=20).reindex(ret.index[mask])
    return calc_rolling_zscore(raw, window=504) * -1.0


def generate_priceFactors5_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()

    _register_factor(raw_factor_df, factor_source_df, "V111_raw", _calc_v111(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V112_raw", _calc_v112())
    _register_factor(raw_factor_df, factor_source_df, "V113_raw", _calc_v113(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V114_raw", _calc_v114())
    _register_factor(raw_factor_df, factor_source_df, "V115_raw", _calc_v115())

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors5 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors5_factors(data_df)
    metadata = metadata_from_priceFactors5_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V071_V142 8.json.
PLAN8_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 8.json"
PLAN8_FACTOR_IDS = [
    "V106",
    "V107",
    "V108",
    "V109",
    "V110",
]
EXTENDED_FACTOR_IDS = PLAN8_FACTOR_IDS + FACTOR_IDS


def _load_plan8_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN8_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN8_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN8_FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id in {"V106", "V109"}:
                _append_note(item, "本脚本使用本地 growth_index.xlsx/value_index.xlsx 的 close 构造 log 成长/价值相对强弱。")
            elif factor_id == "V107":
                _append_note(item, "本脚本按计划记录使用 60 日滚动线性回归斜率。")
            elif factor_id == "V108":
                _append_note(item, "本脚本按计划记录使用 V107 斜率的一阶差分。")
            elif factor_id == "V110":
                _append_note(item, "本脚本按计划记录使用 20 日平滑后再计算 60 日滚动线性回归斜率。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN8_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 8 missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN8_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors5_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _relative_strength(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    aligned = pd.concat([growth_close.rename("growth"), value_close.rename("value")], axis=1).dropna()
    aligned = aligned[(aligned["growth"] > 0) & (aligned["value"] > 0)]
    return np.log(aligned["growth"]) - np.log(aligned["value"])


def _calc_v106(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    return calc_rolling_zscore(_relative_strength(growth_close, value_close), window=504)


def _calc_v107(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    rs = _relative_strength(growth_close, value_close)
    beta1 = _rolling_slope(rs, window=60)
    return calc_rolling_zscore(beta1, window=504)


def _calc_v108(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    rs = _relative_strength(growth_close, value_close)
    beta1 = _rolling_slope(rs, window=60)
    beta2 = beta1 - beta1.shift(1)
    return calc_rolling_zscore(beta2, window=504)


def _calc_v109(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    return _calc_v106(growth_close, value_close)


def _calc_v110(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    rs = _relative_strength(growth_close, value_close)
    rs_smooth = rs.rolling(20, min_periods=20).mean()
    slope = _rolling_slope(rs_smooth, window=60)
    return calc_rolling_zscore(slope, window=504)


def generate_plan8_priceFactors5_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan8_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()

    _register_factor(raw_factor_df, factor_source_df, "V106_raw", _calc_v106(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V107_raw", _calc_v107(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V108_raw", _calc_v108(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V109_raw", _calc_v109(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V110_raw", _calc_v110(growth_close, value_close))

    missing_cols = [factor_id for factor_id in PLAN8_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors5 plan 8 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN8_FACTOR_IDS], records


def generate_extended_priceFactors5_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan8_factor_df, plan8_records = generate_plan8_priceFactors5_factors(data_df)
    plan7_factor_df, plan7_records = generate_priceFactors5_factors(data_df)
    factor_source_df = pd.concat([plan8_factor_df, plan7_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan8_records + plan7_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors5_factors(data_df)
    metadata = metadata_from_extended_priceFactors5_records(selected_records)
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
