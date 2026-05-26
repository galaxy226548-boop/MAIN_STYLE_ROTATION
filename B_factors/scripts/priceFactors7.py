"""Price style factors from completed V071-V142 plan file 11."""

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


OUTPUT_PREFIX = "priceFactors7"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 11.json"

FACTOR_IDS = [
    "V091",
    "V092",
    "V093",
    "V094",
    "V095",
]

INDEX_EOD_TABLE = "index_eod.parquet"
INDEX_WEIGHT_TABLE = "AIndexHS300FreeWeight.parquet"
ADJ_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"
INDUSTRY_INDEX_TABLE = "industry_indice.parquet"
GROWTH_INDEX_FILE = "growth_index.xlsx"
VALUE_INDEX_FILE = "value_index.xlsx"
HS300_PRICE_FILE = "沪深300(000300.SH)-历史价格.xlsx"

CSI100_INDEX = "000903"
CSI500_INDEX = "000905"
GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"

GROWTH_INDUSTRIES = [
    "中信行业指数:计算机",
    "中信行业指数:电子",
    "中信行业指数:医药",
    "中信行业指数:电力设备及新能源",
    "中信行业指数:传媒",
]
VALUE_INDUSTRIES = [
    "中信行业指数:银行",
    "中信行业指数:非银行金融",
    "中信行业指数:石油石化",
    "中信行业指数:煤炭",
    "中信行业指数:房地产",
]

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
            if factor_id == "V091":
                _append_note(
                    item,
                    "本地 index_eod.parquet 无最高/最低价，本脚本使用中证100/中证500开盘指数与收盘指数的绝对差近似每日振幅。",
                )
            elif factor_id == "V092":
                _append_note(
                    item,
                    "本脚本使用 growth_index.xlsx/value_index.xlsx 的 close，并采用 MACD 默认参数 M=12,N=26,L=9；因本地周频 RSW 有效点不足以支持504周标准化，改用104周窗口。",
                )
            elif factor_id == "V093":
                _append_note(item, "本脚本复用 AIndexHS300FreeWeight.parquet 成分和 S_DQ_ADJCLOSE.parquet 复权收盘价宽表计算成长/价值成分股上涨比例。")
            elif factor_id == "V094":
                _append_note(item, "本脚本使用 industry_indice.parquet 中信行业指数与沪深300历史价格，滚动24周 OLS，取最近4周标准化残差均值。")
            elif factor_id == "V095":
                _append_note(item, "本脚本使用 index_eod.parquet 的中证500与中证A100(000903)收盘指数，按 W-FRI 周频计算12周均线偏离。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 11 missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors7_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _safe_ratio(numerator: pd.Series | pd.DataFrame, denominator: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return numerator / denominator.replace(0.0, np.nan)


def _load_index_eod_rows(index_code: str) -> pd.DataFrame:
    df = load_prepared_table(INDEX_EOD_TABLE)
    if "交易所指数代码" not in df.columns:
        raise KeyError(f"{INDEX_EOD_TABLE} must contain '交易所指数代码'; available={list(df.columns)}")
    if isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index, errors="coerce").normalize()
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        raise KeyError(f"{INDEX_EOD_TABLE} must have a DatetimeIndex or date column")

    work = df.copy()
    work["date"] = dates
    work["index_code"] = work["交易所指数代码"].map(_index_code_key)
    out = work[work["index_code"].eq(_index_code_key(index_code))].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{INDEX_EOD_TABLE} missing index_code={index_code!r}")
    return out.sort_values("date")


def _load_index_eod_series(index_code: str, value_col: str, name: str) -> pd.Series:
    rows = _load_index_eod_rows(index_code)
    if value_col not in rows.columns:
        raise KeyError(f"{INDEX_EOD_TABLE} must contain {value_col!r}; available={list(rows.columns)}")
    return _as_float_series(rows[value_col], rows["date"], name)


def _load_growth_close() -> pd.Series:
    return _read_excel_series(GROWTH_INDEX_FILE, "date", "close", "growth_close")


def _load_value_close() -> pd.Series:
    return _read_excel_series(VALUE_INDEX_FILE, "date", "close", "value_close")


def _load_hs300_close() -> pd.Series:
    return _read_excel_series(HS300_PRICE_FILE, "交易日期", "收盘价", "hs300_close")


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


def _component_mean(metric_df: pd.DataFrame, index_code: str) -> pd.Series:
    by_date = _load_component_weights(index_code)
    component_dates = pd.DatetimeIndex(sorted(by_date.keys()))
    out = pd.Series(np.nan, index=metric_df.index, dtype="float64")

    for pos, comp_date in enumerate(component_dates):
        next_date = component_dates[pos + 1] if pos + 1 < len(component_dates) else pd.Timestamp.max
        mask = (metric_df.index >= comp_date) & (metric_df.index < next_date)
        if not mask.any():
            continue
        tickers = [ticker for ticker in by_date[pd.Timestamp(comp_date)].index if ticker in metric_df.columns]
        if tickers:
            out.loc[mask] = metric_df.loc[mask, tickers].mean(axis=1, skipna=True)
    return out


def _calc_v091() -> pd.Series:
    rows_100 = _load_index_eod_rows(CSI100_INDEX)
    rows_500 = _load_index_eod_rows(CSI500_INDEX)
    for col in ["开盘指数", "收盘指数"]:
        if col not in rows_100.columns or col not in rows_500.columns:
            raise KeyError(f"{INDEX_EOD_TABLE} must contain {col!r} for V091")
    amp_100 = _as_float_series((rows_100["开盘指数"] - rows_100["收盘指数"]).abs(), rows_100["date"], "amp_100")
    amp_500 = _as_float_series((rows_500["开盘指数"] - rows_500["收盘指数"]).abs(), rows_500["date"], "amp_500")
    aligned = pd.concat([amp_100.rename("amp_100"), amp_500.rename("amp_500")], axis=1).dropna()
    aligned = aligned[(aligned["amp_100"] > 0) & (aligned["amp_500"] > 0)]
    raw = (np.log(aligned["amp_100"]) - np.log(aligned["amp_500"])) * -1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v092(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_w = growth_close.dropna().sort_index().resample("W-FRI").last()
    value_w = value_close.dropna().sort_index().resample("W-FRI").last()
    r_growth = growth_w.pct_change(fill_method=None)
    r_value = value_w.pct_change(fill_method=None)
    rr = np.log1p(r_growth) - np.log1p(r_value)
    dif = rr.rolling(window=12, min_periods=12).mean() - rr.rolling(window=26, min_periods=26).mean()
    dea = dif.rolling(window=9, min_periods=9).mean()
    rsw = dif - dea
    return calc_rolling_zscore(rsw, window=104)


def _calc_v093() -> pd.Series:
    tickers = _all_tickers_for_indices([GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX])
    close = _load_adj_close_wide(tickers)
    up_flag = (close > close.shift(1)).astype("float64")
    improved_parts = []
    for window in [18, 19, 20, 21, 22]:
        u1 = up_flag.rolling(window=window, min_periods=window).mean()
        u2 = up_flag.shift(1).rolling(window=window, min_periods=window).mean()
        improved_parts.append((u1 + u2) / 2.0)
    improved_udr = sum(improved_parts) / len(improved_parts)
    growth_udr = _component_mean(improved_udr, GROWTH_STYLE_INDEX)
    value_udr = _component_mean(improved_udr, VALUE_STYLE_INDEX)
    raw = growth_udr - value_udr
    return calc_rolling_zscore(raw, window=504)


def _rolling_residual_score(ind_ret: pd.Series, market_ret: pd.Series, window: int = 24, recent: int = 4) -> pd.Series:
    aligned = pd.concat([ind_ret.rename("industry"), market_ret.rename("market")], axis=1).dropna()
    out = pd.Series(np.nan, index=aligned.index, dtype="float64")
    for pos in range(window, len(aligned) + 1):
        block = aligned.iloc[pos - window:pos]
        x = block["market"].to_numpy(dtype="float64")
        y = block["industry"].to_numpy(dtype="float64")
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            continue
        beta, alpha = np.polyfit(x, y, deg=1)
        resid = y - (alpha + beta * x)
        resid_std = float(np.std(resid, ddof=1))
        if resid_std == 0 or np.isnan(resid_std):
            continue
        out.iloc[pos - 1] = float(np.mean(resid[-recent:] / resid_std))
    return out


def _calc_v094(hs300_close: pd.Series) -> pd.Series:
    industry = load_prepared_table(INDUSTRY_INDEX_TABLE)
    industry.index = pd.to_datetime(industry.index, errors="coerce").normalize()
    industry = industry[industry.index.notna()].sort_index()
    missing = [col for col in GROWTH_INDUSTRIES + VALUE_INDUSTRIES if col not in industry.columns]
    if missing:
        raise KeyError(f"{INDUSTRY_INDEX_TABLE} missing industry columns: {missing}")

    industry_w = industry[GROWTH_INDUSTRIES + VALUE_INDUSTRIES].apply(pd.to_numeric, errors="coerce").resample("W-FRI").last()
    industry_ret = industry_w.pct_change(fill_method=None)
    market_ret = hs300_close.dropna().sort_index().resample("W-FRI").last().pct_change(fill_method=None)

    scores = pd.DataFrame(index=industry_ret.index)
    for col in GROWTH_INDUSTRIES + VALUE_INDUSTRIES:
        scores[col] = _rolling_residual_score(industry_ret[col], market_ret).reindex(scores.index)
    growth_score = scores[GROWTH_INDUSTRIES].mean(axis=1, skipna=True)
    value_score = scores[VALUE_INDUSTRIES].mean(axis=1, skipna=True)
    raw = growth_score - value_score
    return calc_rolling_zscore(raw, window=504)


def _calc_v095() -> pd.Series:
    close_500 = _load_index_eod_series(CSI500_INDEX, "收盘指数", "csi500_close")
    close_100 = _load_index_eod_series(CSI100_INDEX, "收盘指数", "csi100_close")
    close_500_w = close_500.dropna().sort_index().resample("W-FRI").last()
    close_100_w = close_100.dropna().sort_index().resample("W-FRI").last()
    ratio = _safe_ratio(close_500_w, close_100_w)
    raw = ratio - ratio.rolling(window=12, min_periods=12).mean()
    return calc_rolling_zscore(raw, window=104)


def generate_priceFactors7_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()
    hs300_close = _load_hs300_close()

    _register_factor(raw_factor_df, factor_source_df, "V091_raw", _calc_v091())
    _register_factor(raw_factor_df, factor_source_df, "V092_raw", _calc_v092(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V093_raw", _calc_v093())
    _register_factor(raw_factor_df, factor_source_df, "V094_raw", _calc_v094(hs300_close))
    _register_factor(raw_factor_df, factor_source_df, "V095_raw", _calc_v095())

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors7 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors7_factors(data_df)
    metadata = metadata_from_priceFactors7_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V071_V142 12.json.
PLAN12_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 12.json"
PLAN12_FACTOR_IDS = [
    "V086",
    "V087",
    "V088",
    "V089",
    "V090",
]
EXTENDED_FACTOR_IDS = PLAN12_FACTOR_IDS + FACTOR_IDS


def _load_plan12_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN12_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN12_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN12_FACTOR_IDS)

    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
            if factor_id == "V086":
                _append_note(item, "本脚本使用 index_eod.parquet 的国证成长/国证价值收盘指数计算20日年化夏普比率差。")
            elif factor_id == "V087":
                _append_note(item, "本脚本使用 growth_index.xlsx/value_index.xlsx 的 close，窗口N按计划取126个交易日。")
            elif factor_id == "V088":
                _append_note(item, "本脚本使用 growth_index.xlsx/value_index.xlsx 的 close，3个月窗口取63个交易日，并按计划乘以-1体现反转方向。")
            elif factor_id == "V089":
                _append_note(item, "本脚本使用 growth_index.xlsx/value_index.xlsx 的 close 计算 log(成长/价值) 并扣除252日均值。")
            elif factor_id == "V090":
                _append_note(item, "本脚本使用 index_eod.parquet 的国证成长/国证价值成份证券成交金额计算资金相对强弱。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN12_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan file 12 missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN12_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors7_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _calc_v086() -> pd.Series:
    growth_close = _load_index_eod_series(GROWTH_STYLE_INDEX, "收盘指数", "growth_index_eod_close")
    value_close = _load_index_eod_series(VALUE_STYLE_INDEX, "收盘指数", "value_index_eod_close")
    growth_ret = growth_close.pct_change(fill_method=None)
    value_ret = value_close.pct_change(fill_method=None)
    growth_sharpe = growth_ret.rolling(20, min_periods=20).mean() / growth_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    value_sharpe = value_ret.rolling(20, min_periods=20).mean() / value_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    raw = growth_sharpe - value_sharpe
    return calc_rolling_zscore(raw, window=504)


def _calc_v087(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_ret = growth_close.pct_change(126, fill_method=None)
    value_ret = value_close.pct_change(126, fill_method=None)
    raw = growth_ret - value_ret
    return calc_rolling_zscore(raw, window=504)


def _calc_v088(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_ret_3m = growth_close.pct_change(63, fill_method=None)
    value_ret_3m = value_close.pct_change(63, fill_method=None)
    raw = (growth_ret_3m - value_ret_3m) * -1.0
    return calc_rolling_zscore(raw, window=504)


def _calc_v089(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    aligned = pd.concat([growth_close.rename("growth"), value_close.rename("value")], axis=1).dropna()
    aligned = aligned[(aligned["growth"] > 0) & (aligned["value"] > 0)]
    log_rs = np.log(_safe_ratio(aligned["growth"], aligned["value"]))
    raw = log_rs - log_rs.rolling(252, min_periods=252).mean()
    return calc_rolling_zscore(raw, window=504)


def _calc_v090() -> pd.Series:
    growth_amt = _load_index_eod_series(GROWTH_STYLE_INDEX, "成份证券成交金额", "growth_amount")
    value_amt = _load_index_eod_series(VALUE_STYLE_INDEX, "成份证券成交金额", "value_amount")
    aligned = pd.concat([growth_amt.rename("growth"), value_amt.rename("value")], axis=1).dropna()
    aligned = aligned[(aligned["growth"] > 0) & (aligned["value"] > 0)]
    log_amt_rs = np.log(_safe_ratio(aligned["growth"], aligned["value"]))
    raw = log_amt_rs - log_amt_rs.rolling(252, min_periods=252).mean()
    return calc_rolling_zscore(raw, window=504)


def generate_plan12_priceFactors7_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan12_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()

    _register_factor(raw_factor_df, factor_source_df, "V086_raw", _calc_v086())
    _register_factor(raw_factor_df, factor_source_df, "V087_raw", _calc_v087(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V088_raw", _calc_v088(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V089_raw", _calc_v089(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V090_raw", _calc_v090())

    missing_cols = [factor_id for factor_id in PLAN12_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors7 plan 12 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, PLAN12_FACTOR_IDS], records


def generate_extended_priceFactors7_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan12_factor_df, plan12_records = generate_plan12_priceFactors7_factors(data_df)
    plan11_factor_df, plan11_records = generate_priceFactors7_factors(data_df)
    factor_source_df = pd.concat([plan12_factor_df, plan11_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan12_records + plan11_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors7_factors(data_df)
    metadata = metadata_from_extended_priceFactors7_records(selected_records)
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
