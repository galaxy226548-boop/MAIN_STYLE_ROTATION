"""Price style factors from completed V159-V166 plan."""

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


OUTPUT_PREFIX = "priceFactors10"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V159_V166.json"

FACTOR_IDS = [
    "V159",
    "V160",
    "V161",
    "V162",
    "V163",
    "V164",
    "V165",
    "V166",
]

ALL_A_VALUATION_FILE = "881001.WI-历史PE-PB-20260518.xlsx"
GROWTH_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx"
VALUE_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx"
HS300_VALUATION_FILE = "000300.SH-历史PE-PB-20260509.xlsx"
ZZ500_VALUATION_FILE = "000905.SH-历史PE-PB-20260509.xlsx"


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{PLAN_PATH} must contain a top-level record list")

    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    wanted = set(FACTOR_IDS)

    for record in payload:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in wanted:
            continue
        item = dict(record)
        item["_source_file"] = source_file
        item["_source_sheet"] = "records"
        item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
        if factor_id in {"V159", "V163"}:
            _append_note(item, "本脚本使用 Wind全A 881001.WI 历史PE-PB文件的市盈率TTM字段。")
        elif factor_id in {"V160", "V161"}:
            _append_note(item, "本脚本使用本地成长指数估值文件代理成长端，沪深300估值文件代理价值端。")
        elif factor_id in {"V162", "V166"}:
            _append_note(item, "本脚本使用沪深300与中证500历史PE-PB文件的市盈率TTM字段。")
        elif factor_id in {"V164", "V165"}:
            _append_note(item, "本脚本使用本地成长/价值风格估值文件的市盈率TTM字段合成EP。")
        if factor_id == "V163":
            _append_note(item, "为遵守项目 event 因子规则，未创6个月新高的日期保持 NaN，不作为负向事件挂载。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V159-V166 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors10_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _read_valuation_series(file_name: str, value_col: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if "交易日期" not in df.columns or value_col not in df.columns:
        raise KeyError(f"{file_name} must contain '交易日期' and {value_col!r}; available={list(df.columns)}")
    dates = pd.to_datetime(df["交易日期"], errors="coerce").dt.normalize()
    values = _as_numeric(df[value_col])
    return _as_float_series(values, dates, name)


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned = pd.concat([numerator.rename("numerator"), denominator.rename("denominator")], axis=1, sort=True)
    ratio = aligned["numerator"].where(aligned["numerator"] > 0) / aligned["denominator"].where(aligned["denominator"] > 0)
    return np.log(ratio.replace([np.inf, -np.inf], np.nan))


def _safe_inverse(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return (1.0 / positive).replace([np.inf, -np.inf], np.nan)


def _calc_v159(all_a_pe: pd.Series) -> pd.Series:
    raw = all_a_pe.shift(1)
    return calc_rolling_zscore(raw, window=504)


def _calc_v160(growth_pe: pd.Series, hs300_pe: pd.Series) -> pd.Series:
    raw = growth_pe - hs300_pe
    return calc_rolling_zscore(raw, window=504)


def _calc_v161(growth_pb: pd.Series, hs300_pb: pd.Series) -> pd.Series:
    raw = growth_pb - hs300_pb
    return calc_rolling_zscore(raw, window=504)


def _calc_v162(hs300_pe: pd.Series, zz500_pe: pd.Series) -> pd.Series:
    raw = _safe_log_ratio(hs300_pe, zz500_pe)
    ma20 = raw.rolling(20, min_periods=20).mean()
    raw_signal = -(raw - ma20)
    return calc_rolling_zscore(raw_signal, window=504)


def _calc_v163(all_a_pe: pd.Series) -> pd.Series:
    rolling_max_6m = all_a_pe.shift(1).rolling(window=126, min_periods=126).max()
    pe_z = calc_rolling_zscore(all_a_pe, window=504)
    rolling_max_z = calc_rolling_zscore(rolling_max_6m, window=504)
    event_strength = pe_z - rolling_max_z
    return event_strength.where(all_a_pe > rolling_max_6m)


def _calc_v164(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    ep_growth = _safe_inverse(growth_pe)
    ep_value = _safe_inverse(value_pe)
    ep_growth_bias = ep_growth - ep_growth.rolling(504, min_periods=504).mean()
    ep_value_bias = ep_value - ep_value.rolling(504, min_periods=504).mean()
    ep_growth_z = calc_rolling_zscore(ep_growth_bias, window=504)
    ep_value_z = calc_rolling_zscore(ep_value_bias, window=504)
    return ep_growth_z - ep_value_z


def _calc_v165(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    raw = _safe_inverse(growth_pe) - _safe_inverse(value_pe)
    return calc_rolling_zscore(raw, window=504)


def _calc_v166(zz500_pe: pd.Series, hs300_pe: pd.Series) -> pd.Series:
    raw = zz500_pe - hs300_pe
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors10_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    all_a_pe = _read_valuation_series(ALL_A_VALUATION_FILE, "市盈率TTM", "all_a_pe_ttm")
    growth_pe = _read_valuation_series(GROWTH_VALUATION_FILE, "市盈率TTM", "growth_pe_ttm")
    value_pe = _read_valuation_series(VALUE_VALUATION_FILE, "市盈率TTM", "value_pe_ttm")
    growth_pb = _read_valuation_series(GROWTH_VALUATION_FILE, "市净率LF", "growth_pb_lf")
    hs300_pe = _read_valuation_series(HS300_VALUATION_FILE, "市盈率TTM", "hs300_pe_ttm")
    hs300_pb = _read_valuation_series(HS300_VALUATION_FILE, "市净率LF", "hs300_pb_lf")
    zz500_pe = _read_valuation_series(ZZ500_VALUATION_FILE, "市盈率TTM", "zz500_pe_ttm")

    _register_factor(raw_factor_df, factor_source_df, "V159_raw", _calc_v159(all_a_pe))
    _register_factor(raw_factor_df, factor_source_df, "V160_raw", _calc_v160(growth_pe, hs300_pe))
    _register_factor(raw_factor_df, factor_source_df, "V161_raw", _calc_v161(growth_pb, hs300_pb))
    _register_factor(raw_factor_df, factor_source_df, "V162_raw", _calc_v162(hs300_pe, zz500_pe))
    _register_factor(raw_factor_df, factor_source_df, "V163_raw", _calc_v163(all_a_pe))
    _register_factor(raw_factor_df, factor_source_df, "V164_raw", _calc_v164(growth_pe, value_pe))
    _register_factor(raw_factor_df, factor_source_df, "V165_raw", _calc_v165(growth_pe, value_pe))
    _register_factor(raw_factor_df, factor_source_df, "V166_raw", _calc_v166(zz500_pe, hs300_pe))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors10 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors10_factors(data_df)
    metadata = metadata_from_priceFactors10_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V167_V174.json.
import pyarrow.parquet as pq


PLAN167_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V167_V174.json"
PLAN167_FACTOR_IDS = [
    "V167",
    "V168",
    "V169",
    "V170",
    "V171",
    "V172",
    "V173",
    "V174",
]
EXTENDED_FACTOR_IDS = FACTOR_IDS + PLAN167_FACTOR_IDS

SW_LARGE_VALUATION_FILE = "801811.SI-历史PE-PB-20260509.xlsx"
SW_SMALL_VALUATION_FILE = "801813.SI-历史PE-PB-20260509.xlsx"
CSI_ALL_VALUATION_FILE = "000985.CSI-历史PE-PB-20260509.xlsx"
CSI2000_VALUATION_FILE = "932000.CSI-历史PE-PB-20260509.xlsx"
PB_TABLE = "pb.parquet"
ASTOCK_DAILY_TABLE = "Astockdaily.parquet"


def _load_plan167_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN167_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{PLAN167_PATH} must contain a top-level record list")

    records: list[dict[str, object]] = []
    source_file = str(PLAN167_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN167_FACTOR_IDS)

    for record in payload:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in wanted:
            continue
        item = dict(record)
        item["_source_file"] = source_file
        item["_source_sheet"] = "records"
        item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
        if factor_id == "V167":
            _append_note(item, "本脚本使用申万大盘801811近似价值、申万小盘801813近似成长。")
        elif factor_id == "V168":
            _append_note(item, "本脚本使用 pb.parquet 个股PB宽表，并用 Astockdaily.parquet 的流通市值按每日截面10%/90%分组。")
        elif factor_id in {"V169", "V170", "V173"}:
            _append_note(item, "本脚本使用本地指数历史PE-PB文件的市净率LF字段。")
        elif factor_id in {"V171", "V172"}:
            _append_note(item, "本脚本使用本地成长/价值风格估值文件的市净率LF字段。")
        elif factor_id == "V174":
            _append_note(item, "本脚本使用本地成长/价值风格估值文件的市盈率TTM字段。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN167_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V167-V174 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN167_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors10_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return metadata_from_priceFactors10_records(records)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].replace(".0", "").zfill(6)


def _prepared_parquet_columns(table_name: str) -> list[str]:
    path = PROJECT_ROOT / "A_data" / "prepared_data" / table_name
    return list(pq.read_schema(path).names)


def _load_pb_panel() -> pd.DataFrame:
    df = pd.read_parquet(PROJECT_ROOT / "A_data" / "prepared_data" / PB_TABLE)
    if "TRADE_DT" not in df.columns:
        raise KeyError(f"{PB_TABLE} must contain TRADE_DT")
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
    df = df.dropna(subset=["TRADE_DT"]).sort_values("TRADE_DT").drop_duplicates(subset=["TRADE_DT"], keep="last")
    df = df.set_index("TRADE_DT")
    df.columns = [_stock_code_key(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


def _load_market_cap_panel(tickers: list[str]) -> pd.DataFrame:
    columns = ["TradingDate", "Symbol", "CirculatedMarketValue"]
    df = pd.read_parquet(PROJECT_ROOT / "A_data" / "prepared_data" / ASTOCK_DAILY_TABLE, columns=columns)
    work = df.copy()
    work["date"] = pd.to_datetime(work["TradingDate"], errors="coerce").dt.normalize()
    work["ticker"] = work["Symbol"].map(_stock_code_key)
    work["mcap"] = pd.to_numeric(work["CirculatedMarketValue"], errors="coerce")
    work = work[work["date"].notna() & work["ticker"].isin(tickers) & work["mcap"].gt(0)].copy()
    if work.empty:
        raise ValueError(f"{ASTOCK_DAILY_TABLE} has no valid market-cap rows matching {PB_TABLE}")
    return work.pivot_table(index="date", columns="ticker", values="mcap", aggfunc="last").astype("float64")


def _calc_v167(sw_large_pb: pd.Series, sw_small_pb: pd.Series) -> pd.Series:
    raw = _safe_ratio(sw_large_pb, sw_small_pb)
    return calc_rolling_zscore(raw, window=504)


def _calc_v168() -> pd.Series:
    pb_panel = _load_pb_panel()
    mcap_panel = _load_market_cap_panel(list(pb_panel.columns))
    common_index = pb_panel.index.intersection(mcap_panel.index).sort_values()
    common_cols = sorted(set(pb_panel.columns).intersection(mcap_panel.columns))
    pb = pb_panel.reindex(index=common_index, columns=common_cols).where(lambda x: x > 0)
    mcap = mcap_panel.reindex(index=common_index, columns=common_cols)

    rank = mcap.rank(axis=1, pct=True)
    small_group_pb_median = pb.where(rank <= 0.1).median(axis=1, skipna=True)
    big_group_pb_median = pb.where(rank >= 0.9).median(axis=1, skipna=True)
    raw = _safe_ratio(small_group_pb_median, big_group_pb_median)
    return calc_rolling_zscore(raw, window=504) * -1.0


def _premium_reversal_factor(premium: pd.Series) -> pd.Series:
    upper = premium.rolling(756, min_periods=756).quantile(0.8)
    lower = premium.rolling(756, min_periods=756).quantile(0.2)
    premium_z = calc_rolling_zscore(premium, window=756)
    upper_z = calc_rolling_zscore(upper, window=756)
    lower_z = calc_rolling_zscore(lower, window=756)

    factor = pd.Series(0.0, index=premium.index, dtype="float64")
    cond_growth = (premium.shift(1) < lower.shift(1)) & (premium > premium.shift(1))
    cond_value = (premium.shift(1) > upper.shift(1)) & (premium < premium.shift(1))
    factor.loc[cond_growth] = (lower_z - premium_z).abs().loc[cond_growth]
    factor.loc[cond_value] = -((premium_z - upper_z).abs().loc[cond_value])
    factor.loc[premium_z.isna()] = np.nan
    return factor.replace([np.inf, -np.inf], np.nan)


def _calc_v169(hs300_pb: pd.Series, csi_all_pb: pd.Series) -> pd.Series:
    premium = hs300_pb - csi_all_pb
    return _premium_reversal_factor(premium)


def _calc_v170(csi2000_pb: pd.Series, csi_all_pb: pd.Series) -> pd.Series:
    premium = csi2000_pb - csi_all_pb
    return _premium_reversal_factor(premium)


def _calc_v171(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    value_qr = value_pb / value_pb.shift(63) - 1.0
    growth_qr = growth_pb / growth_pb.shift(63) - 1.0
    value_diff = value_qr - value_qr.shift(1)
    growth_diff = growth_qr - growth_qr.shift(1)
    raw = growth_diff - value_diff
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v172(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    relative_pb = growth_pb - value_pb
    ma5 = relative_pb.rolling(5, min_periods=5).mean()
    ma240 = relative_pb.rolling(240, min_periods=240).mean()
    raw = ma5 - ma240
    return calc_rolling_zscore(raw, window=504)


def _calc_v173(csi_all_pb: pd.Series) -> pd.Series:
    ma_short = csi_all_pb.rolling(window=20, min_periods=20).mean()
    ma_long = csi_all_pb.rolling(window=120, min_periods=120).mean()
    raw = ma_short - ma_long
    return calc_rolling_zscore(raw, window=504)


def _calc_v174(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    diff = growth_pe - value_pe
    yoy = diff / diff.shift(252) - 1.0
    ma_short = yoy.rolling(window=20, min_periods=20).mean()
    ma_long = yoy.rolling(window=120, min_periods=120).mean()
    raw = ma_short - ma_long
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=504)


def generate_plan167_priceFactors10_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan167_records()

    sw_large_pb = _read_valuation_series(SW_LARGE_VALUATION_FILE, "市净率LF", "sw_large_pb_lf")
    sw_small_pb = _read_valuation_series(SW_SMALL_VALUATION_FILE, "市净率LF", "sw_small_pb_lf")
    hs300_pb = _read_valuation_series(HS300_VALUATION_FILE, "市净率LF", "hs300_pb_lf")
    csi_all_pb = _read_valuation_series(CSI_ALL_VALUATION_FILE, "市净率LF", "csi_all_pb_lf")
    csi2000_pb = _read_valuation_series(CSI2000_VALUATION_FILE, "市净率LF", "csi2000_pb_lf")
    growth_pb = _read_valuation_series(GROWTH_VALUATION_FILE, "市净率LF", "growth_pb_lf")
    value_pb = _read_valuation_series(VALUE_VALUATION_FILE, "市净率LF", "value_pb_lf")
    growth_pe = _read_valuation_series(GROWTH_VALUATION_FILE, "市盈率TTM", "growth_pe_ttm")
    value_pe = _read_valuation_series(VALUE_VALUATION_FILE, "市盈率TTM", "value_pe_ttm")

    _register_factor(raw_factor_df, factor_source_df, "V167_raw", _calc_v167(sw_large_pb, sw_small_pb))
    _register_factor(raw_factor_df, factor_source_df, "V168_raw", _calc_v168())
    _register_factor(raw_factor_df, factor_source_df, "V169_raw", _calc_v169(hs300_pb, csi_all_pb))
    _register_factor(raw_factor_df, factor_source_df, "V170_raw", _calc_v170(csi2000_pb, csi_all_pb))
    _register_factor(raw_factor_df, factor_source_df, "V171_raw", _calc_v171(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V172_raw", _calc_v172(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V173_raw", _calc_v173(csi_all_pb))
    _register_factor(raw_factor_df, factor_source_df, "V174_raw", _calc_v174(growth_pe, value_pe))

    missing_cols = [factor_id for factor_id in PLAN167_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors10 V167-V174 columns missing after generation: {missing_cols}")
    return factor_source_df.loc[:, PLAN167_FACTOR_IDS], records


def generate_extended_priceFactors10_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan159_factor_df, plan159_records = generate_priceFactors10_factors(data_df)
    plan167_factor_df, plan167_records = generate_plan167_priceFactors10_factors(data_df)
    factor_source_df = pd.concat([plan159_factor_df, plan167_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan159_records + plan167_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors10_factors(data_df)
    metadata = metadata_from_extended_priceFactors10_records(selected_records)
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
