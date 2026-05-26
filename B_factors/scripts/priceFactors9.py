"""Price style factors from completed V143-V150 plan."""

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
    read_prepared_series,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "priceFactors9"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V143_V150.json"

FACTOR_IDS = [
    "V143",
    "V144",
    "V145",
    "V146",
    "V147",
    "V148",
    "V149",
    "V150",
]

GROWTH_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx"
VALUE_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx"
HS300_VALUATION_FILE = "000300.SH-历史PE-PB-20260509.xlsx"


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _load_plan_records() -> list[dict[str, object]]:
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    try:
        payload = json.loads(plan_text)
    except json.JSONDecodeError:
        payload = json.loads(plan_text.replace('外资"聪明钱"？', "外资'聪明钱'？"))
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
        if factor_id == "V143":
            _append_note(
                item,
                "本脚本使用成长指数PB_LF减价值指数PB_LF作为风格PB溢价代理，再乘以10年国债收益率日变化方向。",
            )
        elif factor_id in {"V144", "V145", "V146", "V148", "V149", "V150"}:
            _append_note(item, "本脚本使用本地成长/价值风格指数估值文件的日频估值字段。")
        elif factor_id == "V147":
            _append_note(item, "本脚本使用本地沪深300历史PE-PB文件的市盈率TTM字段。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V143-V150 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors9_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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


def _load_growth_valuation(value_col: str, name: str) -> pd.Series:
    return _read_valuation_series(GROWTH_VALUATION_FILE, value_col, name)


def _load_value_valuation(value_col: str, name: str) -> pd.Series:
    return _read_valuation_series(VALUE_VALUATION_FILE, value_col, name)


def _load_hs300_valuation(value_col: str, name: str) -> pd.Series:
    return _read_valuation_series(HS300_VALUATION_FILE, value_col, name)


def _calc_v143(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    cn_10y = read_prepared_series("rate_daily.parquet", "中债国债到期收益率:10年")
    pb_premium = growth_pb - value_pb
    pb_z = calc_rolling_zscore(pb_premium, window=504)
    rate_chg = cn_10y.diff()
    direction = pd.Series(0.0, index=rate_chg.index, dtype="float64")
    direction.loc[rate_chg > 0] = -1.0
    direction.loc[rate_chg < 0] = 1.0
    direction.loc[rate_chg.isna()] = np.nan
    return direction.reindex(pb_z.index, method="ffill") * pb_z


def _calc_v144(growth_pe: pd.Series) -> pd.Series:
    raw = growth_pe.diff()
    return calc_rolling_zscore(raw, window=504)


def _calc_v145(growth_pb: pd.Series) -> pd.Series:
    return calc_rolling_zscore(growth_pb, window=504) * -1.0


def _calc_v146(growth_ps: pd.Series, value_ps: pd.Series) -> pd.Series:
    ps_growth_adj = growth_ps - value_ps
    raw = ps_growth_adj.diff()
    return calc_rolling_zscore(raw, window=504)


def _calc_v147(hs300_pe: pd.Series) -> pd.Series:
    return calc_rolling_zscore(hs300_pe, window=504)


def _calc_v148(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    raw = growth_pe - value_pe
    return calc_rolling_zscore(raw, window=504)


def _calc_v149(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    raw = growth_pb - value_pb
    return calc_rolling_zscore(raw, window=504)


def _calc_v150(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    raw = value_pe - growth_pe
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors9_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_pe = _load_growth_valuation("市盈率TTM", "growth_pe_ttm")
    value_pe = _load_value_valuation("市盈率TTM", "value_pe_ttm")
    growth_pb = _load_growth_valuation("市净率LF", "growth_pb_lf")
    value_pb = _load_value_valuation("市净率LF", "value_pb_lf")
    growth_ps = _load_growth_valuation("市销率TTM", "growth_ps_ttm")
    value_ps = _load_value_valuation("市销率TTM", "value_ps_ttm")
    hs300_pe = _load_hs300_valuation("市盈率TTM", "hs300_pe_ttm")

    _register_factor(raw_factor_df, factor_source_df, "V143_raw", _calc_v143(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V144_raw", _calc_v144(growth_pe))
    _register_factor(raw_factor_df, factor_source_df, "V145_raw", _calc_v145(growth_pb))
    _register_factor(raw_factor_df, factor_source_df, "V146_raw", _calc_v146(growth_ps, value_ps))
    _register_factor(raw_factor_df, factor_source_df, "V147_raw", _calc_v147(hs300_pe))
    _register_factor(raw_factor_df, factor_source_df, "V148_raw", _calc_v148(growth_pe, value_pe))
    _register_factor(raw_factor_df, factor_source_df, "V149_raw", _calc_v149(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V150_raw", _calc_v150(growth_pe, value_pe))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors9 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors9_factors(data_df)
    metadata = metadata_from_priceFactors9_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V151_V158.json.
import pyarrow.parquet as pq


PLAN151_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V151_V158.json"
PLAN151_FACTOR_IDS = [
    "V151",
    "V152",
    "V153",
    "V154",
    "V155",
    "V156",
    "V157",
    "V158",
]
EXTENDED_FACTOR_IDS = FACTOR_IDS + PLAN151_FACTOR_IDS

PE_TABLE = "pe.parquet"
PB_TABLE = "pb.parquet"
ADJ_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"

INDUSTRY_VALUATION_FILES = [
    "399314.SZ-历史PE-PB-20260509.xlsx",
    "399316.SZ-历史PE-PB-20260509.xlsx",
    "801811.SI-历史PE-PB-20260509.xlsx",
    "801813.SI-历史PE-PB-20260509.xlsx",
    "932000.CSI-历史PE-PB-20260509.xlsx",
]


def _load_plan151_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN151_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{PLAN151_PATH} must contain a top-level record list")

    records: list[dict[str, object]] = []
    source_file = str(PLAN151_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN151_FACTOR_IDS)
    for record in payload:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in wanted:
            continue
        item = dict(record)
        item["_source_file"] = source_file
        item["_source_sheet"] = "records"
        item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
        if factor_id in {"V152", "V153"}:
            _append_note(
                item,
                "Astockdaily.parquet 当前无行业分类字段，本脚本使用本地五个行业/宽基指数PE-PB文件的截面变异系数作为行业估值离散度代理。",
            )
        elif factor_id in {"V154", "V155", "V156", "V157"}:
            _append_note(
                item,
                "本脚本使用 pb.parquet、pe.parquet 与 S_DQ_ADJCLOSE.parquet 宽表，按每日PB截面30%/70%分位构建低PB/高PB等权组合。",
            )
        elif factor_id in {"V151", "V158"}:
            _append_note(item, "本脚本使用本地成长/价值风格指数估值文件的日频估值字段。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN151_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V151-V158 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN151_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors9_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return metadata_from_priceFactors9_records(records)


def _prepared_parquet_columns(table_name: str) -> list[str]:
    path = PROJECT_ROOT / "A_data" / "prepared_data" / table_name
    return list(pq.read_schema(path).names)


def _load_wide_numeric_table(table_name: str, tickers: list[str] | None = None) -> pd.DataFrame:
    columns = ["TRADE_DT"] + list(tickers) if tickers is not None else None
    df = pd.read_parquet(PROJECT_ROOT / "A_data" / "prepared_data" / table_name, columns=columns)
    if "TRADE_DT" not in df.columns:
        raise KeyError(f"{table_name} must contain TRADE_DT")
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["TRADE_DT"]).sort_values("TRADE_DT")
    df = df.drop_duplicates(subset=["TRADE_DT"], keep="last").set_index("TRADE_DT")
    return df.apply(pd.to_numeric, errors="coerce").astype("float64")


def _common_stock_tickers(*table_names: str) -> list[str]:
    column_sets = [set(_prepared_parquet_columns(table_name)) - {"TRADE_DT"} for table_name in table_names]
    return sorted(set.intersection(*column_sets))


def _load_stock_valuation_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = _common_stock_tickers(PB_TABLE, PE_TABLE, ADJ_CLOSE_TABLE)
    pb_panel = _load_wide_numeric_table(PB_TABLE, tickers)
    pe_panel = _load_wide_numeric_table(PE_TABLE, tickers)
    close_panel = _load_wide_numeric_table(ADJ_CLOSE_TABLE, tickers)
    common_index = pb_panel.index.intersection(pe_panel.index).intersection(close_panel.index).sort_values()
    return (
        pb_panel.reindex(common_index),
        pe_panel.reindex(common_index),
        close_panel.reindex(common_index),
    )


def _load_industry_valuation_panel(value_col: str, name: str) -> pd.DataFrame:
    series_list = [
        _read_valuation_series(file_name, value_col, Path(file_name).stem)
        for file_name in INDUSTRY_VALUATION_FILES
    ]
    panel = pd.concat(series_list, axis=1, sort=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.name = name
    return panel


def _cross_section_cv(panel: pd.DataFrame) -> pd.Series:
    positive = panel.where(panel > 0)
    mean = positive.mean(axis=1, skipna=True)
    std = positive.std(axis=1, skipna=True)
    return (std / mean.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _calc_low_high_pb_stats(
    pb_panel: pd.DataFrame,
    pe_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
) -> dict[str, pd.Series]:
    pb_positive = pb_panel.where(pb_panel > 0)
    pb_rank = pb_positive.rank(axis=1, pct=True)
    low_pb_mask = pb_rank <= 0.3
    high_pb_mask = pb_rank >= 0.7

    stock_ret = close_panel.pct_change(fill_method=None)
    low_pb_ret = stock_ret.where(low_pb_mask).mean(axis=1, skipna=True)
    high_pb_ret = stock_ret.where(high_pb_mask).mean(axis=1, skipna=True)
    low_pb_pe = pe_panel.where(low_pb_mask & (pe_panel > 0)).mean(axis=1, skipna=True)
    high_pb_pe = pe_panel.where(high_pb_mask & (pe_panel > 0)).mean(axis=1, skipna=True)

    return {
        "low_pb_ret": low_pb_ret,
        "high_pb_ret": high_pb_ret,
        "low_pb_pe": low_pb_pe,
        "high_pb_pe": high_pb_pe,
    }


def _rolling_compound_return(ret: pd.Series, window: int) -> pd.Series:
    return (1.0 + ret).rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0


def _calc_v151(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    raw = value_pb - growth_pb
    return calc_rolling_zscore(raw, window=504)


def _calc_v152() -> pd.Series:
    industry_pe = _load_industry_valuation_panel("市盈率TTM", "industry_pe")
    return calc_rolling_zscore(_cross_section_cv(industry_pe), window=504)


def _calc_v153() -> pd.Series:
    industry_pb = _load_industry_valuation_panel("市净率LF", "industry_pb")
    return calc_rolling_zscore(_cross_section_cv(industry_pb), window=504)


def _calc_v154(stats: dict[str, pd.Series]) -> pd.Series:
    raw = stats["low_pb_ret"] - stats["high_pb_ret"]
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v155(stats: dict[str, pd.Series]) -> pd.Series:
    raw = _rolling_compound_return(stats["low_pb_ret"], window=20)
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v156(stats: dict[str, pd.Series]) -> pd.Series:
    raw = _rolling_compound_return(stats["high_pb_ret"], window=20)
    return calc_rolling_zscore(raw, window=504)


def _calc_v157(stats: dict[str, pd.Series]) -> pd.Series:
    raw = stats["low_pb_pe"] - stats["high_pb_pe"]
    return calc_rolling_zscore(raw, window=504) * -1.0


def _calc_v158(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    raw_centered = (growth_pe - value_pe) - 27.05
    return calc_rolling_zscore(raw_centered, window=504) * -1.0


def generate_plan151_priceFactors9_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan151_records()

    growth_pe = _load_growth_valuation("市盈率TTM", "growth_pe_ttm")
    value_pe = _load_value_valuation("市盈率TTM", "value_pe_ttm")
    growth_pb = _load_growth_valuation("市净率LF", "growth_pb_lf")
    value_pb = _load_value_valuation("市净率LF", "value_pb_lf")
    pb_panel, pe_panel, close_panel = _load_stock_valuation_panels()
    low_high_stats = _calc_low_high_pb_stats(pb_panel, pe_panel, close_panel)

    _register_factor(raw_factor_df, factor_source_df, "V151_raw", _calc_v151(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V152_raw", _calc_v152())
    _register_factor(raw_factor_df, factor_source_df, "V153_raw", _calc_v153())
    _register_factor(raw_factor_df, factor_source_df, "V154_raw", _calc_v154(low_high_stats))
    _register_factor(raw_factor_df, factor_source_df, "V155_raw", _calc_v155(low_high_stats))
    _register_factor(raw_factor_df, factor_source_df, "V156_raw", _calc_v156(low_high_stats))
    _register_factor(raw_factor_df, factor_source_df, "V157_raw", _calc_v157(low_high_stats))
    _register_factor(raw_factor_df, factor_source_df, "V158_raw", _calc_v158(growth_pe, value_pe))

    missing_cols = [factor_id for factor_id in PLAN151_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors9 V151-V158 columns missing after generation: {missing_cols}")
    return factor_source_df.loc[:, PLAN151_FACTOR_IDS], records


def generate_extended_priceFactors9_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan143_factor_df, plan143_records = generate_priceFactors9_factors(data_df)
    plan151_factor_df, plan151_records = generate_plan151_priceFactors9_factors(data_df)
    factor_source_df = pd.concat([plan143_factor_df, plan151_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan143_records + plan151_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors9_factors(data_df)
    metadata = metadata_from_extended_priceFactors9_records(selected_records)
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
