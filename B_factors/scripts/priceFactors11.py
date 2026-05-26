"""Price style factors from completed V175-V182 plan."""

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


OUTPUT_PREFIX = "priceFactors11"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V175_V182.json"

FACTOR_IDS = [
    "V175",
    "V176",
    "V177",
    "V178",
    "V179",
    "V180",
    "V181",
    "V182",
]

GROWTH_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx"
VALUE_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx"
ALL_A_VALUATION_FILE = "881001.WI-历史PE-PB-20260518.xlsx"
BANK_VALUATION_FILE = "801811.SI-历史PE-PB-20260509.xlsx"
NONBANK_VALUATION_FILE = "801813.SI-历史PE-PB-20260509.xlsx"
INDEX_STATEMENT_FILE = "IndexStatement.xlsx"

GROWTH_STYLE_INDEX = "399370.SZ"


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
        if factor_id in {"V175", "V176", "V177", "V180", "V181"}:
            _append_note(item, "本脚本使用本地成长/价值风格指数估值文件的日频估值字段。")
        elif factor_id == "V178":
            _append_note(
                item,
                "本脚本使用 IndexStatement.xlsx 中 399370 国证成长指数的归属母公司股东的权益作为流通股本不可得时的规模代理，并按 PubDate 挂载。",
            )
        elif factor_id == "V179":
            _append_note(item, "本脚本使用 IndexStatement.xlsx 中 399370 国证成长指数 EPS，并按 PubDate 挂载。")
        elif factor_id == "V182":
            _append_note(
                item,
                "本脚本使用 801811.SI 与 801813.SI 历史PE-PB文件的市盈率TTM算术平均代理大金融PE，并与 Wind全A PE 相除。",
            )
        if factor_id == "V176":
            _append_note(item, "原计划 notes 指出需翻转方向以满足 >0=>growth，本实现采用 factor = 0.5 - expanding_quantile_rank。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V175-V182 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors11_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
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
    return text.split(".")[0].zfill(6)


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


def _load_index_statement_series(index_code: str, value_col: str, name: str) -> pd.Series:
    required = ["Indexcd", "Date", "PubDate", value_col]
    df = load_prepared_table(INDEX_STATEMENT_FILE).loc[:, required].copy()
    df["index_code"] = df["Indexcd"].map(_index_code_key)
    df = df[df["index_code"].eq(_index_code_key(index_code))].copy()
    if df.empty:
        raise ValueError(f"{INDEX_STATEMENT_FILE} missing Indexcd={index_code!r}")
    df["PubDate"] = pd.to_datetime(df["PubDate"], errors="coerce")
    df = df[df["PubDate"].notna()].sort_values(["PubDate", "Date"])
    values = _as_numeric(df[value_col])
    return _as_float_series(values, df["PubDate"], name)


def _safe_inverse(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return (1.0 / positive).replace([np.inf, -np.inf], np.nan)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned = pd.concat([numerator.rename("numerator"), denominator.rename("denominator")], axis=1, sort=True)
    ratio = aligned["numerator"] / aligned["denominator"].replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _expanding_quantile_rank(series: pd.Series, min_periods: int = 60) -> pd.Series:
    s = series.astype("float64").sort_index()
    return s.expanding(min_periods=min_periods).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _calc_v175(growth_ps: pd.Series, value_ps: pd.Series) -> pd.Series:
    spread = growth_ps - value_ps
    ma_short = spread.rolling(window=20, min_periods=20).mean()
    ma_long = spread.rolling(window=120, min_periods=120).mean()
    return calc_rolling_zscore(ma_short - ma_long, window=504)


def _calc_v176(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    bp_value = _safe_inverse(value_pb)
    bp_growth = _safe_inverse(growth_pb)
    rank = _expanding_quantile_rank(bp_value - bp_growth, min_periods=60)
    return 0.5 - rank


def _calc_v177(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    bp_growth = _safe_inverse(growth_pb)
    bp_value = _safe_inverse(value_pb)
    rank = _expanding_quantile_rank(bp_growth - bp_value, min_periods=60)
    return 0.5 - rank


def _calc_v178() -> pd.Series:
    equity_proxy = _load_index_statement_series(GROWTH_STYLE_INDEX, "归属母公司股东的权益", "growth_equity_proxy")
    equity_proxy = equity_proxy.where(equity_proxy > 0)
    raw = calc_rolling_zscore(equity_proxy, window=20, min_periods=8)
    return raw * -1.0


def _calc_v179() -> pd.Series:
    eps = _load_index_statement_series(GROWTH_STYLE_INDEX, "EPS", "growth_eps")
    return calc_rolling_zscore(eps, window=20, min_periods=8)


def _calc_v180(growth_dividend_yield: pd.Series) -> pd.Series:
    raw = calc_rolling_zscore(growth_dividend_yield, window=504)
    return raw * -1.0


def _calc_v181(growth_pb: pd.Series) -> pd.Series:
    quarterly_pb = growth_pb.dropna().resample("QE").last()
    spb = calc_rolling_zscore(quarterly_pb, window=20, min_periods=8)
    return (spb - spb.shift(1)) * -1.0


def _calc_v182(bank_pe: pd.Series, nonbank_pe: pd.Series, all_a_pe: pd.Series) -> pd.Series:
    finance_pe = pd.concat([bank_pe.rename("bank"), nonbank_pe.rename("nonbank")], axis=1, sort=True).mean(axis=1)
    raw = _safe_ratio(finance_pe, all_a_pe)
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors11_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_ps = _read_valuation_series(GROWTH_VALUATION_FILE, "市销率TTM", "growth_ps_ttm")
    value_ps = _read_valuation_series(VALUE_VALUATION_FILE, "市销率TTM", "value_ps_ttm")
    growth_pb = _read_valuation_series(GROWTH_VALUATION_FILE, "市净率LF", "growth_pb_lf")
    value_pb = _read_valuation_series(VALUE_VALUATION_FILE, "市净率LF", "value_pb_lf")
    growth_dividend_yield = _read_valuation_series(GROWTH_VALUATION_FILE, "股息率", "growth_dividend_yield")
    all_a_pe = _read_valuation_series(ALL_A_VALUATION_FILE, "市盈率TTM", "all_a_pe_ttm")
    bank_pe = _read_valuation_series(BANK_VALUATION_FILE, "市盈率TTM", "bank_pe_ttm")
    nonbank_pe = _read_valuation_series(NONBANK_VALUATION_FILE, "市盈率TTM", "nonbank_pe_ttm")

    _register_factor(raw_factor_df, factor_source_df, "V175_raw", _calc_v175(growth_ps, value_ps))
    _register_factor(raw_factor_df, factor_source_df, "V176_raw", _calc_v176(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V177_raw", _calc_v177(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V178_raw", _calc_v178())
    _register_factor(raw_factor_df, factor_source_df, "V179_raw", _calc_v179())
    _register_factor(raw_factor_df, factor_source_df, "V180_raw", _calc_v180(growth_dividend_yield))
    _register_factor(raw_factor_df, factor_source_df, "V181_raw", _calc_v181(growth_pb))
    _register_factor(raw_factor_df, factor_source_df, "V182_raw", _calc_v182(bank_pe, nonbank_pe, all_a_pe))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors11 columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_priceFactors11_factors(data_df)
    metadata = metadata_from_priceFactors11_records(selected_records)
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


# Extended factors from working_multiple_factors_plan_completed_V183_V190.json.

PLAN183_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V183_V190.json"
PLAN183_FACTOR_IDS = [
    "V183",
    "V184",
    "V185",
    "V186",
    "V187",
    "V188",
    "V189",
    "V190",
]
EXTENDED_FACTOR_IDS = FACTOR_IDS + PLAN183_FACTOR_IDS

INDEX_EOD_TABLE = "index_eod.parquet"
INDUSTRY_INDEX_TABLE = "industry_indice.parquet"

CONSUMER_INDUSTRY_COLS = [
    "中信行业指数:食品饮料",
    "中信行业指数:家电",
    "中信行业指数:商贸零售",
    "中信行业指数:纺织服装",
    "中信行业指数:农林牧渔",
    "中信行业指数:消费者服务",
    "中信行业指数:医药",
]
FINANCE_INDUSTRY_COLS = [
    "中信行业指数:银行",
    "中信行业指数:非银行金融",
]


def _load_plan183_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN183_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{PLAN183_PATH} must contain a top-level record list")

    records: list[dict[str, object]] = []
    source_file = str(PLAN183_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN183_FACTOR_IDS)

    for record in payload:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in wanted:
            continue
        item = dict(record)
        item["_source_file"] = source_file
        item["_source_sheet"] = "records"
        item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
        if factor_id in {"V183", "V186"}:
            _append_note(item, "本地缺少海通自建消费/非消费PE，本脚本使用成长指数PE减价值指数PE作为可落地代理。")
        elif factor_id in {"V184", "V187"}:
            _append_note(item, "本地缺少海通自建消费/非消费PB，本脚本使用成长指数PB减价值指数PB作为可落地代理。")
        elif factor_id == "V185":
            _append_note(item, "本脚本使用 industry_indice.parquet 中消费类中信行业指数均值与其余中信行业指数均值的20日收益差。")
        elif factor_id == "V188":
            _append_note(item, "本脚本使用 index_eod.parquet 的中文字段：交易所指数代码、收盘指数；000903代理中证100，000905代理中证500。")
        elif factor_id == "V189":
            _append_note(item, "本地缺少FTSE100，本脚本使用 industry_indice.parquet 中恒生指数的22日收益作为港股外盘代理，并按原方向乘以-1。")
        elif factor_id == "V190":
            _append_note(item, "本脚本使用 industry_indice.parquet 中银行+非银行金融均值代理金融指数，其余中信行业指数均值代理非金融指数。")
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN183_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V183-V190 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN183_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors11_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return metadata_from_priceFactors11_records(records)


def _load_industry_index_frame() -> pd.DataFrame:
    df = load_prepared_table(INDUSTRY_INDEX_TABLE)
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((col for col in ["date", "日期", "交易日期"] if col in df.columns), None)
        if date_col is None:
            raise KeyError(f"{INDUSTRY_INDEX_TABLE} must have a DatetimeIndex or a date column")
        df.index = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col])
    df.index = pd.to_datetime(df.index).normalize()
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.apply(pd.to_numeric, errors="coerce").astype("float64")


def _industry_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("中信行业指数:")]


def _mean_price_series(df: pd.DataFrame, columns: list[str], name: str) -> pd.Series:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{INDUSTRY_INDEX_TABLE} missing columns for {name}: {missing}")
    return df.loc[:, columns].mean(axis=1, skipna=True).rename(name)


def _load_index_eod_close(index_code: str, name: str) -> pd.Series:
    df = load_prepared_table(INDEX_EOD_TABLE)
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((col for col in ["date", "日期", "交易日期", "Idxtrd01"] if col in df.columns), None)
        if date_col is None:
            raise KeyError(f"{INDEX_EOD_TABLE} must have a DatetimeIndex or a date column")
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.to_datetime(df.index)
    code_col = "交易所指数代码" if "交易所指数代码" in df.columns else "Indexcd"
    close_col = "收盘指数" if "收盘指数" in df.columns else "Idxtrd05"
    if code_col not in df.columns or close_col not in df.columns:
        raise KeyError(f"{INDEX_EOD_TABLE} must contain index code and close columns; available={list(df.columns)}")
    out = pd.DataFrame(
        {
            "date": dates,
            "index_code": df[code_col].map(_index_code_key),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    target = _index_code_key(index_code)
    out = out[out["index_code"].eq(target) & out["date"].notna()].copy()
    if out.empty:
        raise ValueError(f"{INDEX_EOD_TABLE} missing Indexcd={index_code!r}")
    return _as_float_series(out["close"], out["date"], name)


def _calc_v183(growth_pe: pd.Series, value_pe: pd.Series) -> pd.Series:
    return calc_rolling_zscore(growth_pe - value_pe, window=504)


def _calc_v184(growth_pb: pd.Series, value_pb: pd.Series) -> pd.Series:
    return calc_rolling_zscore(growth_pb - value_pb, window=504)


def _calc_v185(industry_df: pd.DataFrame) -> pd.Series:
    industry_cols = _industry_cols(industry_df)
    nonconsumer_cols = [col for col in industry_cols if col not in CONSUMER_INDUSTRY_COLS]
    consume_close = _mean_price_series(industry_df, CONSUMER_INDUSTRY_COLS, "consume_close")
    nonconsume_close = _mean_price_series(industry_df, nonconsumer_cols, "nonconsume_close")
    raw = consume_close.pct_change(20) - nonconsume_close.pct_change(20)
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=504)


def _calc_v188(csi100_close: pd.Series, csi500_close: pd.Series) -> pd.Series:
    raw = csi100_close.pct_change(10) - csi500_close.pct_change(10)
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=504) * -1.0


def _calc_v189(industry_df: pd.DataFrame) -> pd.Series:
    if "恒生指数" not in industry_df.columns:
        raise KeyError(f"{INDUSTRY_INDEX_TABLE} missing 恒生指数 for V189 FTSE100 proxy")
    raw = industry_df["恒生指数"].pct_change(22)
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=504) * -1.0


def _calc_v190(industry_df: pd.DataFrame) -> pd.Series:
    industry_cols = _industry_cols(industry_df)
    nonfinance_cols = [col for col in industry_cols if col not in FINANCE_INDUSTRY_COLS]
    finance_close = _mean_price_series(industry_df, FINANCE_INDUSTRY_COLS, "finance_close")
    nonfinance_close = _mean_price_series(industry_df, nonfinance_cols, "nonfinance_close")
    raw = np.log(finance_close.where(finance_close > 0)).diff() - np.log(nonfinance_close.where(nonfinance_close > 0)).diff()
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=504) * -1.0


def generate_plan183_priceFactors11_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan183_records()

    growth_pe = _read_valuation_series(GROWTH_VALUATION_FILE, "市盈率TTM", "growth_pe_ttm")
    value_pe = _read_valuation_series(VALUE_VALUATION_FILE, "市盈率TTM", "value_pe_ttm")
    growth_pb = _read_valuation_series(GROWTH_VALUATION_FILE, "市净率LF", "growth_pb_lf")
    value_pb = _read_valuation_series(VALUE_VALUATION_FILE, "市净率LF", "value_pb_lf")
    industry_df = _load_industry_index_frame()
    csi100_close = _load_index_eod_close("000903", "csi100_close")
    csi500_close = _load_index_eod_close("000905", "csi500_close")

    _register_factor(raw_factor_df, factor_source_df, "V183_raw", _calc_v183(growth_pe, value_pe))
    _register_factor(raw_factor_df, factor_source_df, "V184_raw", _calc_v184(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V185_raw", _calc_v185(industry_df))
    _register_factor(raw_factor_df, factor_source_df, "V186_raw", _calc_v183(growth_pe, value_pe))
    _register_factor(raw_factor_df, factor_source_df, "V187_raw", _calc_v184(growth_pb, value_pb))
    _register_factor(raw_factor_df, factor_source_df, "V188_raw", _calc_v188(csi100_close, csi500_close))
    _register_factor(raw_factor_df, factor_source_df, "V189_raw", _calc_v189(industry_df))
    _register_factor(raw_factor_df, factor_source_df, "V190_raw", _calc_v190(industry_df))

    missing_cols = [factor_id for factor_id in PLAN183_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors11 V183-V190 columns missing after generation: {missing_cols}")
    return factor_source_df.loc[:, PLAN183_FACTOR_IDS], records


def generate_extended_priceFactors11_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan175_factor_df, plan175_records = generate_priceFactors11_factors(data_df)
    plan183_factor_df, plan183_records = generate_plan183_priceFactors11_factors(data_df)
    factor_source_df = pd.concat([plan175_factor_df, plan183_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan175_records + plan183_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors11_factors(data_df)
    metadata = metadata_from_extended_priceFactors11_records(selected_records)
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
