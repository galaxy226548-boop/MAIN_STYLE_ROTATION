"""Price style factors from completed V191-V198 plan."""

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
    _register_factor,
    build_threshold_signal_ls_df,
    calc_llt,
    calc_rolling_zscore,
    load_benchmark_index,
    load_default_data,
    mount_factor_source_frame,
    normalize_trade_dt,
    prepared_data_dir,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "priceFactors12"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V191_V198.json"

FACTOR_IDS = [
    "V191",
    "V192",
    "V193",
    "V194",
    "V195",
    "V196",
    "V197",
    "V198",
]

A_INDEX_EOD_TABLE = "AIndexEODPrices.parquet"
ASTOCK_DAILY_TABLE = "Astockdaily.parquet"
MKT_PRICE_TABLE = "mktP.parquet"
GROWTH_COMPONENT_TABLE = "growth_factor_Fri.parquet"
VALUE_COMPONENT_TABLE = "value_factor_Fri.parquet"

CYCLE_INDEX = "399314.SZ"
NONCYCLE_INDEX = "399316.SZ"
MARKET_INDEX = "399370.SZ"
GROWTH_STYLE = "growth"
VALUE_STYLE = "value"

_A_INDEX_CLOSE_CACHE: dict[str, pd.Series] = {}
_STYLE_WEIGHT_CACHE: dict[str, dict[pd.Timestamp, pd.Series]] | None = None


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
        if factor_id in {"V191", "V192", "V193"}:
            _append_note(item, "本脚本使用 AIndexEODPrices.parquet 中 399314.SZ 与 399316.SZ 作为周期/非周期代理。")
        elif factor_id == "V194":
            _append_note(
                item,
                "Astockdaily.parquet 的 Ret 字段不像日收益率；本脚本使用 ChangeRatio 计算月度个股收益截面分化度。",
            )
        elif factor_id == "V195":
            _append_note(
                item,
                "Astockdaily.parquet 的 Ret 字段不像日收益率；本脚本使用 ChangeRatio 复合月度收益，并用 CirculatedMarketValue 月末截面分组。",
            )
        elif factor_id in {"V196", "V197", "V198"}:
            _append_note(
                item,
                "本脚本使用 growth_factor_Fri/value_factor_Fri 的成分权重、mktP.parquet 的 Dretwd 个股收益，并以 399370.SZ 收益作为 CAPM 市场收益代理。",
            )
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V191-V198 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors12_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].replace(".0", "").zfill(6)


def _as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


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


def _relative_cycle_strength() -> pd.Series:
    cycle_close = _load_a_index_close(CYCLE_INDEX)
    noncycle_close = _load_a_index_close(NONCYCLE_INDEX)
    aligned = pd.concat([cycle_close.rename("cycle"), noncycle_close.rename("noncycle")], axis=1, sort=True)
    aligned = aligned[(aligned["cycle"] > 0) & (aligned["noncycle"] > 0)]
    return -(np.log(aligned["cycle"]) - np.log(aligned["noncycle"]))


def _load_monthly_stock_returns() -> pd.Series:
    df = pd.read_parquet(
        prepared_data_dir / ASTOCK_DAILY_TABLE,
        columns=["TradingDate", "Symbol", "ChangeRatio"],
    )
    work = df.dropna(subset=["TradingDate", "Symbol"]).copy()
    work["date"] = pd.to_datetime(work["TradingDate"], errors="coerce").dt.normalize()
    work["month"] = work["date"].dt.to_period("M")
    work["ticker"] = work["Symbol"].map(_stock_code_key)
    work["ret"] = pd.to_numeric(work["ChangeRatio"], errors="coerce")
    work = work[work["month"].notna() & work["ticker"].ne("") & work["ret"].notna()].copy()
    monthly = work.groupby(["month", "ticker"], observed=True)["ret"].apply(lambda s: (1.0 + s).prod() - 1.0)
    monthly.index = monthly.index.set_names(["month", "ticker"])
    return monthly.astype("float64")


def _load_monthly_stock_returns_and_caps() -> pd.DataFrame:
    df = pd.read_parquet(
        prepared_data_dir / ASTOCK_DAILY_TABLE,
        columns=["TradingDate", "Symbol", "ChangeRatio", "CirculatedMarketValue"],
    )
    work = df.dropna(subset=["TradingDate", "Symbol"]).copy()
    work["date"] = pd.to_datetime(work["TradingDate"], errors="coerce").dt.normalize()
    work["month"] = work["date"].dt.to_period("M")
    work["ticker"] = work["Symbol"].map(_stock_code_key)
    work["ret"] = pd.to_numeric(work["ChangeRatio"], errors="coerce")
    work["mcap"] = pd.to_numeric(work["CirculatedMarketValue"], errors="coerce")
    work = work[work["date"].notna() & work["month"].notna() & work["ticker"].ne("")].copy()
    work = work.sort_values(["month", "ticker", "date"])

    monthly_ret = work.dropna(subset=["ret"]).groupby(["month", "ticker"], observed=True)["ret"].apply(
        lambda s: (1.0 + s).prod() - 1.0
    )
    month_end_cap = work.dropna(subset=["mcap"]).groupby(["month", "ticker"], observed=True)["mcap"].last()
    monthly = pd.concat([monthly_ret.rename("ret"), month_end_cap.rename("mcap")], axis=1).reset_index()
    monthly = monthly[monthly["ret"].notna() & monthly["mcap"].gt(0)].copy()
    return monthly


def _month_period_to_timestamp(index: pd.Index) -> pd.DatetimeIndex:
    return pd.PeriodIndex(index, freq="M").to_timestamp(how="end").normalize()


def _load_style_weights() -> dict[str, dict[pd.Timestamp, pd.Series]]:
    global _STYLE_WEIGHT_CACHE
    if _STYLE_WEIGHT_CACHE is not None:
        return _STYLE_WEIGHT_CACHE

    table_by_style = {
        GROWTH_STYLE: GROWTH_COMPONENT_TABLE,
        VALUE_STYLE: VALUE_COMPONENT_TABLE,
    }
    out: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    for style, table_name in table_by_style.items():
        df = pd.read_parquet(prepared_data_dir / table_name, columns=["holding_date", "component", "weight"])
        work = df.dropna(subset=["holding_date", "component", "weight"]).copy()
        work["holding_date"] = pd.to_datetime(work["holding_date"], errors="coerce").dt.normalize()
        work["ticker"] = work["component"].map(_stock_code_key)
        work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
        work = work[work["holding_date"].notna() & work["ticker"].ne("") & work["weight"].gt(0)].copy()

        by_date: dict[pd.Timestamp, pd.Series] = {}
        for dt, date_group in work.groupby("holding_date", sort=True):
            weights = date_group.groupby("ticker")["weight"].sum().astype("float64")
            weights = weights[weights > 0]
            if not weights.empty:
                by_date[pd.Timestamp(dt)] = weights / weights.sum()
        out[style] = by_date

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


def _load_style_stock_return_wide() -> pd.DataFrame:
    ticker_set = set(_all_style_tickers())
    df = pd.read_parquet(
        prepared_data_dir / MKT_PRICE_TABLE,
        columns=["Stkcd", "Trddt", "Dretwd"],
    )
    work = df.dropna(subset=["Stkcd", "Trddt"]).copy()
    work["ticker"] = work["Stkcd"].map(_stock_code_key)
    work = work[work["ticker"].isin(ticker_set)].copy()
    work["date"] = pd.to_datetime(work["Trddt"], errors="coerce").dt.normalize()
    work["ret"] = pd.to_numeric(work["Dretwd"], errors="coerce")
    work = work.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    ret_wide = work.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").sort_index()
    return ret_wide.reindex(columns=_all_style_tickers()).astype("float64")


def _style_composite(metric_df: pd.DataFrame, style: str) -> pd.Series:
    by_date = _load_style_weights()[style]
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
        aligned_weights = weights.reindex(tickers).astype("float64")
        weighted_sum = block.mul(aligned_weights, axis=1).sum(axis=1, min_count=1)
        valid_weight = block.notna().mul(aligned_weights, axis=1).sum(axis=1)
        out.loc[mask] = weighted_sum / valid_weight.replace(0.0, np.nan)
    return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned = pd.concat([numerator.rename("numerator"), denominator.rename("denominator")], axis=1, sort=True)
    ratio = aligned["numerator"] / aligned["denominator"].replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _rolling_beta_df(ret_wide: pd.DataFrame, market_ret: pd.Series, window: int) -> pd.DataFrame:
    aligned_market = market_ret.reindex(ret_wide.index).astype("float64")
    cov = ret_wide.rolling(window=window, min_periods=window).cov(aligned_market)
    var = aligned_market.rolling(window=window, min_periods=window).var()
    return cov.div(var.replace(0.0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan)


def _calc_v191() -> pd.Series:
    return calc_rolling_zscore(_relative_cycle_strength(), window=504)


def _calc_v192() -> pd.Series:
    rn_adj = _relative_cycle_strength()
    beta1_raw = calc_llt(rn_adj, d=30).diff()
    return calc_rolling_zscore(beta1_raw, window=504)


def _calc_v193() -> pd.Series:
    rn_adj = _relative_cycle_strength()
    beta1_adj = calc_llt(rn_adj, d=30).diff()
    beta2_raw = calc_llt(beta1_adj, d=30).diff()
    return calc_rolling_zscore(beta2_raw, window=504)


def _calc_v194() -> pd.Series:
    monthly_ret = _load_monthly_stock_returns()
    divergence = monthly_ret.groupby(level="month", observed=True).std()
    raw = divergence.pct_change(1).replace([np.inf, -np.inf], np.nan)
    raw.index = _month_period_to_timestamp(raw.index)
    return calc_rolling_zscore(raw, window=24, min_periods=12)


def _calc_v195() -> pd.Series:
    monthly = _load_monthly_stock_returns_and_caps()
    monthly["mcap_decile"] = monthly.groupby("month", observed=True)["mcap"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
    )
    small_ret = monthly[monthly["mcap_decile"].eq(0)].groupby("month", observed=True)["ret"].median()
    large_ret = monthly[monthly["mcap_decile"].eq(9)].groupby("month", observed=True)["ret"].median()
    raw = (small_ret - large_ret).replace([np.inf, -np.inf], np.nan)
    raw.index = _month_period_to_timestamp(raw.index)
    return calc_rolling_zscore(raw, window=24, min_periods=12)


def _calc_beta_ratio_factor(window: int) -> pd.Series:
    ret_wide = _load_style_stock_return_wide()
    market_ret = _load_a_index_close(MARKET_INDEX).pct_change(fill_method=None)
    beta_df = _rolling_beta_df(ret_wide, market_ret, window=window)
    growth_beta = _style_composite(beta_df, GROWTH_STYLE)
    value_beta = _style_composite(beta_df, VALUE_STYLE)
    raw = _safe_ratio(growth_beta, value_beta) - 1.0
    return calc_rolling_zscore(raw, window=504)


def generate_priceFactors12_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    _register_factor(raw_factor_df, factor_source_df, "V191_raw", _calc_v191())
    _register_factor(raw_factor_df, factor_source_df, "V192_raw", _calc_v192())
    _register_factor(raw_factor_df, factor_source_df, "V193_raw", _calc_v193())
    _register_factor(raw_factor_df, factor_source_df, "V194_raw", _calc_v194())
    _register_factor(raw_factor_df, factor_source_df, "V195_raw", _calc_v195())
    _register_factor(raw_factor_df, factor_source_df, "V196_raw", _calc_beta_ratio_factor(40))
    _register_factor(raw_factor_df, factor_source_df, "V197_raw", _calc_beta_ratio_factor(80))
    _register_factor(raw_factor_df, factor_source_df, "V198_raw", _calc_beta_ratio_factor(100))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors12 columns missing after generation: {missing_cols}")
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

    factor_source_df, selected_records = generate_priceFactors12_factors(data_df)
    metadata = metadata_from_priceFactors12_records(selected_records)
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


PLAN199_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V199_V206.json"
PLAN199_FACTOR_IDS = [
    "V199",
    "V200",
    "V201",
    "V202",
    "V203",
    "V204",
    "V205",
    "V206",
]
EXTENDED_FACTOR_IDS = FACTOR_IDS + PLAN199_FACTOR_IDS

INDUSTRY_INDEX_TABLE = "industry_indice.parquet"

FIN_REAL_ESTATE_COLS = [
    "中信行业指数:银行",
    "中信行业指数:非银行金融",
    "中信行业指数:房地产",
]
CONSUMER_COLS = [
    "中信行业指数:食品饮料",
    "中信行业指数:医药",
    "中信行业指数:家电",
    "中信行业指数:商贸零售",
    "中信行业指数:农林牧渔",
    "中信行业指数:纺织服装",
    "中信行业指数:消费者服务",
]
CYCLE_COLS = [
    "中信行业指数:煤炭",
    "中信行业指数:有色金属",
    "中信行业指数:钢铁",
    "中信行业指数:基础化工",
    "中信行业指数:建筑",
    "中信行业指数:建材",
    "中信行业指数:石油石化",
]
TMT_COLS = [
    "中信行业指数:计算机",
    "中信行业指数:通信",
    "中信行业指数:传媒",
    "中信行业指数:电子",
    "中信行业指数:电力设备及新能源",
]


def _load_plan199_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN199_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{PLAN199_PATH} must contain a top-level record list")

    records: list[dict[str, object]] = []
    source_file = str(PLAN199_PATH.relative_to(PROJECT_ROOT))
    wanted = set(PLAN199_FACTOR_IDS)

    for record in payload:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in wanted:
            continue
        item = dict(record)
        item["_source_file"] = source_file
        item["_source_sheet"] = "records"
        item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
        if factor_id in {"V199", "V200"}:
            _append_note(
                item,
                "本脚本沿用 V196-V198 的风格成分权重与 mktP.parquet Dretwd，并以 399370.SZ 收益作为 CAPM 市场收益代理计算滚动R²。",
            )
        elif factor_id == "V201":
            _append_note(
                item,
                "Astockdaily.parquet 的 Ret 字段不像日收益率；本脚本使用 ChangeRatio 计算8日累计超额收益，并用 CirculatedMarketValue 做风格成分内流通市值加权扩散指数。",
            )
        elif factor_id in {"V202", "V203", "V204", "V205", "V206"}:
            _append_note(
                item,
                "本脚本使用 industry_indice.parquet 的中信行业指数点位计算日收益，再按季度汇总为四大板块收益排名变化；原始基金重仓超配比例数据本地不可得。",
            )
        records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in PLAN199_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V199-V206 plan missing implemented records: {missing}")
    return sorted(records, key=lambda record: PLAN199_FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_extended_priceFactors12_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return metadata_from_priceFactors12_records(records)


def _rolling_rsq_df(ret_wide: pd.DataFrame, market_ret: pd.Series, window: int) -> pd.DataFrame:
    aligned_market = market_ret.reindex(ret_wide.index).astype("float64")
    cov = ret_wide.rolling(window=window, min_periods=window).cov(aligned_market)
    var_y = ret_wide.rolling(window=window, min_periods=window).var()
    var_x = aligned_market.rolling(window=window, min_periods=window).var()
    rsq = cov.pow(2).div(var_y.mul(var_x, axis=0).replace(0.0, np.nan))
    return rsq.clip(lower=0.0, upper=1.0).replace([np.inf, -np.inf], np.nan)


def _calc_rsq_ratio_factor(window: int) -> pd.Series:
    ret_wide = _load_style_stock_return_wide()
    market_ret = _load_a_index_close(MARKET_INDEX).pct_change(fill_method=None)
    rsq_df = _rolling_rsq_df(ret_wide, market_ret, window=window)
    growth_rsq = _style_composite(rsq_df, GROWTH_STYLE)
    value_rsq = _style_composite(rsq_df, VALUE_STYLE)
    raw = _safe_ratio(growth_rsq, value_rsq) - 1.0
    return calc_rolling_zscore(raw, window=504)


def _load_diffusion_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ticker_set = set(_all_style_tickers())
    df = pd.read_parquet(
        prepared_data_dir / ASTOCK_DAILY_TABLE,
        columns=["TradingDate", "Symbol", "ChangeRatio", "CirculatedMarketValue"],
    )
    work = df.dropna(subset=["TradingDate", "Symbol"]).copy()
    work["date"] = pd.to_datetime(work["TradingDate"], errors="coerce").dt.normalize()
    work["ticker"] = work["Symbol"].map(_stock_code_key)
    work["ret"] = pd.to_numeric(work["ChangeRatio"], errors="coerce")
    work["mcap"] = pd.to_numeric(work["CirculatedMarketValue"], errors="coerce")
    work = work[work["date"].notna() & work["ticker"].ne("")].copy()
    market_ret = work.groupby("date", observed=True)["ret"].mean().sort_index().astype("float64")

    style_work = work[work["ticker"].isin(ticker_set)].copy()
    ret_wide = style_work.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").sort_index()
    mcap_wide = style_work.pivot_table(index="date", columns="ticker", values="mcap", aggfunc="last").sort_index()
    tickers = _all_style_tickers()
    ret_wide = ret_wide.reindex(columns=tickers).astype("float64")
    mcap_wide = mcap_wide.reindex(index=ret_wide.index, columns=tickers).astype("float64")
    return ret_wide, mcap_wide, market_ret


def _style_mcap_weighted_indicator(indicator_df: pd.DataFrame, mcap_df: pd.DataFrame, style: str) -> pd.Series:
    by_date = _load_style_weights()[style]
    component_dates = pd.DatetimeIndex(sorted(by_date.keys()))
    out = pd.Series(np.nan, index=indicator_df.index, dtype="float64")

    for pos, comp_date in enumerate(component_dates):
        next_date = component_dates[pos + 1] if pos + 1 < len(component_dates) else pd.Timestamp.max
        mask = (indicator_df.index >= comp_date) & (indicator_df.index < next_date)
        if not mask.any():
            continue
        tickers = [ticker for ticker in by_date[pd.Timestamp(comp_date)].index if ticker in indicator_df.columns]
        if not tickers:
            continue
        indicator_block = indicator_df.loc[mask, tickers]
        mcap_block = mcap_df.loc[mask, tickers].where(lambda x: x > 0)
        valid_mcap = mcap_block.where(indicator_block.notna())
        numerator = indicator_block.mul(valid_mcap, axis=0).sum(axis=1, min_count=1)
        denominator = valid_mcap.sum(axis=1, min_count=1)
        out.loc[mask] = numerator / denominator.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _calc_v201() -> pd.Series:
    ret_wide, mcap_wide, market_ret = _load_diffusion_inputs()
    excess = ret_wide.sub(market_ret.reindex(ret_wide.index), axis=0)
    cum_excess = excess.rolling(window=8, min_periods=8).sum()
    indicator = pd.DataFrame(np.nan, index=cum_excess.index, columns=cum_excess.columns, dtype="float64")
    indicator.loc[:, :] = (cum_excess > 0).astype("float64")
    indicator = indicator.where(cum_excess.notna())
    growth_di = _style_mcap_weighted_indicator(indicator, mcap_wide, GROWTH_STYLE)
    value_di = _style_mcap_weighted_indicator(indicator, mcap_wide, VALUE_STYLE)
    rel_di = _safe_ratio(growth_di, value_di)
    raw = rel_di.rolling(15, min_periods=15).mean() - rel_di.rolling(60, min_periods=60).mean()
    return calc_rolling_zscore(raw, window=504)


def _load_industry_index_frame() -> pd.DataFrame:
    df = pd.read_parquet(prepared_data_dir / INDUSTRY_INDEX_TABLE)
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


def _mean_checked(df: pd.DataFrame, columns: list[str], name: str) -> pd.Series:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{INDUSTRY_INDEX_TABLE} missing columns for {name}: {missing}")
    return df.loc[:, columns].mean(axis=1, skipna=True).rename(name)


def _quarterly_sector_returns() -> pd.DataFrame:
    industry_close = _load_industry_index_frame()
    industry_ret = industry_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    sector_ret = pd.DataFrame(
        {
            "金融地产": _mean_checked(industry_ret, FIN_REAL_ESTATE_COLS, "金融地产"),
            "消费": _mean_checked(industry_ret, CONSUMER_COLS, "消费"),
            "周期": _mean_checked(industry_ret, CYCLE_COLS, "周期"),
            "TMT": _mean_checked(industry_ret, TMT_COLS, "TMT"),
        }
    )
    return sector_ret.resample("QE").sum(min_count=1).dropna(how="all")


def _calc_rotation_rank_change(method: str = "square", sector: str | None = None) -> pd.Series:
    quarterly_ret = _quarterly_sector_returns()
    ranks = quarterly_ret.rank(axis=1)
    rank_diff = ranks - ranks.shift(1)
    if sector is not None:
        raw = rank_diff[sector].abs()
    elif method == "square":
        raw = rank_diff.pow(2).sum(axis=1, min_count=1)
    elif method == "absolute":
        raw = rank_diff.abs().sum(axis=1, min_count=1)
    else:
        raise ValueError(f"Unsupported rotation method: {method!r}")
    return calc_rolling_zscore(raw.replace([np.inf, -np.inf], np.nan), window=20, min_periods=8)


def generate_plan199_priceFactors12_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan199_records()

    _register_factor(raw_factor_df, factor_source_df, "V199_raw", _calc_rsq_ratio_factor(40))
    _register_factor(raw_factor_df, factor_source_df, "V200_raw", _calc_rsq_ratio_factor(80))
    _register_factor(raw_factor_df, factor_source_df, "V201_raw", _calc_v201())
    _register_factor(raw_factor_df, factor_source_df, "V202_raw", _calc_rotation_rank_change("square"))
    _register_factor(raw_factor_df, factor_source_df, "V203_raw", _calc_rotation_rank_change("absolute"))
    _register_factor(raw_factor_df, factor_source_df, "V204_raw", _calc_rotation_rank_change(sector="金融地产"))
    _register_factor(raw_factor_df, factor_source_df, "V205_raw", _calc_rotation_rank_change(sector="周期"))
    _register_factor(raw_factor_df, factor_source_df, "V206_raw", _calc_rotation_rank_change(sector="TMT"))

    missing_cols = [factor_id for factor_id in PLAN199_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors12 V199-V206 columns missing after generation: {missing_cols}")
    return factor_source_df.loc[:, PLAN199_FACTOR_IDS], records


def generate_extended_priceFactors12_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    plan191_factor_df, plan191_records = generate_priceFactors12_factors(data_df)
    plan199_factor_df, plan199_records = generate_plan199_priceFactors12_factors(data_df)
    factor_source_df = pd.concat([plan191_factor_df, plan199_factor_df], axis=1, sort=True)
    return factor_source_df.loc[:, EXTENDED_FACTOR_IDS], plan191_records + plan199_records


def main_extended() -> None:
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_extended_priceFactors12_factors(data_df)
    metadata = metadata_from_extended_priceFactors12_records(selected_records)
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
