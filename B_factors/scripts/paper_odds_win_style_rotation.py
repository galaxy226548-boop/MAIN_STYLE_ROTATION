"""Factors for 如何从赔率和胜率看成长价值轮动——市场风格轮动系列."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import (
    _data_month_end_series,
    _load_china_macro_series,
    _register_factor,
    _rolling_quantile_rank_year,
    normalize_trade_dt,
    prepared_data_dir,
    read_prepared_series,
)


PAPER_ID = "如何从赔率和胜率看成长价值轮动——市场风格轮动系列"
GROWTH_INDEX_CODE = "399370.SZ"
VALUE_INDEX_CODE = "399371.SZ"
BASE_FACTOR_IDS = ["I001", "I002", "G001", "G002", "P001", "V001", "V002"]
FACTOR_IDS = [*BASE_FACTOR_IDS, "W001", "W002"]


def _calc_return(price: pd.Series, window: int) -> pd.Series:
    return price / price.shift(window) - 1


def _load_long_term_loan_yoy() -> pd.Series:
    yoy_col = "金融机构:人民币:中长期贷款余额:同比"
    balance_col = "中国:金融机构各项贷款余额:中长期:人民币"

    try:
        return read_prepared_series("macro_monthly.parquet", yoy_col)
    except KeyError:
        balance = read_prepared_series("macro_monthly.parquet", balance_col)
        return balance / balance.shift(12) - 1


def _load_style_components() -> dict[str, dict[pd.Timestamp, list[str]]]:
    comp = pd.read_parquet(prepared_data_dir / "IndexComponents.parquet")
    comp["TRADE_DT"] = normalize_trade_dt(comp["TRADE_DT"])
    comp = comp[comp["TRADE_DT"].notna()].copy()
    comp["S_INFO_WINDCODE"] = comp["S_INFO_WINDCODE"].astype(str).str.strip()
    comp["ticker"] = comp["S_CON_WINDCODE"].astype(str).str.slice(0, 6).str.zfill(6)

    out: dict[str, dict[pd.Timestamp, list[str]]] = {}
    for index_code in [GROWTH_INDEX_CODE, VALUE_INDEX_CODE]:
        sub = comp[comp["S_INFO_WINDCODE"].eq(index_code)]
        out[index_code] = {
            dt: sorted(group["ticker"].dropna().unique().tolist())
            for dt, group in sub.groupby("TRADE_DT")
        }
        if not out[index_code]:
            raise ValueError(f"IndexComponents.parquet 中找不到 {index_code} 成分股")
    return out


def _strong_ratio_diff(data_index: pd.DatetimeIndex) -> pd.Series:
    components = _load_style_components()
    all_tickers = sorted(
        {
            ticker
            for by_date in components.values()
            for tickers in by_date.values()
            for ticker in tickers
        }
    )

    mkt = pd.read_parquet(
        prepared_data_dir / "mktP.parquet",
        columns=["Stkcd", "Trddt", "Clsprc"],
    )
    mkt["Trddt"] = normalize_trade_dt(mkt["Trddt"])
    mkt = mkt[mkt["Trddt"].notna()].copy()
    mkt["Stkcd"] = mkt["Stkcd"].astype(str).str.zfill(6)
    mkt = mkt[mkt["Stkcd"].isin(all_tickers)]

    close = (
        mkt.pivot_table(index="Trddt", columns="Stkcd", values="Clsprc", aggfunc="last")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    strong = close.rolling(5, min_periods=5).mean() > close.rolling(20, min_periods=20).mean()

    component_dates = {
        index_code: pd.DatetimeIndex(sorted(by_date.keys()))
        for index_code, by_date in components.items()
    }
    result = pd.Series(np.nan, index=data_index, dtype="float64")
    for dt in data_index:
        if dt not in strong.index:
            continue
        row = strong.loc[dt]
        ratios = {}
        for index_code in [GROWTH_INDEX_CODE, VALUE_INDEX_CODE]:
            dates = component_dates[index_code]
            loc = dates.searchsorted(dt, side="right") - 1
            if loc < 0:
                continue
            tickers = [ticker for ticker in components[index_code][dates[loc]] if ticker in strong.columns]
            if tickers:
                ratios[index_code] = row[tickers].mean(skipna=True)
        if GROWTH_INDEX_CODE in ratios and VALUE_INDEX_CODE in ratios:
            result.at[dt] = ratios[GROWTH_INDEX_CODE] - ratios[VALUE_INDEX_CODE]
    return result


def _direction_score(series: pd.Series, bar: float) -> pd.Series:
    score = pd.Series(np.nan, index=series.index, dtype="float64")
    score.loc[series > bar] = 1.0
    score.loc[series < bar] = -1.0
    score.loc[series == bar] = 0.0
    return score


def generate_paper_odds_win_style_rotation_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)

    cn10y = read_prepared_series("rate_daily.parquet", "中债国债到期收益率:10年")
    i001 = 1 - _rolling_quantile_rank_year(cn10y, year=3)
    _register_factor(raw_factor_df, factor_source_df, "I001_raw", i001)

    us6m = read_prepared_series("rate_daily.parquet", "美国:国债收益率:6个月")
    i002 = 1 - _rolling_quantile_rank_year(us6m, year=3)
    _register_factor(raw_factor_df, factor_source_df, "I002_raw", i002)

    pmi = read_prepared_series("macro_monthly.parquet", "制造业PMI")
    g001_monthly = -(pmi.rolling(3, min_periods=3).mean() - pmi.rolling(36, min_periods=36).mean())
    _register_factor(raw_factor_df, factor_source_df, "G001_raw", _data_month_end_series(g001_monthly, data_index))

    loan_yoy = _load_long_term_loan_yoy()
    g002_monthly = loan_yoy - loan_yoy.rolling(3, min_periods=3).mean()
    _register_factor(raw_factor_df, factor_source_df, "G002_raw", _data_month_end_series(g002_monthly, data_index))

    cpi = _load_china_macro_series("CPI:同比", exclude_contains="核心")
    ppi = _load_china_macro_series("PPI:同比")
    cpi_ppi = (cpi - ppi).dropna().sort_index()
    p001_monthly = cpi_ppi.rolling(3, min_periods=3).mean() - cpi_ppi.rolling(12, min_periods=12).mean()
    _register_factor(raw_factor_df, factor_source_df, "P001_raw", _data_month_end_series(p001_monthly, data_index))

    v001 = _strong_ratio_diff(data_index)
    _register_factor(raw_factor_df, factor_source_df, "V001_raw", v001)

    v002 = _calc_return(data_df["close_g"], 20) - _calc_return(data_df["close_v"], 20)
    _register_factor(raw_factor_df, factor_source_df, "V002_raw", v002)

    score_df = pd.concat(
        [
            _direction_score(factor_source_df["I001"], 0.5).rename("I001"),
            _direction_score(factor_source_df["I002"], 0.5).rename("I002"),
            _direction_score(factor_source_df["G001"], 0.0).rename("G001"),
            _direction_score(factor_source_df["G002"], 0.0).rename("G002"),
            _direction_score(factor_source_df["P001"], 0.0).rename("P001"),
            _direction_score(factor_source_df["V001"], 0.0).rename("V001"),
            _direction_score(factor_source_df["V002"], 0.0).rename("V002"),
        ],
        axis=1,
    )
    w001 = score_df.mean(axis=1, skipna=True)
    w001.loc[score_df.notna().sum(axis=1).eq(0)] = np.nan
    _register_factor(raw_factor_df, factor_source_df, "W001_raw", w001)

    x = w001.clip(-1, 1)
    w002 = 0.5 + np.sign(x) / 2 * (np.exp(np.abs(x)) - 1) / (np.e - 1)
    _register_factor(raw_factor_df, factor_source_df, "W002_raw", w002.clip(0, 1))

    return factor_source_df.loc[:, FACTOR_IDS]
