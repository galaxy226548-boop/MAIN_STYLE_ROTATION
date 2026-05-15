"""Factors for 如何从赔率和胜率看成长价值轮动——市场风格轮动系列."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import (
    _load_china_macro_series,
    _load_china_macro_level_series,
    _register_factor,
    _rolling_quantile_rank_year,
    normalize_trade_dt,
    prepared_data_dir,
    read_prepared_series,
)


PAPER_ID = "如何从赔率和胜率看成长价值轮动——市场风格轮动系列"
GROWTH_INDEX_CODE = "399370.SZ"
VALUE_INDEX_CODE = "399371.SZ"
STYLE_INDEX_CODES = [GROWTH_INDEX_CODE, VALUE_INDEX_CODE]
INDEX_CLOSE_TABLE = "AIndexEODPrices.parquet"
INDEX_COMPONENT_TABLE = "AIndexHS300FreeWeight.parquet"
ADJUSTED_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"
TRADE_DATE_COL = "TRADE_DT"
BASE_FACTOR_IDS = ["I001", "I002", "G001", "G002", "P001", "V001", "V002"]
FACTOR_IDS = [*BASE_FACTOR_IDS, "W001", "W002"]
_ADJUSTED_CLOSE_CACHE: pd.DataFrame | None = None


def _signal_from_diff(diff: pd.Series) -> pd.Series:
    signal = pd.Series(
        np.select([diff > 0, diff < 0], [1, -1], default=0),
        index=diff.index,
        dtype="float64",
    )
    signal.loc[diff.isna()] = np.nan
    return signal


def _load_index_close(index_code: str) -> pd.Series:
    index_price = pd.read_parquet(
        prepared_data_dir / INDEX_CLOSE_TABLE,
        columns=["S_INFO_WINDCODE", TRADE_DATE_COL, "S_DQ_CLOSE"],
        filters=[("S_INFO_WINDCODE", "=", index_code)],
    )
    index_price[TRADE_DATE_COL] = normalize_trade_dt(index_price[TRADE_DATE_COL])
    index_price = index_price[index_price[TRADE_DATE_COL].notna()].copy()
    close = (
        index_price.sort_values(TRADE_DATE_COL)
        .drop_duplicates(subset=TRADE_DATE_COL, keep="last")
        .set_index(TRADE_DATE_COL)["S_DQ_CLOSE"]
    )
    close = pd.to_numeric(close, errors="coerce").sort_index().astype("float64")
    close.name = index_code
    return close


def _load_adjusted_close(tickers: list[str] | set[str] | None = None) -> pd.DataFrame:
    global _ADJUSTED_CLOSE_CACHE
    if _ADJUSTED_CLOSE_CACHE is None:
        close = pd.read_parquet(prepared_data_dir / ADJUSTED_CLOSE_TABLE)
        close[TRADE_DATE_COL] = normalize_trade_dt(close[TRADE_DATE_COL])
        close = close[close[TRADE_DATE_COL].notna()].copy()
        close = close.set_index(TRADE_DATE_COL).sort_index()
        close = close[~close.index.duplicated(keep="last")]

        ticker_cols = [
            col
            for col in close.columns
            if str(col).strip().split(".")[0].isdigit()
        ]
        close = close.loc[:, ticker_cols]
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        close.columns = close.columns.astype(str).str.strip()
        _ADJUSTED_CLOSE_CACHE = close.apply(pd.to_numeric, errors="coerce").astype("float64")

    if tickers is None:
        return _ADJUSTED_CLOSE_CACHE

    selected = [ticker for ticker in sorted(set(tickers)) if ticker in _ADJUSTED_CLOSE_CACHE.columns]
    return _ADJUSTED_CLOSE_CACHE.loc[:, selected].copy()


def _load_long_term_loan_yoy() -> pd.Series:
    yoy_col = "金融机构:人民币:中长期贷款余额:同比"
    return read_prepared_series("macro_monthly.parquet", yoy_col)


def _load_style_components() -> dict[str, dict[pd.Timestamp, list[str]]]:
    out: dict[str, dict[pd.Timestamp, list[str]]] = {}
    for index_code in STYLE_INDEX_CODES:
        comp = pd.read_parquet(
            prepared_data_dir / INDEX_COMPONENT_TABLE,
            columns=["S_INFO_WINDCODE", "S_CON_WINDCODE", TRADE_DATE_COL],
            filters=[("S_INFO_WINDCODE", "=", index_code)],
        )
        comp[TRADE_DATE_COL] = normalize_trade_dt(comp[TRADE_DATE_COL])
        comp = comp[comp[TRADE_DATE_COL].notna()].copy()
        comp["ticker"] = comp["S_CON_WINDCODE"].astype(str).str.strip()
        comp = comp[comp["ticker"].str.fullmatch(r"\d{6}\.(SZ|SH|BJ)", na=False)]

        out[index_code] = {
            dt: sorted(group["ticker"].dropna().unique().tolist())
            for dt, group in comp.groupby(TRADE_DATE_COL)
        }
        if not out[index_code]:
            raise ValueError(f"{INDEX_COMPONENT_TABLE} 中找不到 {index_code} 成分股")
    return out


def _strong_ratio(
    index_code: str,
    close_wide: pd.DataFrame,
    components_by_index: dict[str, dict[pd.Timestamp, list[str]]],
    ma_short: int = 5,
    ma_long: int = 20,
) -> pd.Series:
    components = components_by_index[index_code]
    component_dates = pd.DatetimeIndex(sorted(components.keys()))
    ma_s = close_wide.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = close_wide.rolling(ma_long, min_periods=ma_long).mean()
    is_strong = ma_s > ma_l

    ratio = pd.Series(np.nan, index=close_wide.index, dtype="float64")
    for dt in close_wide.index:
        loc = component_dates.searchsorted(dt, side="right") - 1
        if loc < 0:
            continue
        tickers = [ticker for ticker in components[component_dates[loc]] if ticker in is_strong.columns]
        if tickers:
            ratio.at[dt] = is_strong.loc[dt, tickers].fillna(False).sum() / len(tickers)
    return ratio


def _constituent_momentum_signal(data_index: pd.DatetimeIndex) -> pd.Series:
    components = _load_style_components()
    all_tickers = sorted(
        {
            ticker
            for by_date in components.values()
            for tickers in by_date.values()
            for ticker in tickers
        }
    )
    close = _load_adjusted_close(all_tickers)
    growth_ratio = _strong_ratio(GROWTH_INDEX_CODE, close, components)
    value_ratio = _strong_ratio(VALUE_INDEX_CODE, close, components)
    diff = growth_ratio - value_ratio
    signal = _signal_from_diff(diff.dropna())
    return signal.reindex(data_index)


def _index_momentum_signal(window: int = 20) -> pd.Series:
    growth = _load_index_close(GROWTH_INDEX_CODE).dropna().sort_index()
    value = _load_index_close(VALUE_INDEX_CODE).dropna().sort_index()
    growth_ret = growth.pct_change(window, fill_method=None)
    value_ret = value.pct_change(window, fill_method=None)
    diff = (growth_ret - value_ret).dropna()
    return _signal_from_diff(diff)


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

    pmi = _load_china_macro_level_series("月官方制造业PMI", value_col="今值")
    g001_monthly = -(pmi.rolling(3, min_periods=3).mean() - pmi.rolling(36, min_periods=36).mean())
    _register_factor(raw_factor_df, factor_source_df, "G001_raw", g001_monthly)

    loan_yoy = _load_long_term_loan_yoy()
    g002_monthly = loan_yoy - loan_yoy.rolling(3, min_periods=3).mean()
    _register_factor(raw_factor_df, factor_source_df, "G002_raw", g002_monthly)

    cpi = _load_china_macro_series("CPI:同比", exclude_contains="核心")
    ppi = _load_china_macro_series("PPI:同比")
    cpi_ppi = (cpi - ppi).dropna().sort_index()
    p001_monthly = cpi_ppi.rolling(3, min_periods=3).mean() - cpi_ppi.rolling(12, min_periods=12).mean()
    _register_factor(raw_factor_df, factor_source_df, "P001_raw", p001_monthly)

    v001 = _constituent_momentum_signal(data_index)
    _register_factor(raw_factor_df, factor_source_df, "V001_raw", v001)

    v002 = _index_momentum_signal(window=20)
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
