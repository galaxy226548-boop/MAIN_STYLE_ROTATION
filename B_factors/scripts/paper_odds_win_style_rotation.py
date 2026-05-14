"""Factors for 如何从赔率和胜率看成长价值轮动——市场风格轮动系列."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import (
    _load_china_macro_series,
    _load_china_macro_level_series,
    _register_factor,
    _rolling_quantile_rank_year,
    prepared_data_dir,
    read_prepared_series,
)


PAPER_ID = "如何从赔率和胜率看成长价值轮动——市场风格轮动系列"
GROWTH_STYLE_KEY = "growth"
VALUE_STYLE_KEY = "value"
STYLE_COMPONENT_TABLES = {
    GROWTH_STYLE_KEY: "growth_factor_Fri.parquet",
    VALUE_STYLE_KEY: "value_factor_Fri.parquet",
}
ADJUSTED_CLOSE_TABLE = "S_DQ_ADJCLOSE.parquet"
TRADE_DATE_COL = "TRADE_DT"
BASE_FACTOR_IDS = ["I001", "I002", "G001", "G002", "P001", "V001", "V002"]
FACTOR_IDS = [*BASE_FACTOR_IDS, "W001", "W002"]
_ADJUSTED_CLOSE_CACHE: pd.DataFrame | None = None


def _load_adjusted_close(tickers: list[str] | set[str] | None = None) -> pd.DataFrame:
    global _ADJUSTED_CLOSE_CACHE
    if _ADJUSTED_CLOSE_CACHE is None:
        close = pd.read_parquet(prepared_data_dir / ADJUSTED_CLOSE_TABLE)
        close[TRADE_DATE_COL] = pd.to_datetime(close[TRADE_DATE_COL], errors="coerce").dt.normalize()
        close = close[close[TRADE_DATE_COL].notna()].copy()
        close = close.set_index(TRADE_DATE_COL).sort_index()
        close = close[~close.index.duplicated(keep="last")]

        ticker_cols = {}
        for col in close.columns:
            text = str(col).strip()
            base, sep, exchange = text.partition(".")
            if sep and exchange in {"SZ", "SH", "BJ"} and len(base) == 6 and base.isdigit():
                ticker_cols[col] = base
        close = close.loc[:, list(ticker_cols)].rename(columns=ticker_cols)
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        _ADJUSTED_CLOSE_CACHE = close.apply(pd.to_numeric, errors="coerce").astype("float64")

    if tickers is None:
        return _ADJUSTED_CLOSE_CACHE

    selected = [ticker for ticker in sorted(set(tickers)) if ticker in _ADJUSTED_CLOSE_CACHE.columns]
    return _ADJUSTED_CLOSE_CACHE.loc[:, selected]


def _load_weighted_style_close(table_name: str, adjusted_close: pd.DataFrame) -> pd.Series:
    style_df = pd.read_parquet(
        prepared_data_dir / table_name,
        columns=["holding_date", "component", "weight"],
    )
    style_df["holding_date"] = pd.to_datetime(style_df["holding_date"], errors="coerce").dt.normalize()
    style_df = style_df[style_df["holding_date"].notna()].copy()
    style_df["ticker"] = style_df["component"].astype(str).str.strip().str.slice(0, 6).str.zfill(6)
    style_df = style_df[style_df["ticker"].isin(adjusted_close.columns)]
    style_df["weight"] = pd.to_numeric(style_df["weight"], errors="coerce")

    row_pos = adjusted_close.index.get_indexer(style_df["holding_date"])
    col_pos = adjusted_close.columns.get_indexer(style_df["ticker"])
    valid = (row_pos >= 0) & (col_pos >= 0)
    style_df = style_df.loc[valid].copy()

    close_values = adjusted_close.to_numpy()[row_pos[valid], col_pos[valid]]
    style_df["close"] = close_values
    style_df = style_df[style_df["close"].notna() & style_df["weight"].notna()].copy()

    weighted_close = style_df["close"] * style_df["weight"]
    numerator = weighted_close.groupby(style_df["holding_date"]).sum(min_count=1)
    denominator = style_df["weight"].groupby(style_df["holding_date"]).sum(min_count=1)
    close = (numerator / denominator).replace([np.inf, -np.inf], np.nan).sort_index()
    close.name = table_name.removesuffix(".parquet")
    return close.astype("float64")


def _load_long_term_loan_yoy() -> pd.Series:
    yoy_col = "金融机构:人民币:中长期贷款余额:同比"
    return read_prepared_series("macro_monthly.parquet", yoy_col)


def _load_style_components() -> dict[str, dict[pd.Timestamp, list[str]]]:
    out: dict[str, dict[pd.Timestamp, list[str]]] = {}
    for style_key, table_name in STYLE_COMPONENT_TABLES.items():
        comp = pd.read_parquet(
            prepared_data_dir / table_name,
            columns=["holding_date", "component"],
        )
        comp["holding_date"] = pd.to_datetime(comp["holding_date"], errors="coerce").dt.normalize()
        comp = comp[comp["holding_date"].notna()].copy()
        comp["ticker"] = comp["component"].astype(str).str.strip().str.slice(0, 6).str.zfill(6)
        comp = comp[comp["ticker"].str.fullmatch(r"\d{6}", na=False)]

        out[style_key] = {
            dt: sorted(group["ticker"].dropna().unique().tolist())
            for dt, group in comp.groupby("holding_date")
        }
        if not out[style_key]:
            raise ValueError(f"{table_name} 中找不到可用的 holding_date/component 成分股")
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
    close = _load_adjusted_close(all_tickers)
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
        for style_key in [GROWTH_STYLE_KEY, VALUE_STYLE_KEY]:
            dates = component_dates[style_key]
            loc = dates.searchsorted(dt, side="right") - 1
            if loc < 0:
                continue
            tickers = [ticker for ticker in components[style_key][dates[loc]] if ticker in strong.columns]
            if tickers:
                ratios[style_key] = row[tickers].mean(skipna=True)
        if GROWTH_STYLE_KEY in ratios and VALUE_STYLE_KEY in ratios:
            result.at[dt] = ratios[GROWTH_STYLE_KEY] - ratios[VALUE_STYLE_KEY]
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

    v001 = _strong_ratio_diff(data_index)
    _register_factor(raw_factor_df, factor_source_df, "V001_raw", v001)

    adjusted_close = _load_adjusted_close()
    growth_close = _load_weighted_style_close("growth_factor_Fri.parquet", adjusted_close)
    value_close = _load_weighted_style_close("value_factor_Fri.parquet", adjusted_close)
    v002 = growth_close.pct_change(20, fill_method=None) - value_close.pct_change(20, fill_method=None)
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
