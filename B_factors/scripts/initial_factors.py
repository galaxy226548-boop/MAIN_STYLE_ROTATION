"""Initial factor generation for completed factor_done records with null category."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import (
    Config,
    _YoY,
    _as_numeric,
    _load_china_macro_series,
    _load_macro_all,
    _month_aggregate,
    _read_indicator_series,
    _register_factor,
    _rolling_quantile_rank_year,
    _rolling_sum_ratio_minus_one,
    calc_llt,
    calc_rolling_zscore,
    data_diff,
    data_yoy,
    read_prepared_series,
)


OUTPUT_PREFIX = "initial_factors"

FACTOR_IDS = [
    "D004",
    "D005",
    "D008",
    "F002",
    "F003",
    "F004",
    "F005",
    "G007",
    "G009",
    "G010",
    "G011",
    "G012",
    "G014",
    "G015",
    "G016",
    "G017",
    "I004",
    "I005",
    "I006",
    "I007",
    "I009",
    "I010",
    "I011",
    "I012",
    "I013",
    "L007",
    "L008",
    "L009",
    "L010",
    "L011",
    "L014",
    "L015",
    "L016",
    "L018",
    "L019",
    "L020",
    "L021",
    "L023",
    "L024",
    "O003",
    "O004",
    "O005",
    "O006",
    "O008",
    "O009",
    "P004",
    "P006",
    "P007",
    "P008",
    "P009",
    "P010",
    "P012",
    "P013",
    "P014",
    "V005",
    "V012",
    "V013",
]

SKIPPED_FACTORS = {
    "D006": "factor_done.py 中找不到精确 L003_raw 原始实现",
    "D007": "factor_done.py 中找不到精确 L004_raw 原始实现",
    "D009": "factor_done.py 中找不到精确 L006_raw 原始实现",
    "D011": "factor_done.py 中找不到精确 L008_raw 原始实现",
    "G019": "factor_done.py 中找不到精确 L016_raw 原始实现",
    "I014": "factor_done.py 中只找到近似 L119_1_raw，不作为稳定复现",
    "I015": "factor_done.py 中找不到精确 L124_1_raw 原始实现",
    "I019": "factor_done.py 中找不到精确 L021_raw 原始实现",
    "I020": "factor_done.py 中找不到精确 L022_raw 原始实现",
    "I021": "factor_done.py 中找不到精确 L023_raw 原始实现",
    "I023": "factor_done.py 中找不到精确 L025_raw 原始实现",
    "L012": "factor_done.py 中找不到精确 L32_1_raw 原始实现",
    "L013": "factor_done.py 中找不到精确 L32_2_raw 原始实现",
    "V014": "factor_done.py 中找不到精确 V47_raw 原始实现",
}


def _trailing_time_window(
    series: pd.Series,
    dt: pd.Timestamp,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
) -> pd.Series:
    if months is not None:
        start = dt - pd.DateOffset(months=int(round(months)))
    elif years is not None:
        start = dt - pd.DateOffset(years=int(round(years)))
    elif days is not None:
        start = dt - pd.Timedelta(days=float(days))
    else:
        raise ValueError("one of months, years, or days must be provided")
    return series.loc[(series.index > start) & (series.index <= dt)]


def _time_window_apply(
    series: pd.Series,
    func,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    values = []
    for dt in s.index:
        window = _trailing_time_window(s, dt, months=months, years=years, days=days)
        values.append(np.nan if len(window) < min_periods else func(window))
    return pd.Series(values, index=s.index, dtype="float64")


def _time_mean(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    return _time_window_apply(
        series,
        lambda window: window.mean(),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def _rolling_quantile_value(
    series: pd.Series,
    target_quantile: float,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    return _time_window_apply(
        series,
        lambda window: window.quantile(target_quantile),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def _calc_rolling_zscore_time(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    values = []
    for dt in s.index:
        window = _trailing_time_window(s, dt, months=months, years=years, days=days)
        if len(window) < min_periods:
            values.append(np.nan)
            continue
        rolling_std = window.std()
        if pd.isna(rolling_std) or np.isclose(rolling_std, 0):
            values.append(np.nan)
        else:
            values.append((s.loc[dt] - window.mean()) / rolling_std)
    return pd.Series(values, index=s.index, dtype="float64")


def _data_deviation(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").sort_index()
    rolling_mean = _time_mean(
        s,
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )
    return s - rolling_mean.reindex(s.index)


def _z_data_deviation(
    series: pd.Series,
    *,
    dev_months: int | float | None = None,
    dev_years: int | float | None = None,
    dev_days: int | float | None = None,
    z_months: int | float | None = None,
    z_years: int | float | None = None,
    z_days: int | float | None = None,
    dev_min_periods: int = 1,
    z_min_periods: int = 1,
) -> pd.Series:
    deviation = _data_deviation(
        series,
        months=dev_months,
        years=dev_years,
        days=dev_days,
        min_periods=dev_min_periods,
    )
    return _calc_rolling_zscore_time(
        deviation,
        months=z_months,
        years=z_years,
        days=z_days,
        min_periods=z_min_periods,
    )


def _z_ma_div(
    series: pd.Series,
    short_window: int | float,
    long_window: int | float,
    z_window: int | float,
    *,
    unit: str = "months",
    short_min_periods: int | None = None,
    long_min_periods: int | None = None,
    z_min_periods: int | None = None,
) -> pd.Series:
    if unit == "days":
        short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, int(round(short_window / 30 / 2)))
        long_min_periods = long_min_periods or max(6, int(round(long_window / 30 / 2)))
        z_min_periods = z_min_periods or max(12, int(round(z_window / 30 / 2)))
        short_ma = series.copy() if short_window == 1 else _time_mean(series, days=short_window, min_periods=short_min_periods)
        long_ma = _time_mean(series, days=long_window, min_periods=long_min_periods)
        return _calc_rolling_zscore_time(short_ma - long_ma, days=z_window, min_periods=z_min_periods)

    short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, int(short_window) // 2)
    long_min_periods = long_min_periods or max(3, int(long_window) // 2)
    z_min_periods = z_min_periods or max(6, int(z_window) // 2)
    short_ma = series.copy() if short_window == 1 else _time_mean(series, months=short_window, min_periods=short_min_periods)
    long_ma = _time_mean(series, months=long_window, min_periods=long_min_periods)
    return _calc_rolling_zscore_time(short_ma - long_ma, months=z_window, min_periods=z_min_periods)


def _expectation(
    sub_1: pd.Series,
    sub_2: pd.Series,
    *,
    z_years: int = 3,
    z_min_periods: int = 18,
    up_floor: float | None = None,
    down_ceiling: float | None = None,
    upper_quantile: float | None = None,
    lower_quantile: float | None = None,
    quantile_years: int = 3,
    quantile_min_periods: int = 18,
) -> pd.Series:
    aligned = pd.concat([sub_1.rename("sub_1"), sub_2.rename("sub_2")], axis=1).sort_index()
    aligned = aligned.dropna(subset=["sub_1", "sub_2"])
    surprise = aligned["sub_1"] - aligned["sub_2"]

    if upper_quantile is not None:
        upper_bound = _rolling_quantile_value(
            aligned["sub_1"],
            upper_quantile,
            years=quantile_years,
            min_periods=quantile_min_periods,
        )
    else:
        upper_bound = pd.Series(up_floor, index=aligned.index, dtype="float64")

    if lower_quantile is not None:
        lower_bound = _rolling_quantile_value(
            aligned["sub_1"],
            lower_quantile,
            years=quantile_years,
            min_periods=quantile_min_periods,
        )
    else:
        lower_bound = pd.Series(down_ceiling, index=aligned.index, dtype="float64")

    signal = pd.Series(np.nan, index=aligned.index, dtype="float64")
    positive_mask = (aligned["sub_1"] > aligned["sub_2"]) & (aligned["sub_1"] > upper_bound)
    negative_mask = (aligned["sub_1"] < aligned["sub_2"]) & (aligned["sub_1"] < lower_bound)
    signal.loc[positive_mask | negative_mask] = surprise.loc[positive_mask | negative_mask]
    return _calc_rolling_zscore_time(signal, years=z_years, min_periods=z_min_periods)


def _latest_macro_pair(sub_1: pd.Series, sub_2: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([sub_1.rename("sub_1"), sub_2.rename("sub_2")], axis=1, sort=True).sort_index()
    aligned[["sub_1", "sub_2"]] = aligned[["sub_1", "sub_2"]].ffill()
    return aligned


def _tail_quantile_signal(rank_series: pd.Series, upper: float = 0.75, lower: float = 0.25) -> pd.Series:
    out = pd.Series(0.0, index=rank_series.index, dtype="float64")
    out.loc[rank_series > upper] = rank_series.loc[rank_series > upper] - upper
    out.loc[rank_series < lower] = lower - rank_series.loc[rank_series < lower]
    out.loc[rank_series.isna()] = np.nan
    return out


def _load_macro_series_by_country(
    country: str,
    keyword: str,
    *,
    value_col: str = "今值",
    required_contains: str | list[str] | None = None,
    exclude_contains: str | list[str] | None = None,
    percent_as_ratio: bool = True,
) -> pd.Series:
    macro = _load_macro_all()
    date_col = "日期" if "日期" in macro.columns else macro.columns[2]
    nation_col = "国家/地区" if "国家/地区" in macro.columns else macro.columns[4]
    indicator_col = "指标名称" if "指标名称" in macro.columns else macro.columns[5]

    mask = macro[nation_col].eq(country)
    indicator_text = macro[indicator_col].astype(str)
    mask &= indicator_text.str.contains(keyword, na=False, regex=False)

    required_list = [] if required_contains is None else (
        [required_contains] if isinstance(required_contains, str) else list(required_contains)
    )
    for item in required_list:
        mask &= indicator_text.str.contains(item, na=False, regex=False)

    exclude_list = [] if exclude_contains is None else (
        [exclude_contains] if isinstance(exclude_contains, str) else list(exclude_contains)
    )
    for item in exclude_list:
        mask &= ~indicator_text.str.contains(item, na=False, regex=False)

    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"macro.parquet 中找不到 国家/地区={country!r}, 指标包含={keyword!r}")
    if value_col not in out.columns:
        raise KeyError(f"macro.parquet 中找不到字段 {value_col!r}; available={list(out.columns)}")

    percent_text = (
        out[indicator_col].astype(str).str.contains("%", regex=False, na=False).any()
        or out[value_col].astype(str).str.contains("%", regex=False, na=False).any()
    )
    percent_hint = bool(percent_as_ratio and percent_text)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    sort_cols = [col for col in [date_col, "来源文件", "来源sheet", "文件年月"] if col in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    series = pd.Series(
        _as_numeric(out[value_col], percent_hint=percent_hint).to_numpy(),
        index=out[date_col],
        name=keyword,
        dtype="float64",
    ).sort_index()
    return series[~series.index.duplicated(keep="last")]


def _clean_daily_rate(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").sort_index()


def _read_style_return_series() -> pd.Series:
    raw = pd.read_excel(Config.DATA_DIR / "中证全指(000985.CSI)-历史价格.xlsx")
    raw["交易日期"] = pd.to_datetime(raw["交易日期"], errors="coerce")
    raw = raw.dropna(subset=["交易日期"]).set_index("交易日期").sort_index()
    return _as_numeric(raw["涨跌幅"], percent_hint=True).sort_index()


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).sort_index()
    cov = aligned["y"].rolling(window=window, min_periods=window).cov(aligned["x"])
    var = aligned["x"].rolling(window=window, min_periods=window).var()
    return cov / var


def _load_index_valuation_ratio(value_col: str) -> pd.Series:
    growth = _read_indicator_series(
        "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx",
        value_col,
    ).dropna()
    value = _read_indicator_series(
        "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx",
        value_col,
    ).dropna()
    return growth / value


def _valuation_tail_signal(value_col: str) -> pd.Series:
    quantile_rank = _rolling_quantile_rank_year(_load_index_valuation_ratio(value_col), 5)
    raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0)
    return _month_aggregate(raw.dropna(), how="last")


def _load_credit_spread_5y_factor() -> pd.Series:
    stb_aa_5y = read_prepared_series("rate_daily.parquet", "中债中短期票据到期收益率(AA):5年")
    cdb_5y = read_prepared_series("rate_daily.parquet", "中债国开债到期收益率:5年")
    spread = _clean_daily_rate(stb_aa_5y - cdb_5y).dropna()
    spread_short = spread.rolling(window=5, min_periods=5).mean()
    spread_long = spread.rolling(window=6 * Config.MONTH_DAYS, min_periods=6 * Config.MONTH_DAYS).mean()
    return calc_rolling_zscore(
        spread_long - spread_short,
        window=3 * Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )


def _load_l118_treasury_factor() -> pd.Series:
    rate_10y = read_prepared_series("rate_daily.parquet", "中债国债到期收益率:10年").dropna()
    rate_1y = read_prepared_series("rate_daily.parquet", "中债国债到期收益率:1年").dropna()

    dev_10y = rate_10y.rolling(160, min_periods=160).mean() - rate_10y
    z_10y = calc_rolling_zscore(
        calc_llt(dev_10y, 5),
        window=Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )

    dev_1y = rate_1y.rolling(60, min_periods=60).mean() - rate_1y
    z_1y = calc_rolling_zscore(
        calc_llt(dev_1y, 10),
        window=Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )
    return (z_10y + z_1y) / 2


def _load_v97_crowding_factor(data_df: pd.DataFrame) -> pd.Series:
    turnover_ratio = (
        data_df["turnover_rate_g"].rolling(window=3 * Config.MONTH_DAYS, min_periods=3 * Config.MONTH_DAYS).sum()
        / data_df["turnover_rate_v"].rolling(window=3 * Config.MONTH_DAYS, min_periods=3 * Config.MONTH_DAYS).sum()
    )
    tf_turnover = calc_rolling_zscore(
        calc_llt(np.log(turnover_ratio.replace(0, np.nan)), d=30),
        window=3 * Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )

    vol_ratio = (
        data_df["pct_change_g"].rolling(window=3 * Config.MONTH_DAYS, min_periods=3 * Config.MONTH_DAYS).std()
        / data_df["pct_change_v"].rolling(window=3 * Config.MONTH_DAYS, min_periods=3 * Config.MONTH_DAYS).std()
    )
    tf_volatility = calc_rolling_zscore(
        calc_llt(np.log(vol_ratio.replace(0, np.nan)), d=30),
        window=3 * Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )

    zzqz_return = _read_style_return_series().reindex(data_df.index)
    beta_g = _rolling_beta(data_df["pct_change_g"], zzqz_return, window=3 * Config.MONTH_DAYS)
    beta_v = _rolling_beta(data_df["pct_change_v"], zzqz_return, window=3 * Config.MONTH_DAYS)
    beta_ratio = beta_g / beta_v
    tf_beta = calc_rolling_zscore(
        calc_llt(np.log(beta_ratio.replace(0, np.nan)), d=30),
        window=3 * Config.ANNUAL_TRADING_DAYS,
        min_periods=Config.ANNUAL_TRADING_DAYS,
    )
    return (tf_turnover + tf_volatility + tf_beta) / 3


def generate_initial_factor_generator(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)

    fixed_asset_yoy = _load_china_macro_series("固定资产投资")
    cpi_yoy = _load_china_macro_series("CPI:同比", value_col="今值")
    cpi_forecast = _load_china_macro_series("CPI:同比", value_col="预测值")
    ppi_yoy = _load_china_macro_series("PPI:同比", value_col="今值")
    industrial_yoy = _load_china_macro_series("工业增加值:当月同比", value_col="今值")
    ppi_cpi_spread = ppi_yoy - cpi_yoy

    m2_actual = _load_china_macro_series("M2:同比")
    m2_forecast = _load_china_macro_series("M2:同比", value_col="预测值")
    sf_actual = _load_china_macro_series("社会融资规模存量:同比")
    sf_forecast = _load_china_macro_series("社会融资规模存量:同比", value_col="预测值")
    gdp_actual = _load_china_macro_series("季度GDP:当季同比(%)")
    m1_actual = _load_china_macro_series("M1:同比")
    m0_actual = _load_china_macro_series("M0:同比")

    china_pmi_actual = _load_macro_series_by_country(
        "中国",
        "官方制造业PMI",
        value_col="今值",
        percent_as_ratio=False,
    )
    china_export_actual = _load_macro_series_by_country(
        "中国",
        "出口金额:当月同比",
        value_col="今值",
        exclude_contains="人民币",
    )
    us_cpi_yoy = _load_macro_series_by_country(
        "美国",
        "CPI",
        value_col="今值",
        required_contains="同比",
        exclude_contains="核心",
    )
    us_unemployment = _load_macro_series_by_country(
        "美国",
        "失业率",
        value_col="今值",
        required_contains="季调",
    )
    shibor_1m = read_prepared_series("rate_daily.parquet", "Shibor:1月")
    shibor_3m = read_prepared_series("rate_daily.parquet", "Shibor:3月")
    us_credit_spread_baml = read_prepared_series("rate_daily.parquet", "BAMLH0A0HYM2")
    usd_index = read_prepared_series("exchange_rate_daily.parquet", "美元指数")
    us_t10 = read_prepared_series("rate_daily.parquet", "美国:国债收益率:10年")
    us_t6m = read_prepared_series("rate_daily.parquet", "美国:国债收益率:6个月")
    us_t3m = read_prepared_series("rate_daily.parquet", "美国:国债收益率:3个月")
    us_t1m = read_prepared_series("rate_daily.parquet", "美国:国债收益率:1个月")
    us_t2y = read_prepared_series("rate_daily.parquet", "美国:国债收益率:2年")

    m2_sf_forecast = pd.concat([m2_forecast.rename("sub_1"), sf_forecast.rename("sub_2")], axis=1).sort_index()
    _register_factor(raw_factor_df, factor_source_df, "D004_raw", data_yoy(m2_sf_forecast["sub_2"] / m2_sf_forecast["sub_1"]))

    m2_gdp_latest = _latest_macro_pair(m2_actual, gdp_actual)
    _register_factor(raw_factor_df, factor_source_df, "D005_raw", m2_gdp_latest["sub_1"] - m2_gdp_latest["sub_2"])

    credit_spread_5y = _load_credit_spread_5y_factor()
    _register_factor(raw_factor_df, factor_source_df, "D008_raw", credit_spread_5y)

    _register_factor(
        raw_factor_df,
        factor_source_df,
        "F002_raw",
        _expectation(cpi_yoy, cpi_forecast, up_floor=0.025, down_ceiling=-0.025, z_years=3, z_min_periods=18),
    )
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "F003_raw",
        _expectation(
            cpi_yoy,
            cpi_forecast,
            upper_quantile=0.80,
            lower_quantile=0.20,
            quantile_years=3,
            quantile_min_periods=18,
            z_years=3,
            z_min_periods=18,
        ),
    )
    _register_factor(raw_factor_df, factor_source_df, "F004_raw", cpi_yoy)
    _register_factor(raw_factor_df, factor_source_df, "F005_raw", data_diff(cpi_yoy))
    _register_factor(raw_factor_df, factor_source_df, "G007_raw", fixed_asset_yoy)
    _register_factor(raw_factor_df, factor_source_df, "G009_raw", china_pmi_actual - 50)

    pmi_up_signal = pd.Series(0.0, index=china_pmi_actual.index, dtype="float64")
    pmi_up_zscore = calc_rolling_zscore(50 - china_pmi_actual, window=12, min_periods=6)
    pmi_diff = data_diff(china_pmi_actual)
    pmi_up_signal.loc[pmi_diff > 0] = pmi_up_zscore.loc[pmi_diff > 0]
    _register_factor(raw_factor_df, factor_source_df, "G010_raw", pmi_up_signal)
    _register_factor(raw_factor_df, factor_source_df, "G011_raw", calc_llt(china_pmi_actual - 50, 12))
    _register_factor(raw_factor_df, factor_source_df, "G012_raw", calc_rolling_zscore(china_pmi_actual - 50, window=12, min_periods=6))
    _register_factor(raw_factor_df, factor_source_df, "G014_raw", _rolling_sum_ratio_minus_one(china_pmi_actual, window=3, shift=1))
    _register_factor(raw_factor_df, factor_source_df, "G015_raw", _z_ma_div(china_pmi_actual, 3, 6, 12, unit="months"))
    _register_factor(raw_factor_df, factor_source_df, "G016_raw", calc_llt(china_pmi_actual - 50, 3))

    export_ttm_mean = china_export_actual.rolling(window=12, min_periods=12).mean()
    export_base = pd.Series(export_ttm_mean.copy(), dtype="float64")
    export_growth_mask = export_ttm_mean * export_ttm_mean.shift(1) > 0
    export_base.loc[export_growth_mask] = (export_ttm_mean / export_ttm_mean.shift(1) - 1).loc[export_growth_mask]
    _register_factor(raw_factor_df, factor_source_df, "G017_raw", calc_rolling_zscore(export_base, window=18, min_periods=9))
    _register_factor(raw_factor_df, factor_source_df, "I004_raw", _z_ma_div(shibor_3m, 1, 3 * Config.MONTH_DAYS, Config.ANNUAL_TRADING_DAYS, unit="days"))
    _register_factor(raw_factor_df, factor_source_df, "I005_raw", _z_ma_div(shibor_3m, 5, 250, 252, unit="days"))
    _register_factor(raw_factor_df, factor_source_df, "I006_raw", _z_ma_div(shibor_1m, 5, 250, 252, unit="days"))
    shibor_1m_ma5 = shibor_1m.rolling(window=5, min_periods=5).mean()
    _register_factor(raw_factor_df, factor_source_df, "I007_raw", _tail_quantile_signal(_rolling_quantile_rank_year(shibor_1m_ma5, 3)))
    _register_factor(raw_factor_df, factor_source_df, "I009_raw", _z_ma_div(us_t3m, 60, 240, 252, unit="days"))
    us_t1m_ma10 = us_t1m.rolling(window=10, min_periods=10).mean()
    _register_factor(raw_factor_df, factor_source_df, "I010_raw", _tail_quantile_signal(_rolling_quantile_rank_year(us_t1m_ma10, 1)))
    _register_factor(raw_factor_df, factor_source_df, "I011_raw", _tail_quantile_signal(_rolling_quantile_rank_year(us_t6m, 3)))
    us_t2y_monthly_mean = us_t2y.dropna().rolling(window=20, min_periods=20).mean()
    _register_factor(raw_factor_df, factor_source_df, "I012_raw", calc_rolling_zscore(us_t2y_monthly_mean, window=252, min_periods=126))
    _register_factor(raw_factor_df, factor_source_df, "I013_raw", _load_l118_treasury_factor())

    m1_minus_m2 = m1_actual - m2_actual
    m1_minus_m0 = m1_actual - m0_actual
    _register_factor(raw_factor_df, factor_source_df, "L007_raw", m2_actual)
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "L008_raw",
        _expectation(
            m2_actual,
            m2_forecast,
            upper_quantile=0.80,
            lower_quantile=0.20,
            quantile_years=3,
            quantile_min_periods=18,
            z_years=3,
            z_min_periods=18,
        ),
    )
    _register_factor(raw_factor_df, factor_source_df, "L009_raw", _z_ma_div(m2_actual, 1, Config.ANNUAL_DAYS, 3 * Config.ANNUAL_DAYS, unit="days"))
    m2_forecast_pair = pd.concat(
        [m2_actual.rename("actual"), m2_forecast.rename("forecast")],
        axis=1,
    ).dropna()
    m2_surprise_abs = (m2_forecast_pair["actual"] - m2_forecast_pair["forecast"]).abs()
    _register_factor(raw_factor_df, factor_source_df, "L010_raw", _calc_rolling_zscore_time(m2_surprise_abs, years=3, min_periods=18))
    _register_factor(raw_factor_df, factor_source_df, "L011_raw", data_diff(calc_llt(m2_actual, 5)))
    _register_factor(raw_factor_df, factor_source_df, "L014_raw", _calc_rolling_zscore_time(m1_minus_m2, years=3, min_periods=18))
    _register_factor(raw_factor_df, factor_source_df, "L015_raw", data_yoy(m1_minus_m2))
    _register_factor(raw_factor_df, factor_source_df, "L016_raw", _z_data_deviation(m1_minus_m2, dev_months=3, z_months=12, dev_min_periods=2, z_min_periods=6))
    _register_factor(raw_factor_df, factor_source_df, "L018_raw", _z_ma_div(m1_minus_m2, 3, 6, 12, unit="months"))
    _register_factor(raw_factor_df, factor_source_df, "L019_raw", _z_data_deviation(m1_minus_m2, dev_months=3, z_months=12, dev_min_periods=2, z_min_periods=6))
    _register_factor(raw_factor_df, factor_source_df, "L020_raw", m1_actual)
    _register_factor(raw_factor_df, factor_source_df, "L021_raw", calc_llt(m1_actual, 6))
    _register_factor(raw_factor_df, factor_source_df, "L023_raw", m0_actual)
    _register_factor(raw_factor_df, factor_source_df, "L024_raw", m1_minus_m0)
    _register_factor(raw_factor_df, factor_source_df, "O003_raw", _z_ma_div(us_credit_spread_baml.sort_index().shift(2), 5, 250, 252, unit="days"))
    _register_factor(raw_factor_df, factor_source_df, "O004_raw", data_yoy(calc_llt(usd_index, 5)))
    _register_factor(raw_factor_df, factor_source_df, "O005_raw", _z_ma_div(usd_index, 5, 120, 252, unit="days"))
    _register_factor(raw_factor_df, factor_source_df, "O006_raw", calc_rolling_zscore(us_t10 - calc_llt(us_t10, 12), window=252, min_periods=126))
    _register_factor(raw_factor_df, factor_source_df, "O008_raw", data_diff(us_cpi_yoy))
    _register_factor(raw_factor_df, factor_source_df, "O009_raw", data_diff(us_unemployment))

    ppi_cpi_3m_mean = _time_window_apply(ppi_cpi_spread, lambda window: window.mean(), months=3, min_periods=2)
    ppi_cpi_12m_mean = _time_window_apply(ppi_cpi_spread, lambda window: window.mean(), years=1, min_periods=6)
    _register_factor(raw_factor_df, factor_source_df, "P004_raw", _calc_rolling_zscore_time(ppi_cpi_3m_mean - ppi_cpi_12m_mean, years=3, min_periods=18))
    _register_factor(raw_factor_df, factor_source_df, "P006_raw", ppi_cpi_spread)
    recent_spread_avg = (ppi_cpi_spread + ppi_cpi_spread.shift(1)) / 2
    past_spread_avg = (ppi_cpi_spread.shift(2) + ppi_cpi_spread.shift(3) + ppi_cpi_spread.shift(4)) / 3
    _register_factor(raw_factor_df, factor_source_df, "P007_raw", recent_spread_avg - past_spread_avg)
    _register_factor(raw_factor_df, factor_source_df, "P008_raw", _calc_rolling_zscore_time(ppi_cpi_spread + ppi_cpi_spread.shift(1), years=4, min_periods=24))
    _register_factor(raw_factor_df, factor_source_df, "P009_raw", data_diff(ppi_cpi_spread))
    _register_factor(raw_factor_df, factor_source_df, "P010_raw", ppi_yoy)
    _register_factor(raw_factor_df, factor_source_df, "P012_raw", industrial_yoy)
    _register_factor(raw_factor_df, factor_source_df, "P013_raw", data_diff(calc_llt(industrial_yoy, 5)))
    _register_factor(raw_factor_df, factor_source_df, "P014_raw", _rolling_sum_ratio_minus_one(ppi_yoy, window=5, shift=1))

    _register_factor(raw_factor_df, factor_source_df, "V005_raw", _valuation_tail_signal("市净率LF"))
    tf_logbp = calc_llt(np.log(1 / _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx", "市净率LF")) - np.log(1 / _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx", "市净率LF")), d=30)
    _register_factor(raw_factor_df, factor_source_df, "V012_raw", calc_rolling_zscore(tf_logbp, window=3 * Config.ANNUAL_TRADING_DAYS, min_periods=Config.ANNUAL_TRADING_DAYS))
    _register_factor(raw_factor_df, factor_source_df, "V013_raw", _load_v97_crowding_factor(data_df))

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"initial factor columns missing after generation: {missing_cols}")

    for factor_id, reason in SKIPPED_FACTORS.items():
        print(f"initial_factors skipped {factor_id}: {reason}")

    return factor_source_df.loc[:, FACTOR_IDS]
