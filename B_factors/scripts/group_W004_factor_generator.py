"""W004 factors from factor_done.json records with non-empty category."""

from __future__ import annotations

import pandas as pd

from factor_utils import (
    _YoY,
    _as_numeric,
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


OUTPUT_PREFIX = "W004_factor_generator"
FACTOR_IDS = [
    "G003",
    "F001",
    "D001",
    "L003",
    "L004",
    "L005",
    "G004",
    "O001",
    "O002",
    "G005",
    "D002",
    "V003",
    "I003",
    "G006",
    "L006",
    "P002",
    "V004",
]


def _load_macro_series(
    nation: str,
    keywords: str | list[str],
    value_col: str = "今值",
    exclude_contains: str | list[str] | None = None,
) -> pd.Series:
    macro = _load_macro_all()
    keyword_list = [keywords] if isinstance(keywords, str) else list(keywords)
    exclude_list = [] if exclude_contains is None else (
        [exclude_contains] if isinstance(exclude_contains, str) else list(exclude_contains)
    )

    mask = macro["国家/地区"].eq(nation)
    indicator_text = macro["指标名称"].astype(str)
    keyword_mask = pd.Series(False, index=macro.index)
    for keyword in keyword_list:
        keyword_mask |= indicator_text.str.contains(keyword, regex=False, na=False)
    mask &= keyword_mask
    for item in exclude_list:
        mask &= ~indicator_text.str.contains(item, regex=False, na=False)

    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"macro.parquet 中找不到 国家/地区={nation!r}, 指标包含={keyword_list}")
    if value_col not in out.columns:
        raise KeyError(f"macro.parquet 中找不到字段 {value_col!r}; available={list(out.columns)}")

    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out[out["日期"].notna()].copy()
    sort_cols = [col for col in ["日期", "来源文件", "来源sheet", "文件年月"] if col in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    percent_hint = (
        out["指标名称"].astype(str).str.contains("%", regex=False, na=False).any()
        or out[value_col].astype(str).str.contains("%", regex=False, na=False).any()
    )
    series = pd.Series(
        _as_numeric(out[value_col], percent_hint=percent_hint).to_numpy(),
        index=out["日期"],
        name=keyword_list[0],
    ).sort_index()
    return series[~series.index.duplicated(keep="last")]


def _load_zhao02_index_pe() -> pd.Series:
    growth_pe = _read_indicator_series(
        "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx",
        "市盈率TTM",
    ).dropna()
    value_pe = _read_indicator_series(
        "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx",
        "市盈率TTM",
    ).dropna()
    valuation_ratio = growth_pe / value_pe
    quantile_rank = _rolling_quantile_rank_year(valuation_ratio, 5)
    raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0)
    return _month_aggregate(raw.dropna(), how="last")


def _load_credit_spread_factor() -> pd.Series:
    short_mid_aa_5y = read_prepared_series(
        "rate_daily.parquet",
        "中债中短期票据到期收益率(AA):5年",
    )
    cdb_5y = read_prepared_series("rate_daily.parquet", "中债国开债到期收益率:5年")
    credit_spread = (short_mid_aa_5y - cdb_5y).dropna()
    spread_short = credit_spread.rolling(window=5, min_periods=5).mean()
    spread_long = credit_spread.rolling(window=6 * 20, min_periods=6 * 20).mean()
    return calc_rolling_zscore(
        spread_long - spread_short,
        window=3 * 252,
        min_periods=252,
    )


def generate_W004_factor_generator(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)

    fixed_asset_yoy = _load_macro_series("中国", "固定资产投资")
    _register_factor(raw_factor_df, factor_source_df, "G003_raw", data_diff(fixed_asset_yoy))

    cpi_yoy = _load_macro_series("中国", "CPI:同比")
    cpi_deviation = cpi_yoy - cpi_yoy.rolling(window=6, min_periods=3).mean()
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "F001_raw",
        calc_rolling_zscore(cpi_deviation, window=36, min_periods=18),
    )

    m2_yoy = read_prepared_series("macro_monthly.parquet", "中国:M2:同比")
    sf_yoy = read_prepared_series("macro_monthly.parquet", "中国:社会融资规模存量:同比")
    m2_minus_sf = m2_yoy - sf_yoy
    _register_factor(raw_factor_df, factor_source_df, "D001_raw", data_diff(m2_minus_sf))

    _register_factor(raw_factor_df, factor_source_df, "L003_raw", data_yoy(m2_yoy))

    m1_yoy = read_prepared_series("macro_monthly.parquet", "中国:M1:同比")
    m2_minus_m1 = m2_yoy - m1_yoy
    _register_factor(raw_factor_df, factor_source_df, "L004_raw", data_diff(m2_minus_m1))
    _register_factor(raw_factor_df, factor_source_df, "L005_raw", data_yoy(calc_llt(m1_yoy, 5)))

    pmi = read_prepared_series("macro_monthly.parquet", "制造业PMI")
    _register_factor(raw_factor_df, factor_source_df, "G004_raw", _rolling_sum_ratio_minus_one(pmi - 50, window=3, shift=3))

    us_pmi = _load_macro_series("美国", ["ISM制造业PMI", "制造业PMI"], exclude_contains="非制造业")
    _register_factor(raw_factor_df, factor_source_df, "O001_raw", us_pmi - 50)

    us_nonfarm = _load_macro_series("美国", "非农就业人数")
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "O002_raw",
        calc_rolling_zscore(data_yoy(us_nonfarm), window=24, min_periods=12),
    )

    export_forecast = _load_macro_series("中国", "出口金额:当月同比", value_col="预测值")
    export_actual = _load_macro_series("中国", "出口金额:当月同比", value_col="今值")
    export_aligned = pd.concat(
        [export_forecast.rename("forecast"), export_actual.rename("actual")],
        axis=1,
    ).dropna()
    export_signal = pd.Series(0.0, index=export_aligned.index, dtype="float64")
    export_mask = export_aligned["forecast"] * export_aligned["actual"] < 0
    export_signal.loc[export_mask] = (
        export_aligned["actual"] - export_aligned["forecast"]
    ).loc[export_mask]
    _register_factor(
        raw_factor_df,
        factor_source_df,
        "G005_raw",
        calc_rolling_zscore(export_signal, window=24, min_periods=12),
    )

    _register_factor(raw_factor_df, factor_source_df, "D002_raw", _load_credit_spread_factor())

    _register_factor(raw_factor_df, factor_source_df, "V003_raw", _load_zhao02_index_pe())

    cn_1y = read_prepared_series("rate_daily.parquet", "中债国债到期收益率:1年")
    _register_factor(raw_factor_df, factor_source_df, "I003_raw", _YoY(_month_aggregate(cn_1y, how="average")) * -1)

    retail_yoy = read_prepared_series("macro_monthly.parquet", "中国:社会消费品零售总额:当月同比(1-2月合并)")
    _register_factor(raw_factor_df, factor_source_df, "G006_raw", _rolling_sum_ratio_minus_one(retail_yoy, window=12, shift=11))

    m0_yoy = read_prepared_series("macro_monthly.parquet", "中国:M0:同比")
    _register_factor(raw_factor_df, factor_source_df, "L006_raw", m0_yoy)

    ppi_yoy = _load_macro_series("中国", "PPI:同比")
    _register_factor(raw_factor_df, factor_source_df, "P002_raw", cpi_yoy - ppi_yoy)

    v004 = data_df["close_g"] / data_df["close_g"].shift(20) - data_df["close_v"] / data_df["close_v"].shift(20)
    _register_factor(raw_factor_df, factor_source_df, "V004_raw", v004)

    return factor_source_df.loc[:, FACTOR_IDS]
