from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from Config import Config


FINAL_RAW_FACTOR = "ODDS_WIN_FINAL_raw"
FINAL_FACTOR = "ODDS_WIN_FINAL"
ODDS_WIN_SINGLE_RAW_FACTORS = [
    "OW_F001_CN10Y_Q3Y_raw",
    "OW_F002_US6M_Q3Y_raw",
    "OW_F003_PMI_DIFF_raw",
    "OW_F004_LOAN_YOY_DIFF_raw",
    "OW_F005_CPI_PPI_DIFF_raw",
    "OW_F006_MOM_4W_raw",
    "OW_F007_STRONG_RATIO_raw",
]


def _as_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return pd.to_numeric(
        text.str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def _to_datetime(values) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _read_indicator_series(path: Path, value_col: str, sheet_name=0) -> pd.Series:
    raw = pd.read_excel(path, sheet_name=sheet_name)
    date_col = raw.columns[0]
    out = raw.copy()
    out[date_col] = _to_datetime(out[date_col])
    out = out[out[date_col].notna()].copy()
    if value_col not in out.columns:
        raise KeyError(f"{value_col} not found in {path}. Available: {list(out.columns)}")
    out = out.set_index(date_col).sort_index()
    s = _as_numeric(out[value_col])
    s.name = value_col
    return s[~s.index.duplicated(keep="last")].sort_index()


def _load_china_macro_series(
    path: Path,
    keyword: str,
    *,
    required_contains: Optional[Iterable[str]] = None,
    exclude_contains: Optional[Iterable[str]] = None,
) -> pd.Series:
    macro = pd.read_excel(path, sheet_name="全部记录")
    mask = macro["国家/地区"].eq("中国")
    mask &= macro["指标名称"].astype(str).str.contains(keyword, regex=False, na=False)
    if required_contains is not None:
        for item in required_contains:
            mask &= macro["指标名称"].astype(str).str.contains(item, regex=False, na=False)
    if exclude_contains is not None:
        for item in exclude_contains:
            mask &= ~macro["指标名称"].astype(str).str.contains(item, regex=False, na=False)

    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No China macro rows matched keyword={keyword!r}")

    out["日期"] = _to_datetime(out["日期"])
    out = out[out["日期"].notna()].copy()
    sort_cols = [c for c in ["日期", "来源文件", "来源sheet", "文件年月"] if c in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    s = pd.Series(_as_numeric(out["今值"]).to_numpy(), index=out["日期"], name=keyword)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _month_end_on_data_index(series: pd.Series, data_index: pd.DatetimeIndex) -> pd.Series:
    s = series.dropna().copy().sort_index()
    if s.empty:
        return pd.Series(dtype="float64")
    s.index = pd.to_datetime(s.index).to_period("M")
    s = s[~s.index.duplicated(keep="last")]
    data_index = pd.DatetimeIndex(data_index).sort_values()
    data_month_end = pd.Series(data_index, index=data_index.to_period("M")).groupby(level=0).max()
    target_dates = data_month_end.reindex(s.index)
    valid = target_dates.notna()
    return pd.Series(
        s.loc[valid].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(target_dates.loc[valid].to_numpy()),
        dtype="float64",
    ).sort_index()


def _rolling_quantile_rank_year(series: pd.Series, year: int = 3) -> pd.Series:
    s = series.dropna().sort_index()
    out = []
    dates = s.index
    left = 0
    if len(s) == 0:
        return pd.Series(dtype="float64")
    first_date = dates[0]
    for right, end_date in enumerate(dates):
        start_date = end_date - pd.DateOffset(years=year)
        if first_date > start_date:
            out.append(np.nan)
            continue
        while left < right and dates[left] < start_date:
            left += 1
        window = s.iloc[left : right + 1]
        out.append(window.rank(pct=True).iloc[-1])
    return pd.Series(out, index=dates, dtype="float64")


def _calc_return(price: pd.Series, window: int, return_type: str = Config.RETURN_TYPE) -> pd.Series:
    if return_type == "log":
        return np.log(price / price.shift(window))
    return price / price.shift(window) - 1


def _simple_forward_return(price: pd.Series, horizon: int) -> pd.Series:
    return price.shift(-horizon) / price - 1


def _style_bp_rank(pb_path: Path, data_index: pd.DatetimeIndex) -> pd.Series:
    raw = pd.read_excel(pb_path)
    raw["交易日期"] = _to_datetime(raw["交易日期"])
    raw = raw[raw["交易日期"].notna()].set_index("交易日期").sort_index()
    bp = 1 / _as_numeric(raw["市净率LF"])
    bp = bp.replace([np.inf, -np.inf], np.nan).reindex(data_index)
    return _rolling_quantile_rank_year(bp, year=3).reindex(data_index)


def _expanding_conditional_odds(
    valuation_rank: pd.Series,
    fwd_return: pd.Series,
    *,
    horizon: int = 20,
    min_obs: int = 120,
    bins: int = 5,
) -> pd.Series:
    aligned = pd.concat(
        [valuation_rank.rename("x"), fwd_return.rename("ret")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    out = pd.Series(np.nan, index=aligned.index, dtype="float64")
    valid = aligned.dropna()
    if len(valid) < min_obs:
        return out

    for dt in aligned.index:
        if pd.isna(aligned.at[dt, "x"]):
            continue
        pos = valid.index.searchsorted(dt, side="left") - horizon
        if pos < min_obs:
            continue
        hist = valid.iloc[:pos].copy()
        if len(hist) < min_obs:
            continue
        try:
            hist["bin"] = pd.qcut(hist["x"], q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue

        grouped_rows = []
        for _, g in hist.groupby("bin"):
            pos_ret = g.loc[g["ret"] > 0, "ret"]
            neg_ret = g.loc[g["ret"] < 0, "ret"]
            if len(pos_ret) == 0 or len(neg_ret) == 0:
                continue
            odds = pos_ret.mean() / abs(neg_ret.mean())
            if np.isfinite(odds):
                grouped_rows.append((g["x"].mean(), odds))

        if len(grouped_rows) < 2:
            continue
        reg = pd.DataFrame(grouped_rows, columns=["x", "odds"]).dropna()
        if len(reg) < 2 or np.isclose(reg["x"].std(), 0):
            continue
        beta, alpha = np.polyfit(reg["x"], reg["odds"], deg=1)
        pred = alpha + beta * aligned.at[dt, "x"]
        if np.isfinite(pred):
            out.at[dt] = max(pred, 0.0)

    return out


def _load_style_components(path: Path) -> Dict[str, Dict[pd.Timestamp, list]]:
    comp = pd.read_parquet(path)
    comp["TRADE_DT"] = _to_datetime(comp["TRADE_DT"].astype(str))
    comp = comp[comp["TRADE_DT"].notna()].copy()
    comp["ticker"] = comp["S_CON_WINDCODE"].astype(str).str.slice(0, 6)
    out: Dict[str, Dict[pd.Timestamp, list]] = {}
    for index_code in ["399370.SZ", "399371.SZ"]:
        sub = comp[comp["S_INFO_WINDCODE"].eq(index_code)]
        out[index_code] = {
            dt: sorted(g["ticker"].dropna().unique().tolist())
            for dt, g in sub.groupby("TRADE_DT")
        }
    return out


def _strong_ratio_diff(
    data_index: pd.DatetimeIndex,
    component_path: Path,
    mkt_path: Path,
) -> pd.Series:
    components = _load_style_components(component_path)
    mkt = pd.read_parquet(mkt_path, columns=["Stkcd", "Trddt", "Clsprc"])
    mkt["Trddt"] = _to_datetime(mkt["Trddt"])
    mkt = mkt[mkt["Trddt"].notna()].copy()
    mkt["Stkcd"] = mkt["Stkcd"].astype(str).str.zfill(6)
    close = mkt.pivot_table(index="Trddt", columns="Stkcd", values="Clsprc", aggfunc="last").sort_index()
    strong = close.rolling(5, min_periods=5).mean() > close.rolling(20, min_periods=20).mean()

    comp_dates = {
        key: pd.DatetimeIndex(sorted(value.keys()))
        for key, value in components.items()
    }

    result = pd.Series(np.nan, index=data_index, dtype="float64")
    for dt in data_index:
        if dt not in strong.index:
            continue
        ratios = {}
        row = strong.loc[dt]
        for index_code in ["399370.SZ", "399371.SZ"]:
            dates = comp_dates[index_code]
            loc = dates.searchsorted(dt, side="right") - 1
            if loc < 0:
                continue
            tickers = [x for x in components[index_code][dates[loc]] if x in strong.columns]
            if not tickers:
                continue
            ratios[index_code] = row[tickers].mean(skipna=True)
        if "399370.SZ" in ratios and "399371.SZ" in ratios:
            result.at[dt] = ratios["399370.SZ"] - ratios["399371.SZ"]
    return result


def _direction_score(signal: pd.Series) -> pd.Series:
    score = pd.Series(np.nan, index=signal.index, dtype="float64")
    score.loc[signal > 0] = 1.0
    score.loc[signal < 0] = -1.0
    score.loc[signal == 0] = 0.0
    return score


def _assign(data_df: pd.DataFrame, raw_col: str, series: pd.Series) -> None:
    s = series.copy().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    data_df[raw_col] = s.reindex(data_df.index).astype("float64")


def generate_odds_win_factors(
    data_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    *,
    data_dir: Path | str = Path("data"),
    include_strong_ratio: bool = True,
    mount_final: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    data_dir = Path(data_dir)
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    if market_df is not None:
        market_df = market_df.copy()
        market_df.index = pd.to_datetime(market_df.index)
        market_df = market_df.sort_index()

    idx = pd.DatetimeIndex(data_df.index)

    cn10y = _read_indicator_series(data_dir / "Daily" / "D_国债到期收益率_CN_020104_260409.xlsx", "中债国债到期收益率:10年")
    cn10y_rank = _rolling_quantile_rank_year(cn10y, year=3)
    _assign(data_df, "OW_F001_CN10Y_Q3Y_raw", (0.5 - cn10y_rank).reindex(idx))

    us6m = _read_indicator_series(data_dir / "Daily" / "D_国债收益率_US_530430_260324.xlsx", "美国:国债收益率:6个月")
    us6m_rank = _rolling_quantile_rank_year(us6m, year=3)
    _assign(data_df, "OW_F002_US6M_Q3Y_raw", (0.5 - us6m_rank).reindex(idx))

    macro_path = data_dir / "宏观大事_合并整理.xlsx"
    pmi = _load_china_macro_series(macro_path, "官方制造业PMI").dropna().sort_index()
    pmi_factor = pmi.rolling(3, min_periods=3).mean() - pmi.rolling(36, min_periods=36).mean()
    _assign(data_df, "OW_F003_PMI_DIFF_raw", _month_end_on_data_index(-pmi_factor, idx))

    loan = _read_indicator_series(data_dir / "Monthly" / "DebtData.xlsx", "中国:金融机构各项贷款余额:中长期:人民币")
    loan.index = loan.index.to_period("M").shift(1).to_timestamp("M")
    loan = loan[~loan.index.duplicated(keep="last")].sort_index()
    loan_yoy = loan / loan.shift(12) - 1
    loan_factor = loan_yoy - loan_yoy.rolling(3, min_periods=3).mean()
    _assign(data_df, "OW_F004_LOAN_YOY_DIFF_raw", _month_end_on_data_index(loan_factor, idx))

    cpi = _load_china_macro_series(macro_path, "CPI:同比")
    ppi = _load_china_macro_series(macro_path, "PPI:同比")
    cpi_ppi = (cpi - ppi).dropna().sort_index()
    cpi_ppi_factor = cpi_ppi.rolling(3, min_periods=3).mean() - cpi_ppi.rolling(12, min_periods=12).mean()
    _assign(data_df, "OW_F005_CPI_PPI_DIFF_raw", _month_end_on_data_index(cpi_ppi_factor, idx))

    mom = _calc_return(data_df["close_g"], 20) - _calc_return(data_df["close_v"], 20)
    _assign(data_df, "OW_F006_MOM_4W_raw", mom)

    if include_strong_ratio:
        strong = _strong_ratio_diff(
            idx,
            data_dir / "index_data" / "filtered.parquet",
            data_dir / "Daily" / "mktP" / "mktP.parquet",
        )
        _assign(data_df, "OW_F007_STRONG_RATIO_raw", strong)
    else:
        data_df["OW_F007_STRONG_RATIO_raw"] = np.nan

    base_cols = [
        "OW_F001_CN10Y_Q3Y_raw",
        "OW_F002_US6M_Q3Y_raw",
        "OW_F003_PMI_DIFF_raw",
        "OW_F004_LOAN_YOY_DIFF_raw",
        "OW_F005_CPI_PPI_DIFF_raw",
        "OW_F006_MOM_4W_raw",
        "OW_F007_STRONG_RATIO_raw",
    ]
    score_df = pd.concat([_direction_score(data_df[col]).rename(col) for col in base_cols], axis=1)
    composite = score_df.mean(axis=1, skipna=True)
    valid_count = score_df.notna().sum(axis=1)
    composite.loc[valid_count == 0] = np.nan
    _assign(data_df, "OW_F009_COMPOSITE_SCORE_raw", composite)

    x = composite.clip(-1, 1)
    growth_win_rate = 0.5 + np.sign(x) / 2 * (np.exp(np.abs(x)) - 1) / (np.e - 1)
    growth_win_rate = growth_win_rate.clip(0, 1)
    _assign(data_df, "OW_F010_GROWTH_WIN_RATE_raw", growth_win_rate)

    growth_bp_rank = _style_bp_rank(
        data_dir / "比价" / "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx",
        idx,
    )
    value_bp_rank = _style_bp_rank(
        data_dir / "比价" / "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx",
        idx,
    )
    _assign(data_df, "OW_GROWTH_BP_RANK_3Y_raw", growth_bp_rank)
    _assign(data_df, "OW_VALUE_BP_RANK_3Y_raw", value_bp_rank)

    horizon = 20
    growth_fwd = _simple_forward_return(data_df["close_g"], horizon)
    value_fwd = _simple_forward_return(data_df["close_v"], horizon)
    growth_odds = _expanding_conditional_odds(growth_bp_rank, growth_fwd, horizon=horizon)
    value_odds = _expanding_conditional_odds(value_bp_rank, value_fwd, horizon=horizon)
    _assign(data_df, "OW_F011_GROWTH_ODDS_raw", growth_odds)
    _assign(data_df, "OW_F012_VALUE_ODDS_raw", value_odds)

    value_win_rate = 1 - growth_win_rate
    growth_exp = growth_win_rate * growth_odds - (1 - growth_win_rate)
    value_exp = value_win_rate * value_odds - (1 - value_win_rate)
    _assign(data_df, "OW_F013_GROWTH_EXPECTATION_raw", growth_exp)
    _assign(data_df, "OW_F014_VALUE_EXPECTATION_raw", value_exp)
    _assign(data_df, FINAL_RAW_FACTOR, growth_exp - value_exp)

    summary = []
    for col in [*base_cols, "OW_F009_COMPOSITE_SCORE_raw", "OW_F010_GROWTH_WIN_RATE_raw", "OW_F011_GROWTH_ODDS_raw", "OW_F012_VALUE_ODDS_raw", "OW_F013_GROWTH_EXPECTATION_raw", "OW_F014_VALUE_EXPECTATION_raw", FINAL_RAW_FACTOR]:
        s = data_df[col].replace([np.inf, -np.inf], np.nan)
        summary.append(
            {
                "raw_col": col,
                "non_na": int(s.notna().sum()),
                "first": s.first_valid_index(),
                "last": s.last_valid_index(),
                "min": s.min(),
                "max": s.max(),
            }
        )
    summary_df = pd.DataFrame(summary)

    if mount_final:
        if market_df is None:
            raise ValueError("market_df is required when mount_final=True")
        from factor_utils import merge_factor_to_market

        market_df = merge_factor_to_market(data_df, market_df, FINAL_RAW_FACTOR, factor_type="state")

    return data_df, market_df, summary_df
