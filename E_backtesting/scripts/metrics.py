"""Performance metric helpers copied from the legacy style-rotation script."""

try:
    import pandas as pd
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    pd = None

try:
    import numpy as np
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    np = None

try:
    from Config import Config
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    class Config:
        RETURN_TYPE = "log"
        ANNUAL_TRADING_DAYS = 252
        Tunnels = 5
        FEATURE_LIST = []

annual_days = Config.ANNUAL_TRADING_DAYS
rf = 0.015


def calc_annualized_return(ret_series, holding_days_series, annual_days=annual_days):
    """
    计算年化收益率。

    计算逻辑：
    1. 先把每期 simple return 复利成总收益
    2. 再按总持有天数换算成年化收益

    参数
    ret_series : Series
        每期收益率（simple return）
    holding_days_series : Series
        每期持有天数
    annual_days : float
        年化换算天数

    返回：float，年化收益率（数字单位，非%）
    """
    valid_mask = ret_series.notna() & holding_days_series.notna()

    ret_series = ret_series[valid_mask]
    holding_days_series = holding_days_series[valid_mask]

    if len(ret_series) == 0:
        return np.nan

    total_return = (1 + ret_series).prod() - 1
    total_holding_days = holding_days_series.sum()

    if total_holding_days <= 0:
        return np.nan

    annualized_return = (1 + total_return) ** (annual_days / total_holding_days) - 1
    return annualized_return

def calc_annualized_vol(ret_series, holding_days_series, annual_days=annual_days):
    """
    计算年化波动率。

    计算逻辑：
    1. 先把每期 simple return 转成 log return
    2. 再除以每期 holding_days，得到“日均 log return”
    3. 对日均 log return 求标准差
    4. 乘 sqrt(annual_days)

    参数：
    ret_series : Series
        每期收益率（simple return）
    holding_days_series : Series
        每期持有天数
    annual_days : float
        年化换算天数

    返回：
    float，年化波动率（数字单位，非%）
    """
    valid_mask = ret_series.notna() & holding_days_series.notna()

    ret_series = ret_series[valid_mask]
    holding_days_series = holding_days_series[valid_mask]

    if len(ret_series) < 2:
        return np.nan

    if (holding_days_series <= 0).any():
        return np.nan

    # simple return -> log return
    log_ret_series = np.log(1 + ret_series)

    # 每期 log return 换成“日均 log return”
    daily_log_ret_series = log_ret_series / holding_days_series

    annualized_vol = daily_log_ret_series.std(ddof=1) * np.sqrt(annual_days)
    return annualized_vol

def calc_max_drawdown(nav_series):
    """
    计算最大回撤。

    参数
    nav_series : Series
        净值序列

    返回
    float
        最大回撤（正数，例如 0.15 表示回撤 15%）
    """
    nav_series = nav_series.dropna()

    if len(nav_series) == 0:
        return np.nan

    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1
    max_drawdown = abs(drawdown.min())

    return max_drawdown

def calc_sharpe_ratio(ret_series, holding_days_series, rf, annual_days=annual_days):
    """
    计算夏普比率。

    计算逻辑：
    （年化收益 - 无风险利率） / 年化波动率
    """
    annualized_return = calc_annualized_return(
        ret_series=ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    annualized_vol = calc_annualized_vol(
        ret_series=ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    if pd.isna(annualized_return) or pd.isna(annualized_vol):
        return np.nan

    if annualized_vol == 0:
        return np.nan

    sharpe_ratio = (annualized_return - rf) / annualized_vol
    return sharpe_ratio

def calc_information_ratio(excess_ret_series, holding_days_series, annual_days=annual_days):
    """
    计算信息比率。

    计算逻辑：
    1. 用逐期超额收益序列算年化超额收益
    2. 用逐期超额收益序列算年化波动
    3. 年化超额收益 / 年化超额波动
    """
    annualized_excess_return = calc_annualized_return(
        ret_series=excess_ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    annualized_excess_vol = calc_annualized_vol(
        ret_series=excess_ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    if pd.isna(annualized_excess_return) or pd.isna(annualized_excess_vol):
        return np.nan

    if annualized_excess_vol == 0:
        return np.nan

    information_ratio = annualized_excess_return / annualized_excess_vol
    return information_ratio

def calc_monthly_win_rate(nav_series):
    """
    计算月度胜率。

    计算逻辑：
    1. 把 NAV 按月末重采样
    2. 计算月收益
    3. 统计月收益 > 0 的占比
    """
    nav_series = nav_series.dropna()

    if len(nav_series) == 0:
        return np.nan

    month_end_nav = nav_series.resample("ME").last()
    monthly_ret = month_end_nav.pct_change().dropna()

    if len(monthly_ret) == 0:
        return np.nan

    monthly_win_rate = (monthly_ret > 0).mean()
    return monthly_win_rate

def calc_payoff_ratio(weekly_ret_series):
    """
    计算盈亏比率。

    计算逻辑：
    正收益期平均值 / 负收益期平均值绝对值
    """
    weekly_ret_series = weekly_ret_series.dropna()

    if len(weekly_ret_series) == 0:
        return np.nan

    positive_ret = weekly_ret_series[weekly_ret_series > 0]
    negative_ret = weekly_ret_series[weekly_ret_series < 0]

    if len(positive_ret) == 0 or len(negative_ret) == 0:
        return np.nan

    payoff_ratio = positive_ret.mean() / abs(negative_ret.mean())
    return payoff_ratio

def calc_calmar_ratio(annualized_return, max_drawdown):
    """
    计算卡玛比率。

    计算逻辑：
    年化收益 / 最大回撤
    """
    if pd.isna(annualized_return) or pd.isna(max_drawdown):
        return np.nan

    if max_drawdown == 0:
        return np.nan

    calmar_ratio = annualized_return / max_drawdown
    return calmar_ratio

def calc_market_regime_win_rate(track_result_df):
    """
    计算成长 / 价值波段胜率。

    成长波段胜率：
        在 fwd_ret_g_simple > fwd_ret_v_simple 的区间里，
        period_ret_long_net > 0 的占比

    价值波段胜率：
        在 fwd_ret_g_simple < fwd_ret_v_simple 的区间里，
        period_ret_long_net > 0 的占比

    返回
    ----------
    growth_regime_win_rate : float
    value_regime_win_rate : float
    """
    growth_mask = track_result_df["fwd_ret_g_simple"] > track_result_df["fwd_ret_v_simple"]
    value_mask = track_result_df["fwd_ret_g_simple"] < track_result_df["fwd_ret_v_simple"]

    growth_df = track_result_df[growth_mask]
    value_df = track_result_df[value_mask]

    if len(growth_df) == 0:
        growth_regime_win_rate = np.nan
    else:
        growth_regime_win_rate = (growth_df["period_ret_long_net"] > 0).mean()

    if len(value_df) == 0:
        value_regime_win_rate = np.nan
    else:
        value_regime_win_rate = (value_df["period_ret_long_net"] > 0).mean()

    return growth_regime_win_rate, value_regime_win_rate
