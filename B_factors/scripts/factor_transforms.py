"""因子脚本共享的纯时间序列转换函数。

本模块刻意不包含数据文件名、因子编号、输出前缀、业务方向乘数、
挂载逻辑、信号生成逻辑或任何文件输出副作用。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    """将输入数据整理为浮点时间序列。

    功能：把 series 转为 float64，并使用传入 index 构造 DatetimeIndex；会删除空日期、
        按日期排序，并在重复日期中保留最后一条。
    输入：series，待转换的数据；index，对应的日期序列或日期索引；name，输出序列名称。
    输出：按日期排序、索引唯一的 float64 类型 pd.Series。
    """
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


def positive_series(series: pd.Series) -> pd.Series:
    """保留正值序列。

    功能：将输入转为 float64，并只保留大于 0 的值，其余位置置为 NaN。
    输入：series，待筛选的数值序列。
    输出：仅正值有效的 float64 类型 pd.Series。
    """
    s = series.astype("float64").copy()
    return s.where(s > 0)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """执行安全除法。

    功能：将分母中的 0 替换为 NaN 后再执行逐项相除，避免产生无穷值。
    输入：numerator，分子序列；denominator，分母序列。
    输出：分子除以分母后的 float64 序列，分母为 0 的位置为 NaN。
    """
    denom = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / denom


def rolling_std_breakout(
    series: pd.Series,
    window: int,
    min_periods: int,
    std_multiplier: float = 1.0,
) -> pd.Series:
    """计算滚动标准差突破值。

    功能：比较当前值相对滚动均值的偏离，当绝对偏离超过指定倍数滚动标准差时，
        输出带方向的标准化突破强度。
    输入：series，原始数值序列；window，滚动窗口长度；min_periods，窗口最少有效样本数；
        std_multiplier，突破阈值对应的标准差倍数。
    输出：滚动标准差标准化后的突破强度序列。
    """
    s = series.astype("float64").sort_index()
    rolling_mean = s.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = s.rolling(window=window, min_periods=min_periods).std()
    deviation = s - rolling_mean
    return safe_divide(
        np.sign(deviation) * (deviation.abs() - std_multiplier * rolling_std).clip(lower=0),
        rolling_std,
    )


def rolling_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算滚动窗口内末值百分位排名。

    功能：在每个滚动窗口内，对非空值做百分位排名，并返回窗口最后一个值的排名。
    输入：series，原始数值序列；window，滚动窗口长度；min_periods，窗口最少有效样本数。
    输出：窗口末值百分位排名序列。
    """
    s = series.astype("float64").dropna().sort_index()

    def rank_last(window_values: np.ndarray) -> float:
        """计算当前窗口最后一个非空值的百分位排名。"""
        window_series = pd.Series(window_values).dropna()
        if window_series.empty:
            return np.nan
        return float(window_series.rank(pct=True).iloc[-1])

    return s.rolling(window=window, min_periods=min_periods).apply(rank_last, raw=True)


def rolling_log_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算正值序列对数值的滚动线性斜率。

    功能：对正值取对数，并在每个滚动窗口内用时间位置回归对数值，返回线性斜率。
    输入：series，原始数值序列；window，滚动窗口长度；min_periods，窗口最少有效样本数。
    输出：滚动窗口内对数值相对时间位置的回归斜率序列。
    """
    s = series.astype("float64").dropna().sort_index()
    log_s = np.log(s.where(s > 0))

    def slope_last(window_values: np.ndarray) -> float:
        """计算当前窗口内有效对数值相对时间位置的线性回归斜率。"""
        y = pd.Series(window_values, dtype="float64")
        valid = y.notna()
        if int(valid.sum()) < min_periods:
            return np.nan
        y_valid = y.loc[valid].to_numpy()
        x = np.arange(len(y), dtype="float64")[valid.to_numpy()]
        x_demeaned = x - float(x.mean())
        denominator = float((x_demeaned ** 2).sum())
        if np.isclose(denominator, 0.0):
            return np.nan
        y_demeaned = y_valid - float(y_valid.mean())
        return float((x_demeaned * y_demeaned).sum() / denominator)

    return log_s.rolling(window=window, min_periods=min_periods).apply(slope_last, raw=True)


def rolling_time_corr(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算序列与时间位置的滚动相关系数。

    功能：为序列构造递增时间位置，并计算每个滚动窗口内数值与时间位置的相关性。
    输入：series，原始数值序列；window，滚动窗口长度；min_periods，窗口最少有效样本数。
    输出：滚动时间相关系数序列。
    """
    s = series.astype("float64").dropna().sort_index()
    time_index = pd.Series(np.arange(len(s), dtype="float64"), index=s.index)
    return s.rolling(window=window, min_periods=min_periods).corr(time_index)


def trailing_time_window(
    series: pd.Series,
    dt: pd.Timestamp,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
) -> pd.Series:
    """截取指定日期之前的尾部时间窗口。

    功能：根据 months、years 或 days 之一确定窗口起点，返回 (start, dt] 区间内的数据；
        若未提供任何时间单位，会抛出 ValueError。
    输入：series，带日期索引的序列；dt，窗口结束日期；months、years、days，窗口长度参数。
    输出：位于尾部时间窗口内的 pd.Series。
    """
    if months is not None:
        start = dt - pd.DateOffset(months=int(round(months)))
    elif years is not None:
        start = dt - pd.DateOffset(years=int(round(years)))
    elif days is not None:
        start = dt - pd.Timedelta(days=float(days))
    else:
        raise ValueError("one of months, years, or days must be provided")
    return series.loc[(series.index > start) & (series.index <= dt)]


def time_window_apply(
    series: pd.Series,
    func: Callable[[pd.Series], float],
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """在尾部时间窗口上逐点应用函数。

    功能：对每个日期截取尾部时间窗口，样本数满足 min_periods 时调用 func 计算结果，
        否则返回 NaN。
    输入：series，原始数值序列；func，接收窗口序列并返回标量的函数；
        months、years、days，窗口长度参数；min_periods，窗口最少有效样本数。
    输出：逐日期窗口计算得到的 float64 序列。
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    values = []
    for dt in s.index:
        window = trailing_time_window(s, dt, months=months, years=years, days=days)
        values.append(np.nan if len(window) < min_periods else func(window))
    return pd.Series(values, index=s.index, dtype="float64")


def time_mean(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """计算尾部时间窗口均值。

    功能：对每个日期截取指定长度的尾部时间窗口，并计算窗口均值。
    输入：series，原始数值序列；months、years、days，窗口长度参数；
        min_periods，窗口最少有效样本数。
    输出：逐日期窗口均值序列。
    """
    return time_window_apply(
        series,
        lambda window: window.mean(),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def rolling_quantile_value(
    series: pd.Series,
    target_quantile: float,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """计算尾部时间窗口分位数。

    功能：对每个日期截取指定长度的尾部时间窗口，并计算目标分位数。
    输入：series，原始数值序列；target_quantile，目标分位点；
        months、years、days，窗口长度参数；min_periods，窗口最少有效样本数。
    输出：逐日期窗口分位数序列。
    """
    return time_window_apply(
        series,
        lambda window: window.quantile(target_quantile),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def calc_rolling_zscore_time(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """计算基于尾部时间窗口的滚动 z-score。

    功能：对每个日期截取指定长度的尾部时间窗口，用当前值减窗口均值后除以窗口标准差；
        样本不足或标准差为空/接近 0 时返回 NaN。
    输入：series，原始数值序列；months、years、days，窗口长度参数；
        min_periods，窗口最少有效样本数。
    输出：逐日期滚动 z-score 序列。
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    values = []
    for dt in s.index:
        window = trailing_time_window(s, dt, months=months, years=years, days=days)
        if len(window) < min_periods:
            values.append(np.nan)
            continue
        rolling_std = window.std()
        if pd.isna(rolling_std) or np.isclose(rolling_std, 0):
            values.append(np.nan)
        else:
            values.append((s.loc[dt] - window.mean()) / rolling_std)
    return pd.Series(values, index=s.index, dtype="float64")


def data_deviation(
    series: pd.Series,
    *,
    months: int | float | None = None,
    years: int | float | None = None,
    days: int | float | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """计算数据相对尾部时间窗口均值的偏离。

    功能：先计算指定时间窗口内的滚动均值，再用原序列减去对应均值。
    输入：series，原始数值序列；months、years、days，窗口长度参数；
        min_periods，窗口最少有效样本数。
    输出：原值减滚动均值后的偏离度序列。
    """
    s = pd.to_numeric(series, errors="coerce").sort_index()
    rolling_mean = time_mean(
        s,
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )
    return s - rolling_mean.reindex(s.index)


def z_data_deviation(
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
    """计算偏离度的滚动 z-score。

    功能：先按 dev_* 时间窗口计算原序列相对均值的偏离度，再按 z_* 时间窗口
        对偏离度做滚动 z-score 标准化。
    输入：series，原始数值序列；dev_months、dev_years、dev_days，偏离度窗口长度；
        z_months、z_years、z_days，z-score 窗口长度；dev_min_periods、z_min_periods，
        两阶段窗口最少有效样本数。
    输出：偏离度标准化后的滚动 z-score 序列。
    """
    deviation = data_deviation(
        series,
        months=dev_months,
        years=dev_years,
        days=dev_days,
        min_periods=dev_min_periods,
    )
    return calc_rolling_zscore_time(
        deviation,
        months=z_months,
        years=z_years,
        days=z_days,
        min_periods=z_min_periods,
    )


def z_ma_div(
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
    """计算短长均线差的滚动 z-score。

    功能：分别计算短窗口和长窗口均值，用短均线减长均线，再对差值做滚动 z-score；
        unit 为 "days" 时使用天数窗口，否则使用月度窗口。
    输入：series，原始数值序列；short_window，短均线窗口；long_window，长均线窗口；
        z_window，z-score 窗口；unit，窗口单位；short_min_periods、long_min_periods、
        z_min_periods，各阶段最少有效样本数，未传入时使用函数内默认规则。
    输出：短长均线差标准化后的滚动 z-score 序列。
    """
    if unit == "days":
        short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, int(round(short_window / 30 / 2)))
        long_min_periods = long_min_periods or max(6, int(round(long_window / 30 / 2)))
        z_min_periods = z_min_periods or max(12, int(round(z_window / 30 / 2)))
        short_ma = series.copy() if short_window == 1 else time_mean(series, days=short_window, min_periods=short_min_periods)
        long_ma = time_mean(series, days=long_window, min_periods=long_min_periods)
        return calc_rolling_zscore_time(short_ma - long_ma, days=z_window, min_periods=z_min_periods)

    short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, int(short_window) // 2)
    long_min_periods = long_min_periods or max(3, int(long_window) // 2)
    z_min_periods = z_min_periods or max(6, int(z_window) // 2)
    short_ma = series.copy() if short_window == 1 else time_mean(series, months=short_window, min_periods=short_min_periods)
    long_ma = time_mean(series, months=long_window, min_periods=long_min_periods)
    return calc_rolling_zscore_time(short_ma - long_ma, months=z_window, min_periods=z_min_periods)
