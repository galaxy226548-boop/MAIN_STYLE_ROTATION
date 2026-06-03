"""factor_factory_template_registry 中登记的因子值生成模板。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _params_or_empty(params: Mapping[str, object] | None) -> Mapping[str, object]:
    if params is None:
        return {}
    return params


def _as_sorted_numeric_series(series: pd.Series) -> pd.Series:
    return series.astype("float64").sort_index()


def _rolling_quantile_rank(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """计算当前值在滚动窗口内的百分位排名。"""
    if min_periods is None:
        min_periods = window
    s = _as_sorted_numeric_series(series)

    def rank_last(values: np.ndarray) -> float:
        # 只用窗口内的有效值排名，窗口有效样本不足时保留 NaN。
        valid = values[~np.isnan(values)]
        if len(valid) < min_periods:
            return np.nan
        return float((valid <= valid[-1]).sum() / len(valid))

    return s.rolling(window=window, min_periods=min_periods).apply(rank_last, raw=True)


def _rolling_ols_slope(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """滚动拟合一元线性趋势，返回斜率系数。"""
    if min_periods is None:
        min_periods = window
    s = _as_sorted_numeric_series(series)

    def slope_last(values: np.ndarray) -> float:
        # 用窗口内有效样本做 OLS，x 保留原窗口位置以反映时间间隔。
        valid = ~np.isnan(values)
        if int(valid.sum()) < min_periods:
            return np.nan
        y = values[valid].astype("float64")
        x = np.arange(len(values), dtype="float64")[valid]
        x_demeaned = x - float(x.mean())
        denominator = float((x_demeaned ** 2).sum())
        if np.isclose(denominator, 0.0):
            return np.nan
        y_demeaned = y - float(y.mean())
        return float((x_demeaned * y_demeaned).sum() / denominator)

    return s.rolling(window=window, min_periods=min_periods).apply(slope_last, raw=True)


def rolling_rank_level(sub_1: pd.Series, params: Mapping[str, object] | None = None) -> pd.Series:
    """水平高低模板：用滚动分位衡量当前水平偏高或偏低。"""
    params = _params_or_empty(params)
    lookback_window = int(params.get("lookback_window", 756))
    center = float(params.get("center", 0.5))
    direction = float(params.get("direction", -1))

    # direction=-1 时，高分位映射为负值，低分位映射为正值。
    rank = _rolling_quantile_rank(sub_1, window=lookback_window)
    factor = direction * (rank - center)
    return factor.astype("float64")


def ma_spread_rank(sub_1: pd.Series, params: Mapping[str, object] | None = None) -> pd.Series:
    """趋势方向模板 A：用短长均线差衡量趋势方向，再转为滚动分位。"""
    params = _params_or_empty(params)
    short_window = int(params.get("short_window", 20))
    long_window = int(params.get("long_window", 120))
    rank_window = int(params.get("rank_window", 756))
    center = float(params.get("center", 0.5))
    direction = float(params.get("direction", -1))

    s = _as_sorted_numeric_series(sub_1)
    # 短均线高于长均线代表趋势更强，再通过滚动分位统一量纲。
    short_ma = s.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = s.rolling(window=long_window, min_periods=long_window).mean()
    spread = short_ma - long_ma
    rank = _rolling_quantile_rank(spread, window=rank_window)
    factor = direction * (rank - center)
    return factor.astype("float64")


def ols_slope_rank(sub_1: pd.Series, params: Mapping[str, object] | None = None) -> pd.Series:
    """趋势方向模板 B：用 OLS 斜率衡量趋势方向，再转为滚动分位。"""
    params = _params_or_empty(params)
    slope_window = int(params.get("slope_window", 60))
    rank_window = int(params.get("rank_window", 756))
    center = float(params.get("center", 0.5))
    direction = float(params.get("direction", -1))

    slope_series = _rolling_ols_slope(sub_1, window=slope_window)
    rank = _rolling_quantile_rank(slope_series, window=rank_window)
    factor = direction * (rank - center)
    return factor.astype("float64")


def slope_accel_quadrant(sub_1: pd.Series, params: Mapping[str, object] | None = None) -> pd.Series:
    """趋势加速度模板：在趋势方向基础上叠加趋势加速/减速信息。"""
    params = _params_or_empty(params)
    slope_window = int(params.get("slope_window", 60))
    diff_window = int(params.get("diff_window", 20))
    rank_window = int(params.get("rank_window", 756))
    center = float(params.get("center", 0.5))
    direction = float(params.get("direction", -1))
    weight = float(params.get("accel_weight", 0.5))

    slope = _rolling_ols_slope(sub_1, window=slope_window)
    slope_accel = slope.diff(diff_window)

    # s 表示趋势方向所处分位，a 表示趋势加速度所处分位，均以 center 为中性点。
    s = _rolling_quantile_rank(slope, window=rank_window) - center
    a = _rolling_quantile_rank(slope_accel, window=rank_window) - center

    # 加速度只强化当前趋势方向：上行趋势叠加 |a|，下行趋势扣减 |a|。
    sign_s = np.sign(s)
    raw = s + sign_s * a.abs() * weight

    # 再做一次滚动分位，保持输出和其他模板一致的中心化分位因子。
    rank = _rolling_quantile_rank(raw, window=rank_window)
    factor = direction * (rank - center)
    return factor.astype("float64")
