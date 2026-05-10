"""Return conversion helpers for the style-rotation backtest."""

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


def to_simple_return(ret_series, return_type=Config.RETURN_TYPE):
    """
    把底层收益率统一转换成 simple return，便于后面计算净值、回撤、夏普等指标。
    """
    ret_series = pd.Series(ret_series).astype(float)

    if return_type == "log":
        return np.expm1(ret_series)
    elif return_type == "simple":
        return ret_series
