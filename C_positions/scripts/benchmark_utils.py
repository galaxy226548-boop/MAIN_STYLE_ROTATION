"""Benchmark position helpers copied from the legacy style-rotation script."""

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


def build_benchmark_rebalance_flag(track_index, rebalance_months):
    """
    给单条轨道生成再平衡标记：
    - 第一行一定标记为 1（期初建仓）
    - 之后当“当前日期 >= 上一次再平衡日期 + n个月”时，
      在该轨道上第一个满足条件的交易点标记为 1
    - 其他行标记为 0

    这样可以适配：
    - 周频轨道
    - 节假日导致的日期不整齐
    - “满 n 个月后的首个可交易点”再平衡
    """
    track_index = pd.DatetimeIndex(track_index).sort_values()

    rebalance_flag = pd.Series(0, index=track_index, dtype="int64")

    last_rebalance_dt = None

    for dt in track_index:
        if last_rebalance_dt is None:
            rebalance_flag.loc[dt] = 1 #期初第一次调仓
            last_rebalance_dt = dt
            continue

        next_due_dt = last_rebalance_dt + pd.DateOffset(months=rebalance_months)

        if dt >= next_due_dt:
            rebalance_flag.loc[dt] = 1
            last_rebalance_dt = dt

    return rebalance_flag


def _weights_are_different(weight_g_a, weight_v_a, weight_g_b, weight_v_b, tol):
    return (abs(weight_g_a - weight_g_b) > tol) or (abs(weight_v_a - weight_v_b) > tol)


def build_daily_benchmark_position(
    data_df,
    start_date=None,
    end_date=None,
    benchmark_mode="rebalance_50",
    rebalance_months=2,
    weight_tol=1e-10,
):
    """
    Build the daily benchmark position curve on data_df.index.

    The benchmark is a 50/50 growth-value portfolio. Between rebalance dates,
    weights drift with the two index close prices. On rebalance dates, the
    post-trade weights are reset to 50/50.
    """
    if pd is None:
        raise ImportError("pandas is required to build benchmark positions")

    required_cols = ["close_g", "close_v"]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise KeyError(f"data_df is missing required columns: {missing_cols}")

    if benchmark_mode not in {"buy_and_hold_50", "rebalance_50"}:
        raise ValueError(
            "benchmark_mode must be 'buy_and_hold_50' or 'rebalance_50', "
            f"got {benchmark_mode!r}"
        )

    df = data_df.copy().sort_index()
    df.index = pd.to_datetime(df.index)

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    df = df[required_cols].dropna(how="any")
    if df.empty:
        raise ValueError("No valid rows are available for the requested benchmark window")

    rows = []
    prev_post_weight_g = None
    prev_post_weight_v = None
    prev_close_g = None
    prev_close_v = None
    last_rebalance_dt = None

    for dt, row in df.iterrows():
        close_g = float(row["close_g"])
        close_v = float(row["close_v"])

        if prev_post_weight_g is None:
            pre_trade_weight_g = 0.0
            pre_trade_weight_v = 0.0
            signal_update_flag = 1
        else:
            ret_g = close_g / prev_close_g - 1
            ret_v = close_v / prev_close_v - 1
            end_value_g = prev_post_weight_g * (1 + ret_g)
            end_value_v = prev_post_weight_v * (1 + ret_v)
            total_value = end_value_g + end_value_v
            pre_trade_weight_g = end_value_g / total_value
            pre_trade_weight_v = end_value_v / total_value

            if benchmark_mode == "buy_and_hold_50":
                signal_update_flag = 0
            else:
                next_due_dt = last_rebalance_dt + pd.DateOffset(months=rebalance_months)
                signal_update_flag = int(dt >= next_due_dt)

        if signal_update_flag == 1:
            post_trade_weight_g = 0.5
            post_trade_weight_v = 0.5
        else:
            post_trade_weight_g = pre_trade_weight_g
            post_trade_weight_v = pre_trade_weight_v

        trade_flag = int(
            _weights_are_different(
                pre_trade_weight_g,
                pre_trade_weight_v,
                post_trade_weight_g,
                post_trade_weight_v,
                weight_tol,
            )
        )

        if signal_update_flag == 1:
            last_rebalance_dt = dt

        rows.append(
            {
                "close_g": close_g,
                "close_v": close_v,
                "target_weight_g": 0.5,
                "target_weight_v": 0.5,
                "signal_ls": 0,
                "signal_update_flag": signal_update_flag,
                "pre_trade_weight_g": pre_trade_weight_g,
                "pre_trade_weight_v": pre_trade_weight_v,
                "post_trade_weight_g": post_trade_weight_g,
                "post_trade_weight_v": post_trade_weight_v,
                "trade_flag": trade_flag,
            }
        )

        prev_post_weight_g = post_trade_weight_g
        prev_post_weight_v = post_trade_weight_v
        prev_close_g = close_g
        prev_close_v = close_v

    out = pd.DataFrame(rows, index=df.index)
    out.index.name = df.index.name or "date"
    return out
