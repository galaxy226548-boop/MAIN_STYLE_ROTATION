"""Target-position signal rules copied from the legacy style-rotation script."""

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


def build_state_target_position(track_df, factor_col, bar = 0):
    """
    状态型信号函数：把每一行的因子值映射成目标仓位。

    规则：
        factor > 0          -> 全仓成长  (signal_ls =  1)
        factor < 0          -> 全仓价值  (signal_ls = -1)
        factor == 0 或 NaN  -> 跟随基准曲线

    注意：
        - 这里绝对不能再做 shift，因子在挂载时已经做过 shift(1) 了。
        - NaN 和 0 都按同期基准仓位处理，引用benchmark的持仓情况，不做任何再计算。
        - signal_ls = 0 的含义是模型未发出价值/成长信号，因此也参照基准。
    """

    # 先做一个副本，避免修改传进来的原始 DataFrame
    signal_df = track_df.copy()

    # ---- 第一步：先把所有行都初始化成"中性 50/50" ----
    # 这样 NaN 和 0 就自动被兜底处理了，不用再单独写 elif
    signal_df["target_weight_g"] = 0.5
    signal_df["target_weight_v"] = 0.5
    signal_df["signal_ls"]       = 0

    # ---- 第二步：覆盖"看好成长"的行（因子值严格大于 0） ----
    mask_growth = signal_df[factor_col] > bar
    signal_df.loc[mask_growth, "target_weight_g"] = 1.0
    signal_df.loc[mask_growth, "target_weight_v"] = 0.0
    signal_df.loc[mask_growth, "signal_ls"]       = 1

    # ---- 第三步：覆盖"看好价值"的行（因子值严格小于 0） ----
    mask_value = signal_df[factor_col] < -1 * bar
    signal_df.loc[mask_value, "target_weight_g"] = 0.0
    signal_df.loc[mask_value, "target_weight_v"] = 1.0
    signal_df.loc[mask_value, "signal_ls"]       = -1

    # ---- 第四步：状态型信号每期都重新判断，所以 signal_update_flag 恒为 1 ----
    signal_df["signal_update_flag"] = 1

    return signal_df

def build_event_target_position(track_df, event_col, event_time_col=None, bar = 0):
    """
    触发型事件信号函数：只有本期发生了新事件才更新目标仓位。

    规则：
        本期有新事件，且事件方向 > 0  -> 全仓成长  (signal_ls =  1)
        本期有新事件，且事件方向 < 0  -> 全仓价值  (signal_ls = -1)
        本期有新事件，且事件方向 == 0 -> 引用基准曲线仓位
        本期没有新事件（event_col 为 NaN） -> 延续上一期信号

    重要：
        - 没有新事件时，signal_ls 必须是 NaN，不能是 0。
          因为 signal_ls = 0 已经被定义为"明确触发了中性事件"，
          如果没有事件时也写 0，后面代码就无法区分这两种情况。
        - event_time_col 参数当前版本暂不使用，接口保留供未来扩展。
    """

    # 先做一个副本，避免修改传进来的原始 DataFrame
    signal_df = track_df.copy()

    # ---- 第一步：把所有行都初始化成"无新事件"状态 ----
    # 用 NaN 表示"本期没有新事件，维持上期仓位"
    signal_df["target_weight_g"]   = np.nan
    signal_df["target_weight_v"]   = np.nan
    signal_df["signal_ls"]         = np.nan
    signal_df["signal_update_flag"] = 0       # 没有事件 -> 不更新 -> 0

    # ---- 第二步：识别"本期有新事件"的行 ----
    # 判断标准：event_col 不是 NaN 就算有新事件
    has_event_mask = signal_df[event_col].notna()

    # ---- 第三步：在有新事件的行里，再区分方向 ----

    # 情况 A：新事件看好成长（event_col > bar）
    mask_growth = has_event_mask & (signal_df[event_col] > bar)
    signal_df.loc[mask_growth, "target_weight_g"]    = 1.0
    signal_df.loc[mask_growth, "target_weight_v"]    = 0.0
    signal_df.loc[mask_growth, "signal_ls"]          = 1
    signal_df.loc[mask_growth, "signal_update_flag"] = 1

    # 情况 B：新事件看好价值（event_col < -1 * bar）
    mask_value = has_event_mask & (signal_df[event_col] < -1 * bar)
    signal_df.loc[mask_value, "target_weight_g"]    = 0.0
    signal_df.loc[mask_value, "target_weight_v"]    = 1.0
    signal_df.loc[mask_value, "signal_ls"]          = -1
    signal_df.loc[mask_value, "signal_update_flag"] = 1

    # 情况 C：新事件为中性（event_col 在 [-1*bar, bar]之间，不是 NaN）
    # 注意：这里要用 has_event_mask 做前置过滤，
    # 不能直接写 signal_df[event_col] == 0，
    # 因为 NaN == 0 在 pandas 里会返回 False（而不是 NaN），
    # 单独依赖 == 0 判断有逻辑歧义，加上 has_event_mask 更安全、更清晰
    mask_neutral = has_event_mask & (signal_df[event_col].between(-1 * bar, bar))
    signal_df.loc[mask_neutral, "target_weight_g"]    = 0.5
    signal_df.loc[mask_neutral, "target_weight_v"]    = 0.5
    signal_df.loc[mask_neutral, "signal_ls"]          = 0
    signal_df.loc[mask_neutral, "signal_update_flag"] = 1

    # ---- 第四步：event_time_col 接口预留（当前版本不做任何操作） ----
    # 如果未来需要按照事件发生的具体时间对信号做延迟处理，在这里扩展
    if event_time_col is not None:
        pass  # 当前版本暂不实现，接口保留

    return signal_df
