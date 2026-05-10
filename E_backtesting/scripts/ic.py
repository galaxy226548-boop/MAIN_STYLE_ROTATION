"""Information coefficient analysis helpers copied from the legacy style-rotation script."""

try:
    import pandas as pd
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    pd = None

try:
    import numpy as np
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    np = None

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    pearsonr = None
    spearmanr = None

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    OLS = None
    add_constant = None

try:
    from Config import Config
except Exception:  # pragma: no cover - import-only fallback for minimal environments
    class Config:
        RETURN_TYPE = "log"


def calc_future_horizon_return(ret_series, horizon, return_type = Config.RETURN_TYPE):
    """
    在“同一条轨道内部”，计算未来 horizon 周的累计收益。

    参数
    ----
    ret_series : pd.Series
        单周期未来收益率序列，例如 market_df["fwd_ret_g"]
    horizon : int
        累计窗口长度（未来几周）
    return_type : str
        "log" 或 "simple"，必须与主研究方案一致

    返回
    ----
    pd.Series
        与原索引对齐的“未来 horizon 个调仓周期（未必都是1周）的累计收益率”
    """
    if return_type == "log":
        # 对数收益率可直接相加
        future_ret = (
            ret_series.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .sum()
            .iloc[::-1]
        )
    elif return_type == "simple":
        # 简单收益率必须先转成 (1 + r) 连乘，再减 1
        future_ret = (
            (1 + ret_series).iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .apply(np.prod, raw=True)
            .iloc[::-1]
            - 1
        )

    return future_ret

def calc_ic_stats(x, y, nw_lag=1, min_n=10):
    """
    统一计算：
    1) Pearson IC
    2) Rank IC（Spearman）
    3) 有效样本数 N
    4) 仅针对 Pearson 线性关系的 Newey-West t 值 / p 值

    x : 因子序列
    y : 目标收益序列（如 bucket_target_k / horizon_target_h）
    nw_lag : int, default=1
        Newey-West(HAC) 滞后阶数
    min_n : int, default=10
        最小有效样本数门槛。若清洗缺失值后样本数不足，则直接返回全 NaN

    返回
    - pearson_ic : Pearson 相关系数
    - rank_ic    : Spearman 相关系数
    - N          : 有效样本数
    - nw_t       : Pearson 线性回归中 beta 的 Newey-West t 值
    - nw_p       : Pearson 线性回归中 beta 的 Newey-West p 值

    """

    # 1. 拼接数据并清洗缺失值
    df = pd.DataFrame({
        "x": pd.Series(x),
        "y": pd.Series(y)
    })

    n_total = len(df)
    df = df.dropna()
    n = len(df) #n_valid
    coverage = n / n_total if n_total > 0 else np.nan

    if n < min_n or df["x"].nunique() <= 1 or df["y"].nunique() <= 1:
        return pd.Series({
            "pearson_ic": np.nan,
            "rank_ic": np.nan,
            "N_valid": n,
            "N_total": n_total,
            "coverage": coverage,
            "nw_t": np.nan,
            "nw_p": np.nan
        })

    # 2. 计算 Pearson IC 和 Rank IC
    pearson_ic = pearsonr(df["x"], df["y"])[0]
    rank_ic = spearmanr(df["x"], df["y"])[0]

    # 3. 计算 Pearson 对应线性关系的 NW t / p，回归形式： y = alpha + beta * x + error
    nw_t = np.nan
    nw_p = np.nan

    X = add_constant(df["x"])
    model = OLS(df["y"], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(nw_lag)}
    )

    # x 这一列对应的系数，就是 beta
    beta_name = "x"
    nw_t = model.tvalues.get(beta_name, np.nan)
    nw_p = model.pvalues.get(beta_name, np.nan)

    # -------------------------
    # 4. 输出结果
    # -------------------------
    return pd.Series({
        "pearson_ic": pearson_ic,
        "rank_ic": rank_ic,
        "N_valid": n,
        "N_total": n_total,
        "coverage": coverage,
        "nw_t": nw_t,
        "nw_p": nw_p
    })

def calc_rolling_ic(x, y, window=12, min_n=10):
    """
    逐点滚动计算 Pearson IC。
    
    参数
    ----------
    x : 因子序列
    y : 目标收益序列
    window : int, default=12
        滚动窗口长度。
        这里按“每条轨道周频数据的 3 个月 ≈ 12 个观测点”处理。
    min_n : int, default=10
        每个滚动窗口内最少有效样本数。
        不足时返回 NaN。
    
    返回
    ----------
    rolling_ic : pd.Series
        与输入索引对齐的滚动 IC 序列
    """
    df = pd.DataFrame({
        "x": pd.Series(x),
        "y": pd.Series(y)
    })

    rolling_ic = []

    for i in range(len(df)):
        # 当前点及其之前 window-1 个点
        start_idx = max(0, i - window + 1)
        sub_df = df.iloc[start_idx:i+1].dropna()

        if len(sub_df) < min_n:
            rolling_ic.append(np.nan)
            continue

        if sub_df["x"].nunique() <= 1 or sub_df["y"].nunique() <= 1:
            rolling_ic.append(np.nan)
            continue

        ic = pearsonr(sub_df["x"], sub_df["y"])[0]
        rolling_ic.append(ic)

    rolling_ic = pd.Series(rolling_ic, index=df.index, name="ic")
    return rolling_ic
