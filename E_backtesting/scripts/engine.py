"""Single-track execution engine helpers copied from the legacy style-rotation script."""

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

weight_tol = 1e-10


def calc_pre_trade_actual_weight(
    prev_post_weight_g, prev_post_weight_v, prev_ret_g_simple, prev_ret_v_simple
):
    """
    把“上一期调仓后权重”根据上一期真实资产涨跌，
    推到“本期调仓前的实际权重”。

    输入：
    prev_post_weight_g : float
        上一期调仓后，成长资产权重
    prev_post_weight_v : float
        上一期调仓后，价值资产权重
    prev_ret_g_simple : float
        上一期持有区间内，成长资产 simple return
    prev_ret_v_simple : float
        上一期持有区间内，价值资产 simple return

    输出：
    pre_trade_weight_g : float
        本期调仓前，成长资产的实际权重
    pre_trade_weight_v : float
        本期调仓前，价值资产的实际权重
    """

    end_value_g = prev_post_weight_g * (1 + prev_ret_g_simple)
    end_value_v = prev_post_weight_v * (1 + prev_ret_v_simple)

    total_value = end_value_g + end_value_v

    pre_trade_weight_g = end_value_g / total_value
    pre_trade_weight_v = end_value_v / total_value

    return pre_trade_weight_g, pre_trade_weight_v

def weights_are_different(weight_g_a, weight_v_a, weight_g_b, weight_v_b, tol = weight_tol):
    diff_g = abs(weight_g_a - weight_g_b)
    diff_v = abs(weight_v_a - weight_v_b)

    if (diff_g > tol) or (diff_v > tol):
        return True
    else:
        return False

def decide_post_trade_weight(
    pre_trade_weight_g,
    pre_trade_weight_v,
    signal_ls,
    target_weight_g,
    target_weight_v,
    signal_update_flag,
    prev_effective_signal_ls,
    benchmark_mode,
    weight_tol = weight_tol,
    receiver = "benchmark",  #根据计算对象是策略还是基准，选用"strategy"或"benchmark"
    benchmark_ref_row = None #如果receiver是"strategy"，则需要提供一个benchmark_ref_row，包含该轨道对应日期的benchmark仓位信息，以便在signal_ls=0时直接取用benchmark仓位
):
    """
    根据当前目标仓位、上一期有效仓位、本期是否有新信号，
    决定本期是否交易，以及交易后的仓位。

    输入：
    pre_trade_weight_g / pre_trade_weight_v
        本期调仓前的实际权重

    signal_ls
        本期原始目标方向：
        1  = 全仓成长
        -1 = 全仓价值
        0  = 50/50 中性
        NaN = 本期没有新事件（只会在 event 模式下出现）

    target_weight_g / target_weight_v
        本期原始目标权重

    signal_update_flag
        1 = 本期有新状态 / 新事件
        0 = 本期没有新事件，沿用上一期

    prev_effective_signal_ls
        上一期真正生效的信号方向
        第一行没有上一期，用 NaN 表示

    benchmark_mode
        "buy_and_hold_50" 或 "rebalance_50"

    weight_tol
        权重比较容忍度

    输出：
    ----------
    effective_signal_ls
    post_trade_weight_g
    post_trade_weight_v
    trade_flag
    signal_count
    """

    # =========================================================
    # 情况 A：第一期建仓
    # 判断标准：上一期有效仓位不存在
    # =========================================================
    is_first_row = pd.isna(prev_effective_signal_ls)

    if receiver == "benchmark":
        effective_signal_ls = 0

        # ---- 第一行：期初建仓到 50/50 ----
        if is_first_row:
            post_trade_weight_g = 0.5
            post_trade_weight_v = 0.5
            trade_flag = 1
            signal_count = 1

            return(
                effective_signal_ls,
                post_trade_weight_g,
                post_trade_weight_v,
                trade_flag,
                signal_count
            )

        if benchmark_mode == "buy_and_hold_50":
            post_trade_weight_g = float(pre_trade_weight_g)
            post_trade_weight_v = float(pre_trade_weight_v)

            trade_flag = 0
            signal_count = 1
        
        elif benchmark_mode == "rebalance_50":
            if signal_update_flag == 0:
                post_trade_weight_g = float(pre_trade_weight_g)
                post_trade_weight_v = float(pre_trade_weight_v)
                trade_flag = 0
                signal_count = 1

                return(
                    effective_signal_ls,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    trade_flag,
                    signal_count
                )

            else:
                post_trade_weight_g = 0.5
                post_trade_weight_v = 0.5

                need_trade = weights_are_different(
                    pre_trade_weight_g,
                    pre_trade_weight_v,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    weight_tol
                )

                trade_flag = int(need_trade)
                signal_count = 1

                return(
                    effective_signal_ls,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    trade_flag,
                    signal_count
                )

        else:
            raise ValueError(
                f"benchmark_mode 只能是 'buy_and_hold_50' 或 'rebalance_50'，当前值为：{benchmark_mode}"
            )

    elif receiver == "strategy":

        # -----------------------------------------------------
        # 情况 A：本期没有新信号（event 模式下常见）
        # 规则：维持上期 hold，不调仓
        # -----------------------------------------------------

        if signal_update_flag == 0:

            # 首期如果没有信号，无法维持上期
            if is_first_row:
                effective_signal_ls = 0

                if benchmark_ref_row is not None:
                    post_trade_weight_g = float(benchmark_ref_row["post_trade_weight_g"])
                    post_trade_weight_v = float(benchmark_ref_row["post_trade_weight_v"])
                else:
                    post_trade_weight_g = 0.5
                    post_trade_weight_v = 0.5

                need_trade = weights_are_different(
                    pre_trade_weight_g,
                    pre_trade_weight_v,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    weight_tol
                )
                trade_flag = int(need_trade)
                signal_count = 0

                return(
                    effective_signal_ls,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    trade_flag,
                    signal_count
                )

            # 正常无信号：延续上一期有效信号。这里保留原数值，
            # 以支持非 -1/0/1 的信号强度或仓位方向。
            effective_signal_ls = float(prev_effective_signal_ls)
            
            # 如果上一期已经是中性状态，后续无新事件期间继续跟随基准仓位
            if effective_signal_ls == 0:
                if benchmark_ref_row is None:
                    raise ValueError(
                        "strategy在中性状态延续时需要benchmark_ref_row提供基准仓位信息"
                    )

                post_trade_weight_g = float(benchmark_ref_row["post_trade_weight_g"])
                post_trade_weight_v = float(benchmark_ref_row["post_trade_weight_v"])

                need_trade = weights_are_different(
                    pre_trade_weight_g,
                    pre_trade_weight_v,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    weight_tol
                )

                trade_flag = int(need_trade)
                signal_count = 0

                return (
                    effective_signal_ls,
                    post_trade_weight_g,
                    post_trade_weight_v,
                    trade_flag,
                    signal_count
                )

            # 上一期是成长/价值信号时，才维持漂移仓位
            post_trade_weight_g = float(pre_trade_weight_g)
            post_trade_weight_v = float(pre_trade_weight_v)
            trade_flag = 0
            signal_count = 0

            return (
                effective_signal_ls,
                post_trade_weight_g,
                post_trade_weight_v,
                trade_flag,
                signal_count
            )
        
        # -----------------------------------------------------
        # 情况 B：本期有新信号，但signal_ls = 0
        # 规则：引用基准曲线的仓位
        # -----------------------------------------------------
        if signal_ls == 0:
            if benchmark_ref_row is None:
                raise ValueError(
                    "strategy在signal_ls=0时需要benchmark_ref_row提供基准仓位信息，但当前值为None"
                )
            
            effective_signal_ls = 0
            post_trade_weight_g = float(benchmark_ref_row["post_trade_weight_g"])
            post_trade_weight_v = float(benchmark_ref_row["post_trade_weight_v"])
            signal_count = 1

            need_trade = weights_are_different(
                pre_trade_weight_g,
                pre_trade_weight_v,
                post_trade_weight_g,
                post_trade_weight_v,
                weight_tol
            )
            trade_flag = int(need_trade)

            return(
                effective_signal_ls,
                post_trade_weight_g,
                post_trade_weight_v,
                trade_flag,
                signal_count
            )

        # -----------------------------------------------------
        # 情况 C：本期有新信号，且signal_ls非0
        # 规则：按照signal_ls记录方向，按target_weight进行调仓
        # -----------------------------------------------------

        effective_signal_ls = float(signal_ls)
        post_trade_weight_g = float(target_weight_g)
        post_trade_weight_v = float(target_weight_v)

        need_trade = weights_are_different(
            pre_trade_weight_g,
            pre_trade_weight_v,
            post_trade_weight_g,
            post_trade_weight_v,
            weight_tol
        )
        trade_flag = int(need_trade)
        signal_count = 1

        return(
            effective_signal_ls,
            post_trade_weight_g,
            post_trade_weight_v,
            trade_flag,
            signal_count
        )
        
    else:
        raise ValueError(
            f"receiver 参数只能是 'strategy' 或 'benchmark'，当前值为：{receiver}"
        )

def run_single_track_backtest(
    track_target_df,
    benchmark_mode,
    trans_fee,
    charge_initial_trade,
    receiver = "Undefined",
    benchmark_ref_track_df = None
):
    """
    按日期逐行执行单条轨道的回测。

    输入：
    track_target_df : DataFrame，单条轨道的目标仓位表
    benchmark_mode : str，"buy_and_hold_50" 或 "rebalance_50"
    trans_fee : float，单边手续费率
    charge_initial_trade : bool，True  = 第一期建仓收费，False = 第一期建仓免费
    benchmark_rebalance_months: int or None，benchmark_mode = "rebalance_50" 时需要
    receiver : str，"strategy" 或 "benchmark"，
    benchmark_ref_track_df: DataFrame or None, 仅strategy使用

    输出：
    track_result_df : DataFrame，单轨结果表，每一行代表："从当前调仓日持有到下一次同轨道调仓日"的一个完整持有区间
    """

    # ------ 第一步：初始化 ------
    track_df = track_target_df.copy().sort_index(ascending=True)

    if receiver not in {"strategy", "benchmark"}:
        raise ValueError(
            f"receiver 参数只能是 'strategy' 或 'benchmark'，当前值为：{receiver}"
        )
    
    if benchmark_ref_track_df is not None:
        benchmark_ref_track_df = benchmark_ref_track_df.copy().sort_index(ascending=True)

    # ------ 第二步：逐行循环执行 ------
    result_rows = []

    prev_post_weight_g = np.nan
    prev_post_weight_v = np.nan
    prev_ret_g_simple = np.nan
    prev_ret_v_simple = np.nan
    prev_effective_signal_ls = np.nan

    for dt, row in track_df.iterrows():
        # 第一步：确定本期调仓前实际仓位
        if pd.isna(prev_effective_signal_ls):
            # 第一行没有上一期，从“空仓现金状态”开始建仓
            pre_trade_weight_g = 0.0
            pre_trade_weight_v = 0.0
        else:
            pre_trade_weight_g, pre_trade_weight_v = calc_pre_trade_actual_weight(
                prev_post_weight_g=prev_post_weight_g,
                prev_post_weight_v=prev_post_weight_v,
                prev_ret_g_simple=prev_ret_g_simple,
                prev_ret_v_simple=prev_ret_v_simple
            )

        # 第二步：引用strategy需要的benchmark row
        benchmark_ref_row = None
        if (receiver == "strategy") and (benchmark_ref_track_df is not None):
            if dt not in benchmark_ref_track_df.index:
                raise ValueError(
                    f"receiver 是 'strategy'，且提供了 benchmark_ref_track_df，但在日期 {dt} 没有找到对应的基准数据行，请检查 benchmark_ref_track_df 的索引和日期覆盖情况"
                )
            benchmark_ref_row = benchmark_ref_track_df.loc[dt]

        # 第三步：决定本期调仓后仓位
        (
            effective_signal_ls,
            post_trade_weight_g,
            post_trade_weight_v,
            trade_flag,
            signal_count
        ) = decide_post_trade_weight(
            pre_trade_weight_g=pre_trade_weight_g,
            pre_trade_weight_v=pre_trade_weight_v,
            signal_ls=row["signal_ls"],
            target_weight_g=row["target_weight_g"],
            target_weight_v=row["target_weight_v"],
            signal_update_flag=row["signal_update_flag"],
            prev_effective_signal_ls=prev_effective_signal_ls,
            benchmark_mode=benchmark_mode,
            weight_tol=weight_tol,
            receiver = receiver,
            benchmark_ref_row = benchmark_ref_row
        )

        # 第四步：算本期换手率和费用
        if trade_flag == 1:
            turnover_2way = (
                abs(post_trade_weight_g - pre_trade_weight_g)
                + abs(post_trade_weight_v - pre_trade_weight_v)
            )
        else:
            turnover_2way = 0.0

        # 第一期建仓是否收费，由 charge_initial_trade 控制
        if pd.isna(prev_effective_signal_ls) and (charge_initial_trade is False):
            cost_rate = 0.0
        else:
            cost_rate = turnover_2way * trans_fee

        # 第五步：算本期多头组合收益
        fwd_ret_g_simple = float(row["fwd_ret_g_simple"])
        fwd_ret_v_simple = float(row["fwd_ret_v_simple"])

        period_ret_long_gross = (
            post_trade_weight_g * fwd_ret_g_simple
            + post_trade_weight_v * fwd_ret_v_simple
        )

        period_ret_long_net = (1 - cost_rate) * (1 + period_ret_long_gross) - 1

        # 第六步：算本期多空收益
        period_ret_ls = effective_signal_ls * (fwd_ret_g_simple - fwd_ret_v_simple)

        # 第七步：把所有中间量都存下来
        row_result_1 = {
            "track_id": row["track_id"],
            "next_date": row["next_date"],
            "signal_ls_raw": row["signal_ls"],
            "effective_signal_ls": effective_signal_ls,
            "signal_update_flag": row["signal_update_flag"],
            "pre_trade_weight_g": pre_trade_weight_g,
            "pre_trade_weight_v": pre_trade_weight_v,
            "post_trade_weight_g": post_trade_weight_g,
            "post_trade_weight_v": post_trade_weight_v,
            "trade_flag": trade_flag,
            "signal_count": signal_count,
            "turnover_2way": turnover_2way,
            "cost_rate": cost_rate,
            "fwd_ret_g_simple": fwd_ret_g_simple,
            "fwd_ret_v_simple": fwd_ret_v_simple,
            "holding_days": row["holding_days"],
            "period_ret_long_gross": period_ret_long_gross,
            "period_ret_long_net": period_ret_long_net,
            "period_ret_ls": period_ret_ls,
        }

        result_rows.append(row_result_1)

        # 为下一行更新“上一期状态”
        prev_post_weight_g = post_trade_weight_g
        prev_post_weight_v = post_trade_weight_v
        prev_ret_g_simple = fwd_ret_g_simple
        prev_ret_v_simple = fwd_ret_v_simple
        prev_effective_signal_ls = effective_signal_ls

    # ------ 第三步：整理成结果表 ------
    track_result_df = pd.DataFrame(result_rows, index=track_df.index)
    return track_result_df
