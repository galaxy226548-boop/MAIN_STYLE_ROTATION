"""Summary table helpers copied from the legacy style-rotation script."""

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

try:
    from backtest.metrics import (
        calc_annualized_return,
        calc_annualized_vol,
        calc_max_drawdown,
        calc_sharpe_ratio,
        calc_information_ratio,
        calc_monthly_win_rate,
        calc_payoff_ratio,
        calc_period_win_rate,
        calc_expectancy,
        calc_calmar_ratio,
        calc_market_regime_win_rate,
    )
except Exception:  # pragma: no cover - local script fallback
    from metrics import (
        calc_annualized_return,
        calc_annualized_vol,
        calc_max_drawdown,
        calc_sharpe_ratio,
        calc_information_ratio,
        calc_monthly_win_rate,
        calc_payoff_ratio,
        calc_period_win_rate,
        calc_expectancy,
        calc_calmar_ratio,
        calc_market_regime_win_rate,
    )

annual_days = Config.ANNUAL_TRADING_DAYS
rf = 0.015


def build_track_summary_table(
    strategy_track_result_df,
    benchmark_track_result_df,
    strategy_track_nav,
    benchmark_track_nav,
    rf=rf,
    annual_days=annual_days
):
    """
    为单条轨道生成“按年份 + 全区间”的绩效统计表。

    行：
        每个年份一行
        最后一行为“全区间”

    列：
        trade_count
        signal_count
        ann_ret_long_abs
        ann_ret_ls_abs
        ann_vol_abs
        max_dd_abs
        sharpe_abs
        ann_ret_long_excess
        ann_vol_excess
        max_dd_excess
        sharpe_excess
        information_ratio
        monthly_win_rate
        payoff_ratio
        calmar_ratio
        turnover_2way_pct
        growth_regime_win_rate
        value_regime_win_rate
    """

    # -------------------------------------------------
    # 第一步：基础检查
    # -------------------------------------------------
    assert strategy_track_result_df.index.equals(benchmark_track_result_df.index), \
        "策略结果表和基准结果表的索引不一致，请先检查单轨结果是否对齐"

    # -------------------------------------------------
    # 第二步：定义“计算某一个区间统计值”的内部函数
    # -------------------------------------------------
    def build_one_period_row(
        strategy_sub_df,
        benchmark_sub_df,
        strategy_sub_nav,
        benchmark_sub_nav
    ):
        """
        对某一个时间区间（某一年或全区间）计算一行统计值。
        """

        # ===== 绝对收益部分（策略自身）=====
        trade_count = strategy_sub_df["trade_flag"].sum()
        signal_count = strategy_sub_df["signal_count"].sum()


        ann_ret_long_abs = calc_annualized_return(
            ret_series=strategy_sub_df["period_ret_long_net"],
            holding_days_series=strategy_sub_df["holding_days"],
            annual_days=annual_days
        )

        ann_ret_ls_abs = calc_annualized_return(
            ret_series=strategy_sub_df["period_ret_ls"],
            holding_days_series=strategy_sub_df["holding_days"],
            annual_days=annual_days
        )

        ann_vol_abs = calc_annualized_vol(
            ret_series=strategy_sub_df["period_ret_long_net"],
            holding_days_series=strategy_sub_df["holding_days"],
            annual_days=annual_days
        )

        max_dd_abs = calc_max_drawdown(strategy_sub_nav)

        sharpe_abs = calc_sharpe_ratio(
            ret_series=strategy_sub_df["period_ret_long_net"],
            holding_days_series=strategy_sub_df["holding_days"],
            rf=rf,
            annual_days=annual_days
        )

        # ===== 超额收益部分（策略 - 基准）=====
        period_ret_excess = (
            strategy_sub_df["period_ret_long_net"] -
            benchmark_sub_df["period_ret_long_net"]
        )

        excess_nav = (1 + period_ret_excess).cumprod()

        benchmark_ann_ret_long = calc_annualized_return(
            ret_series=benchmark_sub_df["period_ret_long_net"],
            holding_days_series=benchmark_sub_df["holding_days"],
            annual_days=annual_days
        )

        ann_ret_long_excess = ann_ret_long_abs - benchmark_ann_ret_long

        ann_vol_excess = calc_annualized_vol(
            ret_series=period_ret_excess,
            holding_days_series=strategy_sub_df["holding_days"],
            annual_days=annual_days
        )

        max_dd_excess = calc_max_drawdown(excess_nav)

        sharpe_excess = calc_sharpe_ratio(
            ret_series=period_ret_excess,
            holding_days_series=strategy_sub_df["holding_days"],
            rf=rf,
            annual_days=annual_days
        )

        information_ratio = calc_information_ratio(
            excess_ret_series=period_ret_excess,
            holding_days_series=strategy_sub_df["holding_days"],
            annual_days=annual_days
        )

        # ===== 总体比率 =====
        monthly_win_rate = calc_monthly_win_rate(strategy_sub_nav)

        payoff_ratio = calc_payoff_ratio(
            weekly_ret_series=strategy_sub_df["period_ret_long_net"]
        )

        period_win_rate = calc_period_win_rate(
            period_ret_series=strategy_sub_df["period_ret_long_net"]
        )

        expectancy = calc_expectancy(
            period_ret_series=strategy_sub_df["period_ret_long_net"]
        )

        calmar_ratio = calc_calmar_ratio(
            annualized_return=ann_ret_long_abs,
            max_drawdown=max_dd_abs
        )

        # 这里按“区间内双边换手率求和”统计，并转成百分数展示
        turnover_2way_pct = strategy_sub_df["turnover_2way"].sum() * 100

        # ===== 分市场情况胜率 =====
        growth_regime_win_rate, value_regime_win_rate = calc_market_regime_win_rate(
            strategy_sub_df
        )

        row_dict = {
            "trade_count": trade_count,
            "signal_count": signal_count,
            "ann_ret_long_abs": ann_ret_long_abs,
            "ann_ret_ls_abs": ann_ret_ls_abs,
            "ann_vol_abs": ann_vol_abs,
            "max_dd_abs": max_dd_abs,
            "sharpe_abs": sharpe_abs,
            "ann_ret_long_excess": ann_ret_long_excess,
            "ann_vol_excess": ann_vol_excess,
            "max_dd_excess": max_dd_excess,
            "sharpe_excess": sharpe_excess,
            "information_ratio": information_ratio,
            "monthly_win_rate": monthly_win_rate,
            "payoff_ratio": payoff_ratio,
            "period_win_rate": period_win_rate,
            "expectancy": expectancy,
            "calmar_ratio": calmar_ratio,
            "turnover_2way_pct": turnover_2way_pct,
            "growth_regime_win_rate": growth_regime_win_rate,
            "value_regime_win_rate": value_regime_win_rate,
        }

        return row_dict

    # -------------------------------------------------
    # 第三步：按年份生成统计行
    # -------------------------------------------------
    summary_rows = []

    year_list = sorted(strategy_track_result_df.index.year.unique())

    for year in year_list:
        strategy_year_df = strategy_track_result_df[
            strategy_track_result_df.index.year == year
        ].copy()

        benchmark_year_df = benchmark_track_result_df[
            benchmark_track_result_df.index.year == year
        ].copy()

        strategy_year_nav = strategy_track_nav[
            strategy_track_nav.index.year == year
        ].copy()

        benchmark_year_nav = benchmark_track_nav[
            benchmark_track_nav.index.year == year
        ].copy()

        row_dict = build_one_period_row(
            strategy_sub_df=strategy_year_df,
            benchmark_sub_df=benchmark_year_df,
            strategy_sub_nav=strategy_year_nav,
            benchmark_sub_nav=benchmark_year_nav
        )

        row_dict["period"] = str(year)
        summary_rows.append(row_dict)

    # -------------------------------------------------
    # 第四步：补上“全区间”
    # -------------------------------------------------
    full_row_dict = build_one_period_row(
        strategy_sub_df=strategy_track_result_df,
        benchmark_sub_df=benchmark_track_result_df,
        strategy_sub_nav=strategy_track_nav,
        benchmark_sub_nav=benchmark_track_nav
    )
    full_row_dict["period"] = "全区间"
    summary_rows.append(full_row_dict)

    # -------------------------------------------------
    # 第五步：整理成表
    # -------------------------------------------------
    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.set_index("period")

    # 固定列顺序
    summary_table = summary_table[
        [
            "trade_count",
            "signal_count",
            "ann_ret_long_abs",
            "ann_ret_ls_abs",
            "ann_vol_abs",
            "max_dd_abs",
            "sharpe_abs",
            "ann_ret_long_excess",
            "ann_vol_excess",
            "max_dd_excess",
            "sharpe_excess",
            "information_ratio",
            "monthly_win_rate",
            "payoff_ratio",
            "period_win_rate",
            "expectancy",
            "calmar_ratio",
            "turnover_2way_pct",
            "growth_regime_win_rate",
            "value_regime_win_rate",
        ]
    ]

    return summary_table
