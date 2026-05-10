"""NAV reconstruction helpers copied from the legacy style-rotation script."""

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

n = 10


def build_single_track_daily_nav(track_period_df, daily_asset_ret_df):
    """
    基于单轨“区间结果表”，重建该轨道的真实逐日净值路径。

    逻辑：
    1. 调仓日收盘后先扣当期 cost_rate；
    2. 再按 post_trade_weight_g / post_trade_weight_v 把资金分成两条资产腿；
    3. 区间内每天让两条腿分别按各自资产的日收益率复利滚动；
    4. 到下一个调仓日收盘后，如果下一期有交易费，再在该日扣下一期 cost_rate；
    5. 全过程不做日频再平衡，只做区间内 buy-and-hold。

    输入：
    track_period_df : DataFrame，单轨区间结果表（run_single_track_backtest 的输出子表）
    daily_asset_ret_df : DataFrame，包含 ret_g_daily_simple / ret_v_daily_simple 的日频收益表

    输出：
    nav_series : Series，该轨道的真实逐日净值路径
    """
    track_df = track_period_df.copy().sort_index(ascending=True)

    required_cols = [
        "next_date",
        "cost_rate",
        "post_trade_weight_g",
        "post_trade_weight_v",
        "period_ret_long_gross",
    ]

    missing_cols = [col for col in required_cols if col not in track_df.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"track_period_df 缺少必要列：{missing_cols}")

    nav_records = []
    interval_check_list = []

    # 第一步：先处理第一天（首期调仓日）
    # 约定：净值从 1.0 开始，首期若收费，就在首期调仓日先扣费
    first_row = track_df.iloc[0]
    first_date = track_df.index[0]

    nav = 1.0 * (1 - float(first_row["cost_rate"]))

    leg_g = nav * float(first_row["post_trade_weight_g"])
    leg_v = nav * float(first_row["post_trade_weight_v"])

    # 记录首个调仓日“收盘后、已扣费、已完成调仓”的净值
    nav_records.append((first_date, nav))

    # 第二步：按区间滚动

    track_df = track_period_df.copy().sort_index()
    track_df["next_date"] = pd.to_datetime(track_df["next_date"])

    nat_mask = track_df["next_date"].isna()

    if nat_mask.any():
        nat_dates = track_df.index[nat_mask].tolist()
        print("这些日期的 next_date 是 NaT：", nat_dates)

        # 只允许最后10行(即n的数值)是 NaT；中间行出现 NaT 说明上游 next_date 逻辑有问题
        nat_pos = np.where(nat_mask)[0]
        if not np.all(nat_pos >= len(track_df) - n):
            raise ValueError("next_date 出现了中间缺失，不只是最后一行，请检查上游 next_date 的生成逻辑")

    last_price_date = daily_asset_ret_df.index.max()

    for i in range(len(track_df)):
        row = track_df.iloc[i]
        start_date = track_df.index[i]
        end_date = row["next_date"]
        need_interval_check = True

        is_open_interval = pd.isna(end_date)

        if is_open_interval:
            end_date = last_price_date
            need_interval_check = False

        if end_date not in daily_asset_ret_df.index:
            raise KeyError(
                f"end_date = {end_date} 不在 daily_asset_ret_df.index 中，"
                "请检查 data_df 是否覆盖了完整价格区间"
            )

        # 当前区间开始时（已经扣完本期费用、已经完成本期调仓）的净值
        nav_after_trade_start = leg_g + leg_v

        # 当前区间内的日频收益切片：用 (start_date, end_date]，因为 start_date 收盘已经调仓完成，
        # 因为在start_date这一天的收盘时刻才完成调仓，所以新仓位不吃start_date本身的日收益
        interval_daily_ret_df = daily_asset_ret_df.loc[
            (daily_asset_ret_df.index > start_date) &
            (daily_asset_ret_df.index <= end_date),
            ["ret_g_daily_simple", "ret_v_daily_simple"] #取这个持有区间里的日收益数据，start_date之后、end_date及之前
        ].copy()

        if len(interval_daily_ret_df) == 0:
            raise ValueError(
                f"区间 {start_date} -> {end_date} 没有取到日频收益，"
                "请检查价格表和 next_date 逻辑"
            ) #防御性检查

        nav_pre_next_trade = np.nan

        for dt, daily_row in interval_daily_ret_df.iterrows():
            # 两条资产腿各自滚动，不做日频再平衡
            leg_g = leg_g * (1 + float(daily_row["ret_g_daily_simple"])) #每天对两条腿，拿各自的实际日收益，分别更新
            leg_v = leg_v * (1 + float(daily_row["ret_v_daily_simple"]))

            # 这是“当日收盘、下一次调仓前”的净值
            nav_pre_next_trade = leg_g + leg_v

            # 如果今天正好是下一次调仓日，而且后面还有下一期，
            # 就在今天收盘后扣下一期的交易费，并切到下一期的目标仓位
            if (dt == end_date) and (i < len(track_df) - 1) and (not is_open_interval):
                next_row = track_df.iloc[i + 1]
                next_cost_rate = float(next_row["cost_rate"])

                nav = nav_pre_next_trade * (1 - next_cost_rate)

                leg_g = nav * float(next_row["post_trade_weight_g"])
                leg_v = nav * float(next_row["post_trade_weight_v"])
            else:
                nav = nav_pre_next_trade

            nav_records.append((dt, nav))

        # 第三步：区间一致性校验
        # 这里检查：
        #   用日频两条腿滚出来的“区间总收益”
        #   是否等于 track_period_df 里已经算好的 period_ret_long_gross
        if need_interval_check:
            interval_gross_from_daily = nav_pre_next_trade / nav_after_trade_start - 1

            interval_check_list.append({
                "start_date": start_date,
                "end_date": end_date,
                "interval_gross_from_daily": interval_gross_from_daily,
                "period_ret_long_gross": float(row["period_ret_long_gross"]),
            })

    # 第四步：整理成净值序列
    nav_series = pd.Series(
        data=[value for _, value in nav_records],
        index=pd.Index([dt for dt, _ in nav_records], name=track_df.index.name),
        dtype=float
    )

    # 理论上不应该有重复索引；即使有，也保留“当日收盘后最终状态”
    nav_series = nav_series[~nav_series.index.duplicated(keep="last")]
    nav_series = nav_series.sort_index()

    # 第五步：做一致性校验
    if len(interval_check_list) > 0:
        interval_check_df = pd.DataFrame(interval_check_list)
        interval_check_df["abs_diff"] = (
            interval_check_df["interval_gross_from_daily"] -
            interval_check_df["period_ret_long_gross"]
        ).abs()

        max_abs_diff = interval_check_df["abs_diff"].max()

        assert max_abs_diff < 1e-8, \
            f"逐日净值重建和区间收益不一致，最大误差为 {max_abs_diff}"

    return nav_series
