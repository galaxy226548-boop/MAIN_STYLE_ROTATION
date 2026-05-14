from pathlib import Path
import sys
import re
import warnings
import numpy as np
import pandas as pd
from Utils.DataAccess import *
from config.Consts import Consts

warnings.filterwarnings("ignore")


DATA_DIR = Path(Consts.project_path.value) / "AssetAllocation" / "StyleRotation" / "data"
MACRO_FILE = DATA_DIR / "【更新模板】宏观数据.xlsx"
SIGNAL_RULE_FILE = DATA_DIR / "信号生成方式.xlsx"
OUTPUT_FILE = DATA_DIR / "宏观信号处理结果.xlsx"
MACRO_EVENT_DIR = Path(Consts.project_path.value) / "Portfolio" / "RallyMomentum" / "data" / "宏观大事"

COL_BOND_10Y = "中国:国债到期收益率:10年"
COL_US_6M = "美国:国债收益率:6个月"
COL_LOAN = "中国:金融机构各项贷款余额:中长期:人民币"
COL_CPI = "中国:CPI:当月同比"
COL_PPI = "中国:PPI:当月同比"
COL_PMI = "中国:制造业PMI"

GROWTH_CODE = "CN2370.CNI"
VALUE_CODE = "CN2371.CNI"


def rolling_percentile_by_date(series: pd.Series, years: int = 3, min_periods: int = 120) -> pd.Series:
    """Current value's rank percentile in the latest N-year history."""
    series = series.dropna().sort_index()
    percentile = pd.Series(np.nan, index=series.index, dtype=float)

    for dt, value in series.items():
        window = series.loc[dt - pd.DateOffset(years=years):dt].dropna()
        if len(window) < min_periods:
            continue
        percentile.loc[dt] = (window < value).sum() / len(window)

    return percentile


def monthly_signal_to_daily(monthly_signal: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill monthly signals onto the daily macro template index."""
    monthly_signal = monthly_signal.dropna().sort_index()
    aligned = monthly_signal.reindex(monthly_signal.index.union(daily_index)).sort_index().ffill()
    return aligned.reindex(daily_index).fillna(0)


def load_macro_data(path: Path = MACRO_FILE) -> pd.DataFrame:
    """Read Wind-style macro template and return cleaned numeric data."""
    raw = pd.read_excel(path, sheet_name="Sheet1", header=1)
    data = raw.iloc[4:].copy()
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace(0, np.nan)
    data = data.loc['2009-01-01':]
    return data


def get_monthly_last(series: pd.Series) -> pd.Series:
    """Keep the latest valid observation in each calendar month."""
    return series.dropna().sort_index().resample("M").last().dropna()


def process_macro_signals(macro_data: pd.DataFrame, macro_event: pd.DataFrame) -> pd.DataFrame:
    """生成宏观信号 DataFrame，所有指标的日期均为"实际可获得日期"，不含未来数据：

    - 日频利率类(十年国债 / 6个月美债)：原始序列先滞后 1 个交易日，再算分位数。
    - 月频 PMI / CPI / PPI：先用 macro_event 中的公布日 + 1 交易日重映射月观测，再做滚动/差分。
    - 中长期贷款余额：无事件公布日，按月末 + 15 自然日后的下一个交易日近似可获得日。
    所有处理完的稀疏序列最后 ffill 到 daily_index，保证每个日期上看到的值都已经"对外可见"。
    """
    daily_index = pd.DatetimeIndex(macro_data.index).sort_values()

    def next_trading_day(dt):
        future = daily_index[daily_index > dt]
        return future[0] if len(future) > 0 else None

    def build_release_calendar(events: pd.DataFrame) -> dict:
        """从事件子集中解析"X月XXX"，结合公布年份得到数据期(注意1月公布的是去年12月数据)。
        返回 {数据所属月份的 Period('M'): 最早公布日}。
        """
        calendar = {}
        for name, release in zip(events["指标名称"].astype(str), events["日期"]):
            m = re.search(r"(\d{1,2})月", name)
            if not m:
                continue
            release = pd.Timestamp(release)
            if pd.isna(release):
                continue
            data_month = int(m.group(1))
            year = release.year - (1 if data_month > release.month else 0)
            period = pd.Period(year=year, month=data_month, freq="M")
            if period not in calendar or release < calendar[period]:
                calendar[period] = release
        return calendar

    def reindex_to_releases(monthly: pd.Series, events: pd.DataFrame) -> pd.Series:
        calendar = build_release_calendar(events)
        idx, vals = [], []
        for month_end, val in monthly.items():
            period = pd.Timestamp(month_end).to_period("M")
            release = calendar.get(period)
            if release is None:
                continue
            eff = next_trading_day(release)
            if eff is None:
                continue
            idx.append(eff)
            vals.append(val)
        return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

    def reindex_spread_to_latest_release(left_monthly: pd.Series,
                                         left_events: pd.DataFrame,
                                         right_monthly: pd.Series,
                                         right_events: pd.DataFrame) -> pd.Series:
        """Map a monthly spread to the later effective date of the two releases."""
        left_calendar = build_release_calendar(left_events)
        right_calendar = build_release_calendar(right_events)
        left_values = {pd.Timestamp(dt).to_period("M"): val for dt, val in left_monthly.items()}
        right_values = {pd.Timestamp(dt).to_period("M"): val for dt, val in right_monthly.items()}

        periods = sorted(
            set(left_values)
            & set(right_values)
            & set(left_calendar)
            & set(right_calendar)
        )
        idx, vals = [], []
        for period in periods:
            left_eff = next_trading_day(left_calendar[period])
            right_eff = next_trading_day(right_calendar[period])
            if left_eff is None or right_eff is None:
                continue
            idx.append(max(left_eff, right_eff))
            vals.append(left_values[period] - right_values[period])
        return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

    def reindex_with_fixed_lag(monthly: pd.Series, days: int = 22) -> pd.Series:
        idx, vals = [], []
        for month_end, val in monthly.items():
            eff = next_trading_day(month_end + pd.Timedelta(days=days))
            if eff is None:
                continue
            idx.append(eff)
            vals.append(val)
        return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

    pmi_events = macro_event[macro_event["指标名称"].str.contains("官方制造业PMI", na=False)]
    cpi_events = macro_event[macro_event["指标名称"].str.contains("CPI", na=False)]
    ppi_events = macro_event[macro_event["指标名称"].str.contains("PPI", na=False)]

    # 利率类：日频，滞后 1 个交易日后再做分位数
    bond_10y = macro_data[COL_BOND_10Y].dropna().shift(1).dropna()
    us_6m = macro_data[COL_US_6M].dropna().shift(1).dropna()

    # 月频：按事件公布日重映射(从指标名称提取月份再结合公布年份得到数据期)
    pmi_monthly = get_monthly_last(macro_data[COL_PMI])
    cpi_monthly = get_monthly_last(macro_data[COL_CPI])
    ppi_monthly = get_monthly_last(macro_data[COL_PPI])

    pmi = reindex_to_releases(pmi_monthly, pmi_events)
    cpi_ppi = reindex_spread_to_latest_release(cpi_monthly, cpi_events, ppi_monthly, ppi_events)
    loan_balance = reindex_with_fixed_lag(get_monthly_last(macro_data[COL_LOAN]))

    signal_data = pd.DataFrame(index=daily_index)

    bond_pct = rolling_percentile_by_date(bond_10y)
    signal_data["十年国债到期收益率_分位数"] = bond_pct.reindex(daily_index).ffill()
    us_pct = rolling_percentile_by_date(us_6m)
    signal_data["6个月美债收益率_分位数"] = us_pct.reindex(daily_index).ffill()

    pmi_diff = pmi.rolling(3, min_periods=3).mean() - pmi.rolling(36, min_periods=24).mean()
    signal_data["PMI_差值"] = monthly_signal_to_daily(pmi_diff, daily_index)

    loan_yoy = loan_balance.pct_change(12) * 100
    loan_diff = loan_yoy - loan_yoy.rolling(3, min_periods=3).mean()
    signal_data["中长期贷款同比"] = monthly_signal_to_daily(loan_yoy, daily_index)
    signal_data["中长期贷款同比_差值"] = monthly_signal_to_daily(loan_diff, daily_index)

    cpi_ppi_diff = cpi_ppi.rolling(3, min_periods=3).mean() - cpi_ppi.rolling(12, min_periods=9).mean()
    signal_data["CPI-PPI"] = monthly_signal_to_daily(cpi_ppi, daily_index)
    signal_data["CPI-PPI_差值"] = monthly_signal_to_daily(cpi_ppi_diff, daily_index)

    return signal_data


def merge_macro_event(path: Path = MACRO_EVENT_DIR):
    folder = Path(path)
    frames = []
    for file in sorted(folder.glob('*.xlsx')):
        if file.name.startswith('~$'):
            continue
        try:
            df = pd.read_excel(file, sheet_name='经济数据')
        except ValueError:
            continue
        if df.empty:
            continue
        df['source_file'] = file.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if '日期' in merged.columns:
        merged['日期'] = pd.to_datetime(merged['日期'], errors='coerce')
        merged = merged.sort_values('日期').reset_index(drop=True)
    merged = merged.dropna(subset=['指标名称'])
    merged = merged[merged['国家/地区']=='中国']
    merged = merged.drop_duplicates(subset=['国家/地区','日期','指标名称'])
    return merged


def get_macro_signal(macro_data_processed: pd.DataFrame) -> pd.DataFrame:
    """根据 信号生成方式.xlsx 中的规则把已对齐到可获得日的宏观指标转成 ±1 信号。

    约定：+1 偏成长，-1 偏价值，0 中性。
    - 十年国债到期收益率_分位数 / 6个月美债收益率_分位数: <0.5 → +1, >0.5 → -1
    - PMI_差值(近3月均值-过去36月均值): <0 → +1, >0 → -1
    - 中长期贷款同比_差值(近1月-近3月均值): >0 → +1, <0 → -1
    - CPI-PPI_差值(近3月均值-过去12月均值): >0 → +1, <0 → -1
    """
    df = macro_data_processed
    signals = pd.DataFrame(index=df.index)

    def sign_growth_when_below(series, threshold=0.5):
        return np.select([series < threshold, series > threshold], [1, -1], default=0)

    def sign_growth_when_negative(series):
        return np.select([series < 0, series > 0], [1, -1], default=0)

    def sign_growth_when_positive(series):
        return np.select([series > 0, series < 0], [1, -1], default=0)

    signals["十年国债_信号"] = sign_growth_when_below(df["十年国债到期收益率_分位数"])
    signals["6个月美债_信号"] = sign_growth_when_below(df["6个月美债收益率_分位数"])
    signals["PMI_信号"] = sign_growth_when_negative(df["PMI_差值"])
    signals["中长期贷款_信号"] = sign_growth_when_positive(df["中长期贷款同比_差值"])
    signals["CPI-PPI_信号"] = sign_growth_when_positive(df["CPI-PPI_差值"])
    return signals

def get_mom_signals(index_growth, index_value, window: int = 20) -> pd.DataFrame:
    """近 window 个交易日(默认 20 日 ≈ 4 周) 成长 vs 价值的累计收益率差。

    动量_信号: >0 偏成长 +1, <0 偏价值 -1, =0 中性 0。
    输出格式对齐宏观 signals: 含 合成信号 列。
    """
    growth = index_growth.iloc[:, 0] if isinstance(index_growth, pd.DataFrame) else index_growth
    value = index_value.iloc[:, 0] if isinstance(index_value, pd.DataFrame) else index_value
    growth = growth.dropna().sort_index()
    value = value.dropna().sort_index()

    growth_ret = growth.pct_change(window)
    value_ret = value.pct_change(window)
    diff = (growth_ret - value_ret).dropna()

    signals = pd.DataFrame(index=diff.index)
    signals["动量_信号"] = np.select([diff > 0, diff < 0], [1, -1], default=0)
    signals = signals.astype(int)
    return signals


def _strong_ratio(index_code: str,
                  close_wide: pd.DataFrame,
                  ma_short: int = 5,
                  ma_long: int = 20) -> pd.Series:
    """成分股中 MA{ma_short} > MA{ma_long} 的股票数量占比 (按交易日)。

    成分股名单使用 get_index_comp 历史进出表，按 S_CON_INDATE/S_CON_OUTDATE 构建每日成员掩码。
    """
    members = get_index_comp(index_code=index_code, date='all').reset_index()
    members = members[['S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE']].dropna(subset=['S_CON_WINDCODE'])

    dates = close_wide.index
    cols = close_wide.columns
    end_sentinel = dates[-1]

    cols_set = set(cols)
    member_records = [(r['S_CON_WINDCODE'], pd.Timestamp(r['S_CON_INDATE']),
                       pd.Timestamp(r['S_CON_OUTDATE']) if pd.notna(r['S_CON_OUTDATE']) else end_sentinel)
                      for _, r in members.iterrows() if r['S_CON_WINDCODE'] in cols_set]

    if not member_records:
        return pd.Series(np.nan, index=dates)

    used_stocks = sorted({stk for stk, _, _ in member_records})
    sub_close = close_wide[used_stocks]
    ma_s = sub_close.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = sub_close.rolling(ma_long, min_periods=ma_long).mean()
    is_strong = (ma_s > ma_l)

    mask = pd.DataFrame(False, index=dates, columns=used_stocks)
    for stk, start, end in member_records:
        sel = (dates >= start) & (dates <= end)
        if sel.any():
            mask.loc[sel, stk] = True

    member_count = mask.sum(axis=1).astype(float)
    strong_count = (mask & is_strong.fillna(False)).sum(axis=1).astype(float)
    ratio = strong_count / member_count.replace(0, np.nan)
    return ratio


def get_constituent_mom_signals(growth_index_code: str = "399370.SZ",
                                value_index_code: str = "399371.SZ",
                                ma_short: int = 5,
                                ma_long: int = 20) -> pd.DataFrame:
    """成分股动量信号: 成长/价值指数成分股中 MA5>MA20 的占比之差。

    占比高的一方视为强势风格。+1 偏成长, -1 偏价值, 0 中性。
    """
    close_wide = get_pricedata('close')
    growth_ratio = _strong_ratio(growth_index_code, close_wide, ma_short, ma_long)
    value_ratio = _strong_ratio(value_index_code, close_wide, ma_short, ma_long)

    diff = (growth_ratio - value_ratio).dropna()
    signals = pd.DataFrame(index=diff.index)
    signals["成分股动量_信号"] = np.select([diff > 0, diff < 0], [1, -1], default=0)
    signals = signals.astype(int)
    return signals


def build_style_position(signals: pd.DataFrame,
                         growth_code: str = GROWTH_CODE,
                         value_code: str = VALUE_CODE) -> pd.DataFrame:
    """根据合成信号生成成长/价值指数持仓(long-format: date, code, weight)。

    - 合成信号 > 0: 100% 成长
    - 合成信号 < 0: 100% 价值
    - 合成信号 = 0: 成长/价值 各 50%
    """
    score = signals["合成信号"]
    growth_w = pd.Series(
        np.where(score > 0, 1.0, np.where(score < 0, 0.0, 0.5)),
        index=score.index,
    )
    value_w = 1.0 - growth_w

    # 仅在持仓变化的日期保留
    state = pd.DataFrame({"g": growth_w, "v": value_w})
    changed = (state != state.shift()).any(axis=1)
    state = state.loc[changed]

    growth_pos = pd.DataFrame({
        "date": state.index,
        "code": growth_code,
        "weight": state["g"].values,
    })
    value_pos = pd.DataFrame({
        "date": state.index,
        "code": value_code,
        "weight": state["v"].values,
    })
    position = pd.concat([growth_pos, value_pos], ignore_index=True)
    position = position[position["weight"] > 0]
    position = position.sort_values(["date", "code"]).reset_index(drop=True)
    return position




def backtest_rotation(position: pd.DataFrame,
                      growth_code: str = GROWTH_CODE,
                      value_code: str = VALUE_CODE,
                      index_growth: pd.DataFrame = None,
                      index_value: pd.DataFrame = None) -> pd.DataFrame:
    """根据 signals 生成成长/价值轮动策略净值，滞后一天调仓，基准为成长价值等权 50/50。
    策略日收益 = sum(权重_t × 指数日收益_t)，累乘得到净值。
    """
    if index_growth is None:
        index_growth = get_indexdata(growth_code)
    if index_value is None:
        index_value = get_indexdata(value_code)

    close = pd.concat([index_growth, index_value], axis=1).sort_index()
    close = close.loc[:, [growth_code, value_code]].dropna(how="all")

    # 调仓日权重宽表：rebal_dates × {growth_code, value_code}
    weight_wide = (
        position.pivot_table(index="date", columns="code", values="weight", aggfunc="last")
        .reindex(columns=[growth_code, value_code])
        .fillna(0.0)
    )
    weight_wide.index = pd.to_datetime(weight_wide.index)

    # 对齐到交易日，并滞后一天生效：T 日信号 → T+1 日开始持有
    daily_w = weight_wide.reindex(close.index.union(weight_wide.index)).ffill()
    daily_w = daily_w.reindex(close.index).dropna(how="all").fillna(0.0)

    rets = close.pct_change().reindex(daily_w.index).fillna(0.0)
    strategy_ret = (daily_w * rets).sum(axis=1)
    strategy_nv = (1 + strategy_ret).cumprod()

    bench_ret = rets.mean(axis=1)  # 成长/价值各 50%
    benchmark_nv = (1 + bench_ret).cumprod()

    nv = pd.concat([strategy_nv, benchmark_nv], axis=1)
    nv.columns = ["策略", "基准(成长价值等权)"]
    nv['相对强弱'] = nv['策略']/nv['基准(成长价值等权)']
    return nv


def plot_nv(nv: pd.DataFrame):
    """绘制净值曲线: 策略/基准在左轴，相对强弱在右轴。"""
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    x = nv.index.to_numpy()
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(x, nv["策略"].to_numpy(), label="策略", color="tab:red")
    ax1.plot(x, nv["基准(成长价值等权)"].to_numpy(), label="基准(成长价值等权)", color="tab:blue")
    ax1.set_ylabel("净值")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, nv["相对强弱"].to_numpy(), label="相对强弱", color="tab:green", linestyle="--")
    ax2.set_ylabel("相对强弱")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("成长价值风格轮动")
    plt.show()


#%%
if __name__ == "__main__":
    
    index_growth = get_indexdata(GROWTH_CODE)
    index_value = get_indexdata(VALUE_CODE)
    
    macro_data = load_macro_data()
    macro_data = macro_data.loc['2012-11-01':]
    macro_event = merge_macro_event()
    macro_data_processed = process_macro_signals(macro_data, macro_event)
    macro_signals = get_macro_signal(macro_data_processed)
    mom_signals = get_mom_signals(index_growth,index_value)
    constituent_mom_signals = get_constituent_mom_signals("399370.SZ", "399371.SZ")
    signals = pd.concat([macro_signals,mom_signals,constituent_mom_signals],axis=1)
    
    signals['合成信号'] = signals.sum(axis=1)
    signals = signals.loc['2012-12-31':]
    trading_dates = get_trade_days('2012-12-31',signals.index[-1])
    signals = signals.loc[trading_dates]
    
   
    position = build_style_position(signals,GROWTH_CODE,VALUE_CODE)
    position = adjust_date_col(position)
    
    nv = backtest_rotation(position,GROWTH_CODE,VALUE_CODE,index_growth,index_value)
    plot_nv(nv)
    nv['相对强弱'].plot()
    