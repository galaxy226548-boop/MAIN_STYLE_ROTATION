# ===== AI-Friendly Python Version of Notebook =====
# Source Notebook: /Users/chloezh/Projects/jupyter_to_py_project/input_jupyter/风格轮动信号检验法 copy.ipynb

# ----- Cell 1 (code) -----
# Cell 1: 导入库
# ------------- 标准库 -------------
import os
import glob
import re
import json
from datetime import datetime
from pathlib import Path
import itertools
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
# ------------- 数据分析核心库 -------------
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)
# ------------- 科学计算/统计分析库 -------------
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.regression.rolling import RollingOLS
# ------------- 可视化库 -------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
try:
    import my_plot_helper
except Exception as _my_plot_helper_import_error:
    my_plot_helper = None
    print(f"my_plot_helper import skipped: {_my_plot_helper_import_error}")
# ------------- Parquet和Feather相关库 -------------
from pathlib import Path
from typing import List, Optional, Dict, Any
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pyarrow.compute as pc
# ------------- 程序设定 -------------
from Config import Config
try:
    import src.factor_utils as fu
except Exception as _factor_utils_import_error:
    try:
        import factor_utils as fu
    except Exception as _local_factor_utils_import_error:
        fu = None
        print(f"factor_utils import skipped; notebook-local helpers will be used: {_local_factor_utils_import_error}")


# ----- Cell 2 (code) -----
data_df_path = Path("data_processed/data_df.parquet")
market_df_path = Path("data_processed/market_df.parquet")
if not data_df_path.exists():
    data_df_path = Path("data_df.parquet")
if not market_df_path.exists():
    market_df_path = Path("market_df.parquet")

data_df = pd.read_parquet(data_df_path)
market_df = pd.read_parquet(market_df_path)
data_df.index = pd.to_datetime(data_df.index)
market_df.index = pd.to_datetime(market_df.index)

# raw_factor_df：保留经济含义的原始因子值，例如利率同比、成交额变化率、估值分位数、RSI 差值。
# normalized_factor_df：经过横向/时间序列可比处理后的因子值，例如 winsorize、z-score、rolling z-score、去极值、标准化。
# 若某个因子没有 normalized_factor 处理，则 normalized_factor 沿用 raw_factor。
raw_factor_df = pd.DataFrame(index=data_df.index)
normalized_factor_df = pd.DataFrame(index=data_df.index)

# ----- Cell 3 (markdown) -----
# # 因子生成与挂载

# ----- Cell 4 (markdown) -----
# ## 数据文件清洗

# ----- Cell 5 (code) -----
# 清洗宏观数据的函数
def clean_macro_table(df: pd.DataFrame, nation: str = "美国", indi: str = "PMI") -> pd.DataFrame:
    
    # 复制，避免改原表
    out = df.copy()

    # 1) 先筛选国家/地区
    out = out[out["国家/地区"] == nation]

    # 2) 再筛选指标名称包含 indi 的行
    out = out[out["指标名称"].astype(str).str.contains(indi, na=False)]

    keep_cols = [
        out.columns[2],   # 第3列
        out.columns[3],   # 第4列
        out.columns[4],   # 第5列
        out.columns[5],   # 第6列
        out.columns[7],   # 第8列
        out.columns[8],   # 第9列
        out.columns[9],   # 第10列
        out.columns[-1],  # 最后一列
    ]
    out = out[keep_cols].copy()

    # 4) 重命名列
    out.columns = [
        "date",      # 原第3列：日期
        "time",          # 原第4列：时间
        "nation",        # 原第5列：国家/地区
        "indicator",     # 原第6列：指标
        "prev",          # 原第8列：前值
        "forecast",      # 原第9列：预测值
        "actual",        # 原第10列：今值
        "file_ym",       # 原最后1列：文件年月
    ]

    # 7) 重置索引
    out = out.set_index("date").sort_index(ascending=True)

    # 按同一事件去重：同一天、同一时间、同一国家、同一指标，只保留最新 file_ym
    out = out.drop_duplicates(
        subset=[ "time", "nation", "indicator","prev","forecast","actual"],
        keep="last"
    )

    return out

# ----- Cell 6 (code) -----
# ZHAO todo factor helpers
# Non-Macro_all monthly signals are placed on the last available data_df date of each source month.
# Macro_all factors keep the original row dates from Macro_all.xlsx.

def _find_data_file(file_name):
    file_name = str(file_name)
    direct = Config.DATA_DIR / file_name
    if direct.exists():
        return direct
    candidates = sorted(Config.DATA_DIR.rglob(file_name))
    if not candidates and not file_name.lower().endswith((".xlsx", ".xls")):
        candidates = sorted(Config.DATA_DIR.rglob(file_name + ".xlsx"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find data file under {Config.DATA_DIR}: {file_name}")
    if len(candidates) > 1:
        print(f"Multiple files matched {file_name}, using: {candidates[0]}")
    return candidates[0]


def _as_numeric(series, percent_hint=False):
    text = series.astype(str).str.strip()
    has_percent_sign = text.str.contains("%", regex=False, na=False).any()
    numeric = pd.to_numeric(
        text.str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    if percent_hint or has_percent_sign:
        numeric = numeric / 100
    return numeric


def _read_indicator_series(file_name, value_col, sheet_name=0):
    path = _find_data_file(file_name)
    raw = pd.read_excel(path, sheet_name=sheet_name)
    date_col = raw.columns[0]
    unit_mask = raw[date_col].astype(str).str.strip().eq('单位')
    percent_hint = False
    if value_col in raw.columns and unit_mask.any():
        percent_hint = raw.loc[unit_mask, value_col].astype(str).str.contains("%", regex=False, na=False).any()
    out = raw.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    out = out.set_index(date_col).sort_index(ascending=True)
    if value_col not in out.columns:
        raise KeyError(f"{value_col} not found in {path}. Available columns: {list(out.columns)}")
    s = _as_numeric(out[value_col], percent_hint=percent_hint)
    s.name = value_col
    return s[~s.index.duplicated(keep="last")].sort_index()


def _month_aggregate(series, how="average"):
    s = series.dropna().copy().sort_index()
    grouped = s.groupby(s.index.to_period("M"))

    if how in ["average", "mean"]:
        out = grouped.mean()
    elif how == "last":
        out = grouped.last()
    elif how == "sum":
        out = grouped.sum()

    # 关键：把 index 从“月份”改成该月在原始数据里的最后一个日期
    last_dates = grouped.apply(lambda x: x.index[-1])
    out.index = last_dates.values

    return out


def _data_month_end_series(monthly_series):
    s = monthly_series.dropna().copy().sort_index()
    if len(s) == 0:
        return pd.Series(dtype="float64")
    s.index = pd.to_datetime(s.index).to_period("M")
    s = s[~s.index.duplicated(keep="last")]
    data_index = pd.DatetimeIndex(data_df.index).sort_values()
    last_data_date_by_month = pd.Series(data_index, index=data_index.to_period("M")).groupby(level=0).max()
    target_dates = last_data_date_by_month.reindex(s.index)
    valid = target_dates.notna()
    out = pd.Series(
        s.loc[valid].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(target_dates.loc[valid].to_numpy()),
        dtype="float64",
    )
    return out.sort_index()


def _refreq_month_end(series, how="last"):
    return _data_month_end_series(_month_aggregate(series, how=how))


def _rolling_quantile_rank_year(series, year=5):
    s = series.dropna().sort_index()
    dates = s.index

    out = []
    left = 0
    first_date = dates[0]

    for right in range(len(s)):
        end_date = dates[right]
        start_date = end_date - pd.DateOffset(years=year)

        # 原始数据整体不够5年，才返回 NaN
        if first_date > start_date:
            out.append(np.nan)
            continue

        while left < right and dates[left] < start_date:
            left += 1

        window = s.iloc[left:right + 1]

        pct = window.rank(pct=True).iloc[-1]
        out.append(pct)

    return pd.Series(out, index=s.index)


def _load_macro_all():
    return pd.read_excel(_find_data_file("Macro_all.xlsx"), sheet_name=0)


def _load_china_macro_series(keyword, value_col='今值', required_contains=None, exclude_contains=None):
    macro = _load_macro_all()
    date_col = '日期' if '日期' in macro.columns else macro.columns[2]
    nation_col = '国家/地区' if '国家/地区' in macro.columns else macro.columns[4]
    indicator_col = '指标名称' if '指标名称' in macro.columns else macro.columns[5]
    mask = macro[nation_col].eq('中国')
    mask &= macro[indicator_col].astype(str).str.contains(keyword, na=False, regex=False)
    if required_contains is not None:
        required_list = [required_contains] if isinstance(required_contains, str) else list(required_contains)
        for item in required_list:
            mask &= macro[indicator_col].astype(str).str.contains(item, na=False, regex=False)
    if exclude_contains is not None:
        exclude_list = [exclude_contains] if isinstance(exclude_contains, str) else list(exclude_contains)
        for item in exclude_list:
            mask &= ~macro[indicator_col].astype(str).str.contains(item, na=False, regex=False)
    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No China macro rows matched keyword={keyword!r}")
    percent_hint = (
        out[indicator_col].astype(str).str.contains("%", regex=False, na=False).any()
        or out[value_col].astype(str).str.contains("%", regex=False, na=False).any()
    )
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    sort_cols = [x for x in [date_col, '来源文件', '来源sheet', '文件年月'] if x in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    s = pd.Series(_as_numeric(out[value_col], percent_hint=percent_hint).to_numpy(), index=out[date_col], name=keyword).sort_index()
    dup_count = int(s.index.duplicated(keep=False).sum())
    if dup_count > 0:
        print(f"Macro keyword {keyword!r} matched {dup_count} duplicate-date rows; keeping the last row per date.")
        s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _assign_raw_factor(raw_col, normalized_factor_series, raw_factor_series=None):
    if not raw_col.endswith("_raw"):
        raise ValueError(f"{raw_col} must end with _raw")
    factor_col = raw_col[:-4]

    if raw_factor_series is None:
        raw_factor_series = normalized_factor_series

    raw_factor = raw_factor_series.copy().sort_index()
    raw_factor.index = pd.to_datetime(raw_factor.index)
    raw_factor = raw_factor[~raw_factor.index.duplicated(keep="last")]
    raw_factor_df[factor_col] = raw_factor.reindex(data_df.index).astype("float64")

    normalized_factor = normalized_factor_series.copy().sort_index()
    normalized_factor.index = pd.to_datetime(normalized_factor.index)
    normalized_factor = normalized_factor[~normalized_factor.index.duplicated(keep="last")]
    normalized_factor_df[factor_col] = normalized_factor.reindex(data_df.index).astype("float64")

    # 保留原 notebook 的单因子回测接口：data_df 中的 xxx_raw 仍然使用可挂载的 normalized_factor。
    data_df[raw_col] = normalized_factor_df[factor_col]
    print(
        f"{raw_col} generated:",
        "raw_factor_non_na=", int(raw_factor_df[factor_col].notna().sum()),
        "normalized_factor_non_na=", int(normalized_factor_df[factor_col].notna().sum()),
        "first=", normalized_factor_df[factor_col].first_valid_index(),
        "last=", normalized_factor_df[factor_col].last_valid_index(),
    )
    return data_df[raw_col]


def _rolling_sum_ratio_minus_one(series, window=12, shift=11):
    rolling_sum = series.dropna().sort_index().rolling(window=window, min_periods=window).sum().dropna()
    division = rolling_sum / rolling_sum.shift(shift) - 1
    return division.dropna()


def _YoY(monthly_series):
    s = monthly_series.dropna().sort_index()
    return (s/ s.shift(1) - 1)


def data_diff(series):
    s = series.copy().sort_index()
    return s - s.shift(1)


def _trailing_time_window(series, end_date, months=None, years=None, days=None):
    start_date = pd.Timestamp(end_date)
    if years is not None:
        start_date = start_date - pd.DateOffset(years=years)
    if months is not None:
        start_date = start_date - pd.DateOffset(months=months)
    if days is not None:
        start_date = start_date - pd.Timedelta(days=days)
    return series.loc[(series.index >= start_date) & (series.index <= end_date)]


def _time_window_apply(series, func, months=None, years=None, days=None, min_periods=1):
    s = series.dropna().sort_index()
    out = []
    for dt in s.index:
        window = _trailing_time_window(s, dt, months=months, years=years, days=days)
        if len(window) < min_periods:
            out.append(np.nan)
        else:
            out.append(func(window))
    return pd.Series(out, index=s.index, dtype="float64")


def rolling_quantile_value(series, target_quantile, months=None, years=None, days=None, min_periods=1):
    return _time_window_apply(
        series,
        lambda window: window.quantile(target_quantile),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def calc_rolling_zscore_time(series, months=None, years=None, days=None, min_periods=1):
    s = series.dropna().sort_index()
    out = []
    for dt in s.index:
        window = _trailing_time_window(s, dt, months=months, years=years, days=days)
        if len(window) < min_periods:
            out.append(np.nan)
            continue
        rolling_mean = window.mean()
        rolling_std = window.std()
        current_value = s.loc[dt]
        if pd.isna(current_value) or pd.isna(rolling_std) or np.isclose(rolling_std, 0):
            out.append(np.nan)
        else:
            out.append((current_value - rolling_mean) / rolling_std)
    return pd.Series(out, index=s.index, dtype="float64")


def data_deviation(series, months=None, years=None, days=None, min_periods=1):
    s = series.copy().sort_index()
    rolling_mean = _time_window_apply(
        s,
        lambda window: window.mean(),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )
    return s - rolling_mean.reindex(s.index)


def z_data_deviation(
    series,
    dev_months=None,
    dev_years=None,
    dev_days=None,
    z_months=None,
    z_years=None,
    z_days=None,
    dev_min_periods=1,
    z_min_periods=1,
):
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


def expectation(
    sub_1,
    sub_2,
    *,
    z_years=3,
    z_min_periods=18,
    up_floor=None,
    down_ceiling=None,
    upper_quantile=None,
    lower_quantile=None,
    quantile_years=3,
    quantile_min_periods=18,
):
    aligned = pd.concat([sub_1.rename("sub_1"), sub_2.rename("sub_2")], axis=1).sort_index()
    aligned = aligned.dropna(subset=["sub_1", "sub_2"])
    surprise = aligned["sub_1"] - aligned["sub_2"]

    if upper_quantile is not None:
        upper_bound = rolling_quantile_value(
            aligned["sub_1"],
            upper_quantile,
            years=quantile_years,
            min_periods=quantile_min_periods,
        )
    else:
        upper_bound = pd.Series(up_floor, index=aligned.index, dtype="float64")

    if lower_quantile is not None:
        lower_bound = rolling_quantile_value(
            aligned["sub_1"],
            lower_quantile,
            years=quantile_years,
            min_periods=quantile_min_periods,
        )
    else:
        lower_bound = pd.Series(down_ceiling, index=aligned.index, dtype="float64")

    signal = pd.Series(np.nan, index=aligned.index, dtype="float64")
    positive_mask = (aligned["sub_1"] > aligned["sub_2"]) & (aligned["sub_1"] > upper_bound)
    negative_mask = (aligned["sub_1"] < aligned["sub_2"]) & (aligned["sub_1"] < lower_bound)
    keep_mask = positive_mask | negative_mask
    signal.loc[keep_mask] = surprise.loc[keep_mask]

    return calc_rolling_zscore_time(signal, years=z_years, min_periods=z_min_periods)

#计算z-score
def calc_rolling_zscore(series, window, min_periods=None):
    """
    滚动窗口 zscore。    z_t = (x_t - rolling_mean_t) / rolling_std_t
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    zscore = (series - rolling_mean) / rolling_std
    return zscore


def llt(series, d=30):
    if fu is not None and hasattr(fu, "calc_llt"):
        try:
            return fu.calc_llt(series, d=d)
        except Exception:
            pass

    s = series.astype(float).copy()
    llt_series = pd.Series(index=s.index, dtype="float64")
    alpha = 2 / (d + 1)
    started = False

    for i in range(2, len(s)):
        x_t = s.iloc[i]
        x_t1 = s.iloc[i - 1]
        x_t2 = s.iloc[i - 2]

        if pd.isna(x_t) or pd.isna(x_t1) or pd.isna(x_t2):
            llt_series.iloc[i] = np.nan
            continue

        if not started:
            llt_series.iloc[i - 1] = x_t1
            llt_series.iloc[i] = x_t
            started = True
            continue

        llt_t1 = llt_series.iloc[i - 1]
        llt_t2 = llt_series.iloc[i - 2]

        if pd.isna(llt_t1) or pd.isna(llt_t2):
            llt_series.iloc[i - 1] = x_t1
            llt_series.iloc[i] = x_t
            continue

        llt_series.iloc[i] = (
            (alpha - alpha ** 2 / 4) * x_t
            + (alpha ** 2 / 2) * x_t1
            - (alpha - 3 * alpha ** 2 / 4) * x_t2
            + 2 * (1 - alpha) * llt_t1
            - (1 - alpha) ** 2 * llt_t2
        )

    return llt_series

def merge_factor_to_market(data_df, market_df, raw_factor_col, factor_type="state", track_col="track_id"):
    if raw_factor_col not in data_df.columns:
        raise KeyError(f"{raw_factor_col} is not in data_df.columns")
    if not raw_factor_col.endswith("_raw"):
        raise ValueError(f"{raw_factor_col} must end with _raw")
    factor_col = raw_factor_col[:-4]
    factor_type = str(factor_type).lower()
    if factor_type not in ["state", "event"]:
        raise ValueError(f"factor_type must be 'state' or 'event', got {factor_type!r}")
    if factor_type == "state":
        data_df[factor_col] = data_df[raw_factor_col].shift(1)
        market_df[factor_col] = data_df[factor_col].reindex(market_df.index)
    else:
        if track_col not in market_df.columns:
            raise KeyError(f"event factors require {track_col} in market_df")
        market_df[factor_col] = np.nan
        raw_events = data_df.loc[data_df[raw_factor_col].notna(), raw_factor_col].sort_index()
        track_values = sorted(pd.Series(market_df[track_col]).dropna().unique())
        track_dates = {track_id: pd.DatetimeIndex(market_df.index[market_df[track_col] == track_id]).sort_values() for track_id in track_values}
        for event_date, raw_value in raw_events.items():
            event_date = pd.Timestamp(event_date)
            for track_id, candidate_dates in track_dates.items():
                future_dates = candidate_dates[candidate_dates > event_date]
                if len(future_dates) > 0:
                    market_df.loc[future_dates[0], factor_col] = raw_value
        data_df[factor_col] = market_df[factor_col].reindex(data_df.index)
    if factor_col not in Config.FEATURE_LIST:
        Config.FEATURE_LIST.append(factor_col)
    return market_df


if os.environ.get("ODDS_WIN_ONLY") == "1":
    _script_text = Path(__file__).read_text()
    _tail_marker = "# ----- Cell 237 (markdown) -----"
    _tail_code = _tail_marker + _script_text.rsplit(_tail_marker, 1)[1]
    exec(_tail_code, globals())
    raise SystemExit


# ----- Cell 7 (code) -----
# L23-L46 helper additions
def data_yoy(series):
    s = series.copy().sort_index()
    return s / s.shift(1) - 1


def _time_mean(series, *, months=None, years=None, days=None, min_periods=1):
    return _time_window_apply(
        series,
        lambda window: window.mean(),
        months=months,
        years=years,
        days=days,
        min_periods=min_periods,
    )


def z_MA_div(
    series,
    short_window,
    long_window,
    z_window,
    *,
    unit="months",
    short_min_periods=None,
    long_min_periods=None,
    z_min_periods=None,
):
    if unit == "days":
        short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, int(round(short_window / 30 / 2)))
        long_min_periods = long_min_periods or max(6, int(round(long_window / 30 / 2)))
        z_min_periods = z_min_periods or max(12, int(round(z_window / 30 / 2)))

        short_ma = (
            series.copy()
            if short_window == 1
            else _time_mean(series, days=short_window, min_periods=short_min_periods)
        )
        long_ma = _time_mean(series, days=long_window, min_periods=long_min_periods)
        return calc_rolling_zscore_time(
            short_ma - long_ma,
            days=z_window,
            min_periods=z_min_periods,
        )

    short_min_periods = 1 if short_window == 1 else short_min_periods or max(2, short_window // 2)
    long_min_periods = long_min_periods or max(3, long_window // 2)
    z_min_periods = z_min_periods or max(6, z_window // 2)

    short_ma = (
        series.copy()
        if short_window == 1
        else _time_mean(series, months=short_window, min_periods=short_min_periods)
    )
    long_ma = _time_mean(series, months=long_window, min_periods=long_min_periods)
    return calc_rolling_zscore_time(
        short_ma - long_ma,
        months=z_window,
        min_periods=z_min_periods,
    )


def _latest_macro_pair(sub_1, sub_2):
    aligned = pd.concat(
        [sub_1.rename("sub_1"), sub_2.rename("sub_2")],
        axis=1,
        sort=True,
    ).sort_index()
    aligned[["sub_1", "sub_2"]] = aligned[["sub_1", "sub_2"]].ffill()
    return aligned

# ----- Cell 8 (markdown) -----
# ## ZHAO todo factor generation

# ----- Cell 9 (markdown) -----
# ### index_pb (ZHAO01)

# ----- Cell 10 (code) -----
sub_1 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx', '市净率LF').dropna()
sub_2 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx', '市净率LF').dropna()
valuation_ratio = sub_1 / sub_2
quantile_rank = _rolling_quantile_rank_year(valuation_ratio,5)
ZHAO01_raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0).dropna()
ZHAO01_raw = _month_aggregate(ZHAO01_raw, how="last")
_assign_raw_factor("ZHAO01_raw", ZHAO01_raw, quantile_rank)


# ----- Cell 11 (markdown) -----
# ### index_pe (ZHAO02)

# ----- Cell 12 (code) -----
sub_1 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx', '市盈率TTM').dropna()
sub_2 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx', '市盈率TTM').dropna()
valuation_ratio = sub_1 / sub_2
quantile_rank = _rolling_quantile_rank_year(valuation_ratio,5)
ZHAO02_raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0).dropna()
ZHAO02_raw = _month_aggregate(ZHAO02_raw, how="last")
_assign_raw_factor("ZHAO02_raw", ZHAO02_raw, quantile_rank)

# ----- Cell 13 (markdown) -----
# ### PB_QRD (ZHAO03)

# ----- Cell 14 (code) -----
# 读取 parquet
df_component = pd.read_parquet(r"data_private\index_data\IndexComponents.parquet")
df_pb = pd.read_parquet(r"data_public\比价\pb.parquet")

# ----- Cell 15 (code) -----
# ---- component 表列名 ----
target = "399370.SZ"
comp_date = "TRADE_DT"
comp_code = "S_CON_WINDCODE"
comp_weight = "I_WEIGHT"

# ---- df_pb 表列名 ----
pb_date = "TRADE_DT"

# ----- Cell 16 (code) -----
df_component.head()

# ----- Cell 17 (code) -----
df_pb.head()

# ----- Cell 18 (code) -----
def normalize_trade_dt(series):
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    is_yyyymmdd = text.str.fullmatch(r"\d{8}").fillna(False)

    dt = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    dt.loc[is_yyyymmdd] = pd.to_datetime(
        text.loc[is_yyyymmdd],
        format="%Y%m%d",
        errors="coerce"
    )
    dt.loc[~is_yyyymmdd] = pd.to_datetime(
        text.loc[~is_yyyymmdd],
        errors="coerce"
    )
    return dt.dt.normalize()


df_component[comp_date] = normalize_trade_dt(df_component[comp_date])

df_pb[pb_date] = normalize_trade_dt(df_pb[pb_date])

df_component[comp_code] = df_component[comp_code].astype(str).str.strip()

# ----- Cell 19 (code) -----
# 将 PB 宽表标准化为以 TRADE_DT 为行、股票代码为列的查询表
df_pb_wide = df_pb.copy()

if isinstance(df_pb_wide.index, pd.MultiIndex):
    if pb_date in df_pb_wide.index.names:
        df_pb_wide = df_pb_wide.reset_index()
elif df_pb_wide.index.name == pb_date:
    df_pb_wide = df_pb_wide.reset_index()

if isinstance(df_pb_wide.columns, pd.MultiIndex):
    def _flatten_pb_col(col):
        parts = [
            str(item).strip()
            for item in col
            if pd.notna(item) and str(item).strip() not in ("", "nan")
        ]
        wind_codes = [
            item for item in parts
            if re.match(r"^[0-9A-Z]{6}\.(SH|SZ|BJ)$", item)
        ]
        return wind_codes[0] if wind_codes else (parts[0] if parts else "")

    df_pb_wide.columns = [
        _flatten_pb_col(col) for col in df_pb_wide.columns.to_flat_index()
    ]

pb_date_col = next(
    (col for col in df_pb_wide.columns if str(col).strip().upper() in {pb_date, "DATE"}),
    df_pb_wide.columns[0],
)

df_pb_wide = df_pb_wide.rename(columns={pb_date_col: comp_date})
df_pb_wide[comp_date] = normalize_trade_dt(df_pb_wide[comp_date])

df_pb_wide = (
    df_pb_wide
    .dropna(subset=[comp_date])
    .drop_duplicates(subset=[comp_date], keep="last")
    .set_index(comp_date)
)

df_pb_wide.columns = df_pb_wide.columns.astype(str).str.strip()
df_pb_wide = df_pb_wide.loc[:, ~df_pb_wide.columns.duplicated(keep="last")]

# ----- Cell 20 (code) -----
df_pb_wide.head()

# ----- Cell 21 (code) -----
# 1. 复制，避免污染原表
component_monthly = df_component.copy()
pb_trade_dates = df_pb.copy()

# 2. 标准化日期
component_monthly["TRADE_DT"] = pd.to_datetime(component_monthly["TRADE_DT"])
pb_trade_dates["TRADE_DT"] = pd.to_datetime(pb_trade_dates["TRADE_DT"])

# 3. 提取 df_pb 中所有交易日
trade_dates = (
    pb_trade_dates[["TRADE_DT"]]
    .drop_duplicates()
    .sort_values("TRADE_DT")
)

# 4. 提取每个月末截面的成分股数据
# 注意：这里每个 TRADE_DT 下有很多成分股
component_monthly = component_monthly.sort_values("TRADE_DT")

# 5. 为每个交易日匹配“最近一个已经发生的月末成分股日期”
date_map = pd.merge_asof(
    trade_dates,
    component_monthly[["TRADE_DT"]].drop_duplicates().sort_values("TRADE_DT"),
    on="TRADE_DT",
    direction="backward"
).rename(columns={"TRADE_DT": "TRADE_DT"})

# 上面会覆盖名字，不方便区分，建议重新写一个更清晰版本：
month_end_dates = (
    component_monthly[["TRADE_DT"]]
    .drop_duplicates()
    .sort_values("TRADE_DT")
    .rename(columns={"TRADE_DT": "COMPONENT_DT"})
)

date_map = pd.merge_asof(
    trade_dates.sort_values("TRADE_DT"),
    month_end_dates,
    left_on="TRADE_DT",
    right_on="COMPONENT_DT",
    direction="backward"
)

# 6. 用 COMPONENT_DT 回连月末成分股表
df_component_daily = date_map.merge(
    component_monthly,
    left_on="COMPONENT_DT",
    right_on="TRADE_DT",
    how="left",
    suffixes=("", "_component")
)

# 7. 整理日期列
df_component_daily = df_component_daily.drop(columns=["TRADE_DT_component"])
df_component_daily = df_component_daily.rename(columns={"TRADE_DT": "TRADE_DT"})

# 8. 可选：保留原始成分股截面日期，方便检查
# COMPONENT_DT = 这一天使用的是哪一期月末成分股
df_component_daily = df_component_daily.sort_values(
    ["TRADE_DT", "S_INFO_WINDCODE", "S_CON_WINDCODE"]
).reset_index(drop=True)

# ----- Cell 22 (code) -----
# 先计算399370的市盈率数据
# df_component = df_component[df_component['S_INFO_WINDCODE'] == target].copy()


#df_component[comp_weight] = pd.to_numeric(df_component[comp_weight], errors="coerce")
#df_component[pe_value] = pd.to_numeric(df_component[pe_value], errors="coerce")

row_pos = df_pb_wide.index.get_indexer(df_component_daily[comp_date])
col_pos = df_pb_wide.columns.get_indexer(df_component_daily[comp_code])

pb_values = np.full(len(df_component_daily), np.nan, dtype=object)
valid_pb = (row_pos >= 0) & (col_pos >= 0)

pb_matrix = df_pb_wide.to_numpy()
pb_values[valid_pb] = pb_matrix[row_pos[valid_pb], col_pos[valid_pb]]

df_component_daily["pb_value"] = pd.to_numeric(
    pd.Series(pb_values, index=df_component_daily.index),
    errors="coerce"
)


# ----- Cell 23 (code) -----
df_merged_pb = (
    df_component_daily
    .dropna()
    .reset_index(drop=True)
)

df_merged_pb.head()

# ----- Cell 24 (code) -----
# =========================================================
# 计算两个风格指数每日成分股 PB 的 QRD
# QRD = (q75 - q25) / (q90 - q10)
# 然后计算 QRD 的 20 日滚动均值
# =========================================================

target_index_list = ["399370.SZ"]
df_merged_pb = df_merged_pb[df_merged_pb["S_INFO_WINDCODE"].isin(target_index_list)].copy()

# 2. 定义单日 QRD 计算函数
def calc_qrd(x: pd.Series) -> float:
    x = x.dropna()

    # 样本太少时不计算，阈值可以自己改
    if len(x) < 10:
        return np.nan

    q10 = x.quantile(0.10)
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    q90 = x.quantile(0.90)

    denominator = q90 - q10

    if denominator == 0:
        return np.nan

    return (q75 - q25) / denominator


# 3. 按 指数代码 × 交易日 计算每日 QRD
daily_qrd = (
    df_merged_pb
    .groupby(["S_INFO_WINDCODE", "TRADE_DT"])["pb_value"]
    .apply(calc_qrd)
    .reset_index(name="PB_QRD")
)

# 4. 按指数分别计算 20 日滚动均值
daily_qrd = daily_qrd.sort_values(["S_INFO_WINDCODE", "TRADE_DT"])

daily_qrd["PB_QRD_MA20"] = (
    daily_qrd
    .groupby("S_INFO_WINDCODE")["PB_QRD"]
    .transform(lambda s: s.rolling(window=20, min_periods=20).mean())
)

# =========================================================
# 直接变成时间序列
# =========================================================

qrd_ts = (
    daily_qrd
    .set_index("TRADE_DT")
    .sort_index()[["PB_QRD", "PB_QRD_MA20"]]
).dropna()

qrd_ts.head()

# ----- Cell 25 (code) -----
qrd_ts["ZHAO03"] = qrd_ts["PB_QRD_MA20"]/ qrd_ts["PB_QRD_MA20"].shift(1) - 1

# ----- Cell 26 (code) -----
_assign_raw_factor("ZHAO03_raw", qrd_ts["ZHAO03"])

# ----- Cell 27 (markdown) -----
# ### PB_MAD (ZHAO04)

# ----- Cell 28 (code) -----
df_merged_pb.head()

# ----- Cell 29 (code) -----
df_merged_pb["pb_median"] = df_merged_pb.groupby("TRADE_DT")["pb_value"].transform("median")

# Step 2: 绝对偏差
df_merged_pb["abs_dev"] = (df_merged_pb["pb_value"] - df_merged_pb["pb_median"]).abs()

# Step 3: MAD（按天）
mad_df = (
    df_merged_pb.groupby("TRADE_DT")["abs_dev"]
    .median()
    .rename("PB_MAD")
    .reset_index()
)

df_merged_pb["PB_MAD"] = df_merged_pb.groupby("TRADE_DT")["abs_dev"].transform("median")

# ----- Cell 30 (code) -----
mad_df["PB_MAD_MA20"] = mad_df["PB_MAD"].rolling(window=20, min_periods=20).mean()
mad_df = mad_df.dropna()

# ----- Cell 31 (code) -----
mad_df["ZHAO04"] = mad_df["PB_MAD_MA20"]/ mad_df["PB_MAD_MA20"].shift(1) - 1
mad_df.set_index("TRADE_DT", inplace=True)

# ----- Cell 32 (code) -----
mad_df.tail()

# ----- Cell 33 (code) -----
_assign_raw_factor("ZHAO04_raw", mad_df["ZHAO04"])

# ----- Cell 34 (markdown) -----
# ### 新增中长期人民币贷款 (ZHAO05)
# 日期待调整

# ----- Cell 35 (code) -----
sub_1 = _read_indicator_series('DebtData.xlsx', '中国:金融机构:新增人民币贷款:中长期:当月值')
ZHAO05_raw = _data_month_end_series(_rolling_sum_ratio_minus_one(sub_1, window=12)) 
_assign_raw_factor("ZHAO05_raw", ZHAO05_raw)

# ----- Cell 36 (markdown) -----
# ### PMI (ZHAO06)

# ----- Cell 37 (code) -----
df_PMI = pd.read_excel(Config.DATA_DIR / "宏观大事_合并整理.xlsx")
df_PMI = clean_macro_table(df_PMI, nation="中国", indi="官方制造业PMI")
df_PMI = df_PMI[~df_PMI.index.duplicated(keep="last")]
print(df_PMI.head())
print(df_PMI.dtypes)

# ----- Cell 38 (code) -----
def pmi_yoy_chain(
    df_PMI: pd.DataFrame,
    value_col: str = "actual",
    scale: float = 0.01,
    base_level: float = 100.0,
    drop_dup_keep: str = "last"
) -> pd.DataFrame:
    """
    用 PMI-50 作为“环比增速代理”，链式累乘后再计算 12期同比。

    参数
    ----------
    df_PMI : pd.DataFrame
        index 为发布日期，列至少包含 value_col
    value_col : str
        使用哪一列作为 PMI 值，默认 'actual'
    scale : float
        把 (PMI-50) 映射为“环比代理增速”的缩放系数
        例如 actual=51.3，则环比代理增速 = (51.3-50)*0.01 = 1.3%
    base_level : float
        伪水平序列的起始值
    drop_dup_keep : str
        若 index 重复，保留 'last' 或 'first'

    返回
    ----------
    out : pd.DataFrame
        包含：
        - PMI
        - mom_proxy
        - pseudo_level
        - yoy_chain
    """
    out = df_PMI.copy()

    # 1) 按发布日期排序
    out = out.sort_index()

    # 2) 若 index 重复，只保留最后一个（更符合你之前的习惯）
    out = out[~out.index.duplicated(keep=drop_dup_keep)].copy()

    # 3) 提取 PMI 实际值
    out["PMI"] = pd.to_numeric(out[value_col], errors="coerce")

    # 4) PMI-50 作为“环比代理增速”
    out["mom_proxy"] = (out["PMI"] - 50.0) * scale

    # 5) 链式累乘得到“伪水平”
    #    注意：若某期 mom_proxy 缺失，则该期 pseudo_level 也会缺失
    growth_factor = 1.0 + out["mom_proxy"]
    out["pseudo_level"] = base_level * growth_factor.cumprod()

    # 6) 12期同比
    out["yoy_chain"] = out["pseudo_level"] / out["pseudo_level"].shift(12) - 1.0

    return out

# ----- Cell 39 (code) -----
df_pmi_chain = pmi_yoy_chain(df_PMI, value_col="actual", scale=0.01)
print(df_pmi_chain[["PMI", "mom_proxy", "pseudo_level", "yoy_chain"]].tail())
print(len(df_pmi_chain))

# ----- Cell 40 (code) -----
_assign_raw_factor("ZHAO06_raw", df_pmi_chain["yoy_chain"])

# ----- Cell 41 (markdown) -----
# ### 1年期国债到期收益率 (ZHAO07)

# ----- Cell 42 (code) -----
sub_1 = _read_indicator_series('D_国债到期收益率_CN_020104_260409.xlsx', '中债国债到期收益率:1年')
monthly_avg = _month_aggregate(sub_1, how="average")
ZHAO07_raw = _YoY(monthly_avg)*(-1)
_assign_raw_factor("ZHAO07_raw", ZHAO07_raw)


# ----- Cell 43 (markdown) -----
# ### 2年期美国国债到期收益率 (ZHAO08)

# ----- Cell 44 (code) -----
sub_1 = _read_indicator_series('D_国债收益率_US_530430_260324.xlsx', '美国:国债收益率:2年')
monthly_avg = _month_aggregate(sub_1, how="average")
ZHAO08_raw = _YoY(monthly_avg)*(-1)
_assign_raw_factor("ZHAO08_raw", ZHAO08_raw)


# ----- Cell 45 (markdown) -----
# ### 新增规上工业企业利润总额 (ZHAO09)

# ----- Cell 46 (code) -----
sub_1 = _read_indicator_series('规模以上工业 招证资配.xlsx', '中国:利润总额:规模以上工业企业:累计值')
ZHAO09_raw = _data_month_end_series(_rolling_sum_ratio_minus_one(sub_1, window=12))
_assign_raw_factor("ZHAO09_raw", ZHAO09_raw)


# ----- Cell 47 (markdown) -----
# ### 工业企业产成品存货 (ZHAO10)

# ----- Cell 48 (code) -----
sub_1 = _read_indicator_series('规模以上工业 招证资配.xlsx', '中国:产成品存货:规模以上工业企业:同比')
ZHAO10_raw = _data_month_end_series(0 - sub_1)
_assign_raw_factor("ZHAO10_raw", ZHAO10_raw)


# ----- Cell 49 (markdown) -----
# ### 新增社零 (ZHAO11)

# ----- Cell 50 (code) -----
sub_1 = _load_china_macro_series('社会消费品零售总额')
ZHAO11_raw = _rolling_sum_ratio_minus_one(sub_1, window=12, shift=11)
_assign_raw_factor("ZHAO11_raw", ZHAO11_raw)

# ----- Cell 51 (markdown) -----
# ### 新增出口额（美元） (ZHAO12)
# 缺数据

# ----- Cell 52 (code) -----
sub_1 = _load_china_macro_series('月出口金额:当月值(亿美元)')
ZHAO12_raw = _rolling_sum_ratio_minus_one(sub_1, window=12)
_assign_raw_factor("ZHAO12_raw", ZHAO12_raw)


# ----- Cell 53 (markdown) -----
# ### 一般公共预算支出 (ZHAO13)

# ----- Cell 54 (code) -----
sub_1 = _read_indicator_series('公共预算支出.xlsx', '中国:一般公共预算支出:当月同比(1-2月合并)')
ZHAO13_raw = _data_month_end_series(sub_1)
_assign_raw_factor("ZHAO13_raw", ZHAO13_raw)


# ----- Cell 55 (markdown) -----
# ### 美元兑人民币中间价 (ZHAO14)

# ----- Cell 56 (code) -----
sub_1 = _read_indicator_series('日频汇率.xlsx', '中间价:美元兑人民币')
monthly_avg = _month_aggregate(sub_1, how="average")
ZHAO14_raw = _YoY(monthly_avg)*(-1)
_assign_raw_factor("ZHAO14_raw", ZHAO14_raw)


# ----- Cell 57 (markdown) -----
# ### M0同比 (ZHAO15)

# ----- Cell 58 (code) -----
sub_1 = _load_china_macro_series('月M0:同比(%)')
ZHAO15_raw = sub_1
_assign_raw_factor("ZHAO15_raw", ZHAO15_raw)

# ----- Cell 59 (markdown) -----
# ### M1同比 (ZHAO16)

# ----- Cell 60 (code) -----
sub_1 = _load_china_macro_series('月M1:同比(%)')
ZHAO16_raw = sub_1
_assign_raw_factor("ZHAO16_raw", ZHAO16_raw)

# ----- Cell 61 (markdown) -----
# ### M2同比 (ZHAO17)

# ----- Cell 62 (code) -----
sub_1 = _load_china_macro_series('月M2:同比(%)')
ZHAO17_raw = sub_1
_assign_raw_factor("ZHAO17_raw", ZHAO17_raw)


# ----- Cell 63 (markdown) -----
# ### M1同比-M2同比 (ZHAO18)

# ----- Cell 64 (code) -----
# Follow the ZHAO.xlsx docu order: sub_1 = M2 YoY, sub_2 = M1 YoY.
sub_1 = _load_china_macro_series('月M1:同比(%)')
sub_2 = _load_china_macro_series('月M2:同比(%)')
ZHAO18_raw = sub_1 - sub_2
_assign_raw_factor("ZHAO18_raw", ZHAO18_raw)


# ----- Cell 65 (markdown) -----
# ### PPI (ZHAO20)

# ----- Cell 66 (code) -----
sub_1 = _load_china_macro_series('PPI:同比')
ZHAO20_raw = sub_1
_assign_raw_factor("ZHAO20_raw", ZHAO20_raw)

# ----- Cell 237 (markdown) -----
# ## 因子挂载

# ----- Cell 237-1 (code) -----
from odds_win_rotation_factors import FINAL_RAW_FACTOR, ODDS_WIN_SINGLE_RAW_FACTORS, generate_odds_win_factors

data_df, market_df, odds_win_summary = generate_odds_win_factors(
    data_df=data_df,
    market_df=market_df,
    data_dir=Path("data"),
    include_strong_ratio=True,
    mount_final=False,
)
display(odds_win_summary)

# ----- Cell 238 (code) -----
# Cell 20：设置这次跑的因子
# 可通过环境变量 RAW_FACTOR_NAME 指定任一单因子，例如：
# RAW_FACTOR_NAME=OW_F001_CN10Y_Q3Y_raw python "风格轮动信号检验法 copy.py"
# 若不指定，则默认跑复合最终信号 ODDS_WIN_FINAL_raw。
RAW_FACTOR_NAME = os.environ.get("RAW_FACTOR_NAME", FINAL_RAW_FACTOR)
FACTOR_BAR = 0
signal_type = "state" 
# "state"  : 状态型信号，每个调仓点都重新判断目标仓位（例如 overzero）
# "event"  : 事件型信号，只在信号发生变化时才触发调仓

if RAW_FACTOR_NAME.endswith("_raw"):
    FACTOR_NAME = RAW_FACTOR_NAME[:-4]
else:
    raise ValueError(f"RAW_FACTOR_NAME {RAW_FACTOR_NAME} 不符合命名规范，应该以 '_raw' 结尾")

# ----- Cell 239 (code) -----
# Cell 21：因子检查I
print(data_df[RAW_FACTOR_NAME].dtype)          # 必须是 float
print(data_df[RAW_FACTOR_NAME].isna().sum())   # NaN 不要异常多
print(data_df[RAW_FACTOR_NAME].head())
print(data_df[RAW_FACTOR_NAME].tail())

# ----- Cell 240 (code) -----
data_df[RAW_FACTOR_NAME] = data_df[RAW_FACTOR_NAME].replace([np.inf, -np.inf], np.nan)

# ----- Cell 241 (code) -----
# Cell 22：因子检查II
data_df[RAW_FACTOR_NAME].max(), data_df[RAW_FACTOR_NAME].min(), data_df[RAW_FACTOR_NAME].mean()

# ----- Cell 242 (code) -----
# Cell 23：因子检查III
data_df[RAW_FACTOR_NAME].quantile([0.25, 0.5, 0.75])

# ----- Cell 243 (code) -----
# 1. 先获取你的基准日期
factor_available_date = data_df[RAW_FACTOR_NAME].first_valid_index()

if factor_available_date is not None:
    # 2. 找到这个日期在整个 Index 中的整数位置
    current_loc = data_df.index.get_loc(factor_available_date)
    
    # 3. 检查一下这是否已经是最后一天，防止加 1 后索引越界报错
    if current_loc + 1 < len(data_df.index):
        # 4. 获取下一个位置的日期
        factor_usable_date = data_df.index[current_loc + 1]
        print(f"当前有数据的日期是: {factor_available_date}")
        print(f"实际可操作的下一个日期是: {factor_usable_date}")
    else:
        print(f"当前日期 {factor_available_date} 已经是数据集的最后一天，没有下一个日期了。")
else:
    print("该列全为空，没有有效日期。")

# ----- Cell 244 (code) -----
# Cell 24: 信号滞后与周频挂载
# 1）在 data_df 中生成正式因子列 factor_mom_1m_diff = factor_mom_1m_diff_raw.shift(1)
# 2）把这个正式因子按日期索引挂载到 market_df 中
market_df = merge_factor_to_market(
    data_df=data_df,
    market_df=market_df,
    raw_factor_col= RAW_FACTOR_NAME,
    factor_type = signal_type
)

print("因子挂载完成：")
print("当前 FEATURE_LIST =", Config.FEATURE_LIST)

display(
    market_df[["track_id", FACTOR_NAME]] 
    .dropna(subset=[FACTOR_NAME])
    .tail(Config.Tunnels*2)
)

# ----- Cell 245 (code) -----
# Cell 25: 看眼因子列
display(
    market_df[["track_id", FACTOR_NAME]]
    .dropna(subset=[FACTOR_NAME])
    .tail(Config.Tunnels*2)
)
