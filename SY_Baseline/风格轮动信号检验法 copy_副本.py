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


def _assign_raw_factor(raw_col, series):
    s = series.copy().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    data_df[raw_col] = s.reindex(data_df.index)
    print(f"{raw_col} generated:", "non_na=", int(data_df[raw_col].notna().sum()), "first=", data_df[raw_col].first_valid_index(), "last=", data_df[raw_col].last_valid_index())
    s = s.dropna()
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

# ----- Cell 246 (code) -----
# Cell 26: 删除market_df中冗余的列

# 1. 绝对价格、交易量、暂用不到的指标
cols_to_drop_price_vol = [
    'high_g', 'low_g', 'price_change_g', 'pct_change_g', 'volume_g', 'amount_g', 'turnover_rate_g', 'DDE_g',
    'high_v', 'low_v', 'price_change_v', 'pct_change_v','volume_v', 'amount_v', 'turnover_rate_v', 'DDE_v',
    'RSI_5D_g', 'RSI_10D_g', 'RSI_15D_g','RSI_20D_g', 'RSI_5D_v', 'RSI_10D_v', 'RSI_20D_v','RSI_15D_v',
    'MACD_10_20_5_g', 'MACD_10_20_5_v'
]

# 2. 累计变量
cols_to_drop_cum = [
    'cum_price_change_g', 'cum_pct_change_g', 
    'cum_price_change_v', 'cum_pct_change_v'
]

# 3. 之前用于检查的变量
cols_to_drop_examine = [
    'next_price_g', 'next_price_v'
]

# 合并删除
all_cols_to_drop = cols_to_drop_price_vol + cols_to_drop_cum + cols_to_drop_examine
market_df = market_df.drop(columns=all_cols_to_drop, errors='ignore')

# ----- Cell 247 (code) -----
# Cell 27: 未来累计收益构造工具函数

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

# ----- Cell 248 (code) -----
# Cell 28: 生成 Bucket IC / Horizon IC 目标列

# 1. Bucket IC 目标：未来第 k 周“单周收益差”
for k in Config.BUCKET_WEEKS:
    col_name = f"{Config.BUCKET_PREFIX}{k}"
    market_df[col_name] = (
        market_df
        .groupby("track_id")["target_return_diff"]
        .shift(-(k - 1))
    )

# 2. Horizon IC 目标：未来 h 周“累计收益差”
for h in Config.HORIZON_WEEKS:
    g_col = f"horizon_ret_g_{h}"
    v_col = f"horizon_ret_v_{h}"
    target_col = f"{Config.HORIZON_PREFIX}{h}"

    market_df[g_col] = (
        market_df
        .groupby("track_id")["fwd_ret_g"]
        .transform(lambda s: calc_future_horizon_return(
            ret_series=s,
            horizon=h,
            return_type=Config.RETURN_TYPE
        ))
    )

    market_df[v_col] = (
        market_df
        .groupby("track_id")["fwd_ret_v"]
        .transform(lambda s: calc_future_horizon_return(
            ret_series=s,
            horizon=h,
            return_type=Config.RETURN_TYPE
        ))
    )

    market_df[target_col] = market_df[g_col] - market_df[v_col]

# 3. 抽样检查
check_cols = [
    "track_id",
    "target_return_diff",
    "fwd_ret_g",
    "fwd_ret_v",
]

check_cols += [f"{Config.BUCKET_PREFIX}{k}" for k in Config.BUCKET_WEEKS]
check_cols += [f"{Config.HORIZON_PREFIX}{h}" for h in Config.HORIZON_WEEKS]

print()
print("目标列抽样检查：")
display(market_df[check_cols].head(12))

print()
print("各目标列非空样本数：")
target_cols = [f"{Config.BUCKET_PREFIX}{k}" for k in Config.BUCKET_WEEKS] + [f"{Config.HORIZON_PREFIX}{h}" for h in Config.HORIZON_WEEKS]
display(market_df[target_cols].notna().sum().to_frame("non_null_count").T)

# ----- Cell 249 (code) -----
# Cell 29：切分样本内外
ins_mask = (market_df.index >= Config.INS_START) & (market_df.index <= Config.INS_END)
oos_mask = (market_df.index >= Config.OOS_START) & (market_df.index <= Config.OOS_END)
all_mask = (market_df.index >= Config.ALL_START) & (market_df.index <= Config.ALL_END)
market_df_ins = market_df.loc[ins_mask].copy()
market_df_oos = market_df.loc[oos_mask].copy()
market_df_all = market_df.loc[all_mask].copy()

# ----- Cell 250 (markdown) -----
# # 单因子有效性检验

# ----- Cell 263 (markdown) -----
# ## 信号检验法

# ----- Cell 264 (code) -----
# Cell 41: 参数定义
benchmark_mode = "rebalance_50"
benchmark_rebalance_months = 2 # 每n个月调回一次50%+50%
# "buy_and_hold_50"  : 期初买入 50% 成长 + 50% 价值，之后完全不调仓
# "rebalance_50"     : 按 benchmark_rebalance_months 设定的频率，定期调回 50% 成长 + 50% 价值

# 交易参数设置
trans_fee = 0.002          # 单边手续费（买入或卖出均按此费率收取）
charge_initial_trade = True  # True = 期初建仓也收手续费；False = 期初建仓免费
rf = 0.015          # 无风险利率（年化），用于计算夏普比率等
annual_days = Config.ANNUAL_TRADING_DAYS  # 一年按多少天折算，用于年化收益率和年化波动率，这里仍然使用交易日（而非自然日）
weight_tol = 1e-10 # 两个浮点数权重之差小于 weight_tol，就认为它们相等，不触发调仓

# 输出目录
if benchmark_mode == "rebalance_50":
    strat_suffix = f"{benchmark_mode}_{benchmark_rebalance_months}m"
else:
    strat_suffix = benchmark_mode

STRAT_OUTPUT_DIR = OUTPUT_DIR / f"strategy_{strat_suffix}"
STRAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Cell 265 (code) -----
# Cell 42: 定义策略绩效计算函数

# 为计算绩效指标做准备：把 future return 统一转换成 simple return
def to_simple_return(ret_series, return_type=Config.RETURN_TYPE):
    """
    把底层收益率统一转换成 simple return，便于后面计算净值、回撤、夏普等指标。
    """
    ret_series = pd.Series(ret_series).astype(float)

    if return_type == "log":
        return np.expm1(ret_series)
    elif return_type == "simple":
        return ret_series

# ----- Cell 266 (code) -----
# Cell 43: 生成策略用底表
cols_to_keep = [
    "track_id",
    FACTOR_COL,
    "fwd_ret_g",
    "fwd_ret_v",
    "holding_days",
    "target_return_diff",
    "next_date",
    "target_label",
    "close_g",
    "close_v"
]

work_df = market_df_all[market_df_all.index > factor_available_date].copy().sort_index(ascending=True)
work_df = work_df[cols_to_keep].copy()
work_df["next_date"] = pd.to_datetime(work_df["next_date"])

# 把底层收益率统一转成 simple return，后面净值计算可以统一用"连乘 (1 + r)"

work_df["fwd_ret_g_simple"] = to_simple_return(
    work_df["fwd_ret_g"],
    return_type=Config.RETURN_TYPE
)

work_df["fwd_ret_v_simple"] = to_simple_return(
    work_df["fwd_ret_v"],
    return_type=Config.RETURN_TYPE
)

# 成长 - 价值的 simple return 差，方便后面直接用
work_df["fwd_ret_diff_simple"] = work_df["fwd_ret_g_simple"] - work_df["fwd_ret_v_simple"]

# 检查：work_df[["track_id", "holding_days", "next_date"]]，应该每一行都有 next_date，

display(
    work_df[
        [
            "track_id", FACTOR_COL,
            "fwd_ret_g", "fwd_ret_v",
            "fwd_ret_g_simple", "fwd_ret_v_simple",
            "holding_days", "next_date"
        ]
    ].head(10)
)

# ----- Cell 267 (code) -----
# Cell 44：定义权重漂移函数

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

# ----- Cell 268 (code) -----
# 判断两组权重是否在容忍度下不同
def weights_are_different(weight_g_a, weight_v_a, weight_g_b, weight_v_b, tol = weight_tol):
    diff_g = abs(weight_g_a - weight_g_b)
    diff_v = abs(weight_v_a - weight_v_b)

    if (diff_g > tol) or (diff_v > tol):
        return True
    else:
        return False

# ----- Cell 269 (code) -----
# Cell 45：定义“本期是否交易、交易后仓位是多少”的决策函数

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

            # 正常无信号
            effective_signal_ls = int(prev_effective_signal_ls)
            
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
        # 情况 C：本期有新信号，且signal_ls ∈ {-1,1}
        # 规则：按照signal_ls进行调仓
        # -----------------------------------------------------

        effective_signal_ls = int(signal_ls)
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

# ----- Cell 270 (code) -----
# Cell 46：状态型信号函数定义

# 状态型信号：每一个调仓点都重新看一次因子值，然后判断目标仓位。
# 无论因子是什么值，每一行都必须有一个明确的目标仓位，不允许出现空仓。
#
# 输入：
#   track_df   - 单条轨道的 DataFrame（已经是某一条轨道的子集）
#   factor_col - 因子列的列名（字符串），例如 "factor_mom_1m_diff"
#
# 输出：
#   在 track_df 的副本上新增以下 5 列，并返回这个副本：
#     target_weight_g   : 成长指数的目标权重，取值只能是 0.0 / 0.5 / 1.0
#     target_weight_v   : 价值指数的目标权重，取值只能是 0.0 / 0.5 / 1.0
#     signal_ls         : 多空方向信号，目标方向，+1=全仓成长，-1=全仓价值，0=50/50中性
#     signal_update_flag: 状态型信号每期都重新判断，所以恒为 1

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

# ----- Cell 271 (code) -----
# Cell 47：触发型事件信号函数定义

# 触发型事件信号：只有在"本期发生了新事件"时，才更新目标仓位。
# 没有新事件的行，目标仓位全部用 NaN 表示"维持上期仓位不变"。
#
# 这和状态型信号的关键区别是：
#   状态型：每一期都重新判断，signal_update_flag 恒为 1
#   事件型：只有事件触发时才更新，没有事件时 signal_update_flag = 0，
#           目标仓位列全部为 NaN（不是 0！）
#
# 输入：
#   track_df       - 单条轨道的 DataFrame
#   event_col      - 事件列的列名（字符串）
#                    该列的取值含义：
#                      > 0  -> 看好成长事件
#                      < 0  -> 看好价值事件
#                      == 0 -> 未触发看好价值或成长，但是有数据输入
#                      NaN  -> 本期没有新事件
#   event_time_col - 可选参数，事件发生的具体时间列（字符串或 None）
#                    当前版本暂不使用，保留接口供未来扩展
#
# 输出：
#   在 track_df 的副本上新增以下 5 列，并返回这个副本：
#     target_weight_g   : 有新事件时为 0.0/0.5/1.0；无新事件时为 NaN
#     target_weight_v   : 有新事件时为 0.0/0.5/1.0；无新事件时为 NaN
#     signal_ls         : 有新事件时为 +1/0/-1；无新事件时为 NaN
#     signal_update_flag: 有新事件时为 1；无新事件时为 0

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


# ------ 快速验证：用 work_df 里的 FACTOR_COL 列模拟一个事件列跑一下 ------
# 因为当前阶段还没有正式的事件列，
# 这里用 FACTOR_COL 列来模拟事件（直接复用因子值当事件列），
# 只是为了验证函数输出的列结构是否正确，不代表真实的事件型策略逻辑

_test_track_id_ev  = work_df["track_id"].dropna().unique()[0]
_test_track_df_ev  = work_df[work_df["track_id"] == _test_track_id_ev].copy()
_test_result_ev    = build_event_target_position(_test_track_df_ev, event_col=FACTOR_COL)

print("===== Cell 47 验证：触发型事件信号函数输出结构 =====")
print(f"  测试轨道 track_id = {_test_track_id_ev}")
print(f"  输出列名          : {list(_test_result_ev.columns)}")
print()

# 检查 signal_update_flag 只有 0 和 1 两种值
_flag_values = _test_result_ev["signal_update_flag"].unique()
print(f"  signal_update_flag 出现的所有取值 : {sorted(_flag_values)}")
assert set(_flag_values).issubset({0, 1}), \
    "signal_update_flag 出现了预期之外的值，只允许 0 或 1"

# 检查：有新事件的行，signal_ls 不能是 NaN
_triggered_rows = _test_result_ev[_test_result_ev["signal_update_flag"] == 1]
assert _triggered_rows["signal_ls"].notna().all(), \
    "有新事件的行（signal_update_flag=1）里出现了 NaN 的 signal_ls，请检查函数逻辑"

# 检查：没有新事件的行，signal_ls 必须全部是 NaN
_no_event_rows = _test_result_ev[_test_result_ev["signal_update_flag"] == 0]
assert _no_event_rows["signal_ls"].isna().all(), \
    "没有新事件的行（signal_update_flag=0）里出现了非 NaN 的 signal_ls，请检查函数逻辑"

# 打印触发事件 / 无事件的行数分布
print()
print("  signal_update_flag 各取值的行数分布：")
display(_test_result_ev["signal_update_flag"].value_counts().sort_index(ascending=True).to_frame("行数"))

print()
print("  前 10 行预览（含信号列）：")
_preview_cols_ev = [FACTOR_COL, "target_weight_g", "target_weight_v",
                    "signal_ls", "signal_update_flag"]
display(_test_result_ev[_preview_cols_ev].head(10))

print()
print("注意：以上验证是用 FACTOR_COL 列模拟事件列跑的，")
print("仅用于检查函数输出结构是否正确，不代表真实的事件型策略逻辑。")

# ----- Cell 272 (code) -----
# Cell 48: 构造基准侧目标仓位表（benchmark_target_df），也就是成长-价值 50-50

# "连续两个 50/50 是否真的产生换手"，由后面的执行器根据 benchmark_mode 决定，
#  这张表里不做这个判断。
#
# 输入变量：
#   strategy_target_df - Cell 38 生成的策略侧目标仓位表
#
# 输出变量：
#   benchmark_target_df - 基准侧目标仓位表

# ------ 第一步：复制策略表的索引和公共列 ------
# 只复制"和底层收益有关"的列，目标仓位列之后重新赋值

cols_to_copy = [
    "track_id",
    FACTOR_COL,
    "fwd_ret_g_simple",
    "fwd_ret_v_simple",
    "holding_days",
    "next_date"
]

benchmark_target_df = work_df[cols_to_copy].copy()

# ------ 第二步：定义“每 n 个月再平衡一次”的日期标记函数 ------
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

# ------ 第三步：把所有目标仓位列固定为 50/50 中性 ------
benchmark_target_df["target_weight_g"] = 0.5
benchmark_target_df["target_weight_v"] = 0.5
benchmark_target_df["signal_ls"] = 0

# 先初始化
benchmark_target_df["signal_update_flag"] = 0 #基准的signal_update_flag意为本期是否需要rebalance

# 逐轨生成“每 n 个月再平衡”的标记
for track_id in track_list:
    mask = benchmark_target_df["track_id"] == track_id
    track_index = benchmark_target_df.loc[mask].index

    if benchmark_mode == "buy_and_hold_50":
        rebalance_flag = pd.Series(0, index=track_index, dtype="int64")
        rebalance_flag.iloc[0] = 1  # 期初建仓一次
    
    elif benchmark_mode == "rebalance_50":
        rebalance_flag = build_benchmark_rebalance_flag(
            track_index=track_index,
            rebalance_months=benchmark_rebalance_months
        )
    else:
        raise ValueError(
            f"benchmark_mode 只能是 'buy_and_hold_50' 或 'rebalance_50'，当前值为：{benchmark_mode}"
        )

    benchmark_target_df.loc[mask, "signal_update_flag"] = rebalance_flag.values

benchmark_target_df["signal_update_flag"] = (
    benchmark_target_df["signal_update_flag"].astype(int)
)


display(
    benchmark_target_df.loc[
        benchmark_target_df["track_id"] == 1,
        [
            "track_id",
            "target_weight_g",
            "target_weight_v",
            "signal_ls",
            "signal_update_flag"
        ],
    ].head(20)
)

# ----- Cell 273 (code) -----
# Cell 49: 定义单轨回测器

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

# ----- Cell 274 (code) -----
# Cell 50: 跑完整个基准的五轨回测

# 输入变量：
#   benchmark_target_df
#   track_list
#   run_single_track_backtest
#
# 输出变量：
#   benchmark_period_df

# ------ 第一步：逐轨执行单轨回测 ------
benchmark_result_list = []

for track_id in track_list:
    # 取出当前轨道的数据，并按日期排序
    track_target_df = (
        benchmark_target_df[benchmark_target_df["track_id"] == track_id]
        .copy()
        .sort_index(ascending=True)
    )

    # 调用单轨回测函数
    track_period_df = run_single_track_backtest(
        track_target_df=track_target_df,
        benchmark_mode=benchmark_mode,
        trans_fee=trans_fee,
        charge_initial_trade=charge_initial_trade,
        receiver = "benchmark",
        benchmark_ref_track_df = None
    )

    # 存入列表
    benchmark_result_list.append(track_period_df)

# ------ 第二步：拼接成总表 ------
benchmark_period_df = pd.concat(benchmark_result_list, axis=0)
benchmark_period_df = benchmark_period_df.sort_index(ascending=True)

print("===== Cell 50 验证：benchmark_period_df 构造完成 =====")
print(f"行数 / 列数   : {benchmark_period_df.shape}")
print(f"日期范围      : {benchmark_period_df.index.min().date()} ~ {benchmark_period_df.index.max().date()}")
print(f"轨道列表      : {sorted(benchmark_period_df['track_id'].dropna().astype(int).unique())}")
print()

print("各轨道行数：")
display(benchmark_period_df["track_id"].value_counts().sort_index(ascending=True).to_frame("行数"))

print()
print("前 5 行预览：")
display(benchmark_period_df.head(5))

# ----- Cell 275 (code) -----
# Cell 51：逐轨生成策略目标仓位表（strategy_target_df）

# 这个 cell 的作用：
#   对 work_df 中的每一条轨道（track_id），分别调用对应的信号函数，
#   得到该轨道的目标仓位，最后把所有轨道拼在一起，得到完整的策略目标表。
#
# 输入变量（来自前面的 cell）：
#   work_df                      - 包含所有轨道数据的底表
#   signal_type                  - "state" 或 "event"，决定调用哪个信号函数
#   FACTOR_COL                   - 因子列名，例如 "factor_mom_1m_diff"
#   build_state_target_position  - Cell 36 定义的状态型信号函数
#   build_event_target_position  - Cell 37 定义的触发型事件信号函数
#
# 输出变量：
#   strategy_target_df           - 所有轨道的目标仓位表（已排序）

all_track_results = []   # 用来收集每条轨道的结果，最后一次性 concat

for track_id in track_list:

    # 取出当前轨道的数据，并按日期排序
    track_df = work_df[work_df[TRACK_COL] == track_id].copy().sort_index(ascending=True)

    # 根据 signal_type 决定调用哪个信号函数
    if signal_type == "state":
        # 状态型：每期都重新判断，直接用 FACTOR_COL
        track_result = build_state_target_position(track_df, factor_col=FACTOR_COL, bar = FACTOR_BAR)

    elif signal_type == "event":
        # 事件型：只有事件触发时才更新
        # 注意：正式使用时，event_col 应该换成真正的事件列名
        # 这里暂时用 FACTOR_COL 占位，和 Cell 37 的验证逻辑保持一致
        track_result = build_event_target_position(track_df, event_col=FACTOR_COL, bar = FACTOR_BAR)

    else:
        raise ValueError(f"signal_type 只能是 'state' 或 'event'，当前值是: {signal_type}")

    # 把这条轨道的结果加入列表
    all_track_results.append(track_result)

# ------ 第三步：把所有轨道的结果拼在一起，按日期排序 ------
strategy_target_df = pd.concat(all_track_results, axis=0)
strategy_target_df = strategy_target_df.sort_index(ascending=True)
strategy_target_df["track_id"] = strategy_target_df["track_id"].astype("Int64")

# ----- Cell 276 (code) -----
# Cell 52：执行单轨策略回测


#   输入：
#   strategy_target_df
#   track_list
#   run_single_track_backtest
#
# 输出变量：
#   strategy_period_df

# ------ 第一步：逐轨执行单轨回测 ------
strategy_result_list = []

for track_id in track_list:
    # 取出当前轨道的数据，并按日期排序
    track_target_df = (
        strategy_target_df[strategy_target_df["track_id"] == track_id]
        .copy()
        .sort_index(ascending=True)
    )

    # 同轨benchmark结果表，供strategy的signal_ls == 0 时引用
    benchmark_ref_track_df = (
        benchmark_period_df[benchmark_period_df["track_id"] == track_id]
        .copy()
        .sort_index(ascending=True)
    )

    # 调用单轨回测函数
    track_period_df = run_single_track_backtest(
        track_target_df=track_target_df,
        benchmark_mode=benchmark_mode,
        trans_fee=trans_fee,
        charge_initial_trade=charge_initial_trade,
        receiver = "strategy",
        benchmark_ref_track_df = benchmark_ref_track_df
    )

    # 存入列表
    strategy_result_list.append(track_period_df)

# ------ 第二步：拼接成总表 ------
strategy_period_df = pd.concat(strategy_result_list, axis=0)
strategy_period_df = strategy_period_df.sort_index(ascending=True)

# ------ benchmark / strategy 对齐检查 ------
assert len(benchmark_period_df) == len(strategy_period_df), \
    "benchmark_period_df 与 strategy_period_df 行数不一致，请检查输入表是否对齐"

assert benchmark_period_df.index.equals(strategy_period_df.index), \
    "benchmark_period_df 与 strategy_period_df 的日期索引不一致，请检查输入表是否对齐"

assert (benchmark_period_df["track_id"].values == strategy_period_df["track_id"].values).all(), \
    "benchmark_period_df 与 strategy_period_df 的 track_id 结构不一致，请检查输入表是否对齐"

print("===== Cell 52 验证：strategy_period_df 构造完成 =====")
print(f"行数 / 列数   : {strategy_period_df.shape}")
print(f"日期范围      : {strategy_period_df.index.min().date()} ~ {strategy_period_df.index.max().date()}")
print(f"轨道列表      : {sorted(strategy_period_df['track_id'].dropna().astype(int).unique())}")
print()

print("各轨道行数：")
display(strategy_period_df["track_id"].value_counts().sort_index().to_frame("行数"))

print()
print("前 5 行预览：")
display(strategy_period_df.head(5))

# ----- Cell 277 (code) -----
# Cell 53：构造日频资产收益表，并定义“单轨真实逐日净值重建函数”

n = 10 #允许最后几行的end_date是NaT

# 这里直接用价格算“日频 simple return”
# 注意：净值路径一定要用 simple return 来复利，
# 不要直接拿 log return 去做 (1 + r) 连乘
daily_asset_ret_df = data_df[["close_g", "close_v"]].copy().sort_index(ascending=True)

daily_asset_ret_df["ret_g_daily_simple"] = daily_asset_ret_df["close_g"].pct_change()
daily_asset_ret_df["ret_v_daily_simple"] = daily_asset_ret_df["close_v"].pct_change()

daily_asset_ret_df = daily_asset_ret_df[["ret_g_daily_simple", "ret_v_daily_simple"]]


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


print("===== 逐日净值重建函数已定义 =====")
display(daily_asset_ret_df.head(5))

# ----- Cell 278 (code) -----
# Cell 53：生成单轨净值曲线和五轨组合净值曲线

# 输入变量：
#   strategy_period_df
#   benchmark_period_df
#   track_list
#   daily_asset_ret_df
#   build_single_track_daily_nav

# 输出变量：
#   strategy_track_nav_dict
#   benchmark_track_nav_dict
#   strategy_combo_nav
#   benchmark_combo_nav

# =========================================================
# Part A：单轨真实逐日净值曲线
# =========================================================

strategy_track_nav_dict = {}
benchmark_track_nav_dict = {}

for track_id in track_list:
    print("strategy",track_id)
    # ---------- 策略侧单轨逐日净值 ----------
    strategy_track_df = (
        strategy_period_df[strategy_period_df["track_id"] == track_id]
        .copy()
        .sort_index()
    )

    strategy_track_nav = build_single_track_daily_nav(
        track_period_df=strategy_track_df,
        daily_asset_ret_df=daily_asset_ret_df
    )
    strategy_track_nav.name = f"track_{track_id}"

    strategy_track_nav_dict[track_id] = strategy_track_nav

    print("benchmark",track_id)
    # ---------- 基准侧单轨逐日净值 ----------
    benchmark_track_df = (
        benchmark_period_df[benchmark_period_df["track_id"] == track_id]
        .copy()
        .sort_index()
    )

    benchmark_track_nav = build_single_track_daily_nav(
        track_period_df=benchmark_track_df,
        daily_asset_ret_df=daily_asset_ret_df
    )
    benchmark_track_nav.name = f"track_{track_id}"

    benchmark_track_nav_dict[track_id] = benchmark_track_nav


# =========================================================
# Part B：五轨组合净值曲线
# 逻辑：五条子组合各占 20%，不做跨轨再平衡
# 所以组合 NAV = 五条子组合 NAV 的简单平均
# =========================================================

# ---------- 策略侧五轨组合净值 ----------
strategy_track_nav_df = pd.concat(
    [strategy_track_nav_dict[track_id] for track_id in track_list],
    axis=1,
    join="outer"
).sort_index()

# 只在五条轨道都已经有净值的日期开始算组合
strategy_track_nav_df = strategy_track_nav_df.dropna(how="any")

strategy_combo_nav = strategy_track_nav_df.mean(axis=1)
strategy_combo_nav.name = "strategy_combo_nav"


# ---------- 基准侧五轨组合净值 ----------
benchmark_track_nav_df = pd.concat(
    [benchmark_track_nav_dict[track_id] for track_id in track_list],
    axis=1,
    join="outer"
).sort_index()

benchmark_track_nav_df = benchmark_track_nav_df.dropna(how="any")

benchmark_combo_nav = benchmark_track_nav_df.mean(axis=1)
benchmark_combo_nav.name = "benchmark_combo_nav"


# =========================================================
# 检查
# =========================================================

# 1）每个轨道都应该有一条净值曲线
assert sorted(strategy_track_nav_dict.keys()) == track_list, \
    "strategy_track_nav_dict 的轨道不完整，请检查单轨净值构造逻辑"

assert sorted(benchmark_track_nav_dict.keys()) == track_list, \
    "benchmark_track_nav_dict 的轨道不完整，请检查单轨净值构造逻辑"

# 2）单轨净值索引必须单调递增，且不重复
for track_id in track_list:
    assert strategy_track_nav_dict[track_id].index.is_monotonic_increasing, \
        f"策略 track {track_id} 的净值索引不是升序"

    assert benchmark_track_nav_dict[track_id].index.is_monotonic_increasing, \
        f"基准 track {track_id} 的净值索引不是升序"

    assert strategy_track_nav_dict[track_id].index.duplicated().sum() == 0, \
        f"策略 track {track_id} 的净值索引有重复日期"

    assert benchmark_track_nav_dict[track_id].index.duplicated().sum() == 0, \
        f"基准 track {track_id} 的净值索引有重复日期"

# 3）组合净值序列不能为空
assert len(strategy_combo_nav) > 0, \
    "strategy_combo_nav 为空，请检查五轨日频净值拼接逻辑"

assert len(benchmark_combo_nav) > 0, \
    "benchmark_combo_nav 为空，请检查五轨日频净值拼接逻辑"

# 4）组合净值不能有缺失值
assert strategy_combo_nav.notna().all(), \
    "strategy_combo_nav 存在 NaN，请检查日频净值拼接逻辑"

assert benchmark_combo_nav.notna().all(), \
    "benchmark_combo_nav 存在 NaN，请检查日频净值拼接逻辑"

print("===== Cell 53 验证：真实日频净值曲线已生成 =====")

print(f"策略组合净值长度：{len(strategy_track_nav_df)}")
print(f"基准组合净值长度：{len(benchmark_track_nav_df)}")
print()

print("策略组合净值前 10 行：")
display(strategy_track_nav_df.head(10))

print("基准组合净值前 10 行：")
display(benchmark_track_nav_df.head(10))

# ----- Cell 279 (code) -----
# Cell 54：定义绩效指标函数库

def calc_annualized_return(ret_series, holding_days_series, annual_days=annual_days):
    """
    计算年化收益率。

    计算逻辑：
    1. 先把每期 simple return 复利成总收益
    2. 再按总持有天数换算成年化收益

    参数
    ret_series : Series
        每期收益率（simple return）
    holding_days_series : Series
        每期持有天数
    annual_days : float
        年化换算天数

    返回：float，年化收益率（数字单位，非%）
    """
    valid_mask = ret_series.notna() & holding_days_series.notna()

    ret_series = ret_series[valid_mask]
    holding_days_series = holding_days_series[valid_mask]

    if len(ret_series) == 0:
        return np.nan

    total_return = (1 + ret_series).prod() - 1
    total_holding_days = holding_days_series.sum()

    if total_holding_days <= 0:
        return np.nan

    annualized_return = (1 + total_return) ** (annual_days / total_holding_days) - 1
    return annualized_return


def calc_annualized_vol(ret_series, holding_days_series, annual_days=annual_days):
    """
    计算年化波动率。

    计算逻辑：
    1. 先把每期 simple return 转成 log return
    2. 再除以每期 holding_days，得到“日均 log return”
    3. 对日均 log return 求标准差
    4. 乘 sqrt(annual_days)

    参数：
    ret_series : Series
        每期收益率（simple return）
    holding_days_series : Series
        每期持有天数
    annual_days : float
        年化换算天数

    返回：
    float，年化波动率（数字单位，非%）
    """
    valid_mask = ret_series.notna() & holding_days_series.notna()

    ret_series = ret_series[valid_mask]
    holding_days_series = holding_days_series[valid_mask]

    if len(ret_series) < 2:
        return np.nan

    if (holding_days_series <= 0).any():
        return np.nan

    # simple return -> log return
    log_ret_series = np.log(1 + ret_series)

    # 每期 log return 换成“日均 log return”
    daily_log_ret_series = log_ret_series / holding_days_series

    annualized_vol = daily_log_ret_series.std(ddof=1) * np.sqrt(annual_days)
    return annualized_vol


def calc_max_drawdown(nav_series):
    """
    计算最大回撤。

    参数
    nav_series : Series
        净值序列

    返回
    float
        最大回撤（正数，例如 0.15 表示回撤 15%）
    """
    nav_series = nav_series.dropna()

    if len(nav_series) == 0:
        return np.nan

    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1
    max_drawdown = abs(drawdown.min())

    return max_drawdown


def calc_sharpe_ratio(ret_series, holding_days_series, rf, annual_days=annual_days):
    """
    计算夏普比率。

    计算逻辑：
    （年化收益 - 无风险利率） / 年化波动率
    """
    annualized_return = calc_annualized_return(
        ret_series=ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    annualized_vol = calc_annualized_vol(
        ret_series=ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    if pd.isna(annualized_return) or pd.isna(annualized_vol):
        return np.nan

    if annualized_vol == 0:
        return np.nan

    sharpe_ratio = (annualized_return - rf) / annualized_vol
    return sharpe_ratio


def calc_information_ratio(excess_ret_series, holding_days_series, annual_days=annual_days):
    """
    计算信息比率。

    计算逻辑：
    1. 用逐期超额收益序列算年化超额收益
    2. 用逐期超额收益序列算年化波动
    3. 年化超额收益 / 年化超额波动
    """
    annualized_excess_return = calc_annualized_return(
        ret_series=excess_ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    annualized_excess_vol = calc_annualized_vol(
        ret_series=excess_ret_series,
        holding_days_series=holding_days_series,
        annual_days=annual_days
    )

    if pd.isna(annualized_excess_return) or pd.isna(annualized_excess_vol):
        return np.nan

    if annualized_excess_vol == 0:
        return np.nan

    information_ratio = annualized_excess_return / annualized_excess_vol
    return information_ratio


def calc_monthly_win_rate(nav_series):
    """
    计算月度胜率。

    计算逻辑：
    1. 把 NAV 按月末重采样
    2. 计算月收益
    3. 统计月收益 > 0 的占比
    """
    nav_series = nav_series.dropna()

    if len(nav_series) == 0:
        return np.nan

    month_end_nav = nav_series.resample("ME").last()
    monthly_ret = month_end_nav.pct_change().dropna()

    if len(monthly_ret) == 0:
        return np.nan

    monthly_win_rate = (monthly_ret > 0).mean()
    return monthly_win_rate


def calc_payoff_ratio(weekly_ret_series):
    """
    计算盈亏比率。

    计算逻辑：
    正收益期平均值 / 负收益期平均值绝对值
    """
    weekly_ret_series = weekly_ret_series.dropna()

    if len(weekly_ret_series) == 0:
        return np.nan

    positive_ret = weekly_ret_series[weekly_ret_series > 0]
    negative_ret = weekly_ret_series[weekly_ret_series < 0]

    if len(positive_ret) == 0 or len(negative_ret) == 0:
        return np.nan

    payoff_ratio = positive_ret.mean() / abs(negative_ret.mean())
    return payoff_ratio


def calc_calmar_ratio(annualized_return, max_drawdown):
    """
    计算卡玛比率。

    计算逻辑：
    年化收益 / 最大回撤
    """
    if pd.isna(annualized_return) or pd.isna(max_drawdown):
        return np.nan

    if max_drawdown == 0:
        return np.nan

    calmar_ratio = annualized_return / max_drawdown
    return calmar_ratio


def calc_market_regime_win_rate(track_result_df):
    """
    计算成长 / 价值波段胜率。

    成长波段胜率：
        在 fwd_ret_g_simple > fwd_ret_v_simple 的区间里，
        period_ret_long_net > 0 的占比

    价值波段胜率：
        在 fwd_ret_g_simple < fwd_ret_v_simple 的区间里，
        period_ret_long_net > 0 的占比

    返回
    ----------
    growth_regime_win_rate : float
    value_regime_win_rate : float
    """
    growth_mask = track_result_df["fwd_ret_g_simple"] > track_result_df["fwd_ret_v_simple"]
    value_mask = track_result_df["fwd_ret_g_simple"] < track_result_df["fwd_ret_v_simple"]

    growth_df = track_result_df[growth_mask]
    value_df = track_result_df[value_mask]

    if len(growth_df) == 0:
        growth_regime_win_rate = np.nan
    else:
        growth_regime_win_rate = (growth_df["period_ret_long_net"] > 0).mean()

    if len(value_df) == 0:
        value_regime_win_rate = np.nan
    else:
        value_regime_win_rate = (value_df["period_ret_long_net"] > 0).mean()

    return growth_regime_win_rate, value_regime_win_rate

# ----- Cell 280 (code) -----
# Cell 55 为单个轨道生成“按年份 + 全区间”的统计表

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
            "calmar_ratio",
            "turnover_2way_pct",
            "growth_regime_win_rate",
            "value_regime_win_rate",
        ]
    ]

    return summary_table

# ----- Cell 281 (code) -----
# Cell 56：生成五个轨道的五张表

# 输入变量：
#   strategy_period_df
#   benchmark_period_df
#   strategy_track_nav_dict
#   benchmark_track_nav_dict
#   track_list
#   build_track_summary_table
#
# 输出变量：
#   track_summary_dict

track_summary_dict = {}

for track_id in track_list:
    # 取出该轨道的策略结果表
    strategy_track_result_df = (
        strategy_period_df[strategy_period_df["track_id"] == track_id]
        .copy()
        .sort_index(ascending=True)
    )

    # 取出该轨道的基准结果表
    benchmark_track_result_df = (
        benchmark_period_df[benchmark_period_df["track_id"] == track_id]
        .copy()
        .sort_index(ascending=True)
    )

    # 取出该轨道的净值曲线
    strategy_track_nav = strategy_track_nav_dict[track_id]
    benchmark_track_nav = benchmark_track_nav_dict[track_id]

    # 生成该轨道的年度 + 全区间统计表
    track_summary_df = build_track_summary_table(
        strategy_track_result_df=strategy_track_result_df,
        benchmark_track_result_df=benchmark_track_result_df,
        strategy_track_nav=strategy_track_nav,
        benchmark_track_nav=benchmark_track_nav
    )

    # 存进字典
    track_summary_dict[track_id] = track_summary_df


# ===== 检查 =====
assert len(track_summary_dict) == len(track_list), \
    "track_summary_dict 的表数量不正确，请检查循环生成逻辑"

first_cols = track_summary_dict[track_list[0]].columns.tolist()

for track_id in track_list:
    assert track_id in track_summary_dict, \
        f"track_summary_dict 缺少轨道 {track_id} 的表"

    assert track_summary_dict[track_id].columns.tolist() == first_cols, \
        f"轨道 {track_id} 的列名和其他轨道不一致，请检查汇总函数逻辑"

print("===== Cell 48 验证：track_summary_dict 构造完成 =====")
print(f"共生成 {len(track_summary_dict)} 张表")
print(f"列名：{first_cols}")
print()
print(f"track {track_list[0]} 的表预览：")
display(track_summary_dict[track_list[0]])

# ----- Cell 282 (code) -----
# Cell 57：生成五轨均值表和全区间标准差行

# 输入变量：
#   track_summary_dict
#
# 输出变量：
#   avg_summary_df
#   std_summary_df


# =========================================================
# Part A：五轨均值表
# =========================================================

# 先把五张表按行索引拼起来
all_track_summary_df = pd.concat(
    [track_summary_dict[track_id] for track_id in track_list],
    axis=0,
    keys=track_list,
    names=["track_id", "period"]
)

# 对相同行标签、相同列名逐项取均值
avg_summary_df = all_track_summary_df.groupby(level="period", sort=False).mean()

# 为了确保行顺序和单轨表一致，这里按第一张表的索引顺序重排
avg_summary_df = avg_summary_df.reindex(track_summary_dict[track_list[0]].index)


# =========================================================
# Part B：五轨全区间标准差
# =========================================================

# 只取五张表的“全区间”这一行
full_period_rows = []

for track_id in track_list:
    full_row = track_summary_dict[track_id].loc["全区间"].copy()
    full_period_rows.append(full_row)

full_period_df = pd.DataFrame(full_period_rows, index=track_list)

# 对每一列在五轨上算标准差
std_summary_df = pd.DataFrame(full_period_df.std(axis=0, ddof=1)).T
std_summary_df.index = ["五轨标准差"]


# ===== 检查 =====
assert avg_summary_df.index.tolist() == track_summary_dict[track_list[0]].index.tolist(), \
    "avg_summary_df 的行索引和单轨表不一致，请检查均值表逻辑"

assert len(std_summary_df) == 1, \
    "std_summary_df 只能有一行，请检查标准差行逻辑"

assert std_summary_df.index.tolist() == ["五轨标准差"], \
    "std_summary_df 的行名应固定为“五轨标准差”"

print("===== Cell 49 验证：均值表和标准差行已生成 =====")
print("avg_summary_df：")
display(avg_summary_df)

print()
print("std_summary_df：")
display(std_summary_df)

# ----- Cell 283 (code) -----
# 构造初筛结果表
# 取 avg_summary_df 最后一行（通常是“全区间”）
last_row = avg_summary_df.iloc[-1]

# 六个初筛条件
check_1 = last_row["trade_count"] > 15
check_2 = (last_row["ann_ret_ls_abs"] > 0) or (last_row["ann_ret_long_abs"] > 0)
check_3 = last_row["sharpe_excess"] > 0
check_4 = ( (last_row["growth_regime_win_rate"] > 0.5) and (last_row["value_regime_win_rate"] > 0.5) )
check_5 = last_row["monthly_win_rate"] > 0.55
check_6 = last_row["payoff_ratio"] > 1

# 最终按“六条规则”整理
screen_result_df = pd.DataFrame({
    "condition": [
        "全区间发生调仓次数 > 15",
        "超额年化收益或绝对年化收益 > 0",
        "超额夏普 > 0",
        "成长/价值波段胜率均 > 50%",
        "月度胜率 > 55%",
        "盈亏比 > 1"
    ],
    "is_pass": [
        check_1,
        check_2,
        check_3,
        check_4,
        check_5,
        check_6,
    ]
})

# 统计满足条件的个数
pass_count = int(screen_result_df["is_pass"].sum())

# 用全区间五轨月度胜率均值生成结果前缀，并把前面暂存到 N_xxx 的输出迁移到最终目录
import shutil

result_period_label = avg_summary_df.index[-1]
result_monthly_win_rate = float(
    all_track_summary_df.xs(result_period_label, level="period")["monthly_win_rate"].mean()
)
monthly_win_rate_tag = "MWnan" if pd.isna(result_monthly_win_rate) else f"mw{result_monthly_win_rate * 100:.2f}"
RESULT_PREFIX = f"{pass_count}_{monthly_win_rate_tag}_{FACTOR_COL}"

provisional_output_dir = Path("Result") / f"N_{FACTOR_COL}"
final_output_dir = Path("Result") / RESULT_PREFIX

def _merge_output_dir(src_dir, dst_dir):
    """把临时目录中的已生成结果迁移到最终目录；重复运行时用本轮结果覆盖同名文件。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_dir.iterdir():
        dst_path = dst_dir / src_path.name
        if dst_path.exists():
            if src_path.is_dir() and dst_path.is_dir():
                _merge_output_dir(src_path, dst_path)
                if not any(src_path.iterdir()):
                    src_path.rmdir()
            elif src_path.is_file() and dst_path.is_file():
                src_path.replace(dst_path)
            else:
                if dst_path.is_dir():
                    shutil.rmtree(dst_path)
                else:
                    dst_path.unlink()
                shutil.move(str(src_path), str(dst_path))
        else:
            shutil.move(str(src_path), str(dst_path))
    if src_dir.exists() and not any(src_dir.iterdir()):
        src_dir.rmdir()

if OUTPUT_DIR == provisional_output_dir and OUTPUT_DIR != final_output_dir and OUTPUT_DIR.exists():
    _merge_output_dir(OUTPUT_DIR, final_output_dir)

OUTPUT_DIR = final_output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "strat_suffix" in globals():
    STRAT_OUTPUT_DIR = OUTPUT_DIR / f"strategy_{strat_suffix}"
    STRAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 若前面的 IC 表已经用 N_xxx 命名，迁移后顺手改成同一结果前缀
old_ic_path = OUTPUT_DIR / f"N_{FACTOR_COL}_check.xlsx"
new_ic_path = OUTPUT_DIR / f"{RESULT_PREFIX}_check.xlsx"
if old_ic_path.exists() and old_ic_path != new_ic_path:
    old_ic_path.replace(new_ic_path)

print("===== 初筛检查结果 =====")
print(screen_result_df)
print(f"满足条件个数：{pass_count}")
print(f"全区间五轨月度胜率均值：{result_monthly_win_rate:.2%}")
print(f"最终输出目录：{OUTPUT_DIR}")

# ----- Cell 284 (code) -----
# Cell 58：导出 Excel

# 输入变量：
#   track_summary_dict
#   avg_summary_df
#   std_summary_df
#   screen_result_df
#   pass_count
#   RESULT_PREFIX
#   result_monthly_win_rate
#   STRAT_OUTPUT_DIR
#   FACTOR_COL
#   benchmark_mode
#
# 输出变量：
#   excel_path

excel_path = STRAT_OUTPUT_DIR / f"{RESULT_PREFIX}_{benchmark_mode}_summary.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    
    # sheet1：初筛结果screening

    screen_result_df.to_excel(writer, sheet_name="screening", startrow=0, startcol=0, index=False)
    pass_count_df = pd.DataFrame({
        "condition": ["pass_count", "monthly_win_rate"],
        "is_pass": [pass_count, result_monthly_win_rate]
    })
    pass_count_df.to_excel(writer, sheet_name="screening", startrow=len(screen_result_df) + 2, startcol=0, index=False)

    sheet_name = "summary"
    start_row = 0

    # sheet2：五轨统计summary
    for track_id in track_list:
        table_title = pd.DataFrame([[f"track_{track_id}_summary"]])

        # 写标题
        table_title.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=start_row,
            startcol=0,
            index=False,
            header=False
        )

        # 写表
        track_summary_dict[track_id].to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=start_row + 1,
            startcol=0,
            index=True
        )

        # 更新起始行：标题 1 行 + 表头 1 行 + 数据行数 + 空 1 行
        start_row = start_row + 1 + 1 + len(track_summary_dict[track_id]) + 1

    # ---------- 再写五轨均值表 ----------
    avg_title = pd.DataFrame([["avg_summary"]])
    avg_title.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        startcol=0,
        index=False,
        header=False
    )

    avg_summary_df.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row + 1,
        startcol=0,
        index=True
    )

    start_row = start_row + 1 + 1 + len(avg_summary_df) + 1

    # ---------- 最后写五轨标准差行 ----------
    std_title = pd.DataFrame([["std_summary"]])
    std_title.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row,
        startcol=0,
        index=False,
        header=False
    )

    std_summary_df.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row + 1,
        startcol=0,
        index=True
    )

print(f"===== Cell 58 验证：Excel 已导出 =====")
print(f"保存路径：{excel_path}")

# ----- Cell 285 (code) -----
# Cell 59：画五轨组合净值曲线

# 输入变量：
#   strategy_combo_nav
#   benchmark_combo_nav
#   STRAT_OUTPUT_DIR
#   FACTOR_COL
#   benchmark_mode
#
# 输出变量：
#   combo_nav_path

combo_nav_path = STRAT_OUTPUT_DIR / f"{FACTOR_COL}_{benchmark_mode}_combo_nav.png"

plt.figure(figsize=(12, 6))

plt.plot(
    strategy_combo_nav.index,
    strategy_combo_nav.values,
    label="strategy combo",
    linewidth=2
)

plt.plot(
    benchmark_combo_nav.index,
    benchmark_combo_nav.values,
    label="benchmark combo",
    linewidth=2
)

plt.title(f"Combo NAV - {FACTOR_COL} ({benchmark_mode})")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(combo_nav_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("===== Cell 51 验证：五轨组合净值图已保存 =====")
print(f"保存路径：{combo_nav_path}")

# ----- Cell 286 (code) -----
# Cell 60：画五轨组合 strategy_nav / benchmark_nav 比值曲线

# 输入变量：
#   strategy_combo_nav
#   benchmark_combo_nav
#   STRAT_OUTPUT_DIR
#   FACTOR_COL
#   benchmark_mode
#
# 输出变量：
#   combo_nav_ratio_path
#   combo_nav_ratio

combo_nav_ratio_path = STRAT_OUTPUT_DIR / f"{FACTOR_COL}_{benchmark_mode}_combo_nav_ratio.png"

combo_ratio_df = pd.concat(
    [strategy_combo_nav, benchmark_combo_nav],
    axis=1,
    join="inner"
).sort_index(ascending=True)

combo_ratio_df.columns = ["strategy_combo_nav", "benchmark_combo_nav"]

assert len(combo_ratio_df) > 0, "五轨组合净值对齐后为空，请检查 strategy_combo_nav / benchmark_combo_nav"
assert (combo_ratio_df["benchmark_combo_nav"] != 0).all(), "benchmark_combo_nav 存在 0，无法计算比值"

combo_nav_ratio = (
    combo_ratio_df["strategy_combo_nav"] / combo_ratio_df["benchmark_combo_nav"]
).copy()

combo_nav_ratio.name = "combo_nav_ratio"

plt.figure(figsize=(12, 6))

plt.plot(
    combo_nav_ratio.index,
    combo_nav_ratio.values,
    label="strategy combo / benchmark combo",
    linewidth=2
)

plt.axhline(1.0, linestyle="--", linewidth=1, label="ratio = 1")

plt.title(f"Combo NAV Ratio - {FACTOR_COL} ({benchmark_mode})")
plt.xlabel("Date")
plt.ylabel("NAV Ratio")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(combo_nav_ratio_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----- Cell 287 (code) -----
# Cell 61：画五个轨道各自的净值曲线

# 输入变量：
#   strategy_track_nav_dict
#   benchmark_track_nav_dict
#   track_list
#   STRAT_OUTPUT_DIR
#
# 输出变量：
#   五张图片文件

for track_id in track_list:
    strategy_track_nav = strategy_track_nav_dict[track_id]
    benchmark_track_nav = benchmark_track_nav_dict[track_id]

    track_nav_path = STRAT_OUTPUT_DIR / f"track_{track_id}_nav.png"

    plt.figure(figsize=(12, 6))

    plt.plot(
        strategy_track_nav.index,
        strategy_track_nav.values,
        label=f"strategy track {track_id}",
        linewidth=2
    )

    plt.plot(
        benchmark_track_nav.index,
        benchmark_track_nav.values,
        label=f"benchmark track {track_id}",
        linewidth=2
    )

    plt.title(f"Track {track_id} NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(track_nav_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"track {track_id} 图片已保存到：{track_nav_path}")

# ----- Cell 288 (code) -----
# Cell 62：画五个轨道各自的 strategy_nav / benchmark_nav 比值曲线

# 输入变量：
#   strategy_track_nav_dict
#   benchmark_track_nav_dict
#   track_list
#   STRAT_OUTPUT_DIR
#
# 输出变量：
#   五张图片文件

for track_id in track_list:
    strategy_track_nav = strategy_track_nav_dict[track_id]
    benchmark_track_nav = benchmark_track_nav_dict[track_id]

    track_ratio_df = pd.concat(
        [strategy_track_nav, benchmark_track_nav],
        axis=1,
        join="inner"
    ).sort_index(ascending=True)

    track_ratio_df.columns = ["strategy_nav", "benchmark_nav"]

    assert len(track_ratio_df) > 0, f"track {track_id} 对齐后为空，请检查净值序列"
    assert (track_ratio_df["benchmark_nav"] != 0).all(), \
        f"track {track_id} 的 benchmark_nav 存在 0，无法计算比值"

    track_nav_ratio = (
        track_ratio_df["strategy_nav"] / track_ratio_df["benchmark_nav"]
    ).copy()

    track_nav_ratio.name = f"track_{track_id}_nav_ratio"

    track_nav_ratio_path = STRAT_OUTPUT_DIR / f"track_{track_id}_nav_ratio.png"

    plt.figure(figsize=(12, 6))

    plt.plot(
        track_nav_ratio.index,
        track_nav_ratio.values,
        label=f"track {track_id}: strategy / benchmark",
        linewidth=2
    )

    plt.axhline(1.0, linestyle="--", linewidth=1, label="ratio = 1")

    plt.title(f"Track {track_id} NAV Ratio")
    plt.xlabel("Date")
    plt.ylabel("NAV Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(track_nav_ratio_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"track {track_id} 净值比值图已保存到：{track_nav_ratio_path}")
