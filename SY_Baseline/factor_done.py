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
_assign_raw_factor("ZHAO01_raw", ZHAO01_raw)


# ----- Cell 11 (markdown) -----
# ### index_pe (ZHAO02)

# ----- Cell 12 (code) -----
sub_1 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx', '市盈率TTM').dropna()
sub_2 = _read_indicator_series('D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx', '市盈率TTM').dropna()
valuation_ratio = sub_1 / sub_2
quantile_rank = _rolling_quantile_rank_year(valuation_ratio,5)
ZHAO02_raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0).dropna()
ZHAO02_raw = _month_aggregate(ZHAO02_raw, how="last")
_assign_raw_factor("ZHAO02_raw", ZHAO02_raw)

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
data_df["ZHAO06_raw"] = df_pmi_chain["yoy_chain"].reindex(data_df.index)
print(
    "ZHAO06_raw generated:",
    "non_na=", int(data_df["ZHAO06_raw"].notna().sum()),
    "first=", data_df["ZHAO06_raw"].first_valid_index(),
    "last=", data_df["ZHAO06_raw"].last_valid_index(),
)

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


# ----- Cell 67 (markdown) -----
# ### CPI (ZHAO21)

# ----- Cell 68 (code) -----
sub_1 = _load_china_macro_series('CPI:同比')
ZHAO21_raw = sub_1
_assign_raw_factor("ZHAO21_raw", ZHAO21_raw)


# ----- Cell 69 (markdown) -----
# ### CPI同比-PPI同比 (ZHAO22)

# ----- Cell 70 (code) -----
# Follow the ZHAO.xlsx docu order: sub_1 = CPI YoY, sub_2 = PPI YoY.
sub_1 = _load_china_macro_series('CPI:同比')
sub_2 = _load_china_macro_series('PPI:同比')
ZHAO22_raw = sub_1 - sub_2
_assign_raw_factor("ZHAO22_raw", ZHAO22_raw)


# ----- Cell 71 (markdown) -----
# ### AA3年中票-AAA3年中票 (ZHAO23)

# ----- Cell 72 (code) -----
df_STB_AAA = pd.read_excel(Config.DATA_DIR / "流动性" / "D_AAA级中短期票据到期收益率_CN_110406_260409.xlsx")
df_STB_AAA = df_STB_AAA.iloc[2:,:].copy()
df_STB_AAA.columns = ["date", "STB_AAA_1m", "STB_AAA_2m", "STB_AAA_3m", "STB_AAA_6m", "STB_AAA_9m",
                             "STB_AAA_1y", "STB_AAA_2y", "STB_AAA_3y", "STB_AAA_4y","STB_AAA_5y", 
                            "STB_AAA_6y", "STB_AAA_7y", "STB_AAA_8y", "STB_AAA_9y", "STB_AAA_10y"]
df_STB_AAA.iloc[:, 1:] = df_STB_AAA.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_STB_AAA.iloc[:, 1:] = df_STB_AAA.iloc[:, 1:] / 100
df_STB_AAA['date'] = pd.to_datetime(df_STB_AAA['date'], errors='coerce')
df_STB_AAA.set_index('date', inplace=True)
df_STB_AAA = df_STB_AAA.sort_index(ascending=True)
df_STB_AAA.head(5)

# ----- Cell 73 (code) -----
df_STB_AA = pd.read_excel(Config.DATA_DIR / "流动性" / "D_AA级中短期票据到期收益率_CN_060301_260409.xlsx")
df_STB_AA = df_STB_AA.iloc[2:,:].copy()
df_STB_AA.columns = ["date", "STB_AA_1m", "STB_AA_2m", "STB_AA_3m", "STB_AA_6m", 
                            "STB_AA_9m", "STB_AA_1y", "STB_AA_2y", "STB_AA_3y", "STB_AA_5y", 
                            "STB_AA_6y", "STB_AA_7y", "STB_AA_8y", "STB_AA_9y", "STB_AA_10y"]
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:] / 100
df_STB_AA['date'] = pd.to_datetime(df_STB_AA['date'], errors='coerce')
df_STB_AA.set_index('date', inplace=True)
df_STB_AA = df_STB_AA.sort_index(ascending=True)
df_STB_AA.head(5)

# ----- Cell 74 (code) -----
df_credit_spread = df_STB_AAA.join(df_STB_AA, how='inner')

sub_1 = df_credit_spread["STB_AA_3y"]
sub_2 = df_credit_spread["STB_AAA_3y"]

# ----- Cell 75 (code) -----
monthly_spread = _month_aggregate(sub_1 - sub_2, how="average")
ZHAO23_raw = monthly_spread - monthly_spread.shift(1)
_assign_raw_factor("ZHAO23_raw", ZHAO23_raw)

# ----- Cell 76 (markdown) -----
# ### 10年-1年国债到期收益率 (ZHAO24)

# ----- Cell 77 (code) -----
df_TB = pd.read_excel(Config.DATA_DIR / "流动性" / "D_国债到期收益率_CN_020104_260409.xlsx")
df_TB = df_TB.iloc[2:,:].copy()
df_TB.columns = ["date", "CGB_1y", "CGB_3y", "CGB_5y", "CGB_10y", 
                 "CDB_1m", "CDB_2m", "CDB_3m", "CDB_6m", "CDB_9m", 
                "CDB_1y", "CDB_3y", "CDB_5y", "CDB_6y", "CDB_7y",
                "CDB_8y", "CDB_9y", "CDB_10y"]
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:] / 100
df_TB['date'] = pd.to_datetime(df_TB['date'], errors='coerce')
df_TB.set_index('date', inplace=True)
df_TB = df_TB.sort_index(ascending=True)
df_TB.head(5)

# ----- Cell 78 (code) -----
sub_1 = df_TB["CGB_10y"]
sub_2 = df_TB["CGB_1y"]

# ----- Cell 79 (code) -----
monthly_spread = _month_aggregate(sub_1 - sub_2, how="average")
ZHAO24_raw = monthly_spread - monthly_spread.shift(1)
_assign_raw_factor("ZHAO24_raw", ZHAO24_raw)


# ----- Cell 80 (markdown) -----
# ## L todo factor generation

# ----- Cell 81 (code) -----
# L-series macro raw factors
fixed_asset_yoy = _load_china_macro_series("固定资产投资")
cpi_yoy = _load_china_macro_series("CPI:同比", value_col="今值")
cpi_forecast = _load_china_macro_series("CPI:同比", value_col="预测值")
ppi_yoy = _load_china_macro_series("PPI:同比", value_col="今值")
ppi_forecast = _load_china_macro_series("PPI:同比", value_col="预测值")
industrial_yoy = _load_china_macro_series("工业增加值:当月同比", value_col="今值")
ppi_cpi_spread = ppi_yoy - cpi_yoy

L02_raw = data_diff(fixed_asset_yoy)
_assign_raw_factor("L02_raw", L02_raw)

L03_raw = fixed_asset_yoy.copy()
_assign_raw_factor("L03_raw", L03_raw)

L04_raw = data_deviation(fixed_asset_yoy, months=3, min_periods=2)
_assign_raw_factor("L04_raw", L04_raw)

L05_1_raw = expectation(
    cpi_yoy,
    cpi_forecast,
    up_floor=0.025,
    down_ceiling=-0.025,
    z_years=3,
    z_min_periods=18,
)
_assign_raw_factor("L05_1_raw", L05_1_raw)

L05_2_raw = expectation(
    cpi_yoy,
    cpi_forecast,
    upper_quantile=0.80,
    lower_quantile=0.20,
    quantile_years=3,
    quantile_min_periods=18,
    z_years=3,
    z_min_periods=18,
)
_assign_raw_factor("L05_2_raw", L05_2_raw)

L06_1_raw = cpi_yoy.copy()
_assign_raw_factor("L06_1_raw", L06_1_raw)

L06_2_raw = data_diff(cpi_yoy)
_assign_raw_factor("L06_2_raw", L06_2_raw)

L06_3_raw = _YoY(cpi_yoy)
_assign_raw_factor("L06_3_raw", L06_3_raw)

L07_raw = z_data_deviation(
    cpi_yoy,
    dev_months=6,
    z_years=3,
    dev_min_periods=3,
    z_min_periods=18,
)
_assign_raw_factor("L07_raw", L07_raw)

L08_raw = z_data_deviation(
    ppi_cpi_spread,
    dev_months=6,
    z_years=3,
    dev_min_periods=3,
    z_min_periods=18,
)
_assign_raw_factor("L08_raw", L08_raw)

spread_3m_mean = _time_window_apply(ppi_cpi_spread, lambda window: window.mean(), months=3, min_periods=2)
spread_12m_mean = _time_window_apply(ppi_cpi_spread, lambda window: window.mean(), years=1, min_periods=6)
L09_raw = calc_rolling_zscore_time(spread_3m_mean - spread_12m_mean, years=3, min_periods=18)
_assign_raw_factor("L09_raw", L09_raw)

L10_raw = ppi_forecast - cpi_forecast
_assign_raw_factor("L10_raw", L10_raw)

L11_raw = ppi_cpi_spread.copy()
_assign_raw_factor("L11_raw", L11_raw)

recent_spread_avg = (ppi_cpi_spread + ppi_cpi_spread.shift(1)) / 2
past_spread_avg = (ppi_cpi_spread.shift(2) + ppi_cpi_spread.shift(3) + ppi_cpi_spread.shift(4)) / 3
L12_raw = recent_spread_avg - past_spread_avg
_assign_raw_factor("L12_raw", L12_raw)

L13_raw = calc_rolling_zscore_time(ppi_cpi_spread + ppi_cpi_spread.shift(1), years=4, min_periods=24)
_assign_raw_factor("L13_raw", L13_raw)

L14_raw = data_diff(ppi_cpi_spread)
_assign_raw_factor("L14_raw", L14_raw)

L18_raw = ppi_yoy.copy()
_assign_raw_factor("L18_raw", L18_raw)

L19_raw = industrial_yoy.copy()
_assign_raw_factor("L19_raw", L19_raw)

L20_raw = data_diff(llt(industrial_yoy, 5))
_assign_raw_factor("L20_raw", L20_raw)

L21_raw = data_diff(llt(ppi_yoy, 5))
_assign_raw_factor("L21_raw", L21_raw)

L22_raw = _rolling_sum_ratio_minus_one(ppi_yoy, window=5, shift=1)
_assign_raw_factor("L22_raw", L22_raw)


# ----- Cell 82 (code) -----
# L47-L97 shared loaders
def _load_macro_series_by_country(
    country,
    keyword,
    value_col="今值",
    required_contains=None,
    exclude_contains=None,
    percent_as_ratio=True,
):
    macro = _load_macro_all()
    date_col = "日期" if "日期" in macro.columns else macro.columns[2]
    nation_col = "国家/地区" if "国家/地区" in macro.columns else macro.columns[4]
    indicator_col = "指标名称" if "指标名称" in macro.columns else macro.columns[5]

    mask = macro[nation_col].eq(country)
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
        raise ValueError(f"No macro rows matched country={country!r}, keyword={keyword!r}")
    if value_col not in out.columns:
        raise KeyError(f"{value_col!r} not found in Macro_all columns: {list(out.columns)}")

    percent_text = (
        out[indicator_col].astype(str).str.contains("%", regex=False, na=False).any()
        or out[value_col].astype(str).str.contains("%", regex=False, na=False).any()
    )
    percent_hint = bool(percent_as_ratio and percent_text)

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    sort_cols = [col for col in [date_col, "来源文件", "来源sheet", "文件年月"] if col in out.columns]
    out = out.sort_values(sort_cols, na_position="first")

    values = _as_numeric(out[value_col], percent_hint=percent_hint).to_numpy()
    series = pd.Series(values, index=out[date_col], name=keyword, dtype="float64").sort_index()
    if series.index.duplicated(keep=False).any():
        dup_count = int(series.index.duplicated(keep=False).sum())
        print(
            f"Macro helper matched {dup_count} duplicate-date rows for "
            f"country={country!r}, keyword={keyword!r}; keeping the last row per date."
        )
        series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def _tail_quantile_signal(rank_series, upper=0.75, lower=0.25):
    out = pd.Series(0.0, index=rank_series.index, dtype="float64")
    upper_mask = rank_series > upper
    lower_mask = rank_series < lower
    out.loc[upper_mask] = rank_series.loc[upper_mask] - upper
    out.loc[lower_mask] = lower - rank_series.loc[lower_mask]
    out.loc[rank_series.isna()] = np.nan
    return out


china_pmi_actual = _load_macro_series_by_country(
    "中国",
    "官方制造业PMI",
    value_col="今值",
    percent_as_ratio=False,
)
china_pmi_forecast = _load_macro_series_by_country(
    "中国",
    "官方制造业PMI",
    value_col="预测值",
    percent_as_ratio=False,
)
us_pmi_actual = _load_macro_series_by_country(
    "美国",
    "制造业PMI",
    value_col="今值",
    exclude_contains="服务业",
    percent_as_ratio=False,
)
us_cpi_yoy = _load_macro_series_by_country(
    "美国",
    "CPI",
    value_col="今值",
    required_contains="同比",
    exclude_contains="核心",
)
us_unemployment = _load_macro_series_by_country(
    "美国",
    "失业率",
    value_col="今值",
    required_contains="季调",
)
us_nonfarm = _load_macro_series_by_country(
    "美国",
    "非农就业人口变动",
    value_col="今值",
    required_contains="季调",
    exclude_contains="非农企业",
)
china_export_actual = _load_macro_series_by_country(
    "中国",
    "出口金额:当月同比",
    value_col="今值",
    exclude_contains="人民币",
)
china_export_forecast = _load_macro_series_by_country(
    "中国",
    "出口金额:当月同比",
    value_col="预测值",
    exclude_contains="人民币",
)

shibor_1m = _read_indicator_series(
    "D_21-24_shibor+7天同业利率_CN_020131_260327.xlsx",
    "Shibor:1月",
)
shibor_3m = _read_indicator_series(
    "D_21-24_shibor+7天同业利率_CN_020131_260327.xlsx",
    "Shibor:3月",
)
us_credit_spread_baml = _read_indicator_series(
    "BAMLH0A0HYM2.xlsx",
    "BAMLH0A0HYM2",
    sheet_name="Daily, Close",
)
moody_aaa = _read_indicator_series(
    "D_MoodyCorpBond_US_530430_260324.xlsx",
    "美国:企业债收益率:穆迪Aaa",
)
moody_baa = _read_indicator_series(
    "D_MoodyCorpBond_US_530430_260324.xlsx",
    "美国:企业债收益率:穆迪Baa",
)
usd_index = _read_indicator_series(
    "日频汇率.xlsx",
    "美元指数",
)
us_t10 = _read_indicator_series(
    "D_国债收益率_US_530430_260324.xlsx",
    "美国:国债收益率:10年",
)
us_t6m = _read_indicator_series(
    "D_国债收益率_US_530430_260324.xlsx",
    "美国:国债收益率:6个月",
)
us_t3m = _read_indicator_series(
    "D_国债收益率_US_530430_260324.xlsx",
    "美国:国债收益率:3个月",
)
us_t1m = _read_indicator_series(
    "D_国债收益率_US_530430_260324.xlsx",
    "美国:国债收益率:1个月",
)
us_t2y = _read_indicator_series(
    "D_国债收益率_US_530430_260324.xlsx",
    "美国:国债收益率:2年",
)


# ----- Cell 83 (code) -----
m2_actual = _load_china_macro_series("M2:同比")
m2_forecast = _load_china_macro_series("M2:同比", value_col="预测值")
sf_actual = _load_china_macro_series("社会融资规模存量:同比")
sf_forecast = _load_china_macro_series("社会融资规模存量:同比", value_col="预测值")
gdp_actual = _load_china_macro_series("季度GDP:当季同比(%)")
m1_actual = _load_china_macro_series("M1:同比")
m1_forecast = _load_china_macro_series("M1:同比", value_col="预测值")
m0_actual = _load_china_macro_series("M0:同比")

m2_sf_actual = pd.concat(
    [m2_actual.rename("sub_1"), sf_actual.rename("sub_2")],
    axis=1,
    sort=True,
).sort_index()
m2_sf_forecast = pd.concat(
    [m2_forecast.rename("sub_1"), sf_forecast.rename("sub_2")],
    axis=1,
    sort=True,
).sort_index()
m2_forecast_pair = pd.concat(
    [m2_actual.rename("actual"), m2_forecast.rename("forecast")],
    axis=1,
    sort=True,
).sort_index()
m2_gdp_latest = _latest_macro_pair(m2_actual, gdp_actual)

m1_m2_actual = pd.concat(
    [m2_actual.rename("sub_1"), m1_actual.rename("sub_2")],
    axis=1,
    sort=True,
).sort_index()
m1_forecast_pair = pd.concat(
    [m1_actual.rename("actual"), m1_forecast.rename("forecast")],
    axis=1,
    sort=True,
).sort_index()
m1_m0_actual = pd.concat(
    [m0_actual.rename("sub_1"), m1_actual.rename("sub_2")],
    axis=1,
    sort=True,
).sort_index()

m1_minus_m2 = m1_m2_actual["sub_2"] - m1_m2_actual["sub_1"]
m1_minus_m0 = m1_m0_actual["sub_2"] - m1_m0_actual["sub_1"]


# ----- Cell 84 (markdown) -----
# ### M2-社融同比（趋势周期=1月，影响方向=负） (L23)

# ----- Cell 85 (code) -----
m2_minus_sf = m2_sf_actual["sub_1"] - m2_sf_actual["sub_2"]
L23_raw = data_diff(m2_minus_sf)
_assign_raw_factor("L23_raw", L23_raw)


# ----- Cell 86 (code) -----
L23_raw.dropna().head()

# ----- Cell 87 (markdown) -----
# ### 社融同比/M2同比（社融/M2（一致预期）） (L24_1)

# ----- Cell 88 (code) -----
L24_1_raw = m2_sf_actual["sub_2"] / m2_sf_actual["sub_1"]
_assign_raw_factor("L24_1_raw", L24_1_raw)


# ----- Cell 89 (code) -----
L24_1_raw.dropna().head()

# ----- Cell 90 (markdown) -----
# ### 社融同比/M2同比（社融/M2（一致预期）） (L24_2)

# ----- Cell 91 (code) -----
sf_over_m2_forecast = m2_sf_forecast["sub_2"] / m2_sf_forecast["sub_1"]
L24_2_raw = data_yoy(sf_over_m2_forecast)
_assign_raw_factor("L24_2_raw", L24_2_raw)


# ----- Cell 92 (code) -----
L24_2_raw.dropna().head()

# ----- Cell 93 (markdown) -----
# ### M2同比-名义GDP同比 (L25)

# ----- Cell 94 (code) -----
L25_raw = m2_gdp_latest["sub_1"] - m2_gdp_latest["sub_2"]
_assign_raw_factor("L25_raw", L25_raw)

# ----- Cell 95 (code) -----
L25_raw.dropna().head()

# ----- Cell 96 (markdown) -----
# ### M2 当月同比 (L26)

# ----- Cell 97 (code) -----
L26_raw = m2_actual.copy()
_assign_raw_factor("L26_raw", L26_raw)

# ----- Cell 98 (markdown) -----
# ### M2 一致预期（M2同比超于预期或低于预期） (L27)

# ----- Cell 99 (code) -----
L27_raw = expectation(
    m2_actual,
    m2_forecast,
    upper_quantile=0.80,
    lower_quantile=0.20,
    quantile_years=3,
    quantile_min_periods=18,
    z_years=3,
    z_min_periods=18,
)
_assign_raw_factor("L27_raw", L27_raw)


# ----- Cell 100 (markdown) -----
# ### M2同比增速 (L28)

# ----- Cell 101 (code) -----
L28_raw = data_yoy(m2_actual)
_assign_raw_factor("L28_raw", L28_raw)


# ----- Cell 102 (markdown) -----
# ### M2，短期均线为原始值，长期均线为12月 (L29)

# ----- Cell 103 (code) -----
L29_raw = z_MA_div(
    m2_actual,
    short_window=1,
    long_window=Config.ANNUAL_DAYS,
    z_window=3 * Config.ANNUAL_DAYS,
    unit="days",
)
_assign_raw_factor("L29_raw", L29_raw)


# ----- Cell 104 (markdown) -----
# ### 宏观事件：M2同比超预期/低预期 (L30)

# ----- Cell 105 (code) -----
m2_surprise_abs = (m2_forecast_pair["actual"] - m2_forecast_pair["forecast"]).abs()
L30_raw = calc_rolling_zscore_time(m2_surprise_abs, years=3, min_periods=18)
_assign_raw_factor("L30_raw", L30_raw)


# ----- Cell 106 (markdown) -----
# ### M2同比平滑值环比上升 (L31)

# ----- Cell 107 (code) -----
L31_raw = data_diff(llt(m2_actual, 5))
_assign_raw_factor("L31_raw", L31_raw)


# ----- Cell 108 (markdown) -----
# ### M1-M2剪刀差 (L32)

# ----- Cell 109 (code) -----
L32_raw = m1_minus_m2.copy()
_assign_raw_factor("L32_raw", L32_raw)


# ----- Cell 110 (markdown) -----
# ### M2-M1 当月同比 (L33)

# ----- Cell 111 (code) -----
L33_raw = calc_rolling_zscore_time(m1_minus_m2, years=3, min_periods=18)
_assign_raw_factor("L33_raw", L33_raw)


# ----- Cell 112 (markdown) -----
# ### M2同比-M1同比 (L35_1)

# ----- Cell 113 (code) -----
L35_1_raw = data_yoy(m1_minus_m2)
_assign_raw_factor("L35_1_raw", L35_1_raw)


# ----- Cell 114 (markdown) -----
# ### M2同比-M1同比 (L35_2)

# ----- Cell 115 (code) -----
L35_2_raw = data_diff(m1_minus_m2)
_assign_raw_factor("L35_2_raw", L35_2_raw)

# ----- Cell 116 (markdown) -----
# ### M2同比-M1同比 (L34)

# ----- Cell 117 (code) -----
L34_raw = z_data_deviation(
    m1_minus_m2,
    dev_months=3,
    z_months=12,
    dev_min_periods=2,
    z_min_periods=6,
)
_assign_raw_factor("L34_raw", L34_raw)


# ----- Cell 118 (markdown) -----
# ### M1-M2剪刀差，短期均线为原始值，长期均线为12月 (L36)

# ----- Cell 119 (code) -----
L36_raw = z_MA_div(
    m1_minus_m2,
    short_window=1,
    long_window=12,
    z_window=24,
    unit="months",
)
_assign_raw_factor("L36_raw", L36_raw)


# ----- Cell 120 (markdown) -----
# ### M1-M2短期均线穿长期均线 (L37)

# ----- Cell 121 (code) -----
L37_raw = z_MA_div(
    m1_minus_m2,
    short_window=3,
    long_window=6,
    z_window=12,
    unit="months",
)
_assign_raw_factor("L37_raw", L37_raw)


# ----- Cell 122 (markdown) -----
# ### M2-M1 环比 (L38)

# ----- Cell 123 (code) -----
L38_raw = z_data_deviation(
    m1_minus_m2,
    dev_months=3,
    z_months=12,
    dev_min_periods=2,
    z_min_periods=6,
)
_assign_raw_factor("L38_raw", L38_raw)


# ----- Cell 124 (markdown) -----
# ### M1 当月同比（历史胜率55.16%） (L39)

# ----- Cell 125 (code) -----
L39_raw = m1_actual.copy()
_assign_raw_factor("L39_raw", L39_raw)


# ----- Cell 126 (markdown) -----
# ### M1同比平滑值环比下降 (L40)

# ----- Cell 127 (code) -----
L40_raw = data_yoy(llt(m1_actual, 5))
_assign_raw_factor("L40_raw", L40_raw)


# ----- Cell 128 (markdown) -----
# ### M1同比（趋势周期=6月） (L41)

# ----- Cell 129 (code) -----
L41_raw = llt(m1_actual, 6)
_assign_raw_factor("L41_raw", L41_raw)


# ----- Cell 130 (markdown) -----
# ### 宏观事件：M1同比超预期环比上升 (L42)

# ----- Cell 131 (code) -----
L42_raw = expectation(
    m1_actual,
    m1_forecast,
    upper_quantile=0.60,
    lower_quantile=0.40,
    quantile_years=3,
    quantile_min_periods=18,
    z_years=3,
    z_min_periods=18,
)
if L42_raw.notna().sum() == 0:
    print("L42_raw warning: current condition produced all-NaN values with available M1 forecast history.")
_assign_raw_factor("L42_raw", L42_raw)


# ----- Cell 132 (markdown) -----
# ### M1同比短期均线穿长期均线 (L43)

# ----- Cell 133 (code) -----
L43_raw = z_MA_div(
    m1_actual,
    short_window=3,
    long_window=6,
    z_window=12,
    unit="months",
)
_assign_raw_factor("L43_raw", L43_raw)


# ----- Cell 134 (markdown) -----
# ### M0同比（历史胜率55.95%） (L45)

# ----- Cell 135 (code) -----
L45_raw = m0_actual.copy()
_assign_raw_factor("L45_raw", L45_raw)


# ----- Cell 136 (markdown) -----
# ### M1-M0 当月同比 (L46)

# ----- Cell 137 (code) -----
L46_raw = m1_minus_m0.copy()
_assign_raw_factor("L46_raw", L46_raw)


# ----- Cell 138 (markdown) -----
# ### PMI（制造业PMI） (L47)

# ----- Cell 139 (code) -----
L47_raw = china_pmi_actual - 50
_assign_raw_factor("L47_raw", L47_raw)


# ----- Cell 140 (markdown) -----
# ### PMI（制造业PMI） (L47_1)

# ----- Cell 141 (code) -----
L47_1_signal = pd.Series(0.0, index=china_pmi_actual.index, dtype="float64")
L47_1_zscore = calc_rolling_zscore(50 - china_pmi_actual, window=12, min_periods=6)
L47_1_signal.loc[data_diff(china_pmi_actual) > 0] = L47_1_zscore.loc[data_diff(china_pmi_actual) > 0]
L47_1_raw = L47_1_signal
_assign_raw_factor("L47_1_raw", L47_1_raw)


# ----- Cell 142 (markdown) -----
# ### PMI (12MA) (L49)

# ----- Cell 143 (code) -----
L49_raw = llt(china_pmi_actual - 50, 12)
_assign_raw_factor("L49_raw", L49_raw)


# ----- Cell 144 (markdown) -----
# ### PMI类指标季度均值和上季度均值的差分 (L52)

# ----- Cell 145 (code) -----
L52_raw = _rolling_sum_ratio_minus_one(china_pmi_actual - 50, window=3, shift=3)
_assign_raw_factor("L52_raw", L52_raw)


# ----- Cell 146 (markdown) -----
# ### PMI-50作为环比增速推算同比增速 (L54)

# ----- Cell 147 (code) -----
L54_raw = calc_rolling_zscore(china_pmi_actual - 50, window=12, min_periods=6)
_assign_raw_factor("L54_raw", L54_raw)


# ----- Cell 148 (markdown) -----
# ### PMI 结合预期（PMI数据处于相对高位（51以上），且超预期） (L65)

# ----- Cell 149 (code) -----
L65_aligned = pd.concat(
    [china_pmi_actual.rename("actual"), china_pmi_forecast.rename("forecast")],
    axis=1,
).dropna().sort_index()
L65_signal = pd.Series(np.nan, index=L65_aligned.index, dtype="float64")
L65_positive = (L65_aligned["actual"] > L65_aligned["forecast"]) & (L65_aligned["actual"] > 51)
L65_negative = (L65_aligned["actual"] < L65_aligned["forecast"]) & (L65_aligned["actual"] < 49)
L65_signal.loc[L65_positive | L65_negative] = (
    L65_aligned["actual"] - L65_aligned["forecast"]
).loc[L65_positive | L65_negative]
L65_raw = calc_rolling_zscore(L65_signal, window=12, min_periods=6)
_assign_raw_factor("L65_raw", L65_raw)


# ----- Cell 150 (markdown) -----
# ### PMI近三个月均值-过去三个月均值 (L66)

# ----- Cell 151 (code) -----
L66_raw = _rolling_sum_ratio_minus_one(china_pmi_actual, window=3, shift=1)
_assign_raw_factor("L66_raw", L66_raw)


# ----- Cell 152 (markdown) -----
# ### PMI同比短期均线穿长期均线 (L67)

# ----- Cell 153 (code) -----
L67_raw = z_MA_div(
    china_pmi_actual,
    short_window=3,
    long_window=6,
    z_window=12,
    unit="months",
)
_assign_raw_factor("L67_raw", L67_raw)


# ----- Cell 154 (markdown) -----
# ### PMI (3MA) (L71)

# ----- Cell 155 (code) -----
L71_raw = llt(china_pmi_actual - 50, 3)
_assign_raw_factor("L71_raw", L71_raw)


# ----- Cell 156 (markdown) -----
# ### PMI 美国 (L73)

# ----- Cell 157 (code) -----
L73_raw = us_pmi_actual - 50
_assign_raw_factor("L73_raw", L73_raw)


# ----- Cell 158 (markdown) -----
# ### SHIBOR 3M，短期为最新值，长期为近3月均线 (L74)

# ----- Cell 159 (code) -----
L74_raw = z_MA_div(
    shibor_3m,
    short_window=1,
    long_window=3 * Config.MONTH_DAYS,
    z_window=Config.ANNUAL_TRADING_DAYS,
    unit="days",
)
_assign_raw_factor("L74_raw", L74_raw)


# ----- Cell 160 (markdown) -----
# ### 3个月SHIBOR利率 (L75)

# ----- Cell 161 (code) -----
L75_raw = z_MA_div(
    shibor_3m,
    short_window=5,
    long_window=250,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L75_raw", L75_raw)


# ----- Cell 162 (markdown) -----
# ### SHIBOR利率 (L76)

# ----- Cell 163 (code) -----
L76_raw = z_MA_div(
    shibor_1m,
    short_window=5,
    long_window=250,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L76_raw", L76_raw)


# ----- Cell 164 (markdown) -----
# ### SHIBOR利率近五日均值在过去三年中的百分位 (L77)

# ----- Cell 165 (code) -----
L77_ma5 = shibor_1m.rolling(window=5, min_periods=5).mean()
L77_rank = _rolling_quantile_rank_year(L77_ma5, 3)
L77_raw = _tail_quantile_signal(L77_rank, upper=0.75, lower=0.25)
_assign_raw_factor("L77_raw", L77_raw)


# ----- Cell 166 (markdown) -----
# ### L78

# ----- Cell 167 (code) -----
df_bank_stborrow = pd.read_excel(Config.DATA_DIR / "流动性" / "D_25_7天同业利率_CN_020131_260327.xlsx")
df_bank_stborrow = df_bank_stborrow.iloc[4:,:].copy()
df_bank_stborrow.columns = ["date","df_bank7","df_bank14", "df_bank21"]
df_bank_stborrow.iloc[:, 1:] = df_bank_stborrow.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_bank_stborrow.iloc[:, 1:] = df_bank_stborrow.iloc[:, 1:] / 100
df_bank_stborrow['date'] = pd.to_datetime(df_bank_stborrow['date'], errors = 'coerce')
df_bank_stborrow.set_index('date', inplace=True)
df_bank_stborrow = df_bank_stborrow.sort_index(ascending=True)
df_bank_stborrow.head()

# ----- Cell 168 (code) -----
# 数据中存在零值，但不该是零值，所以把零改为前后两个值的平均数
zero_mask = df_bank_stborrow.eq(0)
df_bank_stborrow = df_bank_stborrow.mask(zero_mask, (df_bank_stborrow.shift(1) + df_bank_stborrow.shift(-1)) / 2)
df_bank_stborrow.head()

# ----- Cell 169 (code) -----
df_bank_stborrow["df_bank7_MA5"] = df_bank_stborrow["df_bank7"].rolling(window=5).mean()
df_bank_stborrow["df_bank7_MA250"] = df_bank_stborrow["df_bank7"].rolling(window= Config.ANNUAL_TRADING_DAYS).mean()

df_bank_stborrow["df_bank7_250_5"] = df_bank_stborrow["df_bank7_MA250"] - df_bank_stborrow["df_bank7_MA5"]

df_bank_stborrow["L78_raw"] = calc_rolling_zscore(
    df_bank_stborrow["df_bank7_250_5"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

df_bank_stborrow["L78_raw"] = df_bank_stborrow["L78_raw"].astype(float)

data_df = data_df.join(df_bank_stborrow["L78_raw"], how='left')

# ----- Cell 170 (markdown) -----
# ### 美国信用利差 (L79_1)

# ----- Cell 171 (code) -----
L79_1_raw = z_MA_div(
    us_credit_spread_baml.sort_index().shift(2),
    short_window=5,
    long_window=250,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L79_1_raw", L79_1_raw)


# ----- Cell 172 (markdown) -----
# ### 美国信用利差 (L79_2)

# ----- Cell 173 (code) -----
L79_2_spread = moody_aaa - moody_baa
L79_2_raw = z_MA_div(
    L79_2_spread,
    short_window=5,
    long_window=250,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L79_2_raw", L79_2_raw)


# ----- Cell 174 (markdown) -----
# ### 美元指数收盘价 (L80)

# ----- Cell 175 (code) -----
L80_raw = data_yoy(llt(usd_index, 5))
_assign_raw_factor("L80_raw", L80_raw)


# ----- Cell 176 (markdown) -----
# ### 美元指数（USDX）和90日/120日均线的关系，短期为现值，长期为90日或120日 (L81_1)

# ----- Cell 177 (code) -----
L81_1_raw = z_MA_div(
    usd_index,
    short_window=5,
    long_window=90,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L81_1_raw", L81_1_raw)


# ----- Cell 178 (markdown) -----
# ### 美元指数（USDX）和90日/120日均线的关系，短期为现值，长期为90日或120日 (L81_2)

# ----- Cell 179 (code) -----
L81_2_raw = z_MA_div(
    usd_index,
    short_window=5,
    long_window=120,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L81_2_raw", L81_2_raw)


# ----- Cell 180 (markdown) -----
# ### 十年到期国债收益率 美国 (L82)

# ----- Cell 181 (code) -----
L82_raw = calc_rolling_zscore(
    us_t10 - llt(us_t10, 12),
    window=252,
    min_periods=126,
)
_assign_raw_factor("L82_raw", L82_raw)


# ----- Cell 182 (markdown) -----
# ### 美国十年期国债到期收益率（趋势周期=12月） (L83)

# ----- Cell 183 (code) -----
L83_raw = data_diff(llt(us_t10, 12))
_assign_raw_factor("L83_raw", L83_raw)


# ----- Cell 184 (markdown) -----
# ### 美国CPI同比（趋势周期=1月） (L85)

# ----- Cell 185 (code) -----
L85_raw = data_diff(us_cpi_yoy)
_assign_raw_factor("L85_raw", L85_raw)


# ----- Cell 186 (markdown) -----
# ### 美债收益率 (L87)

# ----- Cell 187 (code) -----
L87_raw = z_MA_div(
    us_t2y,
    short_window=5,
    long_window=250,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L87_raw", L87_raw)


# ----- Cell 188 (markdown) -----
# ### 失业率 美国 (L88)

# ----- Cell 189 (code) -----
L88_raw = data_diff(us_unemployment)
_assign_raw_factor("L88_raw", L88_raw)


# ----- Cell 190 (markdown) -----
# ### 非农就业人数 (L89)

# ----- Cell 191 (code) -----
L89_raw = calc_rolling_zscore(data_yoy(us_nonfarm), window=24, min_periods=12)
_assign_raw_factor("L89_raw", L89_raw)


# ----- Cell 192 (markdown) -----
# ### 6个月美债收益率当月均值，与过去2个月均值的差值（收益风险比：0.78） (L90)

# ----- Cell 193 (code) -----
L90_short = us_t6m.rolling(window=20, min_periods=20).mean()
L90_long = us_t6m.rolling(window=60, min_periods=60).mean()
L90_raw = calc_rolling_zscore(L90_short - L90_long, window=252, min_periods=126)
_assign_raw_factor("L90_raw", L90_raw)


# ----- Cell 194 (markdown) -----
# ### 3M美债YTM近一季均值和前三季均值比较 (L91)

# ----- Cell 195 (code) -----
L91_raw = z_MA_div(
    us_t3m,
    short_window=60,
    long_window=240,
    z_window=252,
    unit="days",
)
_assign_raw_factor("L91_raw", L91_raw)


# ----- Cell 196 (markdown) -----
# ### 1M美债YTM近10日均值在过去一年中的分位数 (L92)

# ----- Cell 197 (code) -----
L92_ma10 = us_t1m.rolling(window=10, min_periods=10).mean()
L92_rank = _rolling_quantile_rank_year(L92_ma10, 1)
L92_raw = _tail_quantile_signal(L92_rank, upper=0.75, lower=0.25)
_assign_raw_factor("L92_raw", L92_raw)


# ----- Cell 198 (markdown) -----
# ### 6个月美债收益率分位数（近三年） (L93)

# ----- Cell 199 (code) -----
L93_rank = _rolling_quantile_rank_year(us_t6m, 3)
L93_raw = _tail_quantile_signal(L93_rank, upper=0.75, lower=0.25)
_assign_raw_factor("L93_raw", L93_raw)


# ----- Cell 200 (markdown) -----
# ### 美国2年期国债到期收益率月度均值（历史胜率56.35%） (L94)

# ----- Cell 201 (code) -----
L94_monthly_mean = us_t2y.rolling(window=20, min_periods=20).mean()
L94_raw = calc_rolling_zscore(L94_monthly_mean, window=252, min_periods=126)
_assign_raw_factor("L94_raw", L94_raw)


# ----- Cell 202 (markdown) -----
# ### 宏观事件：出口当月同比预期环比下降，实际环比上升 (L96)

# ----- Cell 203 (code) -----
L96_aligned = pd.concat(
    [china_export_forecast.rename("forecast"), china_export_actual.rename("actual")],
    axis=1,
).dropna().sort_index()
L96_signal = pd.Series(0.0, index=L96_aligned.index, dtype="float64")
L96_mask = L96_aligned["forecast"] * L96_aligned["actual"] < 0
L96_signal.loc[L96_mask] = (
    L96_aligned["actual"] - L96_aligned["forecast"]
).loc[L96_mask]
L96_raw = calc_rolling_zscore(L96_signal, window=24, min_periods=12)
_assign_raw_factor("L96_raw", L96_raw)


# ----- Cell 204 (markdown) -----
# ### 新增出口额(美元)TTM同比增速 (L97)

# ----- Cell 205 (code) -----
L97_ttm_mean = china_export_actual.rolling(window=12, min_periods=12).mean()
L97_raw_base = pd.Series(L97_ttm_mean.copy(), dtype="float64")
L97_growth_mask = L97_ttm_mean * L97_ttm_mean.shift(1) > 0
L97_raw_base.loc[L97_growth_mask] = (
    L97_ttm_mean / L97_ttm_mean.shift(1) - 1
).loc[L97_growth_mask]
L97_raw = calc_rolling_zscore(L97_raw_base, window=18, min_periods=9)
_assign_raw_factor("L97_raw", L97_raw)


# ----- Cell 206 (markdown) -----
# ### L005 信用利差

# ----- Cell 207 (code) -----
df_STB_AA = pd.read_excel(Config.DATA_DIR / "流动性" / "D_AA级中短期票据到期收益率_CN_060301_260409.xlsx")
df_STB_AA = df_STB_AA.iloc[2:,:].copy()
df_STB_AA.columns = ["date", "STB_AA_1m", "STB_AA_2m", "STB_AA_3m", "STB_AA_6m", 
                            "STB_AA_9m", "STB_AA_1y", "STB_AA_2y", "STB_AA_3y", "STB_AA_5y", 
                            "STB_AA_6y", "STB_AA_7y", "STB_AA_8y", "STB_AA_9y", "STB_AA_10y"]
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:] / 100
df_STB_AA['date'] = pd.to_datetime(df_STB_AA['date'], errors='coerce')
df_STB_AA.set_index('date', inplace=True)
df_STB_AA = df_STB_AA.sort_index(ascending=True)
df_STB_AA.head(5)

# ----- Cell 208 (code) -----
df_TB = pd.read_excel(Config.DATA_DIR / "流动性" / "D_国债到期收益率_CN_020104_260409.xlsx")
df_TB = df_TB.iloc[2:,:].copy()
df_TB.columns = ["date", "CGB_1y", "CGB_3y", "CGB_5y", "CGB_10y", 
                 "CDB_1m", "CDB_2m", "CDB_3m", "CDB_6m", "CDB_9m", 
                "CDB_1y", "CDB_3y", "CDB_5y", "CDB_6y", "CDB_7y",
                "CDB_8y", "CDB_9y", "CDB_10y"]
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:] / 100
df_TB['date'] = pd.to_datetime(df_TB['date'], errors='coerce')
df_TB.set_index('date', inplace=True)
df_TB = df_TB.sort_index(ascending=True)
df_TB.head(5)

# ----- Cell 209 (code) -----
df_credit_spread = df_STB_AA.join(df_TB, how='inner')
df_credit_spread["cs_5y"] = df_credit_spread["STB_AA_5y"] - df_credit_spread["CDB_5y"]

df_credit_spread["cs_5y_5"] = df_credit_spread["cs_5y"].rolling(window=5).mean()
df_credit_spread["cs_5y_120"] = df_credit_spread["cs_5y"].rolling(window=6 * Config.MONTH_DAYS).mean()

df_credit_spread["cs_5y_120_5"] = df_credit_spread["cs_5y_120"] - df_credit_spread["cs_5y_5"]

df_credit_spread["cs_5y_120_5_z_raw"] = calc_rolling_zscore(
    df_credit_spread["cs_5y_120_5"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

df_credit_spread["L005_raw"] = df_credit_spread["cs_5y_120_5_z_raw"].astype(float)

# ----- Cell 210 (code) -----
data_df = data_df.join(df_credit_spread["L005_raw"], how='left')

# ----- Cell 211 (markdown) -----
# ## 国债到期收益率

# ----- Cell 212 (markdown) -----
# ### L119_1

# ----- Cell 213 (code) -----
# 国债到期收益率数据导入
df_treasury = pd.read_excel(Config.DATA_DIR / "流动性" / "D_国债到期收益率_CN_020104_260409.xlsx")

df_treasury = df_treasury.iloc[2:, :].copy()
df_treasury.iloc[:, 1:] = df_treasury.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_treasury.iloc[:, 1:] = df_treasury.iloc[:, 1:] / 100

df_treasury["指标名称"] = pd.to_datetime(df_treasury["指标名称"], errors="coerce")
df_treasury = df_treasury.set_index("指标名称")
df_treasury = df_treasury.sort_index()

df_treasury.head()

# ----- Cell 214 (code) -----
rate_10y = df_treasury["中债国债到期收益率:10年"]

df_treasury["TB_10y_z"] = fu.calc_rolling_zscore(
    rate_10y,
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods=Config.ANNUAL_TRADING_DAYS,
)

df_treasury["L119_1_raw"] = -df_treasury["TB_10y_z"]

# 拼接到主表
data_df = data_df.join(df_treasury["L119_1_raw"], how="left")

# ----- Cell 215 (markdown) -----
# ### L118

# ----- Cell 216 (code) -----
rate_10y = df_treasury["中债国债到期收益率:10年"]

ma_10y_160 = rate_10y.rolling(160, min_periods=160).mean()
dev_10y = ma_10y_160 - rate_10y

llt_10y = fu.calc_llt(dev_10y, 5)

z_10y = fu.calc_rolling_zscore(
    llt_10y,
    window= Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS,
)

rate_1y = df_treasury["中债国债到期收益率:1年"]

ma_1y_60 = rate_1y.rolling(60, min_periods=60).mean()
dev_1y = ma_1y_60 - rate_1y

llt_1y = fu.calc_llt(dev_1y, 10)

z_1y = fu.calc_rolling_zscore(
    llt_1y,
    window= Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS,
)

df_treasury["L118_raw"] = (z_10y + z_1y) / 2
data_df = data_df.join(df_treasury["L118_raw"], how="left")

# ----- Cell 217 (markdown) -----
# ### L171

# ----- Cell 218 (code) -----
df_STB_AA = pd.read_excel(Config.DATA_DIR / "流动性" / "D_AA级中短期票据到期收益率_CN_060301_260409.xlsx")
df_STB_AA = df_STB_AA.iloc[2:,:].copy()
df_STB_AA.columns = ["date", "STB_AA_1m", "STB_AA_2m", "STB_AA_3m", "STB_AA_6m", 
                            "STB_AA_9m", "STB_AA_1y", "STB_AA_2y", "STB_AA_3y", "STB_AA_5y", 
                            "STB_AA_6y", "STB_AA_7y", "STB_AA_8y", "STB_AA_9y", "STB_AA_10y"]
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_STB_AA.iloc[:, 1:] = df_STB_AA.iloc[:, 1:] / 100
df_STB_AA['date'] = pd.to_datetime(df_STB_AA['date'], errors='coerce')
df_STB_AA.set_index('date', inplace=True)
df_STB_AA = df_STB_AA.sort_index(ascending=True)
df_STB_AA.head(5)

# ----- Cell 219 (code) -----
df_TB = pd.read_excel(Config.DATA_DIR / "流动性" / "D_国债到期收益率_CN_020104_260409.xlsx")
df_TB = df_TB.iloc[2:,:].copy()
df_TB.columns = ["date", "CGB_1y", "CGB_3y", "CGB_5y", "CGB_10y", 
                 "CDB_1m", "CDB_2m", "CDB_3m", "CDB_6m", "CDB_9m", 
                "CDB_1y", "CDB_3y", "CDB_5y", "CDB_6y", "CDB_7y",
                "CDB_8y", "CDB_9y", "CDB_10y"]
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_TB.iloc[:, 1:] = df_TB.iloc[:, 1:] / 100
df_TB['date'] = pd.to_datetime(df_TB['date'], errors='coerce')
df_TB.set_index('date', inplace=True)
df_TB = df_TB.sort_index(ascending=True)
df_TB.head(5)

# ----- Cell 220 (code) -----
df_CB_AAps = pd.read_excel(Config.DATA_DIR / "流动性" / "D_AA+级企业债到期收益率_CN_071011_260409.xlsx")
df_CB_AAps = df_CB_AAps.iloc[2:,:].copy()
df_CB_AAps.columns = ["date", "df_CB_AAps_1m", "df_CB_AAps_2m", "df_CB_AAps_3m", "df_CB_AAps_6m", 
                            "df_CB_AAps_9m", "df_CB_AAps_1y", "df_CB_AAps_2y", "df_CB_AAps_3y", "df_CB_AAps_5y", 
                            "df_CB_AAps_6y", "df_CB_AAps_7y", "df_CB_AAps_8y", "df_CB_AAps_9y", "df_CB_AAps_10y"]
df_CB_AAps.iloc[:, 1:] = df_CB_AAps.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df_CB_AAps.iloc[:, 1:] = df_CB_AAps.iloc[:, 1:] / 100
df_CB_AAps['date'] = pd.to_datetime(df_CB_AAps['date'], errors='coerce')
df_CB_AAps.set_index('date', inplace=True)
df_CB_AAps = df_CB_AAps.sort_index(ascending=True)
df_CB_AAps.head(5)

# ----- Cell 221 (code) -----
df_credit_spread = df_STB_AA.join(df_TB, how='inner')
df_credit_spread["cs_3m"] = df_credit_spread["STB_AA_3m"] - df_credit_spread["CDB_3m"]
df_credit_spread["cs_9m"] = df_credit_spread["STB_AA_9m"] - df_credit_spread["CDB_9m"]

# ----- Cell 222 (code) -----
df_credit_spread["cs_3m_MA120"] = df_credit_spread["cs_3m"].rolling(window=6 * Config.MONTH_DAYS).mean()
df_credit_spread["cs_9m_MA120"] = df_credit_spread["cs_9m"].rolling(window=6 * Config.MONTH_DAYS).mean()

df_credit_spread = df_credit_spread.dropna(subset = ["cs_3m", "cs_9m","cs_3m_MA120", "cs_9m_MA120"])

df_credit_spread["cs_3m_MA120_dev"] = df_credit_spread["cs_3m"] - df_credit_spread["cs_3m_MA120"]
df_credit_spread["cs_9m_MA120_dev"] = df_credit_spread["cs_9m"] - df_credit_spread["cs_9m_MA120"]

df_credit_spread["cs_3m_MA120_raw"] = calc_rolling_zscore(
    df_credit_spread["cs_3m_MA120_dev"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

df_credit_spread["cs_9m_MA120_raw"] = calc_rolling_zscore(
    df_credit_spread["cs_9m_MA120_dev"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

df_credit_spread["cs_3m_MA120_raw"] = df_credit_spread["cs_3m_MA120_raw"].astype(float)
df_credit_spread["cs_9m_MA120_raw"] = df_credit_spread["cs_9m_MA120_raw"].astype(float)

df_credit_spread["L171_raw"] = df_credit_spread["cs_3m_MA120_raw"] - df_credit_spread["cs_9m_MA120_raw"]

# ----- Cell 223 (code) -----
data_df = data_df.join(df_credit_spread["L171_raw"], how='left')

# ----- Cell 224 (markdown) -----
# ## 量价

# ----- Cell 225 (markdown) -----
# ### V33 天风市净率比价 

# ----- Cell 226 (code) -----
df_grow_level = pd.read_excel(Config.DATA_DIR / "比价" / "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx")
df_grow_level.info()

df_zzqz = pd.read_excel(Config.STYLE_INDEX_DIR / "中证全指(000985.CSI)-历史价格.xlsx")
df_grow_level['交易日期'] = pd.to_datetime(df_grow_level['交易日期'], errors='coerce')
df_grow_level.set_index('交易日期', inplace=True)
df_grow_level = df_grow_level.sort_index(ascending=True)

data_df = data_df.join(df_grow_level['市净率LF'].rename('pb_g'), how='left')

# ----- Cell 227 (code) -----
df_value_level = pd.read_excel(Config.DATA_DIR / "比价" / "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx")
df_value_level.info()

df_zzqz = pd.read_excel(Config.STYLE_INDEX_DIR / "中证全指(000985.CSI)-历史价格.xlsx")
df_value_level['交易日期'] = pd.to_datetime(df_value_level['交易日期'], errors='coerce')
df_value_level.set_index('交易日期', inplace=True)
df_value_level = df_value_level.sort_index(ascending=True)

data_df = data_df.join(df_value_level['市净率LF'].rename('pb_v'), how='left')

# ----- Cell 228 (code) -----
data_df['tf_logBP_raw'] = np.log(1 / data_df['pb_v']) - np.log(1 / data_df['pb_g'])
data_df['tf_logBP_raw'] = fu.calc_llt(
    data_df['tf_logBP_raw'],   
    d=30
)
data_df["V33_raw"] = fu.calc_rolling_zscore(
    data_df["tf_logBP_raw"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

# ----- Cell 229 (markdown) -----
# ### V97 天风交易拥挤度

# ----- Cell 230 (code) -----
# 三个月累计换手率、波动率、相对于中证全指的beta

# 换手率z-score
data_df["turnover_ratio_3m_raw"] = (
    data_df["turnover_rate_g"].rolling(window=3 * Config.MONTH_DAYS).sum()
    / data_df["turnover_rate_v"].rolling(window=3 * Config.MONTH_DAYS).sum()
)

data_df["turnover_ratio_3m_log"] = np.log(data_df["turnover_ratio_3m_raw"])

data_df["tf_turnover_3m"] = fu.calc_llt(
    data_df["turnover_ratio_3m_log"],
    d=30
)

data_df["tf_turnover_3m"] = calc_rolling_zscore(
    data_df["tf_turnover_3m"],
    window= 3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

# 波动率 z-score
data_df["vol_ratio_3m_raw"] = (
    data_df["pct_change_g"].rolling(window=3 * Config.MONTH_DAYS).std()
    / data_df["pct_change_v"].rolling(window=3 * Config.MONTH_DAYS).std()
)

data_df["vol_ratio_3m_log"] = np.log(data_df["vol_ratio_3m_raw"])

data_df["tf_volatility_3m"] = fu.calc_llt(
    data_df["vol_ratio_3m_log"],   
    d=30
)

data_df["tf_volatility_3m"] = calc_rolling_zscore(
    data_df["tf_volatility_3m"],
    window= 3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

# ----- Cell 231 (code) -----
check_cols = [
    "turnover_ratio_3m_raw",
    "turnover_ratio_3m_log",
    "tf_turnover_3m",
    "vol_ratio_3m_raw",
    "vol_ratio_3m_log",
    "tf_volatility_3m",
]

summary_list = []

for col in check_cols:
    s = data_df[col]
    summary_list.append({
        "col": col,
        "dtype": s.dtype,
        "n_total": len(s),
        "n_nan": s.isna().sum(),
        "nan_ratio": s.isna().mean(),
        "n_pos_inf": np.isposinf(s).sum(),
        "n_neg_inf": np.isneginf(s).sum(),
        "min": s.replace([np.inf, -np.inf], np.nan).min(),
        "p01": s.replace([np.inf, -np.inf], np.nan).quantile(0.01),
        "p05": s.replace([np.inf, -np.inf], np.nan).quantile(0.05),
        "median": s.replace([np.inf, -np.inf], np.nan).median(),
        "mean": s.replace([np.inf, -np.inf], np.nan).mean(),
        "p95": s.replace([np.inf, -np.inf], np.nan).quantile(0.95),
        "p99": s.replace([np.inf, -np.inf], np.nan).quantile(0.99),
        "max": s.replace([np.inf, -np.inf], np.nan).max(),
        "std": s.replace([np.inf, -np.inf], np.nan).std(),
    })

summary_df = pd.DataFrame(summary_list)
display(summary_df)

# ----- Cell 232 (code) -----
# Beta z-score
df_zzqz = pd.read_excel(Config.STYLE_INDEX_DIR / "中证全指(000985.CSI)-历史价格.xlsx")
df_zzqz['交易日期'] = pd.to_datetime(df_zzqz['交易日期'], errors='coerce')
df_zzqz.set_index('交易日期', inplace=True)
df_zzqz = df_zzqz.sort_index(ascending=True)
df_zzqz["涨跌幅"] = fu.pct_change_to_digit(df_zzqz["涨跌幅"])

# ----- Cell 233 (code) -----
print(df_zzqz["涨跌幅"].head())

# ----- Cell 234 (code) -----
data_df = data_df.join(df_zzqz['涨跌幅'].rename('pct_change_zzqz'), how='left')

X = sm.add_constant(data_df['pct_change_zzqz'])
Y_g = data_df['pct_change_g']
model_g = RollingOLS(Y_g, X, window= 3*Config.MONTH_DAYS)
results_g = model_g.fit()

data_df['beta_g'] = results_g.params['pct_change_zzqz']

Y_v = data_df['pct_change_v']
model_v = RollingOLS(Y_v, X, window= 3*Config.MONTH_DAYS)
results_v = model_v.fit()

data_df['beta_v'] = results_v.params['pct_change_zzqz']

data_df["beta_ratio_3m_raw"] =data_df['beta_g'] / data_df['beta_v']
data_df["beta_ratio_3m_log"] = np.log(data_df["beta_ratio_3m_raw"])

data_df["tf_beta_3m"] = fu.calc_llt(
    data_df["beta_ratio_3m_log"],   
    d=30
)

data_df["tf_beta_3m"] = calc_rolling_zscore(
    data_df["tf_beta_3m"],
    window=3 * Config.ANNUAL_TRADING_DAYS,
    min_periods= Config.ANNUAL_TRADING_DAYS
)

# 总分
data_df["V97_raw"] = (data_df["tf_turnover_3m"] + data_df["tf_volatility_3m"] + data_df["tf_beta_3m"]) / 3

# ----- Cell 235 (markdown) -----
# ### V79 过去一月动量差

# ----- Cell 236 (code) -----
# 生成日频因子底表
# 绑定Config.DEAL_TYPE，更改DEAL_TYPE时此处需修改
# 考虑到交易的实际成交限制，现在Config.DEAL_TYPE="close"，动量依然设定为是前一天的收盘价决定的

data_df["ret_1m_g_raw"] = fu.calc_rolling_return(
    price_series=data_df["close_g"],
    window=Config.MONTH_DAYS,
    return_type=Config.RETURN_TYPE
)

data_df["ret_1m_v_raw"] = fu.calc_rolling_return(
    price_series=data_df["close_v"],
    window=Config.MONTH_DAYS,
    return_type=Config.RETURN_TYPE
)

# 过去一个月动量差 = 价值过去1月收益率 - 成长过去1月收益率
data_df["V79_raw"] =  data_df["ret_1m_g_raw"] - data_df["ret_1m_v_raw"]

print("日频底表因子计算完成：")
display(
    data_df[["close_g", "close_v", "ret_1m_g_raw", "ret_1m_v_raw","V79_raw"]]
    .tail(Config.Tunnels*2)
)

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
