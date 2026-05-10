from pathlib import Path
import sys
import pandas as pd
import numpy as np

# 当前脚本路径：MAIN_STYLE_ROTATION/B_factors/scripts/xxx.py
SCRIPT_PATH = Path(__file__).resolve()

# 项目根目录：MAIN_STYLE_ROTATION
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# 项目配置库config, factor_utils路径
CONFIG_DIR = PROJECT_ROOT / "SY_Baseline" 
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

from Config import Config

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

# 计算收益率
def calc_ret(entry_price, exit_price, return_type=Config.RETURN_TYPE):

    if return_type == "log":
        return np.log(exit_price / entry_price)
    else:
        return exit_price / entry_price - 1
    
# 计算过去 window 个交易日的滚动收益率
def calc_rolling_return(price_series, window=Config.MONTH_DAYS, return_type=Config.RETURN_TYPE):
    """
    price_series : pd.Series
        价格序列，比如收盘价序列
    window : int
        回看窗口，默认使用 Config.MONTH_DAYS
    return_type : str
        "log" 表示对数收益率
        "simple" 表示简单收益率

    返回
    pd.Series
        与原序列同索引的滚动收益率序列
    """
    past_prices = price_series.shift(window)
    rolling_ret = calc_ret(past_prices, price_series, return_type)

    return rolling_ret

# 2) 定义 LLT 函数：常见的二阶低延迟滤波写法
def calc_llt(series, d=30):
    """
    对一条时间序列做 LLT 平滑。
    
    参数
    ----------
    series : pd.Series
        原始序列
    d : int
        LLT 参数，比如 30 表示 LLT(30)
    
    返回
    ----------
    llt_series : pd.Series
        LLT 平滑后的序列
    """
    s = series.astype(float).copy()
    llt = pd.Series(index=s.index, dtype=float)

    alpha = 2 / (d + 1)

    started = False

    for i in range(2, len(s)):
        x_t = s.iloc[i]
        x_t1 = s.iloc[i - 1]
        x_t2 = s.iloc[i - 2]

        # 当前三期原始值必须都有效，否则这一期无法计算
        if pd.isna(x_t) or pd.isna(x_t1) or pd.isna(x_t2):
            llt.iloc[i] = np.nan
            continue

        # 如果还没开始，就在这里初始化
        if not started:
            llt.iloc[i - 1] = x_t1
            llt.iloc[i] = x_t
            started = True
            continue

        llt_t1 = llt.iloc[i - 1]
        llt_t2 = llt.iloc[i - 2]

        # 如果前两期 LLT 缺失，说明中间断过，需要重新初始化
        if pd.isna(llt_t1) or pd.isna(llt_t2):
            llt.iloc[i - 1] = x_t1
            llt.iloc[i] = x_t
            continue

        llt.iloc[i] = (
            (alpha - alpha**2 / 4) * x_t
            + (alpha**2 / 2) * x_t1
            - (alpha - 3 * alpha**2 / 4) * x_t2
            + 2 * (1 - alpha) * llt_t1
            - (1 - alpha) ** 2 * llt_t2
        )

    return llt

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

# 将百分比字符串（如 '1.23%'）转换为数字
def pct_change_to_digit(series):

    s = (
        series.astype(str)
        .str.replace('%', '', regex=False)
    )

    s = pd.to_numeric(s, errors='coerce')

    return s / 100

# 因子挂载
def merge_factor_to_market(data_df, market_df, raw_factor_col, factor_type = "None", track_col = "track_id"):
    """
    将日频原始因子列安全挂载到 market_df 中。factor_type为state或event，决定了如何处理后续挂载

    默认逻辑：
    1. 先把 raw 因子整体 shift(1)，避免未来函数
    2. 去掉 _raw 后缀，生成正式因子列
    3. 按日期索引把正式因子映射到 market_df
    4. 自动做一次抽检：检查 market_df 在 T 日的值是否等于 data_df 在 T-1 日的 raw 值
    5. 把正式因子名加入 Config.FEATURE_LIST
    """
    if raw_factor_col not in data_df.columns:
        raise KeyError(f"{raw_factor_col} 不在 data_df.columns 里，请检查列名是否写对")

    if not raw_factor_col.endswith("_raw"):
        raise ValueError(f"{raw_factor_col} 不是 raw 因子列，挂载前的因子列名必须以 _raw 结尾")

    factor_col = raw_factor_col.replace("_raw", "")

    # 确定因子数据类型
    if factor_type is None:
        factor_type = globals().get("signal_type", "state")
    factor_type = str(factor_type).lower()

    if factor_type not in ["state", "event"]:
        raise ValueError(f"factor_type 只能是 'state' 或 'event'，当前值是: {factor_type}")

    if factor_type == "state":
        # 默认 shift(1) 一天：T-1 日收盘后能看到的信号，用于 T 日交易
        data_df[factor_col] = data_df[raw_factor_col].shift(1)

        # 按日期索引把正式因子映射到 market_df
        market_df[factor_col] = data_df[factor_col].reindex(market_df.index)

        # 自动抽检：随机抽一个非空日期，检查 market_df[T] 是否等于 data_df[T-1] 的 raw 值
        valid_dates = market_df.index[market_df[factor_col].notna()]
        first_date_in_data = data_df.index[0]
        safe_dates = valid_dates[valid_dates > first_date_in_data]

        # 4. 在安全的池子里抽样
        sample_date = safe_dates.to_series().sample(n=1, random_state=42).iloc[0]
        sample_loc = data_df.index.get_loc(sample_date)
        
        prev_date = data_df.index[sample_loc - 1]
        
        market_value = market_df.loc[sample_date, factor_col]
        raw_prev_value = data_df.loc[prev_date, raw_factor_col]

        assert np.isclose(market_value, raw_prev_value, equal_nan=True), (
            f"因子抽检失败：market_df 在 {sample_date} 的 {factor_col} = {market_value}，"
            f"但 data_df 在前一个交易日 {prev_date} 的 {raw_factor_col} = {raw_prev_value}"
        )

    
    elif factor_type == "event":
        if track_col not in market_df.columns:
            raise KeyError(f"event 因子挂载需要 market_df 里有 {track_col} 列")
        
        market_df[factor_col] = np.nan
        raw_events = (
            data_df.loc[data_df[raw_factor_col].notna(), raw_factor_col]
            .copy()
            .sort_index(ascending=True)
        )

        tunnels = Config.Tunnels
        track_dates = {}
        market_track = market_df[track_col]
        for track_id in range(tunnels):
            track_dates[track_id] = pd.DatetimeIndex(
                market_df.index[market_track == track_id]
            ).sort_values()

        check_rows = []
        for event_date, raw_value in raw_events.items():
            event_date = pd.Timestamp(event_date)

            for target_track_id in market_df[track_col].unique():
                candidate_dates = track_dates[target_track_id]
                candidate_dates = candidate_dates[candidate_dates > event_date]

                if len(candidate_dates) == 0:
                    continue

                trade_date = candidate_dates[0]
                market_df.loc[trade_date, factor_col] = raw_value
                check_rows.append({
                    "event_date": event_date,
                    "target_track_id": target_track_id,
                    "trade_date": trade_date,
                    "raw_value": raw_value,
                })

        data_df[factor_col] = market_df[factor_col].reindex(data_df.index)

        if len(check_rows) > 0:
            check_df = pd.DataFrame(check_rows)
            effective_check_df = check_df.drop_duplicates(subset=["trade_date"], keep="last")
            sample_row = effective_check_df.sample(n=1, random_state=42).iloc[0]
            trade_date = sample_row["trade_date"]
            market_value = market_df.loc[trade_date, factor_col]
            raw_value = sample_row["raw_value"]

            assert np.isclose(market_value, raw_value, equal_nan=True), (
                f"event 因子抽检失败：market_df 在 {trade_date} 的 {factor_col} = {market_value}，"
                f"但 data_df 在事件日 {sample_row['event_date']} 的 {raw_factor_col} = {raw_value}"
            )

            same_day_rows = check_df[check_df["trade_date"] == check_df["event_date"]]
            assert same_day_rows.empty, (
                "event 因子挂载失败：事件发生当日不应接收到自己的信号，"
                f"异常日期: {same_day_rows['event_date'].tolist()}"
            )

    # 注册特征
    if factor_col not in Config.FEATURE_LIST:
        Config.FEATURE_LIST.append(factor_col)

    return market_df

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


def align_industrial_dates_to_macro_release(
    industrial_path,
    macro_path,
    macro_indicator_keyword: str = "月工业企业利润:累计同比",
    industrial_date_col: str = "指标名称",
    macro_indicator_col: str = "指标名称",
    macro_date_col: str = "日期",
    macro_country_col: str = "国家/地区",
    macro_country: str | None = None,
    macro_keyword_regex: bool = False,
    months_forward: int = 1,
    output_path=None,
    save: bool = False,
):
    """
    将“规模以上工业 招证资配.xlsx”中的月末统计日期，对齐到 Macro_all 中对应指标的实际发布日期。

    处理逻辑：
    1. 识别工业表里真正的数据行（能被解析为日期的行），跳过前面的元信息行。
    2. 将原日期整体往后推 `months_forward` 个月，记录调整后的年、月。
    3. 在 Macro_all 中筛选“指标名称”包含 `macro_indicator_keyword` 的行。
       若传入 `macro_country`，则额外要求“国家/地区”匹配。
    4. 若 Macro_all 的发布日期年、月与工业表调整后的年、月一致，则用该发布日期替换工业表原日期。

    返回
    ----------
    pd.DataFrame
        包含以下辅助列的结果表：
        - 原日期
        - 调整后日期
        - 调整后年份
        - 调整后月份
        - Macro发布日期
        - 日期匹配状态

    参数
    ----------
    industrial_path : str or Path
        原始工业表路径。
    macro_path : str or Path
        Macro_all 路径。
    macro_country : str, optional
        Macro_all 的国家/地区筛选条件；例如 "中国"。
    macro_keyword_regex : bool, default False
        是否把 `macro_indicator_keyword` 当作正则表达式处理。
    output_path : str or Path, optional
        保存路径；默认与 `industrial_path` 相同。
    save : bool, default False
        True 时写回 Excel，False 时仅返回处理后的 DataFrame。
    """
    industrial_path = Path(industrial_path)
    macro_path = Path(macro_path)
    output_path = Path(output_path) if output_path is not None else industrial_path

    industrial_df = pd.read_excel(industrial_path)
    macro_df = pd.read_excel(macro_path)

    if industrial_date_col not in industrial_df.columns:
        raise KeyError(f"工业表中不存在列: {industrial_date_col}")
    if macro_indicator_col not in macro_df.columns:
        raise KeyError(f"Macro_all 中不存在列: {macro_indicator_col}")
    if macro_date_col not in macro_df.columns:
        raise KeyError(f"Macro_all 中不存在列: {macro_date_col}")
    if macro_country is not None and macro_country_col not in macro_df.columns:
        raise KeyError(f"Macro_all 中不存在列: {macro_country_col}")

    out = industrial_df.copy()
    raw_date_text = out[industrial_date_col].astype(str).str.strip()
    parseable_mask = raw_date_text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    out["原日期"] = pd.NaT
    out.loc[parseable_mask, "原日期"] = pd.to_datetime(
        raw_date_text.loc[parseable_mask],
        errors="coerce",
    )

    data_mask = out["原日期"].notna()
    out["调整后日期"] = pd.NaT
    out.loc[data_mask, "调整后日期"] = (
        out.loc[data_mask, "原日期"] + pd.DateOffset(months=months_forward)
    )
    out["调整后年份"] = out["调整后日期"].dt.year.astype("Int64")
    out["调整后月份"] = out["调整后日期"].dt.month.astype("Int64")

    macro_mask = macro_df[macro_indicator_col].astype(str).str.contains(
        macro_indicator_keyword,
        na=False,
        regex=macro_keyword_regex,
    )
    if macro_country is not None:
        macro_mask = macro_mask & (macro_df[macro_country_col].astype(str) == str(macro_country))

    macro_release_cols = [macro_date_col, macro_indicator_col]
    if macro_country is not None:
        macro_release_cols.append(macro_country_col)

    macro_release = macro_df.loc[macro_mask, macro_release_cols].copy()

    macro_release["Macro发布日期"] = pd.to_datetime(macro_release[macro_date_col], errors="coerce")
    macro_release = macro_release.dropna(subset=["Macro发布日期"])
    macro_release["调整后年份"] = macro_release["Macro发布日期"].dt.year.astype("Int64")
    macro_release["调整后月份"] = macro_release["Macro发布日期"].dt.month.astype("Int64")
    macro_release = macro_release.drop_duplicates()

    duplicate_mask = macro_release.duplicated(subset=["调整后年份", "调整后月份"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = macro_release.loc[
            duplicate_mask,
            ["调整后年份", "调整后月份", "Macro发布日期", macro_indicator_col],
        ]
        raise ValueError(
            "Macro_all 中存在同一年月对应多个发布日期，无法唯一映射：\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    release_calendar = (
        macro_release[["调整后年份", "调整后月份", "Macro发布日期"]]
        .drop_duplicates(subset=["调整后年份", "调整后月份"])
    )

    out = out.merge(
        release_calendar,
        on=["调整后年份", "调整后月份"],
        how="left",
        sort=False,
    )

    matched_mask = data_mask & out["Macro发布日期"].notna()
    out["日期匹配状态"] = "元信息行"
    out.loc[data_mask, "日期匹配状态"] = "未匹配"
    out.loc[matched_mask, "日期匹配状态"] = "已匹配"

    out.loc[matched_mask, industrial_date_col] = out.loc[matched_mask, "Macro发布日期"]

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_excel(output_path, index=False)

    return out


def align_profit_release_dates(
    industrial_path,
    macro_path,
    output_path=None,
    save: bool = False,
):
    """工业企业利润表专用：对齐到 Macro_all 的“月工业企业利润:累计同比”发布日期。"""
    return align_industrial_dates_to_macro_release(
        industrial_path=industrial_path,
        macro_path=macro_path,
        macro_indicator_keyword="月工业企业利润:累计同比",
        output_path=output_path,
        save=save,
    )


def align_rmb_loan_release_dates(
    industrial_path,
    macro_path,
    output_path=None,
    save: bool = False,
):
    """人民币贷款表专用：对齐到 Macro_all 的“中国 + 月新增人民币贷款”发布日期。"""
    return align_industrial_dates_to_macro_release(
        industrial_path=industrial_path,
        macro_path=macro_path,
        macro_indicator_keyword="月新增人民币贷款",
        macro_country="中国",
        output_path=output_path,
        save=save,
    )
