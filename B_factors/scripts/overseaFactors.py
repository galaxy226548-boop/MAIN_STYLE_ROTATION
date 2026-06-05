"""海外相关风格轮动因子。

本文件负责生成 overseaFactors 因子：
1. 从计划表或历史登记文件读取每个因子的 metadata；
2. 按 factor_id 计算各自的原始时间序列；
3. 挂载到项目统一交易日历，并输出标准化因子和 long/short 信号。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from factor_utils import (
    PROJECT_ROOT,
    _register_factor,
    calc_rolling_zscore,
    load_prepared_table,
    read_prepared_series,
)
from factor_metadata import (
    append_record_note as _append_note,
    build_metadata_from_records,
    load_plan_or_generated_records,
    normalize_plan_text as _normalize_plan_text,
)
from factor_pipeline_runner import run_factor_module_pipeline
from factor_transforms import (
    as_float_series as _as_float_series,
    positive_series as _positive_series,
    rolling_log_slope as _rolling_log_slope,
    rolling_rank as _rolling_rank,
    rolling_std_breakout as _rolling_std_breakout,
    rolling_time_corr as _rolling_time_corr,
    safe_divide as _safe_divide,
)


# 输出前缀会写入 B_factors/output，并用于 factor_generated.json 的登记筛选。
OUTPUT_PREFIX = "overseaFactors"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"
C_FACTOR_REGISTRY_PATH = PROJECT_ROOT / "B_factors" / "reference" / "factor_C.json"

# 本模块当前实际实现的因子范围。新增或删除因子时，通常需要同步调整：
# 1. FACTOR_IDS；2. _record_with_actual_fields；3. _calc_oversea_factor。
FACTOR_IDS = [
    "O011",
    "C035",
    "C036",
    "C037",
    "C038",
    "C039",
    "O015",
    "O017",
    "O018",
    "O019",
    "O020",
    "O021",
    "O022",
    "O023",
    "O024",
    "O025",
]
FACTOR_C_IDS = ["C035", "C036", "C037", "C038", "C039"]
FACTOR_O_IDS = [factor_id for factor_id in FACTOR_IDS if factor_id.startswith("O")]

VIX_FILE = "VIX.GI-行情统计-20260509.xlsx"
SPX_FILE = "SPX.GI-行情统计-20260519.xlsx"
NDX_FILE = "NDX.GI-行情统计-20260519.xlsx"
EXCHANGE_TABLE = "exchange_rate_daily.parquet"
OVERSEA_DAILY_TABLE = "factorO_daily.parquet"
OVERSEA_MONTHLY_TABLE = "factorO_monthly.parquet"

HKD_SPOT_COL = "即期汇率定盘价:美元兑港元"
USDCNY_MID_COL = "中间价:美元兑人民币"
USDCNH_COL = "即期汇率:美元兑离岸人民币(USDCNH)"
HKD_SWAP_1M_COL = "掉期点:美元兑港元:1个月"
HIBOR_1M_COL = "HIBOR:1个月"
SOFR_1M_COL = "美国:SOFR期限利率:1个月"
BDI_COL = "波罗的海干散货指数(BDI)"
SOX_COL = "费城半导体指数"
KOREA_SEMI_EXPORT_COL = "韩国:出口金额:半导体:当月值"

ANNUAL_TRADING_DAYS = 252


def _load_plan_records() -> list[dict[str, object]]:
    """读取并整理本模块因子的 metadata 记录。

    O 类因子沿用原 overseaFactors 记录；C035-C039 使用 factor_C.json 中的
    因子定义，避免重命名后丢失 event/state 类型。
    """
    records = load_plan_or_generated_records(
        plan_path=PLAN_PATH,
        generated_path=PROJECT_ROOT / "B_factors" / "output" / "factor_generated.json",
        project_root=PROJECT_ROOT,
        factor_ids=FACTOR_O_IDS,
        output_prefix=OUTPUT_PREFIX,
        record_adjuster=_record_with_actual_fields,
        minimal_record_factory=_minimal_fallback_record,
    )
    records.extend(_load_factor_c_records())
    by_factor_id = {str(record["factor_id"]): record for record in records}
    return [by_factor_id[factor_id] for factor_id in FACTOR_IDS]


def _minimal_fallback_record(factor_id: str) -> dict[str, object]:
    # 如果计划表缺失，但历史登记也不完整，补一份最小可用 metadata，
    # 让已实现的因子仍能执行挂载和输出。这里的字段只服务于程序流程，
    # 不代表已经补全了研究定义。
    return {
        "paper_id": "DIY",
        "factor_id": factor_id,
        "signal_type": "state",
        "bar": 0.0,
        "progress": "done",
        "_source_file": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
        "_source_sheet": "fallback",
    }


def _load_factor_c_records() -> list[dict[str, object]]:
    """从 factor_C.json 读取 C035-C038 的 metadata。"""
    if not C_FACTOR_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Cannot find factor C registry: {C_FACTOR_REGISTRY_PATH}")

    payload = json.loads(C_FACTOR_REGISTRY_PATH.read_text(encoding="utf-8"))
    source_file = str(C_FACTOR_REGISTRY_PATH.relative_to(PROJECT_ROOT))
    wanted = set(FACTOR_C_IDS)
    records: list[dict[str, object]] = []
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("编号") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["paper_id"] = OUTPUT_PREFIX
            item["factor_id"] = factor_id
            item["signal_type"] = str(item.get("数据频率") or "state").strip().lower()
            item["docu"] = Path(str(item.get("数据路径") or "")).name
            item["data_field"] = item.get("字段名")
            item["factor"] = item.get("原数据")
            item["calc_method"] = item.get("测度目标")
            item["progress"] = "done"
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            _append_note(item, "本记录由 overseaFactors.py 按 factor_C.json 的 C 编号生成。")
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_C_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"{C_FACTOR_REGISTRY_PATH.name} missing factor C records: {missing}")
    return sorted(records, key=lambda record: FACTOR_C_IDS.index(str(record["factor_id"])))


def _record_with_actual_fields(record: dict[str, object]) -> dict[str, object]:
    """补齐/修正实际运行所需字段。

    计划表里部分 O 因子的 docu 或 condition 信息不完整。这里根据本脚本实际使用
    的数据源补 docu，并把实现口径写进 notes，便于输出登记时追溯。
    """
    factor_id = str(record["factor_id"])
    if not _normalize_plan_text(record.get("paper_id")):
        record["paper_id"] = "DIY"

    if factor_id in {"O015", "O018", "O020", "O021", "O022", "O023", "O024"}:
        record["docu"] = OVERSEA_DAILY_TABLE
    elif factor_id == "O025":
        record["docu"] = OVERSEA_MONTHLY_TABLE
    elif factor_id == "O016":
        record["docu"] = EXCHANGE_TABLE

    if factor_id in {"O011", "O016", "O017"}:
        _append_note(record, "计划表 condition 存在语法残缺，本脚本按 factor/calc_method/data_field 的业务含义实现。")
    if factor_id == "O013":
        _append_note(record, "本脚本只取完整月份的月末源数据作为事件日，不使用未完整月份。")
    return record


def metadata_from_overseaFactors_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """把计划表记录转换成 mount/build_signal 工具需要的 metadata 字典。"""
    return build_metadata_from_records(records)


def _clean_date_series(series: pd.Series) -> pd.Series:
    """把日期列转成标准化到 00:00:00 的 Timestamp。"""
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _load_price_file_close(file_name: str, name: str) -> pd.Series:
    """读取行情统计类文件的收盘价列，返回正数价格序列。"""
    df = load_prepared_table(file_name)
    if "交易日期" not in df.columns or "收盘价" not in df.columns:
        raise KeyError(f"{file_name} must contain 交易日期 and 收盘价; available={list(df.columns)}")
    return _positive_series(_as_float_series(df["收盘价"], _clean_date_series(df["交易日期"]), name))


def _clean_positive_limited_ffill_series(series: pd.Series, limit: int = 5) -> pd.Series:
    """清理正数序列，并只允许短期前值填充。

    海外日频数据可能存在少量缺口；这里最多 ffill 5 天，避免把长期缺失误当成有效状态。
    """
    s = series.astype("float64").sort_index(ascending=True)
    return s.where(s > 0).ffill(limit=limit)


def _signed_excess_score(value: pd.Series, threshold: float | pd.Series) -> pd.Series:
    """只保留超过阈值的部分，并保留原始方向。

    例：value=0.08、threshold=0.05 时返回 +0.03；
    value=-0.08、threshold=0.05 时返回 -0.03；
    未超过阈值时返回 0。
    """
    threshold_values = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=value.index)
    score = np.sign(value) * (value.abs() - threshold_values).clip(lower=0)
    return score.astype("float64")


def _complete_month_end_values(series: pd.Series) -> pd.Series:
    """保留完整月份的月末值，剔除尚未走完的当月数据。

    O013 使用月末事件口径。如果当前月还没结束，最后一个月末样本可能只是临时最新值，
    这里会把它去掉，避免用未完整月份触发事件。
    """
    s = series.astype("float64").dropna().sort_index()
    if s.empty:
        return s
    month_end_values = s.groupby(s.index.to_period("M")).tail(1)
    today = pd.Timestamp.today().normalize()
    latest_period = month_end_values.index[-1].to_period("M")
    if latest_period == today.to_period("M") and month_end_values.index[-1].normalize() < today + pd.offsets.MonthEnd(0):
        month_end_values = month_end_values.iloc[:-1]
    return month_end_values


def _monthly_average_pct_change(series: pd.Series) -> pd.Series:
    """按自然月求均值并计算月度环比，索引用月末事件日。"""
    s = series.astype("float64").dropna().sort_index()
    monthly_avg = s.groupby(s.index.to_period("M")).mean()
    monthly_avg.index = monthly_avg.index.to_timestamp(how="end").normalize()
    return monthly_avg / monthly_avg.shift(1) - 1


def _calc_oversea_factor(factor_id: str) -> pd.Series:
    """按 factor_id 计算单个海外因子的原始源序列。

    本函数只产出“源因子值”，不负责对齐市场交易日、标准化或生成 long/short 信号。
    这些后处理在 generate_overseaFactors_factors() 和 main() 中完成。
    """
    if factor_id == "O011":
        # VIX 20 日收益率的滚动 z-score。VIX 异常上行通常代表风险偏好下降，
        # 因此返回值前面乘 -1，使“风险冲击”对应偏负方向。
        vix_close = _load_price_file_close(VIX_FILE, "VIX")
        vix_ret_20d = vix_close / vix_close.shift(20) - 1
        vix_zscore = calc_rolling_zscore(vix_ret_20d, window=255, min_periods=120)
        return -np.sign(vix_zscore) * (np.abs(vix_zscore) - 1).clip(lower=0)

    if factor_id == "C035":
        # 美元兑人民币中间价按自然月均值计算环比，再乘 -1；
        # 该因子按月末事件口径挂载。
        usdcny_mid = _positive_series(read_prepared_series(EXCHANGE_TABLE, USDCNY_MID_COL)).sort_index(ascending=True)
        return _monthly_average_pct_change(usdcny_mid) * -1

    if factor_id == "C036":
        # 港币联系汇率区间压力：20 日均值低于 7.77 或高于 7.83 时产生方向信号，
        # 中间区间为 0，缺失数据保持 NaN。
        hkd_spot = _positive_series(read_prepared_series(EXCHANGE_TABLE, HKD_SPOT_COL))
        spot_mean_20d = hkd_spot.rolling(window=20, min_periods=20).mean()
        signal = pd.Series(0.0, index=spot_mean_20d.index, dtype="float64")
        signal.loc[spot_mean_20d < 7.77] = (7.77 - spot_mean_20d.loc[spot_mean_20d < 7.77]) / 0.05
        signal.loc[spot_mean_20d > 7.83] = (7.83 - spot_mean_20d.loc[spot_mean_20d > 7.83]) / 0.05
        signal.loc[spot_mean_20d.isna()] = np.nan
        return signal

    if factor_id == "C037":
        # 美元兑港元 1M 掉期点的月末事件信号：先只保留完整月份月末值，
        # 再判断是否突破过去约一年均值的 1 倍标准差。
        swap_points = _complete_month_end_values(read_prepared_series(EXCHANGE_TABLE, HKD_SWAP_1M_COL))
        return _rolling_std_breakout(
            swap_points,
            window=ANNUAL_TRADING_DAYS,
            min_periods=ANNUAL_TRADING_DAYS,
            std_multiplier=1.0,
        )

    if factor_id == "C038":
        # USDCNH 20 日涨跌幅超过 1.5% 后才计入；乘 -1 表示人民币贬值压力偏负。
        usdcnh = _positive_series(read_prepared_series(EXCHANGE_TABLE, USDCNH_COL)).sort_index(ascending=True)
        usdcnh_ret_20d = usdcnh / usdcnh.shift(20) - 1
        return _signed_excess_score(usdcnh_ret_20d, 0.015) * -1

    if factor_id == "C039":
        # 用港币即期汇率、HIBOR 1M、SOFR 1M 估算理论掉期点，
        # 再计算约一年窗口的标准差突破。
        hkd_spot = _positive_series(read_prepared_series(EXCHANGE_TABLE, HKD_SPOT_COL))
        hibor_1m = read_prepared_series(EXCHANGE_TABLE, HIBOR_1M_COL)
        sofr_1m = read_prepared_series(EXCHANGE_TABLE, SOFR_1M_COL)
        fitted_swap = hkd_spot * (hibor_1m - sofr_1m) / 100 * 30 / 360 * 10000
        return _rolling_std_breakout(
            fitted_swap.dropna(),
            window=ANNUAL_TRADING_DAYS,
            min_periods=ANNUAL_TRADING_DAYS,
            std_multiplier=1.0,
        )

    if factor_id == "O015":
        # BDI 航运景气度突破信号。这里使用 120 日窗口和 2 倍标准差阈值，
        # 再乘 -1 以匹配当前研究方向设定。
        bdi = read_prepared_series(OVERSEA_DAILY_TABLE, BDI_COL)
        return _rolling_std_breakout(bdi, window=120, min_periods=120, std_multiplier=2.0) * -1

    if factor_id == "O017":
        # 纳指相对标普的 20 日超额收益，超过 3% 后计入方向信号。
        ndx_close = _load_price_file_close(NDX_FILE, "NDX")
        spx_close = _load_price_file_close(SPX_FILE, "SPX")
        excess_return = (ndx_close / ndx_close.shift(20) - 1) - (spx_close / spx_close.shift(20) - 1)
        return _signed_excess_score(excess_return.dropna(), 0.03)

    if factor_id == "O018":
        # 费城半导体指数 20 日收益率，超过 5% 后计入方向信号。
        sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
        sox_ret_20d = sox / sox.shift(20) - 1
        return _signed_excess_score(sox_ret_20d, 0.05)

    if factor_id == "O019":
        # 费城半导体指数相对标普的 20 日超额收益，超过 5% 后计入方向信号。
        sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
        spx_close = _load_price_file_close(SPX_FILE, "SPX")
        excess_return = (sox / sox.shift(20) - 1) - (spx_close / spx_close.shift(20) - 1)
        return _signed_excess_score(excess_return.dropna(), 0.05)

    # O020-O024 都基于 SOX 趋势类特征：先清理 SOX 价格，再复用斜率、排名、
    # z-score、趋势相关性等工具，把不同趋势形态映射到 [-1, 1] 左右的排名信号。
    sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
    log_sox = np.log(sox.where(sox > 0))
    if factor_id == "O020":
        # 60 日 log 价格趋势斜率，在约 5 年窗口内做百分位排名。
        slope_60d = _rolling_log_slope(sox, window=60, min_periods=60)
        return _rolling_rank(slope_60d, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O021":
        # 短期趋势斜率减长期趋势斜率，刻画趋势加速度，再做长窗口百分位排名。
        slope_acceleration = (
            _rolling_log_slope(sox, window=20, min_periods=20)
            - _rolling_log_slope(sox, window=120, min_periods=120)
        )
        return _rolling_rank(slope_acceleration, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O022":
        # log SOX 相对 250 日均值的 z-score，再做长窗口百分位排名。
        log_sox_valid = log_sox.dropna()
        long_mean = log_sox_valid.rolling(window=250, min_periods=250).mean()
        long_std = log_sox_valid.rolling(window=250, min_periods=250).std()
        trend_zscore = _safe_divide(log_sox_valid - long_mean, long_std)
        return _rolling_rank(trend_zscore, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O023":
        # 60 日趋势斜率乘以趋势相关性的平方，相当于用 R^2 对斜率做质量加权。
        slope_60d = _rolling_log_slope(sox, window=60, min_periods=60)
        corr_60d = _rolling_time_corr(log_sox, window=60, min_periods=60)
        r2_weighted_slope = slope_60d * corr_60d.pow(2)
        return _rolling_rank(r2_weighted_slope, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O024":
        # 20/60/120 日三个趋势斜率分别排名后取平均，形成多窗口趋势综合信号。
        rank_20d = _rolling_rank(_rolling_log_slope(sox, window=20, min_periods=20), window=1250, min_periods=500)
        rank_60d = _rolling_rank(_rolling_log_slope(sox, window=60, min_periods=60), window=1250, min_periods=500)
        rank_120d = _rolling_rank(_rolling_log_slope(sox, window=120, min_periods=120), window=1250, min_periods=500)
        return (rank_20d + rank_60d + rank_120d) / 3 * 2 - 1

    if factor_id == "O025":
        # 韩国半导体出口金额同比：超过过去 36 个月均值 +/- 1 倍标准差后才计入。
        exports = read_prepared_series(OVERSEA_MONTHLY_TABLE, KOREA_SEMI_EXPORT_COL)
        export_yoy = exports / exports.shift(12) - 1
        yoy_mean = export_yoy.rolling(window=36, min_periods=24).mean()
        yoy_std = export_yoy.rolling(window=36, min_periods=24).std()
        deviation = export_yoy - yoy_mean
        return _safe_divide(np.sign(deviation) * (deviation.abs() - yoy_std).clip(lower=0), yoy_std)

    raise KeyError(f"Unsupported factor_id: {factor_id}")


def generate_overseaFactors_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """生成所有 overseaFactors 原始因子列。

    data_df 只提供项目统一的日期索引；每个因子的真实数据源在 _calc_oversea_factor()
    内按需读取。_register_factor 会把外部源序列对齐到 data_df 的日期索引。
    """
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    by_factor_id = {str(record["factor_id"]): record for record in records}
    for factor_id in FACTOR_IDS:
        if factor_id not in by_factor_id:
            continue
        factor_series = _calc_oversea_factor(factor_id)
        # 注册时使用 {factor_id}_raw 作为临时列名；factor_utils 内部会落到正式 factor_id 列。
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"overseaFactors factor columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], records


def generate_overseaFactors_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    """给 build_factor_matrix.py 等上层入口调用的轻量接口，只返回源因子矩阵。"""
    factor_source_df, _records = generate_overseaFactors_factors(data_df)
    return factor_source_df


def _print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
    """打印生成结果摘要，用于人工检查非空数量和首尾有效日期。"""
    print(f"{label} mounted_normalized_factor_df shape:", mounted_factor_df.shape)
    print(f"{label} signal_ls_df shape:", signal_ls_df.shape)
    print(f"{label} factor columns:", list(mounted_factor_df.columns))
    print(f"{label} factor non-null summary:")
    for factor_col in mounted_factor_df.columns:
        series = mounted_factor_df[factor_col]
        print(
            factor_col,
            "non_na=", int(series.notna().sum()),
            "first=", series.first_valid_index(),
            "last=", series.last_valid_index(),
        )


def main() -> None:
    """命令行入口：海外因子的公式和 metadata 留在本模块，公共后处理交给 runner。"""
    # initial_factors.py 没有独立 main，本轮不接入；这里只迁移已有完整入口的海外因子模块。
    run_factor_module_pipeline(
        output_prefix=OUTPUT_PREFIX,
        generate_factors=generate_overseaFactors_factors,
        metadata_builder=metadata_from_overseaFactors_records,
        print_summary=_print_factor_output_summary,
    )


if __name__ == "__main__":
    main()
