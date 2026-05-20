"""Oversea style-rotation factors from working_multiple_factors_plan.json."""

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
    build_threshold_signal_ls_df,
    calc_rolling_zscore,
    load_benchmark_index,
    load_default_data,
    load_prepared_table,
    mount_factor_source_frame,
    read_prepared_series,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "overseaFactors"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

FACTOR_IDS = [
    "O011",
    "O012",
    "O013",
    "O014",
    "O015",
    "O016",
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

VIX_FILE = "VIX.GI-行情统计-20260509.xlsx"
SPX_FILE = "SPX.GI-行情统计-20260519.xlsx"
NDX_FILE = "NDX.GI-行情统计-20260519.xlsx"
EXCHANGE_TABLE = "exchange_rate_daily.parquet"
OVERSEA_DAILY_TABLE = "factorO_daily.parquet"
OVERSEA_MONTHLY_TABLE = "factorO_monthly.parquet"

HKD_SPOT_COL = "即期汇率定盘价:美元兑港元"
USDCNH_COL = "即期汇率:美元兑离岸人民币(USDCNH)"
HKD_SWAP_1M_COL = "掉期点:美元兑港元:1个月"
HIBOR_1M_COL = "HIBOR:1个月"
SOFR_1M_COL = "美国:SOFR期限利率:1个月"
BDI_COL = "波罗的海干散货指数(BDI)"
SOX_COL = "费城半导体指数"
KOREA_SEMI_EXPORT_COL = "韩国:出口金额:半导体:当月值"

ANNUAL_TRADING_DAYS = 252


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in FACTOR_IDS:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            records.append(_record_with_actual_fields(item))

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"working_multiple_factors_plan.json missing implemented records: {missing}")
    return records


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _record_with_actual_fields(record: dict[str, object]) -> dict[str, object]:
    factor_id = str(record["factor_id"])
    if not _normalize_plan_text(record.get("paper_id")):
        record["paper_id"] = "DIY"

    if factor_id in {"O015", "O018", "O020", "O021", "O022", "O023", "O024"}:
        record["docu"] = OVERSEA_DAILY_TABLE
    elif factor_id == "O025":
        record["docu"] = OVERSEA_MONTHLY_TABLE
    elif factor_id in {"O012", "O013", "O014", "O016"}:
        record["docu"] = EXCHANGE_TABLE

    if factor_id in {"O011", "O016", "O017"}:
        _append_note(record, "计划表 condition 存在语法残缺，本脚本按 factor/calc_method/data_field 的业务含义实现。")
    if factor_id == "O013":
        _append_note(record, "本脚本只取完整月份的月末源数据作为事件日，不使用未完整月份。")
    return record


def metadata_from_overseaFactors_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _clean_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


def _load_price_file_close(file_name: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if "交易日期" not in df.columns or "收盘价" not in df.columns:
        raise KeyError(f"{file_name} must contain 交易日期 and 收盘价; available={list(df.columns)}")
    return _positive_series(_as_float_series(df["收盘价"], _clean_date_series(df["交易日期"]), name))


def _positive_series(series: pd.Series) -> pd.Series:
    s = series.astype("float64").copy()
    return s.where(s > 0)


def _clean_positive_limited_ffill_series(series: pd.Series, limit: int = 5) -> pd.Series:
    s = series.astype("float64").sort_index(ascending=True)
    return s.where(s > 0).ffill(limit=limit)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / denom


def _signed_excess_score(value: pd.Series, threshold: float | pd.Series) -> pd.Series:
    threshold_values = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=value.index)
    score = np.sign(value) * (value.abs() - threshold_values).clip(lower=0)
    return score.astype("float64")


def _rolling_std_breakout(series: pd.Series, window: int, min_periods: int, std_multiplier: float = 1.0) -> pd.Series:
    s = series.astype("float64").sort_index()
    rolling_mean = s.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = s.rolling(window=window, min_periods=min_periods).std()
    deviation = s - rolling_mean
    return _safe_divide(
        np.sign(deviation) * (deviation.abs() - std_multiplier * rolling_std).clip(lower=0),
        rolling_std,
    )


def _rolling_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = series.astype("float64").dropna().sort_index()

    def rank_last(window_values: np.ndarray) -> float:
        window_series = pd.Series(window_values).dropna()
        if window_series.empty:
            return np.nan
        return float(window_series.rank(pct=True).iloc[-1])

    return s.rolling(window=window, min_periods=min_periods).apply(rank_last, raw=True)


def _rolling_log_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = series.astype("float64").dropna().sort_index()
    log_s = np.log(s.where(s > 0))

    def slope_last(window_values: np.ndarray) -> float:
        y = pd.Series(window_values, dtype="float64")
        valid = y.notna()
        if int(valid.sum()) < min_periods:
            return np.nan
        y_valid = y.loc[valid].to_numpy()
        x = np.arange(len(y), dtype="float64")[valid.to_numpy()]
        x_demeaned = x - float(x.mean())
        denominator = float((x_demeaned ** 2).sum())
        if np.isclose(denominator, 0.0):
            return np.nan
        y_demeaned = y_valid - float(y_valid.mean())
        return float((x_demeaned * y_demeaned).sum() / denominator)

    return log_s.rolling(window=window, min_periods=min_periods).apply(slope_last, raw=True)


def _rolling_time_corr(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = series.astype("float64").dropna().sort_index()
    time_index = pd.Series(np.arange(len(s), dtype="float64"), index=s.index)
    return s.rolling(window=window, min_periods=min_periods).corr(time_index)


def _complete_month_end_values(series: pd.Series) -> pd.Series:
    s = series.astype("float64").dropna().sort_index()
    if s.empty:
        return s
    month_end_values = s.groupby(s.index.to_period("M")).tail(1)
    today = pd.Timestamp.today().normalize()
    latest_period = month_end_values.index[-1].to_period("M")
    if latest_period == today.to_period("M") and month_end_values.index[-1].normalize() < today + pd.offsets.MonthEnd(0):
        month_end_values = month_end_values.iloc[:-1]
    return month_end_values


def _calc_oversea_factor(factor_id: str) -> pd.Series:
    if factor_id == "O011":
        vix_close = _load_price_file_close(VIX_FILE, "VIX")
        vix_ret_20d = vix_close / vix_close.shift(20) - 1
        vix_zscore = calc_rolling_zscore(vix_ret_20d, window=ANNUAL_TRADING_DAYS, min_periods=120)
        return _signed_excess_score(vix_zscore, 2.0)

    if factor_id == "O012":
        hkd_spot = _positive_series(read_prepared_series(EXCHANGE_TABLE, HKD_SPOT_COL))
        spot_mean_20d = hkd_spot.rolling(window=20, min_periods=20).mean()
        signal = pd.Series(0.0, index=spot_mean_20d.index, dtype="float64")
        signal.loc[spot_mean_20d < 7.77] = (7.77 - spot_mean_20d.loc[spot_mean_20d < 7.77]) / 0.05
        signal.loc[spot_mean_20d > 7.83] = (7.83 - spot_mean_20d.loc[spot_mean_20d > 7.83]) / 0.05
        signal.loc[spot_mean_20d.isna()] = np.nan
        return signal

    if factor_id == "O013":
        swap_points = _complete_month_end_values(read_prepared_series(EXCHANGE_TABLE, HKD_SWAP_1M_COL))
        return _rolling_std_breakout(
            swap_points,
            window=ANNUAL_TRADING_DAYS,
            min_periods=ANNUAL_TRADING_DAYS,
            std_multiplier=1.0,
        )

    if factor_id == "O014":
        usdcnh = _positive_series(read_prepared_series(EXCHANGE_TABLE, USDCNH_COL)).sort_index(ascending=True)
        usdcnh_ret_20d = usdcnh / usdcnh.shift(20) - 1
        return _signed_excess_score(usdcnh_ret_20d, 0.015) * -1

    if factor_id == "O015":
        bdi = read_prepared_series(OVERSEA_DAILY_TABLE, BDI_COL)
        return _rolling_std_breakout(bdi, window=120, min_periods=120, std_multiplier=2.0) * -1

    if factor_id == "O016":
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

    if factor_id == "O017":
        ndx_close = _load_price_file_close(NDX_FILE, "NDX")
        spx_close = _load_price_file_close(SPX_FILE, "SPX")
        excess_return = (ndx_close / ndx_close.shift(20) - 1) - (spx_close / spx_close.shift(20) - 1)
        return _signed_excess_score(excess_return.dropna(), 0.03)

    if factor_id == "O018":
        sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
        sox_ret_20d = sox / sox.shift(20) - 1
        return _signed_excess_score(sox_ret_20d, 0.05)

    if factor_id == "O019":
        sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
        spx_close = _load_price_file_close(SPX_FILE, "SPX")
        excess_return = (sox / sox.shift(20) - 1) - (spx_close / spx_close.shift(20) - 1)
        return _signed_excess_score(excess_return.dropna(), 0.05)

    sox = _clean_positive_limited_ffill_series(read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL))
    log_sox = np.log(sox.where(sox > 0))
    if factor_id == "O020":
        slope_60d = _rolling_log_slope(sox, window=60, min_periods=60)
        return _rolling_rank(slope_60d, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O021":
        slope_acceleration = (
            _rolling_log_slope(sox, window=20, min_periods=20)
            - _rolling_log_slope(sox, window=120, min_periods=120)
        )
        return _rolling_rank(slope_acceleration, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O022":
        log_sox_valid = log_sox.dropna()
        long_mean = log_sox_valid.rolling(window=250, min_periods=250).mean()
        long_std = log_sox_valid.rolling(window=250, min_periods=250).std()
        trend_zscore = _safe_divide(log_sox_valid - long_mean, long_std)
        return _rolling_rank(trend_zscore, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O023":
        slope_60d = _rolling_log_slope(sox, window=60, min_periods=60)
        corr_60d = _rolling_time_corr(log_sox, window=60, min_periods=60)
        r2_weighted_slope = slope_60d * corr_60d.pow(2)
        return _rolling_rank(r2_weighted_slope, window=1250, min_periods=500) * 2 - 1
    if factor_id == "O024":
        rank_20d = _rolling_rank(_rolling_log_slope(sox, window=20, min_periods=20), window=1250, min_periods=500)
        rank_60d = _rolling_rank(_rolling_log_slope(sox, window=60, min_periods=60), window=1250, min_periods=500)
        rank_120d = _rolling_rank(_rolling_log_slope(sox, window=120, min_periods=120), window=1250, min_periods=500)
        return (rank_20d + rank_60d + rank_120d) / 3 * 2 - 1

    if factor_id == "O025":
        exports = read_prepared_series(OVERSEA_MONTHLY_TABLE, KOREA_SEMI_EXPORT_COL)
        export_yoy = exports / exports.shift(12) - 1
        yoy_mean = export_yoy.rolling(window=36, min_periods=24).mean()
        yoy_std = export_yoy.rolling(window=36, min_periods=24).std()
        deviation = export_yoy - yoy_mean
        return _safe_divide(np.sign(deviation) * (deviation.abs() - yoy_std).clip(lower=0), yoy_std)

    raise KeyError(f"Unsupported factor_id: {factor_id}")


def generate_overseaFactors_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
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
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"overseaFactors factor columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], records


def generate_overseaFactors_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    factor_source_df, _records = generate_overseaFactors_factors(data_df)
    return factor_source_df


def _print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
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
    validate_prepared_mapping()
    data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

    factor_source_df, selected_records = generate_overseaFactors_factors(data_df)
    metadata = metadata_from_overseaFactors_records(selected_records)
    mounted_normalized_factor_df = mount_factor_source_frame(
        factor_source_df=factor_source_df,
        market_df=market_df,
        benchmark_index=benchmark_index,
        metadata=metadata,
    )
    signal_ls_df = build_threshold_signal_ls_df(mounted_normalized_factor_df, metadata)
    output_paths = save_factor_outputs(
        mounted_normalized_factor_df=mounted_normalized_factor_df,
        signal_ls_df=signal_ls_df,
        missing_bar_defaults=[],
        output_prefix=OUTPUT_PREFIX,
        write_empty_missing_bar_file=False,
    )

    for label, path in output_paths.items():
        print(f"{label} saved to:", path)
    generated_path = save_generated_factor_records(selected_records, OUTPUT_PREFIX)
    print("generated records saved to:", generated_path)
    _print_factor_output_summary(OUTPUT_PREFIX, mounted_normalized_factor_df, signal_ls_df)


if __name__ == "__main__":
    main()
