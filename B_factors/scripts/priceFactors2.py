"""Additional price and valuation factors from completed V071-V142 plan files."""

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
    _as_numeric,
    _register_factor,
    build_threshold_signal_ls_df,
    calc_rolling_zscore,
    load_benchmark_index,
    load_default_data,
    load_prepared_table,
    mount_factor_source_frame,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)
from priceFactors1 import _as_float_series, _load_growth_close, _load_value_close


OUTPUT_PREFIX = "priceFactors2"
PLAN_PATHS = [
    PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 1.json",
    PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan_completed_V071_V142 2.json",
]

FACTOR_IDS = [
    "V136",
    "V137",
    "V138",
    "V139",
    "V140",
    "V141",
    "V142",
]

CSI_ALL_PRICE_FILE = "中证全指(000985.CSI)-历史价格.xlsx"
GROWTH_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx"
VALUE_VALUATION_FILE = "D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx"

# Local substitutes for the five industry PE series requested by V140/V141.
# The project has prepared index valuation files but no reliable stock-to-industry
# mapping for the original Shanghai industry ETF universe.
INDUSTRY_VALUATION_FILES = [
    "399314.SZ-历史PE-PB-20260509.xlsx",
    "399316.SZ-历史PE-PB-20260509.xlsx",
    "801811.SI-历史PE-PB-20260509.xlsx",
    "801813.SI-历史PE-PB-20260509.xlsx",
    "932000.CSI-历史PE-PB-20260509.xlsx",
]


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _append_note(record: dict[str, object], note: str) -> None:
    existing = _normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def _load_plan_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    wanted = set(FACTOR_IDS)
    for plan_path in PLAN_PATHS:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        source_file = str(plan_path.relative_to(PROJECT_ROOT))
        for sheet_name, sheet_meta in payload.get("sheets", {}).items():
            for record in sheet_meta.get("records", []):
                factor_id = str(record.get("factor_id") or "")
                if factor_id not in wanted:
                    continue
                item = dict(record)
                item["_source_file"] = source_file
                item["_source_sheet"] = sheet_name
                item["signal_type"] = _normalize_plan_text(item.get("signal_type")) or "state"
                if factor_id in {"V140", "V141"}:
                    _append_note(
                        item,
                        "本脚本使用 prepared_data 中五个指数估值文件的市盈率TTM均值作为行业PE近似；"
                        "原研报上证行业ETF分行业PE口径仍需人工确认。",
                    )
                if factor_id in {"V138", "V139", "V142"}:
                    _append_note(item, "本脚本按 event 因子规则使用 NaN 表示非触发日。")
                records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"completed V071-V142 plan files missing implemented records: {missing}")
    return sorted(records, key=lambda record: FACTOR_IDS.index(str(record["factor_id"])))


def metadata_from_priceFactors2_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _read_excel_series(file_name: str, date_col: str, value_col: str, name: str) -> pd.Series:
    df = load_prepared_table(file_name)
    if date_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"{file_name} must contain {date_col!r} and {value_col!r}; available={list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    values = _as_numeric(df[value_col])
    return _as_float_series(values, dates, name)


def _load_csi_all_amount() -> pd.Series:
    return _read_excel_series(CSI_ALL_PRICE_FILE, "交易日期", "成交额(元,CNY)", "csi_all_amount")


def _load_growth_pe() -> pd.Series:
    return _read_excel_series(GROWTH_VALUATION_FILE, "交易日期", "市盈率TTM", "growth_pe_ttm")


def _load_value_pe() -> pd.Series:
    return _read_excel_series(VALUE_VALUATION_FILE, "交易日期", "市盈率TTM", "value_pe_ttm")


def _rolling_rank_pct(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    s = series.astype("float64").sort_index()

    def rank_last(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) < min_periods:
            return np.nan
        return float((valid <= valid[-1]).sum() / len(valid))

    return s.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _month_end_last(series: pd.Series) -> pd.Series:
    s = series.astype("float64").dropna().sort_index()
    out = s.groupby(s.index.to_period("M")).last()
    out.index = s.groupby(s.index.to_period("M")).apply(lambda group: group.index[-1]).to_numpy()
    return out.sort_index()


def _load_industry_pe_proxy() -> pd.Series:
    series_list = [
        _read_excel_series(file_name, "交易日期", "市盈率TTM", Path(file_name).stem)
        for file_name in INDUSTRY_VALUATION_FILES
    ]
    aligned = pd.concat(series_list, axis=1, sort=True)
    return aligned.mean(axis=1, skipna=True).dropna().rename("industry_pe_proxy")


def _calc_v136(growth_close: pd.Series, value_close: pd.Series) -> pd.Series:
    growth_ret = growth_close.pct_change(fill_method=None)
    value_ret = value_close.pct_change(fill_method=None)
    rel_vol = growth_ret.rolling(60, min_periods=60).std() / value_ret.rolling(60, min_periods=60).std()
    upper = rel_vol.rolling(252, min_periods=252).quantile(0.95)
    lower = rel_vol.rolling(252, min_periods=252).quantile(0.05)

    rel_nav = growth_close / value_close
    ma5_close = rel_nav.rolling(5, min_periods=5).mean()
    ma20_close = rel_nav.rolling(20, min_periods=20).mean()

    rel_vol_z = calc_rolling_zscore(rel_vol, window=252)
    upper_z = calc_rolling_zscore(upper, window=252)
    lower_z = calc_rolling_zscore(lower, window=252)
    ma_diff_z = calc_rolling_zscore(ma5_close - ma20_close, window=252)

    factor = pd.Series(0.0, index=rel_vol.index, dtype="float64")
    cond_value_crowded = (rel_vol < lower) & (ma5_close > ma20_close)
    factor.loc[cond_value_crowded] = (
        (lower_z - rel_vol_z).loc[cond_value_crowded].abs()
        * ma_diff_z.loc[cond_value_crowded].abs()
    )

    cond_growth_crowded = (rel_vol > upper) & (ma5_close < ma20_close)
    factor.loc[cond_growth_crowded] = -(
        (rel_vol_z - upper_z).loc[cond_growth_crowded].abs()
        * ma_diff_z.loc[cond_growth_crowded].abs()
    )
    factor.loc[rel_vol.isna() | upper.isna() | lower.isna()] = np.nan
    return factor


def _calc_v137() -> pd.Series:
    amount = _load_csi_all_amount()
    raw = amount.rolling(63, min_periods=63).mean() - amount.rolling(504, min_periods=504).mean()
    return calc_rolling_zscore(raw, window=504)


def _calc_month_event(index: pd.DatetimeIndex, month: int) -> pd.Series:
    dates = pd.DatetimeIndex(index).sort_values()
    factor = pd.Series(np.nan, index=dates, dtype="float64")
    is_month = dates.month == month
    progress = pd.Series(dates.day / dates.days_in_month, index=dates, dtype="float64")
    factor.loc[is_month] = -progress.loc[is_month]
    return factor


def _calc_v140() -> pd.Series:
    industry_pe = _load_industry_pe_proxy()
    return _month_end_last(industry_pe)


def _calc_v141() -> pd.Series:
    sub_pe_monthly = _calc_v140()
    pe_mean = sub_pe_monthly.rolling(window=36, min_periods=36).mean()
    raw = sub_pe_monthly / pe_mean.replace(0.0, np.nan) - 1.0
    return calc_rolling_zscore(raw, window=36)


def _calc_v142() -> pd.Series:
    pe_ratio = _load_growth_pe() / _load_value_pe().replace(0.0, np.nan)
    quantile_rank = _rolling_rank_pct(pe_ratio, window=756, min_periods=756)
    raw = -(quantile_rank - 0.5)

    factor = pd.Series(np.nan, index=quantile_rank.index, dtype="float64")
    cond_value = quantile_rank > 0.8
    cond_growth = quantile_rank < 0.2
    factor.loc[cond_value | cond_growth] = raw.loc[cond_value | cond_growth]
    return factor


def generate_priceFactors2_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    growth_close = _load_growth_close()
    value_close = _load_value_close()

    _register_factor(raw_factor_df, factor_source_df, "V136_raw", _calc_v136(growth_close, value_close))
    _register_factor(raw_factor_df, factor_source_df, "V137_raw", _calc_v137())
    _register_factor(raw_factor_df, factor_source_df, "V138_raw", _calc_month_event(data_index, 6))
    _register_factor(raw_factor_df, factor_source_df, "V139_raw", _calc_month_event(data_index, 12))
    _register_factor(raw_factor_df, factor_source_df, "V140_raw", _calc_v140())
    _register_factor(raw_factor_df, factor_source_df, "V141_raw", _calc_v141())
    _register_factor(raw_factor_df, factor_source_df, "V142_raw", _calc_v142())

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"priceFactors2 columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], records


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

    factor_source_df, selected_records = generate_priceFactors2_factors(data_df)
    metadata = metadata_from_priceFactors2_records(selected_records)
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
