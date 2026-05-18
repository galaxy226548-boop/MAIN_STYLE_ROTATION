"""Inflation and growth factors from working_multiple_factors_plan.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from factor_utils import (
    PROJECT_ROOT,
    MACRO_REVISION_POLICY_USE_NEXT_PREV,
    _as_numeric,
    _load_china_macro_level_series,
    _load_china_macro_series,
    _load_macro_all,
    _register_factor,
    build_threshold_signal_ls_df,
    calc_rolling_zscore,
    load_benchmark_index,
    load_default_data,
    mount_factor_source_frame,
    read_prepared_series,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "flationNgrowth_factors"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

FACTOR_IDS = [
    "F007",
    "F008",
    "F009",
    "F010",
    "F011",
    "F012",
    "F013",
    "F014",
    "F015",
    "F016",
    "F017",
    "F018",
    "F019",
    "F020",
    "F021",
    "F022",
    "F023",
    "F024",
    "F025",
    "F026",
    "F029",
    "G002",
    "G025",
    "G026",
    "G027",
    "G028",
    "G029",
    "G030",
    "G031",
    "G032",
    "G033",
    "G034",
    "G035",
    "G036",
    "G037",
    "G038",
    "G039",
    "G042",
    "G043",
    "G044",
    "G045",
    "G046",
    "G048",
    "G049",
    "G050",
    "G051",
    "G052",
    "G053",
    "G054",
    "G055",
    "G056",
    "G057",
    "G058",
    "G059",
    "G060",
    "G061",
    "G062",
    "G063",
    "G064",
    "G065",
    "G066",
    "G067",
    "G068",
    "G069",
    "G070",
]

SKIPPED_UNKNOWN_FACTOR_IDS = {"F027", "F028", "G040", "G041", "G047"}

TABLE_NAME_REPLACEMENTS = {
    "FactorF_monthly.parquet": "factorF_monthly.parquet",
    "FactorG_monthly.parquet": "factorG_monthly.parquet",
}

INFER_MISSING_ACTUAL_FROM_NEXT_PREV = True
MACRO_REVISION_POLICY = MACRO_REVISION_POLICY_USE_NEXT_PREV


def _normalize_plan_text(value: object) -> str:
    return str(value or "").strip()


def _is_unknown_or_todo(value: object) -> bool:
    return _normalize_plan_text(value).lower() in {"", "unknown", "todo"}


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id in SKIPPED_UNKNOWN_FACTOR_IDS:
                continue
            if factor_id not in FACTOR_IDS:
                continue
            if _is_unknown_or_todo(record.get("docu")) or _is_unknown_or_todo(record.get("data_field")):
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"working_multiple_factors_plan.json missing implemented records: {missing}")
    return records


def _metadata_from_plan_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _table_name_from_docu(docu: object) -> str:
    path_text = _normalize_plan_text(docu)
    if not path_text:
        raise ValueError("empty docu")
    table_name = Path(path_text).name
    return TABLE_NAME_REPLACEMENTS.get(table_name, table_name)


def _split_fields(data_field: object) -> list[str]:
    return [field.strip() for field in _normalize_plan_text(data_field).split(",") if field.strip()]


def _macro_calendar_series(
    keyword: str,
    *,
    value_col: str = "今值",
    level: bool = False,
    required_contains: str | list[str] | None = None,
    exclude_contains: str | list[str] | None = None,
) -> pd.Series:
    if level:
        if required_contains is not None or exclude_contains is not None:
            return _load_macro_level_series_filtered(
                keyword,
                value_col=value_col,
                required_contains=required_contains,
                exclude_contains=exclude_contains,
            )
        return _load_china_macro_level_series(
            keyword,
            value_col=value_col,
            infer_missing_actual_from_next_prev=INFER_MISSING_ACTUAL_FROM_NEXT_PREV,
            revision_policy=MACRO_REVISION_POLICY,
        )
    return _load_china_macro_series(
        keyword,
        value_col=value_col,
        required_contains=required_contains,
        exclude_contains=exclude_contains,
        infer_missing_actual_from_next_prev=INFER_MISSING_ACTUAL_FROM_NEXT_PREV,
        revision_policy=MACRO_REVISION_POLICY,
    )


def _load_macro_level_series_filtered(
    keyword: str,
    *,
    value_col: str = "今值",
    required_contains: str | list[str] | None = None,
    exclude_contains: str | list[str] | None = None,
) -> pd.Series:
    macro = _load_macro_all()
    mask = macro["国家/地区"].eq("中国")
    indicator_text = macro["指标名称"].astype(str)
    mask &= indicator_text.str.contains(keyword, na=False, regex=False)
    if required_contains is not None:
        required_list = [required_contains] if isinstance(required_contains, str) else list(required_contains)
        for item in required_list:
            mask &= indicator_text.str.contains(item, na=False, regex=False)
    if exclude_contains is not None:
        exclude_list = [exclude_contains] if isinstance(exclude_contains, str) else list(exclude_contains)
        for item in exclude_list:
            mask &= ~indicator_text.str.contains(item, na=False, regex=False)

    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"macro.parquet 中找不到中国宏观指标：{keyword!r}")
    if value_col not in out.columns:
        raise KeyError(f"macro.parquet 中找不到字段 {value_col!r}; available={list(out.columns)}")
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out[out["日期"].notna()].copy()
    sort_cols = [col for col in ["日期", "来源文件", "来源sheet", "文件年月"] if col in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    series = pd.Series(
        pd.to_numeric(out[value_col], errors="coerce").to_numpy(),
        index=out["日期"],
        name=keyword,
        dtype="float64",
    ).sort_index()
    return series[~series.index.duplicated(keep="last")]


def _load_macro_keyword_from_plan(field: str, *, value_col: str = "今值") -> pd.Series:
    if "官方制造业PMI" in field:
        return _macro_calendar_series("官方制造业PMI", value_col=value_col, level=True)
    if "工业增加值" in field:
        return _macro_calendar_series("工业增加值", value_col=value_col)
    if "出口金额:当月同比" in field:
        return _macro_calendar_series("出口金额:当月同比", value_col=value_col, exclude_contains="进出口")
    if "出口金额:当月值" in field:
        return _macro_calendar_series("出口金额:当月值", value_col=value_col, exclude_contains="进出口")
    if "进口金额:当月同比" in field:
        return _macro_calendar_series("进口金额:当月同比", value_col=value_col)
    if "社会消费品零售总额" in field:
        return _macro_calendar_series("社会消费品零售总额", value_col=value_col)
    if "季度GDP" in field:
        return _macro_calendar_series("GDP:当季同比", value_col=value_col)
    if "发电量" in field:
        return _macro_calendar_series("发电量", value_col=value_col)
    if "PPI" in field:
        return _macro_calendar_series("PPI:同比", value_col=value_col)
    if "CPI" in field:
        return _macro_calendar_series("CPI:同比", value_col=value_col)
    raise ValueError(f"Unsupported macro plan data_field: {field!r}")


def _load_record_series(record: dict[str, object]) -> list[pd.Series]:
    factor_id = str(record["factor_id"])
    table_name = _table_name_from_docu(record.get("docu"))
    fields = _split_fields(record.get("data_field"))

    if table_name == "macro.parquet":
        if factor_id in {"G033", "G042", "G046"}:
            keyword_field = fields[0]
            return [
                _load_macro_keyword_from_plan(keyword_field, value_col="今值"),
                _load_macro_keyword_from_plan(keyword_field, value_col="预测值"),
            ]
        return [_load_macro_keyword_from_plan(fields[0])]

    return [read_prepared_series(table_name, field) for field in fields]


def _new_high_low_magnitude(series: pd.Series, window: int) -> pd.Series:
    previous_high = series.rolling(window=window, min_periods=window).max().shift(1)
    previous_low = series.rolling(window=window, min_periods=window).min().shift(1)
    out = pd.Series(0.0, index=series.index, dtype="float64")
    high_mask = series > previous_high
    low_mask = series < previous_low
    out.loc[high_mask] = series.loc[high_mask] - previous_high.loc[high_mask]
    out.loc[low_mask] = previous_low.loc[low_mask] - series.loc[low_mask]
    out.loc[previous_high.isna() | previous_low.isna()] = np.nan
    return out


def _bollinger_event(series: pd.Series, window: int = 6) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    lower = rolling_mean - 2 * rolling_std
    upper = rolling_mean + 2 * rolling_std
    out = pd.Series(0.0, index=series.index, dtype="float64")
    below_mask = series < lower
    above_mask = series > upper
    out.loc[below_mask] = lower.loc[below_mask] - series.loc[below_mask]
    out.loc[above_mask] = upper.loc[above_mask] - series.loc[above_mask]
    out.loc[rolling_mean.isna() | rolling_std.isna()] = np.nan
    return out


def _growth_new_high_low_with_forecast(actual: pd.Series, forecast: pd.Series, window: int = 6) -> pd.Series:
    previous_high = actual.rolling(window=window, min_periods=window).max().shift(1)
    previous_low = actual.rolling(window=window, min_periods=window).min().shift(1)
    out = pd.Series(0.0, index=actual.index.union(forecast.index).sort_values(), dtype="float64")
    actual = actual.reindex(out.index)
    forecast = forecast.reindex(out.index)
    previous_high = previous_high.reindex(out.index)
    previous_low = previous_low.reindex(out.index)
    high_mask = (actual > previous_high) & (actual > forecast)
    low_mask = (actual < previous_low) & (actual < forecast)
    out.loc[high_mask] = actual.loc[high_mask] - previous_high.loc[high_mask]
    out.loc[low_mask] = previous_low.loc[low_mask] - actual.loc[low_mask]
    out.loc[previous_high.isna() | previous_low.isna()] = np.nan
    return out


def _export_expectation_reversal(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    aligned = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).sort_index()
    actual_diff = aligned["actual"] - aligned["actual"].shift(1)
    forecast_diff = aligned["forecast"] - aligned["forecast"].shift(1)
    out = pd.Series(0.0, index=aligned.index, dtype="float64")
    good_mask = (forecast_diff < 0) & (actual_diff > 0)
    bad_mask = (forecast_diff > 0) & (actual_diff < 0)
    out.loc[good_mask] = (-forecast_diff + actual_diff).loc[good_mask]
    out.loc[bad_mask] = (forecast_diff - actual_diff).loc[bad_mask] * -1
    out.loc[actual_diff.isna() | forecast_diff.isna()] = np.nan
    return out


def _pmi_expectation_event(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    aligned = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).sort_index()
    out = pd.Series(0.0, index=aligned.index, dtype="float64")
    low_mask = (aligned["actual"] < 49) & (aligned["actual"] < aligned["forecast"])
    high_mask = (aligned["actual"] > 51) & (aligned["actual"] > aligned["forecast"])
    out.loc[low_mask] = (aligned["forecast"] - aligned["actual"]).loc[low_mask]
    out.loc[high_mask] = (aligned["forecast"] - aligned["actual"]).loc[high_mask]
    out.loc[aligned.isna().any(axis=1)] = np.nan
    return out


def _norm_ppf_pmi_factor(series: pd.Series) -> pd.Series:
    probability = (series / 100.0).clip(0.001, 0.999)
    normalized = pd.Series(norm.ppf(probability), index=series.index, dtype="float64")
    upper = normalized.expanding(min_periods=12).quantile(0.66)
    lower = normalized.expanding(min_periods=12).quantile(0.33)
    out = pd.Series(0.0, index=series.index, dtype="float64")
    high_mask = normalized > upper
    low_mask = normalized < lower
    out.loc[high_mask] = normalized.loc[high_mask] - upper.loc[high_mask]
    out.loc[low_mask] = lower.loc[low_mask] - normalized.loc[low_mask]
    out.loc[upper.isna() | lower.isna()] = np.nan
    return out * -1


def _calc_factor(factor_id: str, series_list: list[pd.Series]) -> pd.Series:
    sub_1 = series_list[0]
    sub_2 = series_list[1] if len(series_list) > 1 else None

    if factor_id in {"F007", "F018", "G028", "G055", "G061", "G064"}:
        return sub_1
    if factor_id in {"F008", "F023", "G034", "G045"}:
        return _new_high_low_magnitude(sub_1, window=6)
    if factor_id in {"F009", "G002", "G051", "G057", "G066", "G070"}:
        return sub_1 - sub_1.rolling(window=3, min_periods=3).mean()
    if factor_id == "F010":
        return sub_1.shift(1) / sub_1 - 1
    if factor_id == "F011":
        return calc_rolling_zscore(sub_1 / sub_1.shift(1) - 1, window=12)
    if factor_id in {"F012", "F013", "F020", "G049", "G056"}:
        return sub_1 - sub_1.shift(1)
    if factor_id in {"F014", "F021", "G043", "G044", "G063"}:
        return (sub_1 - sub_1.shift(1)).shift(2)
    if factor_id in {"F015", "F022"}:
        return sub_1.shift(3) - sub_1.shift(2)
    if factor_id == "F016":
        return sub_1 - sub_1.rolling(window=12, min_periods=12).mean()
    if factor_id in {"F017", "G032", "G054"}:
        return sub_1.rolling(window=6, min_periods=6).mean() - sub_1.rolling(window=12, min_periods=12).mean()
    if factor_id == "F019":
        return sub_1 - sub_1.rolling(window=2, min_periods=2).mean()
    if factor_id in {"F024", "G067"}:
        out = _bollinger_event(sub_1, window=6)
        return out * -1 if factor_id == "G067" else out
    if factor_id == "F025":
        return sub_1.rolling(window=12, min_periods=12).mean() - sub_1.rolling(window=5, min_periods=5).mean()
    if factor_id == "F026":
        return sub_1.rolling(window=6, min_periods=6).mean() - sub_1.rolling(window=3, min_periods=3).mean()
    if factor_id == "F029":
        return sub_1 - sub_1.rolling(window=2, min_periods=2).mean()
    if factor_id == "G025":
        return _norm_ppf_pmi_factor(sub_1)
    if factor_id == "G026":
        return sub_1.rolling(window=36, min_periods=36).mean() - sub_1.rolling(window=3, min_periods=3).mean()
    if factor_id == "G027":
        return calc_rolling_zscore((sub_1 - 50) * -1, window=6)
    if factor_id == "G029":
        return sub_1.sub(50).rolling(window=3, min_periods=3).mean()
    if factor_id == "G030":
        return sub_1.sub(50).rolling(window=12, min_periods=12).mean()
    if factor_id == "G031":
        return sub_1 / sub_1.shift(12) - 1
    if factor_id == "G033":
        if sub_2 is None:
            raise ValueError("G033 requires forecast series")
        return _pmi_expectation_event(sub_1, sub_2)
    if factor_id == "G035":
        return sub_1.shift(1) - sub_1
    if factor_id == "G036":
        rolling_mean = sub_1.rolling(window=3, min_periods=3).mean()
        return rolling_mean.shift(1) - rolling_mean
    if factor_id == "G037":
        rolling_mean = sub_1.rolling(window=3, min_periods=3).mean()
        return rolling_mean.shift(3) - rolling_mean
    if factor_id == "G038":
        return sub_1.rolling(window=8, min_periods=8).mean() - sub_1.rolling(window=16, min_periods=16).mean()
    if factor_id == "G039":
        return sub_1.diff() * -1
    if factor_id == "G042":
        if sub_2 is None:
            raise ValueError("G042 requires forecast series")
        return _export_expectation_reversal(sub_1, sub_2)
    if factor_id == "G046":
        if sub_2 is None:
            raise ValueError("G046 requires forecast series")
        return _growth_new_high_low_with_forecast(sub_1, sub_2, window=6)
    if factor_id == "G048":
        return sub_1 * -1
    if factor_id == "G050":
        return sub_1.diff() * -1
    if factor_id == "G052":
        return (sub_1 - sub_1.shift(1)).shift(3)
    if factor_id == "G053":
        return sub_1.shift(4) - sub_1.shift(3)
    if factor_id == "G058":
        return sub_1.shift(2)
    if factor_id == "G059":
        return sub_1 - sub_1.rolling(window=9, min_periods=9).mean()
    if factor_id == "G060":
        return (sub_1 - sub_1.shift(1)).shift(2)
    if factor_id == "G062":
        return sub_1 / sub_1.shift(1) - 1
    if factor_id == "G065":
        return sub_1 / sub_1.shift(1) - 1
    if factor_id == "G068":
        return _new_high_low_magnitude(sub_1, window=6) * -1
    if factor_id == "G069":
        if sub_2 is None:
            raise ValueError("G069 requires two PMI component series")
        return (sub_1 - 50) - (sub_2 - 50)

    raise KeyError(f"Unsupported factor_id: {factor_id}")


def generate_flationNgrowth_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    by_factor_id = {str(record["factor_id"]): record for record in records}
    for factor_id in FACTOR_IDS:
        record = by_factor_id[factor_id]
        factor_series = _calc_factor(factor_id, _load_record_series(record))
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"flationNgrowth factor columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_flationNgrowth_factors(data_df)
    metadata = _metadata_from_plan_records(selected_records)
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
