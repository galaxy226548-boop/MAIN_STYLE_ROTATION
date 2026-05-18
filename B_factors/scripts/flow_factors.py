"""Flow/macro/bond factors from working_multiple_factors_plan.json."""

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
    _load_macro_all,
    _register_factor,
    _rolling_quantile_rank_year,
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


OUTPUT_PREFIX = "flow_factors"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

FACTOR_IDS = [
    "C001",
    "C002",
    "C003",
    "C004",
    "C005",
    "C009",
    "C012",
    "C015",
    "C017",
    "C018",
    "C019",
    "C024",
    "C026",
    "C030",
    "C031",
    "C032",
    "D009",
    "D010",
    "D011",
    "D012",
    "D013",
    "D014",
    "D015",
    "D016",
    "D017",
    "D018",
    "D019",
    "D020",
    "D021",
    "D022",
    "D023",
    "D025",
]

DATA_FIELD_REPLACEMENTS = {
    "新成立基金份额:偏股混合型基金": "新成立基金份额:混合型基金:偏股混合型基金",
}

FACTOR_FIELD_REPLACEMENTS = {
    "C026": {
        "中债国债到期收益率:1年": "中债国债到期收益率:10年",
        "美国国债收益率:1年": "美国:国债收益率:10年",
    },
    "D009": {
        "中债企业债到期收益率(AAA):5年": "中债中短期票据到期收益率(AAA):5年",
    },
}


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
            factor_id = record.get("factor_id")
            if factor_id not in FACTOR_IDS:
                continue
            if _is_unknown_or_todo(record.get("docu")) or _is_unknown_or_todo(record.get("data_field")):
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


def _record_with_actual_fields(record: dict[str, object]) -> dict[str, object]:
    factor_id = str(record["factor_id"])
    original_data_field = _normalize_plan_text(record.get("data_field"))
    fields = [field.strip() for field in original_data_field.split(",")]
    replacements = dict(DATA_FIELD_REPLACEMENTS)
    replacements.update(FACTOR_FIELD_REPLACEMENTS.get(factor_id, {}))
    actual_fields = [replacements.get(field, field) for field in fields]

    if not _normalize_plan_text(record.get("paper_id")):
        record["paper_id"] = "DIY"

    if actual_fields != fields:
        replacement_text = "; ".join(
            f"{old} -> {new}" for old, new in zip(fields, actual_fields, strict=True) if old != new
        )
        notes = _normalize_plan_text(record.get("notes"))
        suffix = f"本脚本因本地 prepared 数据口径可得性，实际使用字段替代：{replacement_text}。原 data_field：{original_data_field}。"
        record["notes"] = f"{notes} {suffix}".strip()
        record["data_field"] = ", ".join(actual_fields)

    return record


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
    return Path(path_text).name


def _load_china_macro_keyword_series(keyword: str, value_col: str = "今值") -> pd.Series:
    macro = _load_macro_all()
    mask = macro["国家/地区"].eq("中国")
    mask &= macro["指标名称"].astype(str).str.contains(keyword, regex=False, na=False)
    out = macro.loc[mask].copy()
    if out.empty:
        raise ValueError(f"macro.parquet 中找不到中国宏观指标：{keyword!r}")
    if value_col not in out.columns:
        raise KeyError(f"macro.parquet 中找不到字段 {value_col!r}; available={list(out.columns)}")

    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out[out["日期"].notna()].copy()
    sort_cols = [col for col in ["日期", "来源文件", "来源sheet", "文件年月"] if col in out.columns]
    out = out.sort_values(sort_cols, na_position="first")
    percent_hint = (
        out["指标名称"].astype(str).str.contains("%", regex=False, na=False).any()
        or out[value_col].astype(str).str.contains("%", regex=False, na=False).any()
    )
    series = pd.Series(
        _as_numeric(out[value_col], percent_hint=percent_hint).to_numpy(),
        index=out["日期"],
        name=keyword,
        dtype="float64",
    ).sort_index()
    return series[~series.index.duplicated(keep="last")]


def _load_record_series(record: dict[str, object]) -> list[pd.Series]:
    table_name = _table_name_from_docu(record.get("docu"))
    fields = [field.strip() for field in _normalize_plan_text(record.get("data_field")).split(",")]
    if table_name == "macro.parquet":
        return [_load_china_macro_keyword_series(field) for field in fields]
    return [read_prepared_series(table_name, field) for field in fields]


def _event_high_low_magnitude(series: pd.Series, window: int) -> pd.Series:
    previous_high = series.rolling(window=window, min_periods=window).max().shift(1)
    previous_low = series.rolling(window=window, min_periods=window).min().shift(1)
    out = pd.Series(0.0, index=series.index, dtype="float64")
    high_mask = series > previous_high
    low_mask = series < previous_low
    out.loc[high_mask] = series.loc[high_mask] - previous_high.loc[high_mask]
    out.loc[low_mask] = previous_low.loc[low_mask] - series.loc[low_mask]
    out.loc[previous_high.isna() | previous_low.isna()] = np.nan
    return out


def _new_high_low_direction(series: pd.Series, window: int) -> pd.Series:
    previous_high = series.rolling(window=window, min_periods=window).max().shift(1)
    previous_low = series.rolling(window=window, min_periods=window).min().shift(1)
    out = pd.Series(0.0, index=series.index, dtype="float64")
    out.loc[series > previous_high] = 1.0
    out.loc[series < previous_low] = -1.0
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


def _credit_spread_mean_gap(sub_1: pd.Series, sub_2: pd.Series, window: int = 9) -> pd.Series:
    spread = sub_1 - sub_2
    return spread.rolling(window=window, min_periods=window).mean() - spread


def _credit_spread_quantile(sub_1: pd.Series, sub_2: pd.Series) -> pd.Series:
    spread_mean = (sub_1 - sub_2).rolling(window=5, min_periods=5).mean()
    return 0.5 - _rolling_quantile_rank_year(spread_mean, year=2)


def _pct_change_zscore(series: pd.Series, window: int = 12) -> pd.Series:
    return calc_rolling_zscore(series / series.shift(1) - 1, window=window)


def _calc_factor(factor_id: str, series_list: list[pd.Series]) -> pd.Series:
    sub_1 = series_list[0]
    sub_2 = series_list[1] if len(series_list) > 1 else None

    if factor_id == "C001":
        return calc_rolling_zscore(sub_1, window=12)
    if factor_id == "C002":
        return _event_high_low_magnitude(sub_1, window=6)
    if factor_id in {"C003", "C017", "D022"}:
        return sub_1 - sub_1.rolling(window=3, min_periods=3).mean()
    if factor_id == "C004":
        return sub_1 - sub_1.rolling(window=6, min_periods=6).mean()
    if factor_id == "C005":
        return sub_1 - sub_1.shift(1)
    if factor_id == "C009":
        return calc_rolling_zscore(sub_1, window=6) * -1
    if factor_id == "C012":
        return sub_1.shift(1) - sub_1
    if factor_id in {"C015", "C024"}:
        return calc_rolling_zscore(sub_1, window=6)
    if factor_id == "C018":
        return sub_1 - sub_1.rolling(window=3, min_periods=3).mean()
    if factor_id == "C019":
        return sub_1.rolling(window=12, min_periods=12).mean() - sub_1
    if factor_id == "C026":
        if sub_2 is None:
            raise ValueError("C026 requires two series")
        spread = sub_1 - sub_2
        return spread.shift(3).rolling(window=6, min_periods=6).mean() - spread.rolling(window=3, min_periods=3).mean()
    if factor_id == "C030":
        return sub_1 - sub_1.rolling(window=6, min_periods=6).mean()
    if factor_id == "C031":
        return _pct_change_zscore(sub_1, window=12)
    if factor_id == "C032":
        if sub_2 is None:
            raise ValueError("C032 requires two series")
        return _pct_change_zscore(sub_1 + sub_2, window=12)
    if factor_id in {"D009", "D010", "D011", "D013"}:
        if sub_2 is None:
            raise ValueError(f"{factor_id} requires two series")
        return _credit_spread_mean_gap(sub_1, sub_2, window=9)
    if factor_id in {"D012", "D014", "D015"}:
        if sub_2 is None:
            raise ValueError(f"{factor_id} requires two series")
        return _credit_spread_quantile(sub_1, sub_2)
    if factor_id == "D016":
        if sub_2 is None:
            raise ValueError("D016 requires two series")
        spread = (sub_1 - sub_2).dropna()
        return spread.rolling(window=250, min_periods=250).mean() - spread.rolling(window=5, min_periods=5).mean()
    if factor_id in {"D017", "D018"}:
        if sub_2 is None:
            raise ValueError(f"{factor_id} requires two series")
        spread = sub_1 - sub_2
        return spread.shift(1) - spread
    if factor_id == "D019":
        return _new_high_low_direction(sub_1, window=6)
    if factor_id == "D020":
        return _bollinger_event(sub_1, window=6)
    if factor_id == "D021":
        return sub_1 - sub_1.rolling(window=2, min_periods=2).mean().shift(1)
    if factor_id == "D023":
        recent = (sub_1 * 2 + sub_1.shift(1)) / 3
        base = (sub_1.shift(12) + sub_1.shift(11) + sub_1.shift(10)) / 3
        return (recent / base - 1) * -1
    if factor_id == "D025":
        if sub_2 is None:
            raise ValueError("D025 requires two series")
        spread = sub_1 - sub_2
        return spread - spread.rolling(window=12, min_periods=12).mean()

    raise KeyError(f"Unsupported factor_id: {factor_id}")


def generate_flow_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
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
        raise ValueError(f"flow factor columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_flow_factors(data_df)
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
