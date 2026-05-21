"""Additional profit and statement factors from working_multiple_factors_plan.json."""

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
    load_benchmark_index,
    load_default_data,
    load_prepared_table,
    mount_factor_source_frame,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)
from profitFactors import (
    ASHARE_PROFIT_TABLE,
    GROWTH_STYLE_INDEX,
    INDEX_STATEMENT_FILE,
    INDEX_WEIGHT_FILE,
    VALUE_STYLE_INDEX,
    _as_float_series,
    _calc_stock_ttm_yoy_change,
    _index_code_key,
    _index_statement_report_frame,
    _load_ashare_profit_metric,
    _load_parent_net_profit,
    _normalize_plan_text,
    _quarterly_ttm,
    _read_index_weights,
    _stock_code_key,
    _ttm_yoy,
    _weighted_average,
    _weighted_stock_metric_by_index,
)


OUTPUT_PREFIX = "profitFactors2"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

IMPLEMENTED_FACTOR_IDS = [
    "P033",
    "P035",
    "P037",
    "P038",
    "P039",
    "P040",
    "P041",
    "P042",
    "P043",
    "P044",
    "P045",
    "P046",
    "P047",
    "P049",
    "P050",
    "P051",
    "P052",
    "P053",
    "P054",
    "P055",
    "P056",
    "P057",
    "P058",
    "P059",
    "P060",
]
FACTOR_IDS = IMPLEMENTED_FACTOR_IDS

UNIMPLEMENTED_FACTORS = {
    "P036": "计划注明 Wind 全A非金融、石油石化 ROE 数据本地不可获得，未用代理口径伪造。",
    "P048": "计划中 EPS、25日前 EPS、当日收盘价数据字段均为 unknown，未用季度 EPS 伪造日频预期因子。",
}

CSI_ALL_INDEX = "000985.SH"
CONSENSUS_FILES = {
    "P046": ("Con_np_change_60d.feather", "Con_np_change_60d"),
    "P047": ("Con_np_change_20d.feather", "Con_np_change_20d"),
    "P051": ("Con_np_change_60d.feather", "Con_np_change_60d"),
}

_INDEX_STATEMENT_CACHE: dict[tuple[str, ...], pd.DataFrame] = {}
_WEIGHTS_CACHE: dict[tuple[str, ...], pd.DataFrame] = {}
_PUBDATE_MAP_CACHE: pd.DataFrame | None = None


def _load_plan_records() -> list[dict[str, object]]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    source_file = str(PLAN_PATH.relative_to(PROJECT_ROOT))
    wanted = set(IMPLEMENTED_FACTOR_IDS)
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in wanted:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in IMPLEMENTED_FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"working_multiple_factors_plan.json missing implemented records: {missing}")
    return records


def metadata_from_profitFactors2_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "event",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _load_index_statement(columns: list[str]) -> pd.DataFrame:
    required = ["Indexcd", "Date", "PubDate"]
    usecols = tuple(dict.fromkeys(required + columns))
    if usecols not in _INDEX_STATEMENT_CACHE:
        df = load_prepared_table(INDEX_STATEMENT_FILE).loc[:, list(usecols)].copy()
        df["index_code"] = df["Indexcd"].map(_index_code_key)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["PubDate"] = pd.to_datetime(df["PubDate"], errors="coerce")
        df = df[df["Date"].notna() & df["PubDate"].notna()].copy()
        _INDEX_STATEMENT_CACHE[usecols] = df.sort_values(["index_code", "Date", "PubDate"])
    return _INDEX_STATEMENT_CACHE[usecols].copy()


def _index_report_frame(index_code: str, value_col: str) -> pd.DataFrame:
    df = _load_index_statement([value_col])
    out = df[df["index_code"].eq(_index_code_key(index_code))].copy()
    if out.empty:
        raise ValueError(f"{INDEX_STATEMENT_FILE} missing Indexcd={index_code!r}")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out[["Date", "PubDate", value_col]].drop_duplicates("Date", keep="last").sort_values("Date")


def _series_on_pubdate(index_code: str, value_col: str) -> pd.Series:
    frame = _index_report_frame(index_code, value_col)
    return _as_float_series(frame[value_col], frame["PubDate"], value_col)


def _report_series(index_code: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    frame = _index_report_frame(index_code, value_col)
    report_s = pd.Series(
        pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype="float64"),
        index=pd.to_datetime(frame["Date"]),
        name=value_col,
    ).sort_index()
    pub_dates = pd.Series(pd.to_datetime(frame["PubDate"]).to_numpy(), index=report_s.index)
    return report_s, pub_dates


def _with_pubdate(report_series: pd.Series, pub_dates: pd.Series) -> pd.Series:
    out = report_series.copy()
    out.index = pd.to_datetime(pub_dates.reindex(out.index).to_numpy())
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _ma_diff(series: pd.Series, short_window: int = 13, long_window: int = 26) -> pd.Series:
    s = series.astype("float64").sort_index()
    return s.rolling(short_window, min_periods=short_window).mean() - s.rolling(
        long_window,
        min_periods=long_window,
    ).mean()


def _style_report_spread(value_col: str) -> tuple[pd.Series, pd.Series]:
    growth, growth_pub = _report_series(GROWTH_STYLE_INDEX, value_col)
    value, _value_pub = _report_series(VALUE_STYLE_INDEX, value_col)
    return growth - value, growth_pub


def _style_ratio_spread(numerator_col: str, denominator_col: str) -> pd.Series:
    growth_num, growth_pub = _report_series(GROWTH_STYLE_INDEX, numerator_col)
    growth_den, _ = _report_series(GROWTH_STYLE_INDEX, denominator_col)
    value_num, _ = _report_series(VALUE_STYLE_INDEX, numerator_col)
    value_den, _ = _report_series(VALUE_STYLE_INDEX, denominator_col)
    spread = _safe_ratio(growth_num, growth_den) - _safe_ratio(value_num, value_den)
    return _with_pubdate(spread, growth_pub)


def _full_index_ma_factor(value_col: str, short_window: int = 13, long_window: int = 26, yoy: bool = False) -> pd.Series:
    series, pub_dates = _report_series(CSI_ALL_INDEX, value_col)
    base = _ttm_yoy(series) if yoy else series
    return _with_pubdate(_ma_diff(base, short_window, long_window), pub_dates)


def _style_ma_factor(value_col: str, short_window: int = 13, long_window: int = 26) -> pd.Series:
    spread, pub_dates = _style_report_spread(value_col)
    return _with_pubdate(_ma_diff(spread, short_window, long_window), pub_dates)


def _style_ttm_yoy_ma_factor(value_col: str, short_window: int = 13, long_window: int = 26) -> pd.Series:
    growth, growth_pub = _report_series(GROWTH_STYLE_INDEX, value_col)
    value, _ = _report_series(VALUE_STYLE_INDEX, value_col)
    spread = _ttm_yoy(growth) - _ttm_yoy(value)
    return _with_pubdate(_ma_diff(spread, short_window, long_window), growth_pub)


def _read_index_weights_cached(target_index_codes: list[str] | tuple[str, ...]) -> pd.DataFrame:
    key = tuple(sorted(_index_code_key(code) for code in target_index_codes))
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = _read_index_weights(list(target_index_codes))
    return _WEIGHTS_CACHE[key].copy()


def _weighted_metric_by_index_cached(metric_df: pd.DataFrame, target_index_code: str) -> pd.DataFrame:
    target_key = _index_code_key(target_index_code)
    weights = _read_index_weights_cached([target_index_code])
    weights = weights[weights["index_code"].eq(target_key)].copy()

    rows: list[dict[str, object]] = []
    report_dates = sorted(pd.to_datetime(metric_df["Accper"].dropna().unique()))
    weight_dates = pd.DatetimeIndex(weights["TRADE_DT"].dropna().sort_values().unique())
    for report_date, report_df in metric_df.groupby("Accper", sort=True):
        eligible = weight_dates[weight_dates <= pd.Timestamp(report_date)]
        if len(eligible) == 0:
            continue
        latest_date = eligible[-1]
        merged = weights[weights["TRADE_DT"].eq(latest_date)].merge(report_df, on="stock_code", how="inner")
        value = _weighted_average(merged, "value")
        if pd.isna(value):
            continue
        rows.append(
            {
                "Accper": pd.Timestamp(report_date),
                "PubDate": pd.to_datetime(merged["PubDate"], errors="coerce").max(),
                "value": value,
            }
        )
    return pd.DataFrame(rows).sort_values("Accper") if rows else pd.DataFrame(columns=["Accper", "PubDate", "value"])


def _style_weighted_spread(metric_df: pd.DataFrame) -> pd.Series:
    growth = _weighted_metric_by_index_cached(metric_df, GROWTH_STYLE_INDEX)
    value = _weighted_metric_by_index_cached(metric_df, VALUE_STYLE_INDEX)
    growth_s = pd.Series(growth["value"].to_numpy(dtype="float64"), index=pd.to_datetime(growth["Accper"]))
    value_s = pd.Series(value["value"].to_numpy(dtype="float64"), index=pd.to_datetime(value["Accper"]))
    factor = growth_s - value_s
    pub_dates = growth.drop_duplicates("Accper", keep="last").set_index("Accper")["PubDate"]
    return _with_pubdate(factor, pub_dates)


def _load_stock_metric_from_prepared(table_name: str, value_expr: str | list[str], value_name: str = "value") -> pd.DataFrame:
    cols = ["Stkcd", "Accper", "Typrep"]
    if isinstance(value_expr, str):
        value_cols = [value_expr]
    else:
        value_cols = list(value_expr)
    df = load_prepared_table(table_name)
    required = cols + value_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing columns: {missing}")
    out = df.loc[df["Typrep"].astype(str).eq("A"), required].copy()
    out["stock_code"] = out["Stkcd"].map(_stock_code_key)
    out["Accper"] = pd.to_datetime(out["Accper"], errors="coerce")
    if isinstance(value_expr, str):
        out[value_name] = pd.to_numeric(out[value_expr], errors="coerce")
    else:
        numeric = out[value_cols].apply(pd.to_numeric, errors="coerce")
        out[value_name] = numeric.iloc[:, 0]
        for col in value_cols[1:]:
            out[value_name] = out[value_name] + numeric[col]
    out = out[out["stock_code"].notna() & out["Accper"].notna()].copy()
    out = out.sort_values(["stock_code", "Accper"]).drop_duplicates(["stock_code", "Accper"], keep="last")
    pub_dates = _pubdate_map()
    out = out.merge(pub_dates, on=["stock_code", "Accper"], how="left")
    return out[["stock_code", "Accper", "PubDate", value_name]].rename(columns={value_name: "value"}).dropna(
        subset=["PubDate", "value"]
    )


def _pubdate_map() -> pd.DataFrame:
    global _PUBDATE_MAP_CACHE
    if _PUBDATE_MAP_CACHE is None:
        df = load_prepared_table(ASHARE_PROFIT_TABLE)
        out = df.loc[df["Typrep"].astype(str).eq("A"), ["Stkcd", "Accper", "PubDate"]].copy()
        out["stock_code"] = out["Stkcd"].map(_stock_code_key)
        out["Accper"] = pd.to_datetime(out["Accper"], errors="coerce")
        out["PubDate"] = pd.to_datetime(out["PubDate"], errors="coerce")
        out = out.dropna(subset=["stock_code", "Accper", "PubDate"])
        _PUBDATE_MAP_CACHE = out.sort_values(["stock_code", "Accper", "PubDate"]).drop_duplicates(
            ["stock_code", "Accper"],
            keep="last",
        )[["stock_code", "Accper", "PubDate"]]
    return _PUBDATE_MAP_CACHE.copy()


def _stock_group_transform(metric_df: pd.DataFrame, transform: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for stock_code, group in metric_df.groupby("stock_code", sort=False):
        group = group.sort_values("Accper")
        s = pd.Series(group["value"].to_numpy(dtype="float64"), index=pd.to_datetime(group["Accper"]))
        if transform == "diff":
            value = s - s.shift(1)
        elif transform == "yoy":
            value = s / s.shift(4) - 1.0
        elif transform == "yoy_change":
            yoy = s / s.shift(4) - 1.0
            value = yoy - yoy.shift(1)
        else:
            raise ValueError(f"Unsupported stock transform: {transform}")
        part = pd.DataFrame({"stock_code": stock_code, "Accper": value.index, "value": value.to_numpy(dtype="float64")})
        part = part.merge(group[["Accper", "PubDate"]].drop_duplicates("Accper"), on="Accper", how="left")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stock_code", "Accper", "PubDate", "value"])
    return out.dropna(subset=["value", "PubDate"])


def _load_is_ratio_metric(numerator_col: str, denominator_col: str) -> pd.DataFrame:
    num = _load_stock_metric_from_prepared("Astock_IS.parquet", numerator_col).rename(columns={"value": "numerator"})
    den = _load_stock_metric_from_prepared("Astock_IS.parquet", denominator_col).rename(columns={"value": "denominator"})
    merged = num.merge(den[["stock_code", "Accper", "denominator"]], on=["stock_code", "Accper"], how="inner")
    merged["value"] = merged["numerator"] / merged["denominator"].replace(0, np.nan)
    return merged[["stock_code", "Accper", "PubDate", "value"]].dropna(subset=["value"])


def _load_bs_parent_equity_to_debt_metric() -> pd.DataFrame:
    debt_cols = ["A002101000", "A002125000", "A002201000", "A002203000"]
    equity = _load_stock_metric_from_prepared("BS_AexcluST1F.parquet", "A003100000").rename(columns={"value": "equity"})
    debt = _load_stock_metric_from_prepared("BS_AexcluST1F.parquet", debt_cols).rename(columns={"value": "debt"})
    merged = equity.merge(debt[["stock_code", "Accper", "debt"]], on=["stock_code", "Accper"], how="inner")
    merged["value"] = merged["equity"] / merged["debt"].replace(0, np.nan)
    return merged[["stock_code", "Accper", "PubDate", "value"]].dropna(subset=["value"])


def _weighted_consensus_style_spread(file_name: str, value_col: str) -> pd.Series:
    path = PROJECT_ROOT / "A_data" / "prepared_data" / file_name
    consensus = pd.read_feather(path, columns=["TRADE_DT", "S_INFO_WINDCODE", value_col])
    consensus["TRADE_DT"] = pd.to_datetime(consensus["TRADE_DT"].astype(str), format="%Y%m%d", errors="coerce")
    consensus["stock_code"] = consensus["S_INFO_WINDCODE"].map(_stock_code_key)
    consensus[value_col] = pd.to_numeric(consensus[value_col], errors="coerce")
    consensus = consensus[consensus["TRADE_DT"].notna() & consensus[value_col].notna()].copy()

    weights = _read_index_weights_cached([GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX])
    merged = weights.merge(consensus[["TRADE_DT", "stock_code", value_col]], on=["TRADE_DT", "stock_code"], how="inner")
    if merged.empty:
        raise ValueError(f"{file_name} cannot merge with style index weights")
    weighted = (
        merged.groupby(["index_code", "TRADE_DT"], sort=True)
        .apply(lambda group: _weighted_average(group, value_col), include_groups=False)
        .rename("weighted_value")
        .reset_index()
    )
    pivot = weighted.pivot(index="TRADE_DT", columns="index_code", values="weighted_value")
    growth = _index_code_key(GROWTH_STYLE_INDEX)
    value = _index_code_key(VALUE_STYLE_INDEX)
    if growth not in pivot.columns or value not in pivot.columns:
        raise ValueError(f"{file_name} missing weighted style columns")
    return (pivot[growth] - pivot[value]).dropna().sort_index()


def _calc_p033() -> pd.Series:
    return _full_index_ma_factor("净利润", yoy=True)


def _calc_p035() -> pd.Series:
    spread, pub_dates = _style_report_spread("一致预测ROE(FY1)")
    spread = spread.replace(0, np.nan)
    return _with_pubdate(spread - spread.shift(1), pub_dates)


def _calc_p037() -> pd.Series:
    spread, pub_dates = _style_report_spread("ROE")
    return _with_pubdate(spread, pub_dates)


def _calc_p038() -> pd.Series:
    return _full_index_ma_factor("营业收入", yoy=True)


def _calc_p039() -> pd.Series:
    return _full_index_ma_factor("ROE")


def _calc_p040() -> pd.Series:
    return _full_index_ma_factor("资产负债率")


def _calc_p041() -> pd.Series:
    return _full_index_ma_factor("流动比率", short_window=13, long_window=5)


def _calc_p042() -> pd.Series:
    parent_profit = _calc_stock_ttm_yoy_change(_load_parent_net_profit())
    investment_return = _stock_group_transform(_load_ashare_profit_metric("F053202B"), "diff")
    roe_yoy = _stock_group_transform(_load_ashare_profit_metric("F050501B"), "yoy")
    roe_ttm_change = _stock_group_transform(_load_ashare_profit_metric("F050504C"), "diff")
    roa_yoy = _stock_group_transform(_load_ashare_profit_metric("F050201B"), "yoy")
    net_margin_yoy = _stock_group_transform(_load_is_ratio_metric("B002000000", "B001101000"), "yoy")

    components = [
        _style_weighted_spread(parent_profit).rename("parent_profit"),
        _style_weighted_spread(investment_return).rename("investment_return"),
        _style_weighted_spread(roe_yoy).rename("roe_yoy"),
        _style_weighted_spread(roe_ttm_change).rename("roe_ttm_change"),
        _style_weighted_spread(roa_yoy).rename("roa_yoy"),
        _style_weighted_spread(net_margin_yoy).rename("net_margin_yoy"),
    ]
    return pd.concat(components, axis=1).mean(axis=1, skipna=False).dropna().sort_index()


def _calc_p043() -> pd.Series:
    ratio_spread = _style_ratio_spread("营业利润", "利润总额")
    return (_ma_diff(ratio_spread, 1, 26) * -1.0).dropna()


def _calc_p044() -> pd.Series:
    return _style_weighted_spread(_load_bs_parent_equity_to_debt_metric()) * -1.0


def _calc_p045() -> pd.Series:
    growth_in = _series_on_pubdate(GROWTH_STYLE_INDEX, "营业外收入")
    growth_out = _series_on_pubdate(GROWTH_STYLE_INDEX, "营业外支出")
    growth_profit = _series_on_pubdate(GROWTH_STYLE_INDEX, "利润总额")
    value_in = _series_on_pubdate(VALUE_STYLE_INDEX, "营业外收入")
    value_out = _series_on_pubdate(VALUE_STYLE_INDEX, "营业外支出")
    value_profit = _series_on_pubdate(VALUE_STYLE_INDEX, "利润总额")
    growth_ratio = _safe_ratio(growth_in - growth_out, growth_profit)
    value_ratio = _safe_ratio(value_in - value_out, value_profit)
    return ((growth_ratio - value_ratio) * -1.0).dropna().sort_index()


def _calc_p046() -> pd.Series:
    file_name, value_col = CONSENSUS_FILES["P046"]
    return _weighted_consensus_style_spread(file_name, value_col)


def _calc_p047() -> pd.Series:
    file_name, value_col = CONSENSUS_FILES["P047"]
    return _weighted_consensus_style_spread(file_name, value_col)


def _calc_p049() -> pd.Series:
    spread, pub_dates = _style_report_spread("应收账款周转天数")
    return _with_pubdate((spread - spread.shift(1)) * -1.0, pub_dates)


def _calc_p050() -> pd.Series:
    equity_spread = _style_ratio_spread("归属母公司股东的权益", "资产负债率")
    return ((equity_spread - equity_spread.shift(1)) * -1.0).dropna().sort_index()


def _calc_p051() -> pd.Series:
    file_name, value_col = CONSENSUS_FILES["P051"]
    return _weighted_consensus_style_spread(file_name, value_col).diff().dropna()


def _calc_p052() -> pd.Series:
    return _full_index_ma_factor("销售毛利率")


def _calc_p053() -> pd.Series:
    return _style_ma_factor("销售毛利率")


def _calc_p054() -> pd.Series:
    return _style_ma_factor("存货周转率", short_window=5, long_window=13)


def _calc_p055() -> pd.Series:
    return _style_ma_factor("应收账款周转率", short_window=5, long_window=13)


def _calc_p056() -> pd.Series:
    return _style_ma_factor("总资产周转率", short_window=5, long_window=13)


def _calc_p057() -> pd.Series:
    return _style_ma_factor("ROE")


def _calc_p058() -> pd.Series:
    return _style_ttm_yoy_ma_factor("归母净利润")


def _calc_p059() -> pd.Series:
    growth_profit, growth_pub = _report_series(GROWTH_STYLE_INDEX, "归母净利润")
    growth_revenue, _ = _report_series(GROWTH_STYLE_INDEX, "营业收入")
    value_profit, _ = _report_series(VALUE_STYLE_INDEX, "归母净利润")
    value_revenue, _ = _report_series(VALUE_STYLE_INDEX, "营业收入")
    spread = _safe_ratio(_quarterly_ttm(growth_profit), _quarterly_ttm(growth_revenue)) - _safe_ratio(
        _quarterly_ttm(value_profit),
        _quarterly_ttm(value_revenue),
    )
    return _with_pubdate(_ma_diff(spread), growth_pub)


def _calc_p060() -> pd.Series:
    return _style_ttm_yoy_ma_factor("归母净利润")


def generate_profitFactors2_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    calculators = {
        "P033": _calc_p033,
        "P035": _calc_p035,
        "P037": _calc_p037,
        "P038": _calc_p038,
        "P039": _calc_p039,
        "P040": _calc_p040,
        "P041": _calc_p041,
        "P042": _calc_p042,
        "P043": _calc_p043,
        "P044": _calc_p044,
        "P045": _calc_p045,
        "P046": _calc_p046,
        "P047": _calc_p047,
        "P049": _calc_p049,
        "P050": _calc_p050,
        "P051": _calc_p051,
        "P052": _calc_p052,
        "P053": _calc_p053,
        "P054": _calc_p054,
        "P055": _calc_p055,
        "P056": _calc_p056,
        "P057": _calc_p057,
        "P058": _calc_p058,
        "P059": _calc_p059,
        "P060": _calc_p060,
    }
    for factor_id in IMPLEMENTED_FACTOR_IDS:
        factor_series = calculators[factor_id]()
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in IMPLEMENTED_FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"profitFactors2 columns missing after generation: {missing_cols}")

    for factor_id, reason in UNIMPLEMENTED_FACTORS.items():
        print(f"Skipping {factor_id}: {reason}")

    return factor_source_df.loc[:, IMPLEMENTED_FACTOR_IDS], records


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

    factor_source_df, selected_records = generate_profitFactors2_factors(data_df)
    metadata = metadata_from_profitFactors2_records(selected_records)
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
