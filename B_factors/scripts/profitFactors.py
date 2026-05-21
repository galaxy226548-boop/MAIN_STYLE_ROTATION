"""Profit and earnings style-rotation factors from working_multiple_factors_plan.json."""

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
    read_prepared_series,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


OUTPUT_PREFIX = "profitFactors"
PLAN_PATH = PROJECT_ROOT / "B_factors" / "reference" / "working_multiple_factors_plan.json"

FACTOR_IDS = ["P061", "P018", "P019", "P021", "P022", "P023"]

INDEX_STATEMENT_FILE = "IndexStatement.xlsx"
CONSENSUS_NP_GROWTH_FILE = "Con_np_yoy_roll.feather"
INDEX_WEIGHT_FILE = "AIndexHS300FreeWeight.parquet"
ASHARE_PROFIT_TABLE = "Ashare_profit.parquet"
IS_AEXCLUST1F_FILE = PROJECT_ROOT / "A_data" / "data" / "update" / "IS_AexcluST1F.xlsx"
MAX_PARENT_NET_PROFIT_PUBDATE_MISSING_RATIO = 0.01

GROWTH_STYLE_INDEX = "399370.SZ"
VALUE_STYLE_INDEX = "399371.SZ"
CSI500_INDEX = "000905.SH"
CSI300_INDEX = "000300.SH"
CHINEXT_INDEX = "399006.SZ"

_PARENT_NET_PROFIT_CACHE: pd.DataFrame | None = None


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
            records.append(item)

    found = {str(record["factor_id"]) for record in records}
    missing = [factor_id for factor_id in FACTOR_IDS if factor_id not in found]
    if missing:
        raise ValueError(f"working_multiple_factors_plan.json missing implemented records: {missing}")
    return records


def metadata_from_profitFactors_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(record["factor_id"]): {
            "signal_type": _normalize_plan_text(record.get("signal_type")) or "state",
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def _index_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].zfill(6)


def _stock_code_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".")[0].zfill(6)


def _as_float_series(series: pd.Series, index: pd.Series | pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(index), name=name)
    out = out[out.index.notna()].sort_index()
    return out[~out.index.duplicated(keep="last")].astype("float64")


def _load_index_statement(columns: list[str]) -> pd.DataFrame:
    required = ["Indexcd", "Date", "PubDate"]
    usecols = list(dict.fromkeys(required + columns))
    df = load_prepared_table(INDEX_STATEMENT_FILE).loc[:, usecols].copy()
    df["index_code"] = df["Indexcd"].map(_index_code_key)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["PubDate"] = pd.to_datetime(df["PubDate"], errors="coerce")
    df = df[df["Date"].notna() & df["PubDate"].notna()].copy()
    return df.sort_values(["index_code", "Date", "PubDate"])


def _index_statement_series(index_code: str, value_col: str) -> pd.Series:
    df = _load_index_statement([value_col])
    out = df[df["index_code"].eq(_index_code_key(index_code))].copy()
    if out.empty:
        raise ValueError(f"{INDEX_STATEMENT_FILE} missing Indexcd={index_code!r}")
    return _as_float_series(out[value_col], out["PubDate"], value_col)


def _index_statement_report_frame(index_code: str, value_col: str) -> pd.DataFrame:
    df = _load_index_statement([value_col])
    out = df[df["index_code"].eq(_index_code_key(index_code))].copy()
    if out.empty:
        raise ValueError(f"{INDEX_STATEMENT_FILE} missing Indexcd={index_code!r}")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out[["Date", "PubDate", value_col]].drop_duplicates("Date", keep="last").sort_values("Date")


def _quarterly_ttm(cumulative: pd.Series) -> pd.Series:
    s = cumulative.astype("float64").sort_index()
    annual_mask = s.index.quarter == 4
    return s.where(annual_mask, s + s.shift(1) - s.shift(4))


def _ttm_yoy(cumulative: pd.Series) -> pd.Series:
    ttm = _quarterly_ttm(cumulative)
    return ttm / ttm.shift(4) - 1.0


def _read_index_weights(target_index_codes: list[str] | tuple[str, ...]) -> pd.DataFrame:
    path = PROJECT_ROOT / "A_data" / "prepared_data" / INDEX_WEIGHT_FILE
    target_keys = {_index_code_key(code) for code in target_index_codes}
    columns = ["S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT", "I_WEIGHT"]

    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        candidate_codes = []
        for key in target_keys:
            candidate_codes.extend([f"{key}.SH", f"{key}.SZ", f"{key}.CSI", key])
        table = dataset.to_table(
            columns=columns,
            filter=pc.field("S_INFO_WINDCODE").isin(candidate_codes),
        )
        df = table.to_pandas()
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        keys = df["S_INFO_WINDCODE"].map(_index_code_key)
        df = df[keys.isin(target_keys)].copy()

    if df.empty:
        raise ValueError(f"{INDEX_WEIGHT_FILE} missing target index weights: {sorted(target_keys)}")
    df["index_code"] = df["S_INFO_WINDCODE"].map(_index_code_key)
    df = df[df["index_code"].isin(target_keys)].copy()
    df["stock_code"] = df["S_CON_WINDCODE"].map(_stock_code_key)
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str), format="%Y%m%d", errors="coerce")
    df["I_WEIGHT"] = pd.to_numeric(df["I_WEIGHT"], errors="coerce")
    df = df[df["TRADE_DT"].notna() & df["stock_code"].notna() & df["I_WEIGHT"].notna()].copy()
    if df.empty:
        raise ValueError(f"{INDEX_WEIGHT_FILE} has no usable rows for target index weights: {sorted(target_keys)}")
    return df.sort_values(["index_code", "TRADE_DT", "stock_code"])


def _weighted_average(group: pd.DataFrame, value_col: str, weight_col: str = "I_WEIGHT") -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float((values.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def _latest_weights_for_dates(weights: pd.DataFrame, report_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    sorted_dates = sorted(pd.to_datetime(report_dates))
    weight_dates = pd.DatetimeIndex(weights["TRADE_DT"].dropna().sort_values().unique())
    for report_date in sorted_dates:
        eligible = weight_dates[weight_dates <= report_date]
        if len(eligible) == 0:
            continue
        latest_date = eligible[-1]
        out[pd.Timestamp(report_date)] = weights[weights["TRADE_DT"].eq(latest_date)].copy()
    return out


def _weighted_stock_metric_by_index(metric_df: pd.DataFrame, target_index_code: str) -> pd.DataFrame:
    target_key = _index_code_key(target_index_code)
    weights = _read_index_weights([target_index_code])
    weights = weights[weights["index_code"].eq(target_key)].copy()
    by_date = _latest_weights_for_dates(weights, metric_df["Accper"].dropna().unique().tolist())

    rows: list[dict[str, object]] = []
    for report_date, report_df in metric_df.groupby("Accper", sort=True):
        weights_on_date = by_date.get(pd.Timestamp(report_date))
        if weights_on_date is None:
            continue
        merged = weights_on_date.merge(report_df, on="stock_code", how="inner")
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
    return pd.DataFrame(rows).sort_values("Accper")


def _load_ashare_profit_metric(value_col: str) -> pd.DataFrame:
    df = load_prepared_table(ASHARE_PROFIT_TABLE)
    required = ["Stkcd", "Accper", "Typrep", "PubDate", value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{ASHARE_PROFIT_TABLE} missing columns: {missing}")
    out = df.loc[df["Typrep"].astype(str).eq("A"), required].copy()
    out["stock_code"] = out["Stkcd"].map(_stock_code_key)
    out["Accper"] = pd.to_datetime(out["Accper"], errors="coerce")
    out["PubDate"] = pd.to_datetime(out["PubDate"], errors="coerce")
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out = out[out["stock_code"].notna() & out["Accper"].notna() & out["PubDate"].notna()].copy()
    out = out.sort_values(["stock_code", "Accper", "PubDate"]).drop_duplicates(["stock_code", "Accper"], keep="last")
    return out[["stock_code", "Accper", "PubDate", "value"]].sort_values(["stock_code", "Accper"])


def _load_stock_industry_map() -> pd.DataFrame:
    df = load_prepared_table(ASHARE_PROFIT_TABLE)
    required = ["Stkcd", "Accper", "Typrep", "Indcd"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{ASHARE_PROFIT_TABLE} missing columns: {missing}")
    out = df.loc[df["Typrep"].astype(str).eq("A"), required].copy()
    out["stock_code"] = out["Stkcd"].map(_stock_code_key)
    out["Accper"] = pd.to_datetime(out["Accper"], errors="coerce")
    out["Indcd"] = out["Indcd"].astype("string").str.strip()
    out = out[out["stock_code"].notna() & out["Accper"].notna() & out["Indcd"].notna() & out["Indcd"].ne("")].copy()
    return out.sort_values(["stock_code", "Accper"]).drop_duplicates(["stock_code", "Accper"], keep="last")


def _load_parent_net_profit() -> pd.DataFrame:
    global _PARENT_NET_PROFIT_CACHE
    if _PARENT_NET_PROFIT_CACHE is not None:
        return _PARENT_NET_PROFIT_CACHE.copy()

    if not IS_AEXCLUST1F_FILE.exists():
        raise FileNotFoundError(f"找不到归母净利润源文件：{IS_AEXCLUST1F_FILE}")
    usecols = ["Stkcd", "Accper", "Typrep", "B002000101"]
    df = pd.read_excel(IS_AEXCLUST1F_FILE, usecols=usecols, skiprows=[1, 2])
    df = df[df["Typrep"].astype(str).eq("A")].copy()
    df["stock_code"] = df["Stkcd"].map(_stock_code_key)
    df["Accper"] = pd.to_datetime(df["Accper"], errors="coerce")
    df["net_profit"] = pd.to_numeric(df["B002000101"], errors="coerce")
    df = df[df["stock_code"].notna() & df["Accper"].notna()].copy()
    df = df.sort_values(["stock_code", "Accper"]).drop_duplicates(["stock_code", "Accper"], keep="last")
    pub_map = load_prepared_table(ASHARE_PROFIT_TABLE)
    pub_map = pub_map.loc[pub_map["Typrep"].astype(str).eq("A"), ["Stkcd", "Accper", "PubDate"]].copy()
    pub_map["stock_code"] = pub_map["Stkcd"].map(_stock_code_key)
    pub_map["Accper"] = pd.to_datetime(pub_map["Accper"], errors="coerce")
    pub_map["PubDate"] = pd.to_datetime(pub_map["PubDate"], errors="coerce")
    pub_map = pub_map.dropna(subset=["stock_code", "Accper", "PubDate"])
    pub_map = pub_map.sort_values(["stock_code", "Accper", "PubDate"]).drop_duplicates(
        ["stock_code", "Accper"],
        keep="last",
    )
    df = df.merge(pub_map[["stock_code", "Accper", "PubDate"]], on=["stock_code", "Accper"], how="left")
    missing_pubdate_ratio = float(df["PubDate"].isna().mean()) if len(df) else 0.0
    if missing_pubdate_ratio > MAX_PARENT_NET_PROFIT_PUBDATE_MISSING_RATIO:
        print(
            "Warning: IS_AexcluST1F parent net profit rows failed to match prepared PubDate from "
            f"{ASHARE_PROFIT_TABLE}: missing_ratio={missing_pubdate_ratio:.2%}. "
            "Rows without PubDate will not produce event dates."
        )
    elif missing_pubdate_ratio > 0:
        print(
            "Warning: IS_AexcluST1F parent net profit rows have a small prepared PubDate "
            f"missing ratio: {missing_pubdate_ratio:.2%}"
        )
    _PARENT_NET_PROFIT_CACHE = df[["stock_code", "Accper", "PubDate", "net_profit"]].copy()
    return _PARENT_NET_PROFIT_CACHE.copy()


def _calc_stock_ttm_yoy_change(profit_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for stock_code, group in profit_df.groupby("stock_code", sort=False):
        s = pd.Series(group["net_profit"].to_numpy(dtype="float64"), index=group["Accper"]).sort_index()
        yoy = _ttm_yoy(s)
        yoy_change = yoy - yoy.shift(1)
        part = pd.DataFrame(
            {
                "stock_code": stock_code,
                "Accper": yoy_change.index,
                "value": yoy_change.to_numpy(dtype="float64"),
            }
        )
        part = part.merge(group[["Accper", "PubDate"]].drop_duplicates("Accper"), on="Accper", how="left")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stock_code", "Accper", "value"])
    return out.dropna(subset=["value"])


def _calc_stock_ttm_yoy(profit_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for stock_code, group in profit_df.groupby("stock_code", sort=False):
        s = pd.Series(group["net_profit"].to_numpy(dtype="float64"), index=group["Accper"]).sort_index()
        yoy = _ttm_yoy(s)
        part = pd.DataFrame(
            {
                "stock_code": stock_code,
                "Accper": yoy.index,
                "value": yoy.to_numpy(dtype="float64"),
            }
        )
        part = part.merge(group[["Accper", "PubDate"]].drop_duplicates("Accper"), on="Accper", how="left")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stock_code", "Accper", "value"])
    return out.dropna(subset=["value"])


def _calc_p061() -> pd.Series:
    fields = ["营业收入", "归属母公司股东的权益", "归母净利润"]
    growth = _load_index_statement(fields)
    growth = growth[growth["index_code"].eq(_index_code_key(GROWTH_STYLE_INDEX))]
    value = _load_index_statement(fields)
    value = value[value["index_code"].eq(_index_code_key(VALUE_STYLE_INDEX))]

    rows = []
    for field in fields:
        growth_s = _as_float_series(growth[field], growth["Date"], f"growth_{field}")
        value_s = _as_float_series(value[field], value["Date"], f"value_{field}")
        diff = growth_s / growth_s.shift(4) - value_s / value_s.shift(4)
        rows.append(diff.rename(field))
    diff_df = pd.concat(rows, axis=1)
    score_pos = (diff_df > 0).sum(axis=1)
    score_neg = (diff_df < 0).sum(axis=1)
    event = pd.Series(0.0, index=diff_df.index, dtype="float64")
    event.loc[score_pos >= 2] = 1.0
    event.loc[score_neg >= 2] = -1.0
    event.loc[diff_df.notna().sum(axis=1) < 2] = np.nan

    pub_dates = growth.drop_duplicates("Date", keep="last").set_index("Date")["PubDate"]
    event.index = pd.to_datetime(pub_dates.reindex(event.index).to_numpy())
    event = event[event.index.notna()].sort_index()
    return event[~event.index.duplicated(keep="last")]


def _calc_p018() -> pd.Series:
    weights = _read_index_weights([CSI500_INDEX, CSI300_INDEX])
    consensus_path = PROJECT_ROOT / "A_data" / "prepared_data" / CONSENSUS_NP_GROWTH_FILE
    consensus = pd.read_feather(consensus_path, columns=["TRADE_DT", "S_INFO_WINDCODE", "Con_np_yoy_roll"])
    consensus["TRADE_DT"] = pd.to_datetime(consensus["TRADE_DT"].astype(str), format="%Y%m%d", errors="coerce")
    consensus["stock_code"] = consensus["S_INFO_WINDCODE"].map(_stock_code_key)
    consensus["Con_np_yoy_roll"] = pd.to_numeric(consensus["Con_np_yoy_roll"], errors="coerce")
    consensus = consensus[consensus["TRADE_DT"].notna() & consensus["Con_np_yoy_roll"].notna()].copy()

    merged = weights.merge(
        consensus[["TRADE_DT", "stock_code", "Con_np_yoy_roll"]],
        on=["TRADE_DT", "stock_code"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("P018 cannot merge consensus net profit growth with index weights")
    weighted = (
        merged.groupby(["index_code", "TRADE_DT"], sort=True)
        .apply(lambda group: _weighted_average(group, "Con_np_yoy_roll"), include_groups=False)
        .rename("weighted_growth")
        .reset_index()
    )
    pivot = weighted.pivot(index="TRADE_DT", columns="index_code", values="weighted_growth")
    csi500 = _index_code_key(CSI500_INDEX)
    csi300 = _index_code_key(CSI300_INDEX)
    if csi500 not in pivot.columns or csi300 not in pivot.columns:
        raise ValueError("P018 missing weighted series for CSI500 or CSI300")
    return (pivot[csi500] - pivot[csi300]).dropna().sort_index()


def _calc_p019() -> pd.Series:
    return read_prepared_series("macro_monthly.parquet", "中国:利润总额:规模以上工业企业:累计同比") * -1.0


def _calc_p021() -> pd.Series:
    csi500 = _index_statement_report_frame(CSI500_INDEX, "归母净利润")
    csi300 = _index_statement_report_frame(CSI300_INDEX, "归母净利润")
    invalid_ratio = float((csi300["归母净利润"].isna() | csi300["归母净利润"].eq(0)).mean())
    if invalid_ratio > 0.2:
        print(f"P021 IndexStatement 沪深300归母净利润无效值占比过高：{invalid_ratio:.2%}; using component-weighted fallback.")
        profit = _load_parent_net_profit()
        stock_yoy = _calc_stock_ttm_yoy(profit)
        csi500_weighted = _weighted_stock_metric_by_index(stock_yoy, CSI500_INDEX)
        csi300_weighted = _weighted_stock_metric_by_index(stock_yoy, CSI300_INDEX)
        csi500_s = pd.Series(csi500_weighted["value"].to_numpy(dtype="float64"), index=csi500_weighted["Accper"])
        csi300_s = pd.Series(csi300_weighted["value"].to_numpy(dtype="float64"), index=csi300_weighted["Accper"])
        factor = csi500_s - csi300_s
        pub_dates = csi500_weighted.drop_duplicates("Accper", keep="last").set_index("Accper")["PubDate"]
        factor.index = pd.to_datetime(pub_dates.reindex(factor.index).to_numpy())
        factor = factor[factor.index.notna()].sort_index()
        return factor[~factor.index.duplicated(keep="last")]
    csi500_yoy = _ttm_yoy(pd.Series(csi500["归母净利润"].to_numpy(dtype="float64"), index=csi500["Date"]))
    csi300_yoy = _ttm_yoy(pd.Series(csi300["归母净利润"].to_numpy(dtype="float64"), index=csi300["Date"]))
    factor = csi500_yoy - csi300_yoy
    pub_dates = csi500.drop_duplicates("Date", keep="last").set_index("Date")["PubDate"]
    factor.index = pd.to_datetime(pub_dates.reindex(factor.index).to_numpy())
    factor = factor[factor.index.notna()].sort_index()
    return factor[~factor.index.duplicated(keep="last")]


def _calc_p022() -> pd.Series:
    profit = _load_parent_net_profit()
    stock_factor = _calc_stock_ttm_yoy_change(profit)
    industry_map = _load_stock_industry_map()
    mapped = stock_factor.merge(industry_map[["stock_code", "Accper", "Indcd"]], on=["stock_code", "Accper"], how="inner")
    mapped = mapped.dropna(subset=["value", "Indcd"])
    industry_factor = mapped.groupby(["Accper", "Indcd"], sort=True)["value"].mean().rename("industry_value").reset_index()

    pub_by_report = profit.groupby("Accper")["PubDate"].max()

    weights = _read_index_weights([GROWTH_STYLE_INDEX, VALUE_STYLE_INDEX])
    industry_history = industry_map[["stock_code", "Accper", "Indcd"]].sort_values(["stock_code", "Accper"])
    weighted_rows = []
    for index_code, group in weights.groupby("index_code", sort=False):
        merged_parts = []
        for stock_code, stock_weights in group.groupby("stock_code", sort=False):
            industry_rows = industry_history[industry_history["stock_code"].eq(stock_code)]
            if industry_rows.empty:
                continue
            joined = pd.merge_asof(
                stock_weights.sort_values("TRADE_DT"),
                industry_rows.sort_values("Accper"),
                left_on="TRADE_DT",
                right_on="Accper",
                direction="backward",
            )
            merged_parts.append(joined)
        if not merged_parts:
            continue
        merged_industry = pd.concat(merged_parts, ignore_index=True).dropna(subset=["Indcd"])
        industry_weights = (
            merged_industry.groupby(["index_code", "TRADE_DT", "Indcd"], sort=True)["I_WEIGHT"].sum().reset_index()
        )
        industry_weights["industry_weight"] = industry_weights["I_WEIGHT"] / industry_weights.groupby(
            ["index_code", "TRADE_DT"]
        )["I_WEIGHT"].transform("sum")
        weighted_rows.append(industry_weights[["index_code", "TRADE_DT", "Indcd", "industry_weight"]])
    if not weighted_rows:
        raise ValueError("P022 cannot build style index industry weights")
    style_industry_weights = pd.concat(weighted_rows, ignore_index=True)

    rows: list[dict[str, object]] = []
    for report_date, factor_rows in industry_factor.groupby("Accper", sort=True):
        report_date = pd.Timestamp(report_date)
        row = {"Accper": report_date, "PubDate": pub_by_report.get(report_date)}
        for index_code in [_index_code_key(GROWTH_STYLE_INDEX), _index_code_key(VALUE_STYLE_INDEX)]:
            index_weights = style_industry_weights[style_industry_weights["index_code"].eq(index_code)]
            eligible_dates = pd.DatetimeIndex(index_weights["TRADE_DT"].dropna().sort_values().unique())
            eligible_dates = eligible_dates[eligible_dates <= report_date]
            if len(eligible_dates) == 0:
                row[index_code] = np.nan
                continue
            latest_weight_date = eligible_dates[-1]
            latest_weights = index_weights[index_weights["TRADE_DT"].eq(latest_weight_date)]
            merged = latest_weights.merge(factor_rows, on="Indcd", how="inner")
            row[index_code] = _weighted_average(merged, "industry_value", "industry_weight")
        rows.append(row)

    style_df = pd.DataFrame(rows).dropna(subset=["PubDate"])
    growth = _index_code_key(GROWTH_STYLE_INDEX)
    value = _index_code_key(VALUE_STYLE_INDEX)
    factor = pd.Series((style_df[growth] - style_df[value]).to_numpy(dtype="float64"), index=pd.to_datetime(style_df["PubDate"]))
    return factor.dropna().sort_index()


def _calc_p023_from_index_statement() -> pd.Series | None:
    chinext = _index_statement_report_frame(CHINEXT_INDEX, "ROE")
    csi300 = _index_statement_report_frame(CSI300_INDEX, "ROE")
    csi300_invalid = float((csi300["ROE"].isna() | csi300["ROE"].eq(0)).mean())
    chinext_invalid = float((chinext["ROE"].isna() | chinext["ROE"].eq(0)).mean())
    if csi300_invalid > 0.2 or chinext_invalid > 0.2:
        print(
            "P023 IndexStatement ROE invalid ratio too high; using component-weighted fallback:",
            f"399006={chinext_invalid:.2%}",
            f"000300={csi300_invalid:.2%}",
        )
        return None
    chinext_roe = pd.Series(chinext["ROE"].to_numpy(dtype="float64"), index=chinext["Date"])
    csi300_roe = pd.Series(csi300["ROE"].to_numpy(dtype="float64"), index=csi300["Date"])
    factor = (chinext_roe - chinext_roe.shift(4)) - (csi300_roe - csi300_roe.shift(4))
    pub_dates = chinext.drop_duplicates("Date", keep="last").set_index("Date")["PubDate"]
    factor.index = pd.to_datetime(pub_dates.reindex(factor.index).to_numpy())
    factor = factor[factor.index.notna()].sort_index()
    return factor[~factor.index.duplicated(keep="last")]


def _calc_p023() -> pd.Series:
    direct = _calc_p023_from_index_statement()
    if direct is not None:
        return direct

    metric = _load_ashare_profit_metric("F050102B")
    chinext = _weighted_stock_metric_by_index(metric, CHINEXT_INDEX)
    csi300 = _weighted_stock_metric_by_index(metric, CSI300_INDEX)
    chinext_s = pd.Series(chinext["value"].to_numpy(dtype="float64"), index=chinext["Accper"])
    csi300_s = pd.Series(csi300["value"].to_numpy(dtype="float64"), index=csi300["Accper"])
    factor = (chinext_s - chinext_s.shift(4)) - (csi300_s - csi300_s.shift(4))
    pub_dates = chinext.drop_duplicates("Accper", keep="last").set_index("Accper")["PubDate"]
    factor.index = pd.to_datetime(pub_dates.reindex(factor.index).to_numpy())
    factor = factor[factor.index.notna()].sort_index()
    return factor[~factor.index.duplicated(keep="last")]


def generate_profitFactors_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    records = _load_plan_records()

    calculators = {
        "P061": _calc_p061,
        "P018": _calc_p018,
        "P019": _calc_p019,
        "P021": _calc_p021,
        "P022": _calc_p022,
        "P023": _calc_p023,
    }
    for factor_id in FACTOR_IDS:
        factor_series = calculators[factor_id]()
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"profit factor columns missing after generation: {missing_cols}")

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

    factor_source_df, selected_records = generate_profitFactors_factors(data_df)
    metadata = metadata_from_profitFactors_records(selected_records)
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
