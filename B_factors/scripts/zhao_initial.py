"""Initial ZHAO factor source generation."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from factor_utils import (
    _YoY,
    _find_data_file,
    _load_china_macro_series,
    _load_macro_all,
    _month_aggregate,
    _read_indicator_series,
    _register_factor,
    _rolling_quantile_rank_year,
    _rolling_sum_ratio_minus_one,
    calc_qrd,
    clean_macro_table,
    normalize_trade_dt,
    pmi_yoy_chain,
)


def generate_zhao_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    # 保留旧 _register_factor 调用形态中的占位参数，实际只写入 factor_source_df。
    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)

    # ### index_pb (ZHAO01)
    sub_1 = _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx", "市净率LF").dropna()
    sub_2 = _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx", "市净率LF").dropna()
    valuation_ratio = sub_1 / sub_2
    quantile_rank = _rolling_quantile_rank_year(valuation_ratio, 5)
    ZHAO01_raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0).dropna()
    ZHAO01_raw = _month_aggregate(ZHAO01_raw, how="last")
    _register_factor(raw_factor_df, factor_source_df, "ZHAO01_raw", quantile_rank, ZHAO01_raw)

    # ### index_pe (ZHAO02)
    sub_1 = _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_growth_100104_260325.xlsx", "市盈率TTM").dropna()
    sub_2 = _read_indicator_series("D_PE+PB+DI+PS+PC+调仓_value_100104_260325.xlsx", "市盈率TTM").dropna()
    valuation_ratio = sub_1 / sub_2
    quantile_rank = _rolling_quantile_rank_year(valuation_ratio, 5)
    ZHAO02_raw = (0.1 - quantile_rank).clip(lower=0) - (quantile_rank - 0.9).clip(lower=0).dropna()
    ZHAO02_raw = _month_aggregate(ZHAO02_raw, how="last")
    _register_factor(raw_factor_df, factor_source_df, "ZHAO02_raw", quantile_rank, ZHAO02_raw)

    # ### PB_QRD (ZHAO03)
    df_component = pd.read_parquet(_find_data_file("IndexComponents.parquet"))
    df_pb = pd.read_parquet(_find_data_file("pb.parquet"))
    target = "399370.SZ"
    comp_date = "TRADE_DT"
    comp_code = "S_CON_WINDCODE"
    pb_date = "TRADE_DT"
    df_component[comp_date] = normalize_trade_dt(df_component[comp_date])
    df_pb[pb_date] = normalize_trade_dt(df_pb[pb_date])
    df_component[comp_code] = df_component[comp_code].astype(str).str.strip()

    df_pb_wide = df_pb.copy()
    if isinstance(df_pb_wide.index, pd.MultiIndex):
        if pb_date in df_pb_wide.index.names:
            df_pb_wide = df_pb_wide.reset_index()
    elif df_pb_wide.index.name == pb_date:
        df_pb_wide = df_pb_wide.reset_index()

    if isinstance(df_pb_wide.columns, pd.MultiIndex):
        def _flatten_pb_col(col):
            parts = [str(item).strip() for item in col if pd.notna(item) and str(item).strip() not in ("", "nan")]
            wind_codes = [item for item in parts if re.match(r"^[0-9A-Z]{6}\.(SH|SZ|BJ)$", item)]
            return wind_codes[0] if wind_codes else (parts[0] if parts else "")

        df_pb_wide.columns = [_flatten_pb_col(col) for col in df_pb_wide.columns.to_flat_index()]

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

    component_monthly = df_component.copy()
    pb_trade_dates = df_pb.copy()
    component_monthly["TRADE_DT"] = pd.to_datetime(component_monthly["TRADE_DT"])
    pb_trade_dates["TRADE_DT"] = pd.to_datetime(pb_trade_dates["TRADE_DT"])
    trade_dates = pb_trade_dates[["TRADE_DT"]].drop_duplicates().sort_values("TRADE_DT")
    component_monthly = component_monthly.sort_values("TRADE_DT")
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
        direction="backward",
    )
    df_component_daily = date_map.merge(
        component_monthly,
        left_on="COMPONENT_DT",
        right_on="TRADE_DT",
        how="left",
        suffixes=("", "_component"),
    )
    df_component_daily = df_component_daily.drop(columns=["TRADE_DT_component"])
    df_component_daily = df_component_daily.rename(columns={"TRADE_DT": "TRADE_DT"})
    df_component_daily = df_component_daily.sort_values(
        ["TRADE_DT", "S_INFO_WINDCODE", "S_CON_WINDCODE"]
    ).reset_index(drop=True)

    row_pos = df_pb_wide.index.get_indexer(df_component_daily[comp_date])
    col_pos = df_pb_wide.columns.get_indexer(df_component_daily[comp_code])
    pb_values = np.full(len(df_component_daily), np.nan, dtype=object)
    valid_pb = (row_pos >= 0) & (col_pos >= 0)
    pb_matrix = df_pb_wide.to_numpy()
    pb_values[valid_pb] = pb_matrix[row_pos[valid_pb], col_pos[valid_pb]]
    df_component_daily["pb_value"] = pd.to_numeric(pd.Series(pb_values, index=df_component_daily.index), errors="coerce")
    df_merged_pb = df_component_daily.dropna().reset_index(drop=True)

    target_index_list = [target]
    df_merged_pb = df_merged_pb[df_merged_pb["S_INFO_WINDCODE"].isin(target_index_list)].copy()
    daily_qrd = (
        df_merged_pb
        .groupby(["S_INFO_WINDCODE", "TRADE_DT"])["pb_value"]
        .apply(calc_qrd)
        .reset_index(name="PB_QRD")
    )
    daily_qrd = daily_qrd.sort_values(["S_INFO_WINDCODE", "TRADE_DT"])
    daily_qrd["PB_QRD_MA20"] = (
        daily_qrd
        .groupby("S_INFO_WINDCODE")["PB_QRD"]
        .transform(lambda s: s.rolling(window=20, min_periods=20).mean())
    )
    qrd_ts = daily_qrd.set_index("TRADE_DT").sort_index()[["PB_QRD", "PB_QRD_MA20"]].dropna()
    qrd_ts["ZHAO03"] = qrd_ts["PB_QRD_MA20"] / qrd_ts["PB_QRD_MA20"].shift(1) - 1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO03_raw", qrd_ts["ZHAO03"])

    # ### PB_MAD (ZHAO04)
    df_merged_pb["pb_median"] = df_merged_pb.groupby("TRADE_DT")["pb_value"].transform("median")
    df_merged_pb["abs_dev"] = (df_merged_pb["pb_value"] - df_merged_pb["pb_median"]).abs()
    mad_df = df_merged_pb.groupby("TRADE_DT")["abs_dev"].median().rename("PB_MAD").reset_index()
    df_merged_pb["PB_MAD"] = df_merged_pb.groupby("TRADE_DT")["abs_dev"].transform("median")
    mad_df["PB_MAD_MA20"] = mad_df["PB_MAD"].rolling(window=20, min_periods=20).mean()
    mad_df = mad_df.dropna()
    mad_df["ZHAO04"] = mad_df["PB_MAD_MA20"] / mad_df["PB_MAD_MA20"].shift(1) - 1
    mad_df.set_index("TRADE_DT", inplace=True)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO04_raw", mad_df["ZHAO04"])

    # ### 新增中长期人民币贷款 (ZHAO05)
    sub_1 = _read_indicator_series("DebtData.xlsx", "中国:金融机构:新增人民币贷款:中长期:当月值")
    ZHAO05_raw = _rolling_sum_ratio_minus_one(sub_1, window=12)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO05_raw", ZHAO05_raw)

    # ### PMI (ZHAO06)
    df_PMI = _load_macro_all()
    df_PMI = clean_macro_table(df_PMI, nation="中国", indi="官方制造业PMI")
    df_PMI = df_PMI[~df_PMI.index.duplicated(keep="last")]
    df_pmi_chain = pmi_yoy_chain(df_PMI, value_col="actual", scale=0.01)
    ZHAO06_raw = df_pmi_chain["yoy_chain"]
    _register_factor(raw_factor_df, factor_source_df, "ZHAO06_raw", ZHAO06_raw)

    # ### 1年期国债到期收益率 (ZHAO07)
    sub_1 = _read_indicator_series("D_国债到期收益率_CN_020104_260409.xlsx", "中债国债到期收益率:1年")
    monthly_avg = _month_aggregate(sub_1, how="average")
    ZHAO07_raw = _YoY(monthly_avg) * (-1)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO07_raw", ZHAO07_raw)

    # ### 2年期美国国债到期收益率 (ZHAO08)
    sub_1 = _read_indicator_series("D_国债收益率_US_530430_260324.xlsx", "美国:国债收益率:2年")
    monthly_avg = _month_aggregate(sub_1, how="average")
    ZHAO08_raw = _YoY(monthly_avg) * (-1)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO08_raw", ZHAO08_raw)

    # ### 新增规上工业企业利润总额 (ZHAO09)s
    sub_1 = _read_indicator_series("规模以上工业 招证资配.xlsx", "中国:利润总额:规模以上工业企业:累计值")
    ZHAO09_raw = _rolling_sum_ratio_minus_one(sub_1, window=12)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO09_raw", ZHAO09_raw)

    # ### 工业企业产成品存货 (ZHAO10)
    sub_1 = _read_indicator_series("规模以上工业 招证资配.xlsx", "中国:产成品存货:规模以上工业企业:同比")
    ZHAO10_raw = 0 - sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO10_raw", ZHAO10_raw)

    # ### 新增社零 (ZHAO11)
    sub_1 = _load_china_macro_series("社会消费品零售总额")
    ZHAO11_raw = _rolling_sum_ratio_minus_one(sub_1, window=12, shift=11)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO11_raw", ZHAO11_raw)

    # ### 新增出口额（美元） (ZHAO12)
    sub_1 = _load_china_macro_series("月出口金额:当月值(亿美元)")
    ZHAO12_raw = _rolling_sum_ratio_minus_one(sub_1, window=12)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO12_raw", ZHAO12_raw)

    # ### 一般公共预算支出 (ZHAO13)
    sub_1 = _read_indicator_series("公共预算支出.xlsx", "中国:一般公共预算支出:当月同比(1-2月合并)")
    ZHAO13_raw = sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO13_raw", ZHAO13_raw)

    # ### 美元兑人民币中间价 (ZHAO14)
    sub_1 = _read_indicator_series("日频汇率.xlsx", "中间价:美元兑人民币")
    monthly_avg = _month_aggregate(sub_1, how="average")
    ZHAO14_raw = _YoY(monthly_avg) * (-1)
    _register_factor(raw_factor_df, factor_source_df, "ZHAO14_raw", ZHAO14_raw)

    # ### M0同比 (ZHAO15)
    sub_1 = _load_china_macro_series("月M0:同比(%)")
    ZHAO15_raw = sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO15_raw", ZHAO15_raw)

    # ### M1同比 (ZHAO16)
    sub_1 = _load_china_macro_series("月M1:同比(%)")
    ZHAO16_raw = sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO16_raw", ZHAO16_raw)

    # ### M2同比 (ZHAO17)
    sub_1 = _load_china_macro_series("月M2:同比(%)")
    ZHAO17_raw = sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO17_raw", ZHAO17_raw)

    # ### M1同比-M2同比 (ZHAO18)
    sub_1 = _load_china_macro_series("月M1:同比(%)")
    sub_2 = _load_china_macro_series("月M2:同比(%)")
    ZHAO18_raw = sub_1 - sub_2
    _register_factor(raw_factor_df, factor_source_df, "ZHAO18_raw", ZHAO18_raw)

    # ### PPI (ZHAO20)
    sub_1 = _load_china_macro_series("PPI:同比")
    ZHAO20_raw = sub_1
    _register_factor(raw_factor_df, factor_source_df, "ZHAO20_raw", ZHAO20_raw)

    return factor_source_df
