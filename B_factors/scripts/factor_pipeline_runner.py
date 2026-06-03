"""因子模块独立运行时的公共 pipeline 入口。"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from factor_utils import (
    build_threshold_signal_ls_df,
    load_benchmark_index,
    load_default_data,
    mount_factor_source_frame,
    save_factor_outputs,
    save_generated_factor_records,
    validate_prepared_mapping,
)


FactorRecords = list[dict[str, object]]
FactorMetadata = dict[str, dict[str, object]]


def run_factor_module_pipeline(
    *,
    output_prefix: str,
    generate_factors: Callable[[pd.DataFrame], tuple[pd.DataFrame, FactorRecords]],
    metadata_builder: Callable[[FactorRecords], FactorMetadata],
    print_summary: Callable[[str, pd.DataFrame, pd.DataFrame], None] | None = None,
    missing_bar_defaults: list[object] | None = None,
    write_empty_missing_bar_file: bool = False,
) -> None:
    """执行单个因子模块的标准挂载、信号和保存流程。

    这个 runner 只负责源因子矩阵生成之后的公共 pipeline；单因子公式、
    metadata 修正、因子清单维护仍留在各自模块内。默认参数保持当前独立
    模块的行为：没有额外 missing bar 文件，空 missing bar 文件也不写出。
    """
    validate_prepared_mapping()
    data_df, _market_df = load_default_data()

    factor_source_df, selected_records = generate_factors(data_df)
    metadata = metadata_builder(selected_records)
    run_factor_output_pipeline(
        output_prefix=output_prefix,
        factor_source_df=factor_source_df,
        metadata=metadata,
        selected_records=selected_records,
        print_summary=print_summary,
        missing_bar_defaults=missing_bar_defaults,
        write_empty_missing_bar_file=write_empty_missing_bar_file,
    )


def print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
    """打印因子输出摘要。

    功能：展示挂载因子矩阵、信号矩阵的形状、列名，以及每列非空数量和首末有效日期。
    输入：label，输出前缀；mounted_factor_df，挂载后的标准化因子；signal_ls_df，多空信号。
    输出：无返回值；摘要打印到 stdout。
    """
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


def run_factor_output_pipeline(
    *,
    output_prefix: str,
    factor_source_df: pd.DataFrame,
    metadata: FactorMetadata,
    selected_records: FactorRecords,
    print_summary: Callable[[str, pd.DataFrame, pd.DataFrame], None] | None = None,
    missing_bar_defaults: list[object] | None = None,
    write_empty_missing_bar_file: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行源因子矩阵之后的公共输出流程。

    功能：校验 prepared 数据映射，读取市场日历和基准日历，挂载源因子，
        生成 signal_ls，保存 parquet/xlsx 输出，并更新 factor_generated.json。
    输入：output_prefix，输出文件前缀；factor_source_df，源因子矩阵；
        metadata，挂载和信号所需 metadata；selected_records，需要登记的记录；
        print_summary，摘要打印函数；missing_bar_defaults，bar 缺失默认记录；
        write_empty_missing_bar_file，是否在无缺失 bar 时也写说明文件。
    输出：挂载后的因子矩阵和 signal_ls 矩阵。
    """
    validate_prepared_mapping()
    _data_df, market_df = load_default_data()
    benchmark_index = load_benchmark_index()

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
        missing_bar_defaults=[] if missing_bar_defaults is None else missing_bar_defaults,
        output_prefix=output_prefix,
        write_empty_missing_bar_file=write_empty_missing_bar_file,
    )

    for label, path in output_paths.items():
        print(f"{label} saved to:", path)
    generated_path = save_generated_factor_records(selected_records, output_prefix)
    print("generated records saved to:", generated_path)
    summary_printer = print_factor_output_summary if print_summary is None else print_summary
    summary_printer(output_prefix, mounted_normalized_factor_df, signal_ls_df)
    return mounted_normalized_factor_df, signal_ls_df
