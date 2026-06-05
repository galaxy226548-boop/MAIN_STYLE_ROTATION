"""批量生成组合信号、运行回测 pipeline，并汇总结果。

默认读取:
    F_grouping/reference/grouping_batches.xlsx

输出:
    F_grouping/output/grouping_batch_result_summary.xlsx
    F_grouping/output/grouping_batch_run_log.xlsx

示例:
    python F_grouping/scripts/run_grouping_batches.py --dry-run
    python F_grouping/scripts/run_grouping_batches.py --combo W021
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SCRIPT_DIR = SCRIPT_PATH.parent
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_factor_group_interactive as factor_group_builder
import build_signal_altogether_binary_interactive as altogether_binary_builder
import build_signal_group_binary_interactive as signal_group_binary_builder
import build_signal_group_interactive as signal_group_builder


BATCH_PATH = PROJECT_ROOT / "F_grouping" / "reference" / "grouping_batches.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "F_grouping" / "output"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "grouping_batch_result_summary.xlsx"
LOG_OUTPUT_PATH = OUTPUT_DIR / "grouping_batch_run_log.xlsx"
PIPELINE_SCRIPT = PROJECT_ROOT / "G_engine" / "run_factor_pipeline.py"
POSITION_OUTPUT_ROOT = PROJECT_ROOT / "C_positions" / "output" / "factor_positions"
BACKTEST_OUTPUT_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"
IC_OUTPUT_ROOT = PROJECT_ROOT / "D_analysis" / "IC_output"

GROUPED_SHEETS = {"字母分类", "变量分类"}
SAMPLE_SHEETS = ["ins", "oos", "all"]
SUMMARY_FILE_PATTERN = "*_rebalance_50_summary.xlsx"

SUMMARY_METRIC_COLUMNS = [
    "trade_count",
    "signal_count",
    "ann_ret_long_abs",
    "ann_ret_ls_abs",
    "ann_vol_abs",
    "max_dd_abs",
    "sharpe_abs",
    "ann_ret_long_excess",
    "ann_vol_excess",
    "max_dd_excess",
    "sharpe_excess",
    "information_ratio",
    "monthly_win_rate",
    "payoff_ratio",
    "period_win_rate",
    "expectancy",
    "calmar_ratio",
    "turnover_2way",
    "growth_regime_win_rate",
    "value_regime_win_rate",
]


@dataclass
class BatchRow:
    """保存 Excel 中一行组合定义。"""

    sheet_name: str
    row_number: int
    group_id: str
    groups: OrderedDict[str, list[str]]
    raw_group_values: OrderedDict[str, str]

    @property
    def factor_names(self) -> list[str]:
        """按 Excel 顺序展开并去重底层因子。"""
        result: list[str] = []
        seen: set[str] = set()
        for factors in self.groups.values():
            for factor in factors:
                if factor not in seen:
                    result.append(factor)
                    seen.add(factor)
        return result


@dataclass
class Artifact:
    """记录一组可回测的组合产物。"""

    kind: str
    run_name: str
    factor_path: Path
    signal_path: Path
    final_factor: str
    subgroup_factors: list[str]
    output_paths: dict[str, Path]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 F_grouping 组合、运行回测并汇总结果。")
    parser.add_argument("--batch-path", type=Path, default=BATCH_PATH, help="组合批次 Excel 路径。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="汇总输出目录。")
    parser.add_argument("--sheet", action="append", help="只运行指定 sheet，可重复传入。默认运行全部 sheet。")
    parser.add_argument("--combo", action="append", help="只运行指定组合编号，例如 W021，可重复传入。")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不生成、不回测、不汇总。")
    parser.add_argument("--skip-build", action="store_true", help="跳过组合产物生成，直接使用已有 output_COMB 产物。")
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过 pipeline，只尝试汇总已有回测结果。")
    parser.add_argument("--skip-subgroups", action="store_true", help="不回测组内平均后的一级组列。")
    parser.add_argument("--limit", type=int, help="最多运行前 N 个组合，便于试跑。")
    return parser.parse_args()


def python_executable() -> str:
    """优先使用项目虚拟环境。"""
    if LOCAL_PYTHON.exists():
        return str(LOCAL_PYTHON)
    return sys.executable


def resolve_project_path(path: Path) -> Path:
    """把相对路径解析到项目根目录。"""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_factor_cell(value: object) -> list[str]:
    """解析单元格中的因子列表。"""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("，", ",").replace("\n", ",")
    if "," in text:
        parts = text.split(",")
    else:
        parts = text.split()
    return [part.strip().strip("'").strip('"') for part in parts if part.strip()]


def read_batch_rows(batch_path: Path, selected_sheets: set[str] | None, selected_combos: set[str] | None) -> list[BatchRow]:
    """读取 grouping_batches.xlsx 并转成批处理定义。"""
    if not batch_path.exists():
        raise FileNotFoundError(f"找不到组合批次 Excel: {batch_path}")

    excel = pd.ExcelFile(batch_path)
    sheet_names = [name for name in excel.sheet_names if selected_sheets is None or name in selected_sheets]
    rows: list[BatchRow] = []

    for sheet_name in sheet_names:
        df = pd.read_excel(batch_path, sheet_name=sheet_name)
        if df.empty:
            continue
        id_col = df.columns[0]
        group_columns = [column for column in df.columns[1:] if not str(column).startswith("Unnamed")]

        for row_idx, row in df.iterrows():
            group_id = str(row.get(id_col, "")).strip()
            if not group_id or group_id.lower() == "nan":
                continue
            if selected_combos is not None and group_id not in selected_combos:
                continue

            groups: OrderedDict[str, list[str]] = OrderedDict()
            raw_values: OrderedDict[str, str] = OrderedDict()
            for column in group_columns:
                factors = parse_factor_cell(row.get(column))
                if not factors:
                    continue
                group_name = str(column).strip()
                groups[group_name] = factors
                raw_values[group_name] = ", ".join(factors)

            if groups:
                rows.append(
                    BatchRow(
                        sheet_name=sheet_name,
                        row_number=int(row_idx) + 2,
                        group_id=group_id,
                        groups=groups,
                        raw_group_values=raw_values,
                    )
                )

    return rows


def write_outputs(
    factor_df: pd.DataFrame,
    signal_ls_df: pd.DataFrame,
    output_paths: dict[str, Path],
    record: dict[str, Any],
    writer_func: Any,
) -> None:
    """统一写 parquet、xlsx 和记录 JSON。"""
    output_paths["factor_parquet"].parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_parquet(output_paths["factor_parquet"])
    signal_ls_df.to_parquet(output_paths["signal_parquet"])
    writer_func(factor_df, output_paths["factor_xlsx"])
    writer_func(signal_ls_df, output_paths["signal_xlsx"])
    output_paths["record_json"].write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_factor_group(row: BatchRow, dry_run: bool, skip_build: bool) -> Artifact:
    """生成 factor group 产物。"""
    output_paths = factor_group_builder.build_output_paths(row.group_id)
    subgroup_factors = [f"{row.group_id}_{group_name}" for group_name in row.groups]
    artifact = Artifact(
        kind="factor",
        run_name=f"{row.group_id}_factor",
        factor_path=output_paths["factor_parquet"],
        signal_path=output_paths["signal_parquet"],
        final_factor=row.group_id,
        subgroup_factors=subgroup_factors,
        output_paths=output_paths,
        warnings=[],
    )
    if dry_run or skip_build:
        return artifact

    input_files = factor_group_builder.list_input_files(factor_group_builder.INPUT_DIR)
    source_df, scan_record, scan_warnings = factor_group_builder.collect_selected_factors(row.factor_names, input_files)
    factor_df, groups, factor_warnings = factor_group_builder.build_factor_matrix(
        source_df=source_df,
        factor_group=row.group_id,
        custom_groups=row.groups,
    )
    signal_ls_df = factor_group_builder.build_signal_ls_matrix(factor_df)
    warnings = scan_warnings + factor_warnings
    record = factor_group_builder.build_record(
        factor_group=row.group_id,
        requested_factor_names=row.factor_names,
        source_df=source_df,
        factor_df=factor_df,
        groups=groups,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )
    record["batch_sheet"] = row.sheet_name
    record["batch_row_number"] = row.row_number
    record["batch_grouping_columns"] = row.raw_group_values
    write_outputs(factor_df, signal_ls_df, output_paths, record, factor_group_builder.write_factor_frame_xlsx)
    artifact.warnings = warnings
    return artifact


def build_signal_group(row: BatchRow, dry_run: bool, skip_build: bool) -> Artifact:
    """生成 signal group 产物。"""
    base_group, signal_group = signal_group_builder.normalize_factor_group(row.group_id)
    output_paths = signal_group_builder.build_output_paths(signal_group)
    subgroup_factors = [f"{base_group}_{group_name}{signal_group_builder.SIGNAL_SUFFIX}" for group_name in row.groups]
    artifact = Artifact(
        kind="signal",
        run_name=f"{base_group}_signal",
        factor_path=output_paths["factor_parquet"],
        signal_path=output_paths["signal_parquet"],
        final_factor=signal_group,
        subgroup_factors=subgroup_factors,
        output_paths=output_paths,
        warnings=[],
    )
    if dry_run or skip_build:
        return artifact

    input_files = signal_group_builder.list_input_files(signal_group_builder.INPUT_DIR)
    source_df, scan_record, scan_warnings = signal_group_builder.collect_selected_factors(row.factor_names, input_files)
    factor_df, groups, factor_warnings = signal_group_builder.build_factor_matrix(
        source_df=source_df,
        base_group=base_group,
        signal_group=signal_group,
        custom_groups=row.groups,
    )
    warnings = scan_warnings + factor_warnings
    signal_ls_df = signal_group_builder.build_signal_ls_matrix(factor_df)
    record = signal_group_builder.build_record(
        input_factor_group=row.group_id,
        base_group=base_group,
        signal_group=signal_group,
        requested_factor_names=row.factor_names,
        source_df=source_df,
        factor_df=factor_df,
        groups=groups,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )
    record["batch_sheet"] = row.sheet_name
    record["batch_row_number"] = row.row_number
    record["batch_grouping_columns"] = row.raw_group_values
    write_outputs(factor_df, signal_ls_df, output_paths, record, signal_group_builder.write_factor_frame_xlsx)
    artifact.warnings = warnings
    return artifact


def build_signal_group_binary(row: BatchRow, dry_run: bool, skip_build: bool) -> Artifact:
    """生成先组内平均、再组间平均、最后二值化的 signal 产物。"""
    base_group, signal_group, binary_group = signal_group_binary_builder.normalize_factor_group(row.group_id)
    output_paths = signal_group_binary_builder.build_output_paths(binary_group)
    subgroup_factors = [f"{base_group}_{group_name}{signal_group_binary_builder.SIGNAL_SUFFIX}" for group_name in row.groups]
    artifact = Artifact(
        kind="signal_binary",
        run_name=f"{base_group}_signal_binary",
        factor_path=output_paths["factor_parquet"],
        signal_path=output_paths["signal_parquet"],
        final_factor=binary_group,
        subgroup_factors=subgroup_factors,
        output_paths=output_paths,
        warnings=[],
    )
    if dry_run or skip_build:
        return artifact

    input_files = signal_group_binary_builder.list_input_files(signal_group_binary_builder.INPUT_DIR)
    source_df, scan_record, scan_warnings = signal_group_binary_builder.collect_selected_factors(row.factor_names, input_files)
    factor_df, groups, factor_warnings = signal_group_binary_builder.build_factor_matrix(
        source_df=source_df,
        base_group=base_group,
        signal_group=signal_group,
        binary_group=binary_group,
        custom_groups=row.groups,
    )
    warnings = scan_warnings + factor_warnings
    signal_ls_df = signal_group_binary_builder.build_signal_ls_matrix(factor_df)
    record = signal_group_binary_builder.build_record(
        input_factor_group=row.group_id,
        base_group=base_group,
        signal_group=signal_group,
        binary_group=binary_group,
        requested_factor_names=row.factor_names,
        source_df=source_df,
        factor_df=factor_df,
        signal_ls_df=signal_ls_df,
        groups=groups,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )
    record["batch_sheet"] = row.sheet_name
    record["batch_row_number"] = row.row_number
    record["batch_grouping_columns"] = row.raw_group_values
    write_outputs(factor_df, signal_ls_df, output_paths, record, signal_group_binary_builder.write_factor_frame_xlsx)
    artifact.warnings = warnings
    return artifact


def build_altogether_binary(row: BatchRow, dry_run: bool, skip_build: bool) -> Artifact:
    """生成全体因子直接平均后二值化的 signal 产物。"""
    base_group, binary_group = altogether_binary_builder.normalize_factor_group(row.group_id)
    output_paths = altogether_binary_builder.build_output_paths(binary_group)
    artifact = Artifact(
        kind="binary",
        run_name=f"{base_group}_binary",
        factor_path=output_paths["factor_parquet"],
        signal_path=output_paths["signal_parquet"],
        final_factor=binary_group,
        subgroup_factors=[],
        output_paths=output_paths,
        warnings=[],
    )
    if dry_run or skip_build:
        return artifact

    input_files = altogether_binary_builder.list_input_files(altogether_binary_builder.INPUT_DIR)
    source_df, scan_record, scan_warnings = altogether_binary_builder.collect_selected_factors(row.factor_names, input_files)
    factor_df, factor_warnings = altogether_binary_builder.build_factor_matrix(
        source_df=source_df,
        binary_group=binary_group,
    )
    warnings = scan_warnings + factor_warnings
    signal_ls_df = altogether_binary_builder.build_signal_ls_matrix(
        factor_df=factor_df,
        source_cols=row.factor_names,
        binary_group=binary_group,
        warnings=warnings,
    )
    record = altogether_binary_builder.build_record(
        input_factor_group=row.group_id,
        base_group=base_group,
        binary_group=binary_group,
        requested_factor_names=row.factor_names,
        source_df=source_df,
        factor_df=factor_df,
        signal_ls_df=signal_ls_df,
        scan_record=scan_record,
        warnings=warnings,
        output_paths=output_paths,
    )
    record["batch_sheet"] = row.sheet_name
    record["batch_row_number"] = row.row_number
    record["batch_grouping_columns"] = row.raw_group_values
    write_outputs(factor_df, signal_ls_df, output_paths, record, altogether_binary_builder.write_factor_frame_xlsx)
    artifact.warnings = warnings
    return artifact


def build_artifacts(row: BatchRow, dry_run: bool, skip_build: bool) -> list[Artifact]:
    """按四种组合方式生成产物。"""
    return [
        build_factor_group(row, dry_run=dry_run, skip_build=skip_build),
        build_signal_group(row, dry_run=dry_run, skip_build=skip_build),
        build_signal_group_binary(row, dry_run=dry_run, skip_build=skip_build),
        build_altogether_binary(row, dry_run=dry_run, skip_build=skip_build),
    ]


def clean_run_outputs(run_names: set[str]) -> list[str]:
    """覆盖重跑前清理本批次 run_name 的旧输出。"""
    removed: list[str] = []
    for root in [POSITION_OUTPUT_ROOT, BACKTEST_OUTPUT_ROOT, IC_OUTPUT_ROOT]:
        for run_name in sorted(run_names):
            path = root / run_name
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path))
    return removed


def pipeline_command(artifact: Artifact, sample: str, factor: str | None = None) -> list[str]:
    """组装 run_factor_pipeline.py 命令。"""
    command = [
        python_executable(),
        str(PIPELINE_SCRIPT),
        "--signal-path",
        str(artifact.signal_path),
        "--factor-path",
        str(artifact.factor_path),
        "--run-name",
        artifact.run_name,
        "--sample",
        sample,
    ]
    if factor is not None:
        command.extend(["--factor", factor])
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    """执行命令；dry-run 时只打印。"""
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_pipeline_for_artifacts(artifacts: list[Artifact], dry_run: bool, skip_subgroups: bool) -> list[dict[str, Any]]:
    """对组合总列和一级组列运行 ins/oos/all 回测。"""
    logs: list[dict[str, Any]] = []
    for artifact in artifacts:
        factors: list[str | None] = [None]
        if not skip_subgroups:
            factors.extend(artifact.subgroup_factors)

        for factor in factors:
            for sample in ["both", "all"]:
                command = pipeline_command(artifact, sample=sample, factor=factor)
                label = factor or artifact.final_factor
                log_row = {
                    "kind": artifact.kind,
                    "run_name": artifact.run_name,
                    "factor": label,
                    "sample": sample,
                    "command": " ".join(command),
                    "status": "success" if not dry_run else "dry_run",
                    "error": "",
                }
                try:
                    run_command(command, dry_run=dry_run)
                except Exception as exc:  # noqa: BLE001
                    log_row["status"] = "failed"
                    log_row["error"] = str(exc)
                    logs.append(log_row)
                    raise
                logs.append(log_row)
    return logs


def find_summary_file(run_name: str, sample: str, factor_name: str) -> Path | None:
    """查找指定 run/sample/factor 的回测 summary 文件。"""
    sample_dir = BACKTEST_OUTPUT_ROOT / run_name / sample
    if not sample_dir.exists():
        return None
    matches = sorted(
        path
        for path in sample_dir.rglob(SUMMARY_FILE_PATTERN)
        if path.name.endswith(f"_{factor_name}_rebalance_50_summary.xlsx")
    )
    if not matches:
        return None
    if len(matches) > 1:
        # 覆盖重跑会先清理目录；若仍有多个，取最新修改时间的文件并在 file_name 中保留路径。
        matches = sorted(matches, key=lambda item: item.stat().st_mtime)
    return matches[-1]


def read_screening(path: Path) -> dict[str, Any]:
    """读取 screening sheet 中的 pass 和核心统计。"""
    df = pd.read_excel(path, sheet_name="screening")
    if df.shape[1] < 2:
        return {}
    key_col = df.columns[0]
    value_col = df.columns[1]
    key_series = df[key_col].astype(str).str.strip()
    result: dict[str, Any] = {}

    pass_rows = df.loc[key_series == "pass_count", value_col]
    if not pass_rows.empty:
        result["pass_count"] = pd.to_numeric(pass_rows.iloc[-1], errors="coerce")

    for metric in ["monthly_win_rate", "period_win_rate", "payoff_ratio", "expectancy"]:
        rows = df.loc[key_series == metric, value_col]
        if not rows.empty:
            result[f"{metric}.1"] = pd.to_numeric(rows.iloc[-1], errors="coerce")

    return result


def read_avg_summary_last_row(path: Path) -> dict[str, Any]:
    """从 summary sheet 定位 avg_summary，并取最后一行。"""
    raw_df = pd.read_excel(path, sheet_name="summary", header=None)
    positions = raw_df.eq("avg_summary")
    if not positions.any().any():
        return {}

    row_idx, col_idx = positions.stack()[positions.stack()].index[0]
    header_idx = int(row_idx) + 1
    data_start = header_idx + 1
    headers = raw_df.iloc[header_idx, col_idx:].tolist()
    # avg_summary 后面会接一个空行和 std_summary，汇总只取 avg_summary 自己的区块。
    first_col = raw_df.iloc[data_start:, col_idx]
    blank_offsets = first_col[first_col.isna()].index
    data_end = int(blank_offsets[0]) if len(blank_offsets) > 0 else len(raw_df)
    data = raw_df.iloc[data_start:data_end, col_idx : col_idx + len(headers)].copy()
    data.columns = headers
    data = data.loc[data["period"].notna()].copy()
    if data.empty:
        return {}

    last_row = data.iloc[-1].to_dict()
    result: dict[str, Any] = {}
    for column in SUMMARY_METRIC_COLUMNS:
        source_column = "turnover_2way_pct" if column == "turnover_2way" else column
        if source_column in last_row:
            value = pd.to_numeric(last_row[source_column], errors="coerce")
            # 原回测 summary 写的是百分数，附件汇总的 turnover_2way 使用未乘 100 的口径。
            if column == "turnover_2way":
                value = value / 100
            result[column] = value
    return result


def summary_record(
    row: BatchRow,
    artifact: Artifact,
    factor_name: str,
    sample: str,
) -> dict[str, Any]:
    """生成汇总表中的一行。"""
    summary_path = find_summary_file(artifact.run_name, sample, factor_name)
    document = factor_name
    record: dict[str, Any] = {
        "Document": document,
        "file_name": "" if summary_path is None else summary_path.parent.name,
        "batch_sheet": row.sheet_name,
        "group_id": row.group_id,
        "kind": artifact.kind,
        "run_name": artifact.run_name,
        "composite": ", ".join(row.factor_names),
    }
    for group_name, raw_value in row.raw_group_values.items():
        record[group_name] = raw_value

    if summary_path is None:
        record["summary_status"] = "missing"
        return record

    record["summary_status"] = "success"
    record.update(read_avg_summary_last_row(summary_path))
    record.update(read_screening(summary_path))
    return record


def collect_summary(rows: list[BatchRow], artifacts_by_group: dict[str, list[Artifact]], skip_subgroups: bool) -> dict[str, pd.DataFrame]:
    """汇总 ins/oos/all 三个 sheet。"""
    sample_records: dict[str, list[dict[str, Any]]] = {sample: [] for sample in SAMPLE_SHEETS}
    for row in rows:
        for artifact in artifacts_by_group.get(row.group_id, []):
            factor_names = [artifact.final_factor]
            if not skip_subgroups:
                factor_names.extend(artifact.subgroup_factors)
            for factor_name in factor_names:
                for sample in SAMPLE_SHEETS:
                    sample_records[sample].append(summary_record(row, artifact, factor_name, sample))

    return {
        sample: pd.DataFrame(records)
        for sample, records in sample_records.items()
    }


def write_summary_workbook(summary_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """写入三口径汇总工作簿。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_prefix = [
        "Document",
        "file_name",
        *SUMMARY_METRIC_COLUMNS,
        "pass_count",
        "monthly_win_rate.1",
        "period_win_rate.1",
        "payoff_ratio.1",
        "expectancy.1",
        "composite",
        "batch_sheet",
        "group_id",
        "kind",
        "run_name",
        "summary_status",
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sample in SAMPLE_SHEETS:
            df = summary_frames.get(sample, pd.DataFrame()).copy()
            if not df.empty:
                ordered = [column for column in preferred_prefix if column in df.columns]
                rest = [column for column in df.columns if column not in ordered]
                df = df.loc[:, [*ordered, *rest]]
            df.to_excel(writer, sheet_name=sample, index=False)


def write_log(logs: list[dict[str, Any]], output_path: Path) -> None:
    """写入运行日志。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_excel(output_path, index=False)


def main() -> None:
    args = parse_args()
    batch_path = resolve_project_path(args.batch_path)
    output_dir = resolve_project_path(args.output_dir)
    summary_output_path = output_dir / SUMMARY_OUTPUT_PATH.name
    log_output_path = output_dir / LOG_OUTPUT_PATH.name

    rows = read_batch_rows(
        batch_path=batch_path,
        selected_sheets=set(args.sheet) if args.sheet else None,
        selected_combos=set(args.combo) if args.combo else None,
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("没有找到可运行的组合定义。")

    print(f"待处理组合数: {len(rows)}")
    artifacts_by_group: dict[str, list[Artifact]] = {}
    logs: list[dict[str, Any]] = []

    for row in rows:
        print(f"\n[组合] {row.sheet_name} row={row.row_number} {row.group_id}: {row.factor_names}")
        artifacts = build_artifacts(row, dry_run=args.dry_run, skip_build=args.skip_build)
        artifacts_by_group[row.group_id] = artifacts
        for artifact in artifacts:
            print(
                f"- {artifact.kind}: run_name={artifact.run_name}, "
                f"final={artifact.final_factor}, subgroups={artifact.subgroup_factors}"
            )
            logs.append(
                {
                    "kind": artifact.kind,
                    "run_name": artifact.run_name,
                    "factor": artifact.final_factor,
                    "sample": "build",
                    "command": "",
                    "status": "dry_run" if args.dry_run else "built",
                    "error": "",
                    "warnings": " | ".join(artifact.warnings),
                }
            )

    if not args.dry_run and not args.skip_pipeline:
        run_names = {artifact.run_name for artifacts in artifacts_by_group.values() for artifact in artifacts}
        removed = clean_run_outputs(run_names)
        for path in removed:
            print(f"已清理旧输出: {path}")

    if not args.skip_pipeline:
        for artifacts in artifacts_by_group.values():
            logs.extend(
                run_pipeline_for_artifacts(
                    artifacts=artifacts,
                    dry_run=args.dry_run,
                    skip_subgroups=args.skip_subgroups,
                )
            )

    if not args.dry_run:
        summary_frames = collect_summary(rows, artifacts_by_group, skip_subgroups=args.skip_subgroups)
        write_summary_workbook(summary_frames, summary_output_path)
        write_log(logs, log_output_path)
        print(f"\n汇总已写入: {summary_output_path}")
        print(f"日志已写入: {log_output_path}")
    else:
        print("\ndry-run 完成，未写入文件。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
