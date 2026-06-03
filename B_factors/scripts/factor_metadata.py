"""因子 metadata 记录的共享辅助函数。

本模块只处理 JSON metadata 记录，以及挂载/信号辅助函数使用的小型
metadata 字典。不读取 prepared market data，不计算因子，不挂载因子，
不生成信号，不保存输出，也不更新 factor_generated.json。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path


Record = dict[str, object]
RecordAdjuster = Callable[[Record], Record]


def normalize_plan_text(value: object) -> str:
    """规范化计划记录中的文本字段。

    功能：将任意输入值转成去除首尾空白的字符串；空值按空字符串处理。
    输入：value，计划记录中的任意字段值。
    输出：清理后的字符串。
    """
    return str(value or "").strip()


def append_record_note(record: Record, note: str) -> None:
    """向记录追加备注文本。

    功能：读取 record 中已有的 notes 字段，将新备注追加到末尾并写回记录。
    输入：record，待更新的 metadata 记录；note，要追加的备注文本。
    输出：无返回值；函数会原地更新 record["notes"]。
    """
    existing = normalize_plan_text(record.get("notes"))
    record["notes"] = f"{existing} {note}".strip()


def build_metadata_from_records(
    records: Sequence[Record],
    default_signal_type: str = "state",
) -> dict[str, dict[str, object]]:
    """从计划记录构建因子 metadata 映射。

    功能：按 factor_id 汇总记录，生成挂载/信号流程需要的 metadata 字典。
    输入：records，包含 factor_id、signal_type、factor、progress 等字段的记录序列；
        default_signal_type，signal_type 缺失或为空时使用的默认信号类型。
    输出：以 factor_id 字符串为键、metadata 字典为值的映射。
    """
    return {
        str(record["factor_id"]): {
            "signal_type": normalize_plan_text(record.get("signal_type")) or default_signal_type,
            "bar": 0.0,
            "factor": record.get("factor"),
            "progress": record.get("progress"),
        }
        for record in records
    }


def load_plan_records(
    *,
    plan_path: Path,
    project_root: Path,
    factor_ids: Sequence[str],
    record_adjuster: RecordAdjuster,
) -> list[Record]:
    """从计划 JSON 中读取指定因子的 metadata 记录。

    功能：读取 plan_path，筛选 factor_ids 指定的记录，补充来源文件和 sheet 信息，
        并通过 record_adjuster 做标准化；若缺少指定因子记录，会抛出 ValueError。
    输入：plan_path，计划 JSON 路径；project_root，项目根目录；factor_ids，需读取的因子编号；
        record_adjuster，记录标准化/修正函数。
    输出：标准化后的 metadata 记录列表。
    """
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    factor_id_set = set(factor_ids)
    source_file = str(plan_path.relative_to(project_root))
    records: list[Record] = []
    for sheet_name, sheet_meta in payload.get("sheets", {}).items():
        for record in sheet_meta.get("records", []):
            factor_id = str(record.get("factor_id") or "")
            if factor_id not in factor_id_set:
                continue
            item = dict(record)
            item["_source_file"] = source_file
            item["_source_sheet"] = sheet_name
            records.append(record_adjuster(item))

    _raise_if_missing_records(records, factor_ids, plan_path.name)
    return records


def load_plan_or_generated_records(
    *,
    plan_path: Path,
    generated_path: Path,
    project_root: Path,
    factor_ids: Sequence[str],
    output_prefix: str,
    record_adjuster: RecordAdjuster,
    minimal_record_factory: Callable[[str], Record],
) -> list[Record]:
    """从计划文件或已生成记录中读取指定因子的 metadata。

    功能：优先从 plan_path 读取记录；若计划文件不存在，则从 generated_path 中筛选
        指定 output_prefix 的记录，并为缺失因子创建最小记录；若计划文件存在但缺少
        指定因子记录，会抛出 ValueError。
    输入：plan_path，计划 JSON 路径；generated_path，factor_generated.json 路径；
        project_root，项目根目录；factor_ids，需读取的因子编号；output_prefix，生成输出前缀；
        record_adjuster，记录标准化/修正函数；minimal_record_factory，最小记录构造函数。
    输出：标准化后的 metadata 记录列表。
    """
    factor_id_set = set(factor_ids)
    if plan_path.exists():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        source_file = str(plan_path.relative_to(project_root))
        records_source = (
            dict(record, _source_file=source_file, _source_sheet=sheet_name)
            for sheet_name, sheet_meta in payload.get("sheets", {}).items()
            for record in sheet_meta.get("records", [])
        )
    else:
        payload = json.loads(generated_path.read_text(encoding="utf-8"))
        records_source = (
            dict(record)
            for record in payload.get("records", [])
            if record.get("_generated_output_prefix") == output_prefix
        )

    records: list[Record] = []
    for record in records_source:
        factor_id = str(record.get("factor_id") or "")
        if factor_id not in factor_id_set:
            continue
        records.append(record_adjuster(dict(record)))

    missing = _missing_factor_ids(records, factor_ids)
    if missing and not plan_path.exists():
        records.extend(record_adjuster(minimal_record_factory(factor_id)) for factor_id in missing)
        missing = []
    if missing:
        raise ValueError(f"{plan_path.name} missing implemented records: {missing}")
    return records


def _missing_factor_ids(records: Sequence[Record], factor_ids: Sequence[str]) -> list[str]:
    """找出记录中缺失的因子编号。

    功能：比较 records 已包含的 factor_id 与目标 factor_ids，返回尚未覆盖的编号。
    输入：records，已读取的 metadata 记录；factor_ids，期望存在的因子编号序列。
    输出：缺失的 factor_id 列表，顺序与 factor_ids 保持一致。
    """
    found = {str(record["factor_id"]) for record in records}
    return [factor_id for factor_id in factor_ids if factor_id not in found]


def _raise_if_missing_records(records: Sequence[Record], factor_ids: Sequence[str], plan_name: str) -> None:
    """检查指定因子记录是否齐全。

    功能：确认 records 覆盖所有 factor_ids；若存在缺失编号，会抛出 ValueError。
    输入：records，已读取的 metadata 记录；factor_ids，期望存在的因子编号序列；
        plan_name，用于错误信息的计划文件名。
    输出：无返回值；记录齐全时直接结束。
    """
    missing = _missing_factor_ids(records, factor_ids)
    if missing:
        raise ValueError(f"{plan_name} missing implemented records: {missing}")
