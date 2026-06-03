"""注册表驱动的贴现率实验室因子。

本脚本按一个直观顺序生成 I_laboratory 因子：
1. 读取 registry，得到要生成的因子清单；
2. 对每个因子读取对应 prepared 数据字段；
3. 调用 factor_factory_templates.py 中登记的模板函数计算源因子；
4. 所有源因子生成完后，一起交给项目统一 pipeline 挂载、生成信号并保存。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import factor_factory_templates
from factor_metadata import build_metadata_from_records
from factor_pipeline_runner import run_factor_module_pipeline
from factor_utils import PROJECT_ROOT, _register_factor, read_prepared_series


OUTPUT_PREFIX = "I_laboratory"
REGISTRY_PATH = PROJECT_ROOT / "B_factors" / "reference" / "factor_factory_factors_registry.json"
DEFAULT_SIGNAL_TYPE = "state"

TemplateFunc = Callable[[pd.Series, Mapping[str, object] | None], pd.Series]
TEMPLATE_FUNCTIONS: dict[str, TemplateFunc] = {
    name: template_func
    for name, template_func in vars(factor_factory_templates).items()
    if callable(template_func) and not name.startswith("_")
}


def _required_text(record: Mapping[str, object], field_name: str) -> str:
    """读取必填文本字段，缺失时直接报错。"""
    value = str(record.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"registry record missing required field {field_name!r}: {record}")
    return value


def _parse_default_params(record: Mapping[str, object]) -> dict[str, object]:
    """把 registry 中的 default_params JSON 字符串解析为模板参数。"""
    raw_params = str(record.get("default_params") or "").strip()
    if not raw_params:
        return {}
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        factor_id = record.get("编号")
        raise ValueError(f"{factor_id} default_params is not valid JSON: {raw_params}") from exc
    if not isinstance(params, dict):
        factor_id = record.get("编号")
        raise ValueError(f"{factor_id} default_params must decode to a JSON object")
    return params


def _normalize_factor_record(
    raw_record: Mapping[str, object],
    *,
    source_file: str,
    source_sheet: str,
) -> dict[str, object]:
    """把一行中文 registry 记录整理成后续 pipeline 能识别的因子记录。"""
    record = dict(raw_record)
    factor_id = _required_text(record, "编号")
    data_path = _required_text(record, "数据路径")
    source_field = _required_text(record, "字段名")
    template_name = _required_text(record, "template")
    signal_type = str(record.get("数据频率") or DEFAULT_SIGNAL_TYPE).strip() or DEFAULT_SIGNAL_TYPE

    record["_source_file"] = source_file
    record["_source_sheet"] = source_sheet
    record["factor_id"] = factor_id
    record["signal_type"] = signal_type
    record["bar"] = 0.0
    record["factor"] = record.get("测度目标")
    record["progress"] = "done"
    record["paper_id"] = OUTPUT_PREFIX
    record["docu"] = Path(data_path).name
    record["data_field"] = source_field
    record["_factory_template"] = template_name
    record["_factory_params"] = _parse_default_params(record)
    return record


def load_factor_plan() -> list[dict[str, object]]:
    """读取 registry，返回按原始顺序排列的 I_laboratory 因子清单。"""
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    source_file = str(REGISTRY_PATH.relative_to(PROJECT_ROOT))
    factor_plan_records: list[dict[str, object]] = []

    for source_sheet, sheet_meta in payload.get("sheets", {}).items():
        for raw_record in sheet_meta.get("records", []):
            factor_plan_records.append(
                _normalize_factor_record(
                    raw_record,
                    source_file=source_file,
                    source_sheet=source_sheet,
                )
            )

    if not factor_plan_records:
        raise ValueError(f"{REGISTRY_PATH.name} does not contain any factor records")

    factor_ids = [str(record["factor_id"]) for record in factor_plan_records]
    duplicated = sorted({factor_id for factor_id in factor_ids if factor_ids.count(factor_id) > 1})
    if duplicated:
        raise ValueError(f"{REGISTRY_PATH.name} duplicated factor_id values: {duplicated}")

    return factor_plan_records


FACTOR_PLAN_RECORDS = load_factor_plan()
FACTOR_IDS = [str(record["factor_id"]) for record in FACTOR_PLAN_RECORDS]


def load_factor_input(record: Mapping[str, object]) -> pd.Series:
    """根据单个因子记录读取 prepared 数据字段。"""
    data_path = _required_text(record, "数据路径")
    prepared_table = Path(data_path).name
    if not prepared_table:
        raise ValueError(f"cannot infer prepared table name from 数据路径={data_path!r}")

    source_field = _required_text(record, "字段名")

    # 只保留真实观测点进入模板计算；这里不做 ffill，避免扩展状态有效期。
    source_series = read_prepared_series(prepared_table, source_field).dropna()
    return source_series


def metadata_from_I_laboratory_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """把 I_laboratory 因子记录转换为挂载和信号流程需要的 metadata。"""
    return build_metadata_from_records(records, default_signal_type=DEFAULT_SIGNAL_TYPE)


def generate_I_laboratory_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """按“清单 -> 输入 -> 模板 -> 源因子矩阵”的顺序生成全部 I_laboratory 因子。"""
    data_df = data_df.copy()
    data_df.index = pd.to_datetime(data_df.index)
    data_df = data_df.sort_index()
    data_index = pd.DatetimeIndex(data_df.index)

    raw_factor_df = pd.DataFrame(index=data_index)
    factor_source_df = pd.DataFrame(index=data_index)
    factor_plan_records = [dict(record) for record in FACTOR_PLAN_RECORDS]

    for record in factor_plan_records:
        factor_id = str(record["factor_id"])
        template_name = str(record["_factory_template"])
        template_params = record.get("_factory_params")

        # 第一步：按 registry 中的数据路径和字段名读取原始输入序列。
        source_series = load_factor_input(record)

        # 第二步：按 registry 的 template 名称找到 factor_factory_templates.py 里的函数。
        template_func = TEMPLATE_FUNCTIONS.get(template_name)
        if template_func is None:
            raise KeyError(f"Unsupported factor factory template: {template_name!r}")

        # 第三步：直接调用模板函数生成因子值，例如：
        # rolling_rank_level(source_series, {"lookback_window": 756, "direction": -1})
        factor_series = template_func(source_series, template_params)
        factor_series.name = factor_id
        factor_series = factor_series.astype("float64")

        # 第四步：注册到项目标准源因子矩阵，列名会落到 factor_id。
        _register_factor(raw_factor_df, factor_source_df, f"{factor_id}_raw", factor_series)

    missing_cols = [factor_id for factor_id in FACTOR_IDS if factor_id not in factor_source_df.columns]
    if missing_cols:
        raise ValueError(f"I_laboratory factor columns missing after generation: {missing_cols}")

    return factor_source_df.loc[:, FACTOR_IDS], factor_plan_records


def generate_I_laboratory_factor_source_frame(data_df: pd.DataFrame) -> pd.DataFrame:
    """给上层入口调用的轻量接口，只返回源因子矩阵。"""
    factor_source_df, _records = generate_I_laboratory_factors(data_df)
    return factor_source_df


def _print_factor_output_summary(label: str, mounted_factor_df: pd.DataFrame, signal_ls_df: pd.DataFrame) -> None:
    """打印输出摘要，方便人工确认生成结果。"""
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
    """命令行入口：独立生成 I_laboratory 因子输出。"""
    run_factor_module_pipeline(
        output_prefix=OUTPUT_PREFIX,
        generate_factors=generate_I_laboratory_factors,
        metadata_builder=metadata_from_I_laboratory_records,
        print_summary=_print_factor_output_summary,
    )


if __name__ == "__main__":
    main()
