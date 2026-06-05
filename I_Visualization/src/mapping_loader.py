"""
mapping_loader.py
-----------------
负责：
  1. 解析 mapping JSON，提取 clean_data -> raw file_name 的对应关系
  2. 根据 mapping（或回退同名逻辑）构建覆盖表所需的 rows
"""

import json
from pathlib import Path


# ── 公开接口 1：加载 mapping ───────────────────────────────────────────────────

def load_mapping(mapping_json_path: str) -> tuple[dict | None, str | None]:
    """
    读取 mapping JSON，返回 (mapping_dict, error_message)。
    mapping_dict 结构：{ cleaned_filename: [raw_filename, ...] }
    失败时 mapping_dict 为 None，error_message 说明原因。
    """
    path = Path(mapping_json_path.strip()).expanduser()

    if not path.exists():
        return None, f"文件不存在：{path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败：{e}"
    except Exception as e:
        return None, f"读取失败：{e}"

    # 从所有 sheets 的 records 中提取映射关系
    mapping: dict[str, list[str]] = {}
    sheets = data.get("sheets", {})

    if not isinstance(sheets, dict):
        return None, "JSON 结构异常：'sheets' 字段不是对象"

    INVALID_CLEAN = {"none", "nan", "", "n/a", "na"}

    for sheet_name, sheet_body in sheets.items():
        records = sheet_body.get("records", [])
        if not isinstance(records, list):
            continue
        for rec in records:
            clean_name = rec.get("clean_data", "").strip()
            raw_name   = rec.get("file_name",  "").strip()
            # 跳过无效记录：clean_data 为空、'none'、'nan' 等占位符，或 raw 文件名为空
            if clean_name.lower() in INVALID_CLEAN or not raw_name:
                continue
            mapping.setdefault(clean_name, []).append(raw_name)

    if not mapping:
        return None, "mapping JSON 中未找到任何有效的 clean_data / file_name 记录"

    return mapping, None


# ── 公开接口 2：构建覆盖表 rows ───────────────────────────────────────────────

def build_file_coverage(
    raw_map:     dict[str, Path],   # { filename: full_path }
    cleaned_map: dict[str, Path],   # { filename: full_path }
    mapping:     dict[str, list[str]] | None,
) -> list[dict]:
    """
    返回覆盖表的 rows，每行包含：
      file_name, raw_exists, cleaned_exists, status,
      match_method, mapped_raw_files, mapping_status
    """
    if mapping:
        return _build_with_mapping(raw_map, cleaned_map, mapping)
    else:
        return _build_same_name(raw_map, cleaned_map)


# ── 内部：json_mapping 路径 ───────────────────────────────────────────────────

def _build_with_mapping(
    raw_map:     dict[str, Path],
    cleaned_map: dict[str, Path],
    mapping:     dict[str, list[str]],
) -> list[dict]:

    rows = []
    handled_cleaned = set()

    # 遍历 mapping 中登记的 cleaned 文件
    for cleaned_name, raw_names in sorted(mapping.items()):
        handled_cleaned.add(cleaned_name)
        in_cleaned = cleaned_name in cleaned_map

        raw_found  = [r for r in raw_names if r in raw_map]
        in_raw     = len(raw_found) > 0

        rows.append({
            "file_name":        cleaned_name,
            "raw_exists":       in_raw,
            "cleaned_exists":   in_cleaned,
            "status":           _get_status(in_raw, in_cleaned),
            "match_method":     "json_mapping",
            "mapped_raw_files": ", ".join(raw_names),
            "mapping_status":   _get_mapping_status(raw_names, raw_map),
        })

    # 未出现在 mapping 中的 cleaned 文件 → 同名兜底
    for cleaned_name in sorted(cleaned_map.keys() - handled_cleaned):
        in_raw = cleaned_name in raw_map
        rows.append({
            "file_name":        cleaned_name,
            "raw_exists":       in_raw,
            "cleaned_exists":   True,
            "status":           _get_status(in_raw, True),
            "match_method":     "same_name",
            "mapped_raw_files": cleaned_name if in_raw else "—",
            "mapping_status":   "✅ raw 存在" if in_raw else "⚠️ raw 缺失",
        })

    # mapping 中登记的所有 raw 文件名集合
    mapped_raw_names = {r for names in mapping.values() for r in names}

    # raw 目录中存在、但既不在 mapping 里也不在 cleaned 目录里的文件 → 未清洗
    for raw_name in sorted(raw_map.keys()):
        if raw_name not in mapped_raw_names and raw_name not in cleaned_map:
            rows.append({
                "file_name":        raw_name,
                "raw_exists":       True,
                "cleaned_exists":   False,
                "status":           "⏳ 未清洗",
                "match_method":     "same_name",
                "mapped_raw_files": raw_name,
                "mapping_status":   "⚠️ 未在 mapping 中登记",
            })

    return rows


# ── 内部：same_name 路径 ──────────────────────────────────────────────────────

def _build_same_name(
    raw_map:     dict[str, Path],
    cleaned_map: dict[str, Path],
) -> list[dict]:

    all_files = raw_map.keys() | cleaned_map.keys()
    rows = []
    for name in sorted(all_files):
        in_raw, in_cleaned = name in raw_map, name in cleaned_map
        rows.append({
            "file_name":        name,
            "raw_exists":       in_raw,
            "cleaned_exists":   in_cleaned,
            "status":           _get_status(in_raw, in_cleaned),
            "match_method":     "same_name",
            "mapped_raw_files": name if in_raw else "—",
            "mapping_status":   "✅ raw 存在" if in_raw else ("⚠️ raw 缺失" if in_cleaned else "—"),
        })
    return rows


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _get_status(in_raw: bool, in_cleaned: bool) -> str:
    if in_raw and in_cleaned: return "✅ 已清洗"
    elif in_raw:              return "⏳ 未清洗"
    else:                     return "👻 孤儿清洗文件"


def _get_mapping_status(raw_names: list[str], raw_map: dict[str, Path]) -> str:
    found   = [r for r in raw_names if r in raw_map]
    missing = [r for r in raw_names if r not in raw_map]
    if not missing:
        return "✅ raw 存在"
    elif not found:
        return f"❌ raw 全部缺失：{', '.join(missing)}"
    else:
        return f"⚠️ 部分缺失：{', '.join(missing)}"