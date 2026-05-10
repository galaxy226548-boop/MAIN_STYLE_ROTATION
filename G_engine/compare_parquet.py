#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_parquet_safe.py

用途：
1. 比较两个 parquet 文件的行数、列数、磁盘大小；
2. 找出“较大文件”比“较小文件”多出来的列；
3. 在指定 key 列的前提下，找出“较大文件”比“较小文件”多出来的行 key；
4. 全程尽量避免一次性把大 parquet 读入内存。

推荐用法：
python compare_parquet_safe.py \
  --file-a path/to/big.parquet \
  --file-b path/to/small.parquet \
  --keys TRADE_DT S_INFO_WINDCODE \
  --out-dir compare_output

如果暂时不知道 key 列，可以先只看 metadata 和列差异：
python compare_parquet_safe.py --file-a a.parquet --file-b b.parquet --out-dir compare_output
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# =========================
# 1. 基础工具
# =========================

def get_parquet_meta(path: Path) -> dict:
    """
    只读取 parquet metadata，不读取实际数据。
    这一步通常很快、很省内存。
    """
    pf = pq.ParquetFile(path)
    schema_names = pf.schema_arrow.names

    return {
        "path": str(path),
        "disk_size_mb": path.stat().st_size / 1024 / 1024,
        "num_rows": pf.metadata.num_rows,
        "num_columns": len(schema_names),
        "columns": schema_names,
    }


def choose_larger_file(meta_a: dict, meta_b: dict) -> Tuple[str, dict, str, dict]:
    """
    默认按磁盘大小判断“较大文件”。
    如果磁盘大小相同，则按行数 * 列数判断。
    """
    size_a = meta_a["disk_size_mb"]
    size_b = meta_b["disk_size_mb"]

    if size_a > size_b:
        return "A", meta_a, "B", meta_b
    if size_b > size_a:
        return "B", meta_b, "A", meta_a

    cells_a = meta_a["num_rows"] * meta_a["num_columns"]
    cells_b = meta_b["num_rows"] * meta_b["num_columns"]

    if cells_a >= cells_b:
        return "A", meta_a, "B", meta_b
    return "B", meta_b, "A", meta_a


def write_list_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =========================
# 2. key 行比较：分批读 + SQLite
# =========================

def normalize_key_value(x) -> str:
    """
    把 key 值转成稳定字符串，避免日期、None、数字类型在拼接时出错。
    注意：如果你的 key 列里有浮点数，不建议作为 key。
    """
    if x is None:
        return "<NA>"
    return str(x)


def make_key_hash(values: Sequence[object]) -> str:
    """
    多个 key 列合并后做 hash。
    SQLite 里只存 hash，可以比存长字符串更省空间。
    """
    raw = "\x1f".join(normalize_key_value(v) for v in values)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def table_to_key_rows(batch: pa.RecordBatch, keys: Sequence[str]) -> Iterable[Tuple[str, Tuple[str, ...]]]:
    """
    从一个 RecordBatch 里生成：
    1. key_hash：用于 join / diff；
    2. key_values：用于最终输出样例。
    """
    cols = [batch.column(batch.schema.get_field_index(k)).to_pylist() for k in keys]
    n = batch.num_rows

    for i in range(n):
        key_values = tuple(normalize_key_value(col[i]) for col in cols)
        key_hash = make_key_hash(key_values)
        yield key_hash, key_values


def create_sqlite_db(db_path: Path, keys: Sequence[str]) -> sqlite3.Connection:
    """
    用 SQLite 做外部存储，避免把所有 key 放进 Python set 导致内存爆掉。
    """
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = FILE;")

    key_cols_sql = ", ".join([f"key_{i} TEXT" for i in range(len(keys))])

    conn.execute(f"""
        CREATE TABLE big_keys (
            key_hash TEXT PRIMARY KEY,
            {key_cols_sql}
        )
    """)

    conn.execute("""
        CREATE TABLE small_keys (
            key_hash TEXT PRIMARY KEY
        )
    """)

    return conn


def insert_keys_from_parquet(
    conn: sqlite3.Connection,
    parquet_path: Path,
    keys: Sequence[str],
    table_name: str,
    batch_size: int = 100_000,
) -> int:
    """
    只读取 key 列，分批插入 SQLite。
    table_name:
      - big_keys: 存 key_hash + key 值，方便输出差异样例
      - small_keys: 只存 key_hash，用于判断是否存在
    """
    dataset = ds.dataset(str(parquet_path), format="parquet")
    scanner = dataset.scanner(columns=list(keys), batch_size=batch_size)

    total = 0

    if table_name == "big_keys":
        placeholders = ", ".join(["?"] * (1 + len(keys)))
        insert_sql = f"INSERT OR IGNORE INTO big_keys VALUES ({placeholders})"

        for batch in scanner.to_batches():
            rows = [(h, *vals) for h, vals in table_to_key_rows(batch, keys)]
            conn.executemany(insert_sql, rows)
            total += len(rows)
            conn.commit()

    elif table_name == "small_keys":
        insert_sql = "INSERT OR IGNORE INTO small_keys VALUES (?)"

        for batch in scanner.to_batches():
            rows = [(h,) for h, _ in table_to_key_rows(batch, keys)]
            conn.executemany(insert_sql, rows)
            total += len(rows)
            conn.commit()

    else:
        raise ValueError("table_name must be 'big_keys' or 'small_keys'.")

    return total


def export_extra_big_keys(
    conn: sqlite3.Connection,
    keys: Sequence[str],
    out_csv: Path,
    limit: Optional[int] = None,
) -> int:
    """
    导出较大文件中有、小文件中没有的 key。
    limit=None 表示全部导出；limit=1000 表示只导出前 1000 条。
    """
    key_cols = ", ".join([f"b.key_{i}" for i in range(len(keys))])

    limit_sql = "" if limit is None else f"LIMIT {int(limit)}"

    query = f"""
        SELECT {key_cols}
        FROM big_keys b
        LEFT JOIN small_keys s
        ON b.key_hash = s.key_hash
        WHERE s.key_hash IS NULL
        {limit_sql}
    """

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(list(keys))

        for row in conn.execute(query):
            writer.writerow(row)
            count += 1

    return count


def count_extra_big_keys(conn: sqlite3.Connection) -> int:
    query = """
        SELECT COUNT(*)
        FROM big_keys b
        LEFT JOIN small_keys s
        ON b.key_hash = s.key_hash
        WHERE s.key_hash IS NULL
    """
    return conn.execute(query).fetchone()[0]


# =========================
# 3. 主流程
# =========================

def compare_parquet(
    file_a: Path,
    file_b: Path,
    out_dir: Path,
    keys: Optional[List[str]] = None,
    batch_size: int = 100_000,
    export_limit: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_a = get_parquet_meta(file_a)
    meta_b = get_parquet_meta(file_b)

    larger_label, larger_meta, smaller_label, smaller_meta = choose_larger_file(meta_a, meta_b)

    larger_path = Path(larger_meta["path"])
    smaller_path = Path(smaller_meta["path"])

    # 1. 输出基础信息
    summary_rows = [
        ["A", meta_a["path"], f'{meta_a["disk_size_mb"]:.2f}', meta_a["num_rows"], meta_a["num_columns"]],
        ["B", meta_b["path"], f'{meta_b["disk_size_mb"]:.2f}', meta_b["num_rows"], meta_b["num_columns"]],
        ["larger_by_disk_size", larger_label, "", "", ""],
        ["smaller_by_disk_size", smaller_label, "", "", ""],
    ]
    write_list_csv(
        out_dir / "01_file_summary.csv",
        ["file_label", "path_or_label", "disk_size_mb", "num_rows", "num_columns"],
        summary_rows,
    )

    # 2. 输出列差异
    cols_a = set(meta_a["columns"])
    cols_b = set(meta_b["columns"])
    larger_cols = set(larger_meta["columns"])
    smaller_cols = set(smaller_meta["columns"])

    write_list_csv(
        out_dir / "02_columns_in_A_not_in_B.csv",
        ["column"],
        [[c] for c in sorted(cols_a - cols_b)],
    )
    write_list_csv(
        out_dir / "03_columns_in_B_not_in_A.csv",
        ["column"],
        [[c] for c in sorted(cols_b - cols_a)],
    )
    write_list_csv(
        out_dir / "04_columns_in_larger_not_in_smaller.csv",
        ["column"],
        [[c] for c in sorted(larger_cols - smaller_cols)],
    )

    # 3. 如果没有 key，只做 metadata 和列比较，不做行比较
    if not keys:
        print("已完成 metadata 和列差异比较。")
        print("未提供 --keys，因此没有做行差异比较。")
        print(f"输出目录：{out_dir}")
        return

    # 4. 检查 key 是否存在
    missing_in_larger = [k for k in keys if k not in larger_cols]
    missing_in_smaller = [k for k in keys if k not in smaller_cols]

    if missing_in_larger or missing_in_smaller:
        raise ValueError(
            "指定的 key 列不存在：\n"
            f"larger 缺失：{missing_in_larger}\n"
            f"smaller 缺失：{missing_in_smaller}"
        )

    # 5. 行差异：只读 key 列，分批写入 SQLite
    db_path = out_dir / "parquet_key_compare.sqlite"
    conn = create_sqlite_db(db_path, keys)

    print(f"正在读取较大文件 key 列：{larger_path}")
    inserted_big = insert_keys_from_parquet(
        conn=conn,
        parquet_path=larger_path,
        keys=keys,
        table_name="big_keys",
        batch_size=batch_size,
    )

    print(f"正在读取较小文件 key 列：{smaller_path}")
    inserted_small = insert_keys_from_parquet(
        conn=conn,
        parquet_path=smaller_path,
        keys=keys,
        table_name="small_keys",
        batch_size=batch_size,
    )

    extra_count = count_extra_big_keys(conn)

    extra_csv = out_dir / "05_rows_in_larger_not_in_smaller_keys.csv"
    exported_count = export_extra_big_keys(
        conn=conn,
        keys=keys,
        out_csv=extra_csv,
        limit=export_limit,
    )

    row_summary = [
        ["larger_label", larger_label],
        ["larger_path", str(larger_path)],
        ["smaller_label", smaller_label],
        ["smaller_path", str(smaller_path)],
        ["keys", "|".join(keys)],
        ["scanned_larger_rows", inserted_big],
        ["scanned_smaller_rows", inserted_small],
        ["extra_key_count_in_larger_not_in_smaller", extra_count],
        ["exported_extra_key_rows", exported_count],
        ["export_limit", "ALL" if export_limit is None else export_limit],
    ]

    write_list_csv(
        out_dir / "06_row_compare_summary.csv",
        ["item", "value"],
        row_summary,
    )

    conn.close()

    print("比较完成。")
    print(f"输出目录：{out_dir}")
    print(f"较大文件比小文件多出的 key 数量：{extra_count}")
    print(f"已导出 key 差异文件：{extra_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-safe comparison for two parquet files.")

    parser.add_argument("--file-a", required=True, type=Path, help="第一个 parquet 文件路径")
    parser.add_argument("--file-b", required=True, type=Path, help="第二个 parquet 文件路径")
    parser.add_argument("--out-dir", default=Path("compare_parquet_output"), type=Path, help="输出目录")

    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help=(
            "用于判断行是否相同的 key 列。"
            "例如：--keys TRADE_DT S_INFO_WINDCODE。"
            "如果不提供，则只比较行列数、磁盘大小和列差异。"
        ),
    )

    parser.add_argument(
        "--batch-size",
        default=100_000,
        type=int,
        help="分批读取 parquet 的 batch size。内存紧张时可以调小，例如 20000。",
    )

    parser.add_argument(
        "--export-limit",
        default=None,
        type=int,
        help=(
            "最多导出多少条较大文件多出来的 key。"
            "不设置则全部导出；如果差异很多，建议先设 1000 看样例。"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    compare_parquet(
        file_a=args.file_a,
        file_b=args.file_b,
        out_dir=args.out_dir,
        keys=args.keys,
        batch_size=args.batch_size,
        export_limit=args.export_limit,
    )