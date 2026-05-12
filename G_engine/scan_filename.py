from pathlib import Path
import pandas as pd


# ========== 1. 配置路径 ==========
# 当前脚本所在文件夹：MAIN_STYLE_ROTATION/F_engine
SCRIPT_DIR = Path(__file__).resolve().parent

# 项目根目录：MAIN_STYLE_ROTATION
PROJECT_ROOT = SCRIPT_DIR.parent

# 数据文件夹：MAIN_STYLE_ROTATION/F_grouping
DATA_DIR = PROJECT_ROOT / "F_grouping" / "input_COMB"

# ========== 2. 辅助函数：读取第一行所有值 ==========
def read_first_row_values(file_path: Path, suffix: str, sheet_name=None):
    """
    读取 Excel 某个 sheet、CSV 文件或 Parquet 文件的第一行/表头。

    对 Excel / CSV：
    - 返回第一行所有值

    对 Parquet：
    - 返回列名列表，即表头
    """

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            nrows=1,
        )

        if df.empty:
            return []

        values = df.iloc[0, :].tolist()

    elif suffix == ".csv":
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                nrows=1,
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path,
                header=None,
                nrows=1,
                encoding="gbk",
            )

        if df.empty:
            return []

        values = df.iloc[0, :].tolist()

    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
        values = df.columns.tolist()

    else:
        return []

    # 把 pandas 里的 NaN 转成 None
    values = [None if pd.isna(x) else x for x in values]

    return values


# ========== 3. 递归读取所有文件 ==========
def scan_files_with_sheets_and_first_row(data_dir: Path) -> pd.DataFrame:
    """
    递归扫描 data_dir 下的所有文件，包括子文件夹中的文件。

    对 Excel 文件：
    - 每个 sheet 占一行
    - first_raw_values 返回该 sheet 第一行所有值

    对 CSV 文件：
    - 每个 CSV 占一行
    - sheet_name 记为 "__csv__"
    - first_raw_values 返回该 CSV 第一行所有值

    对其他文件：
    - 每个文件占一行
    - sheet_name 为空
    - first_raw_values 为空列表
    """

    if not data_dir.exists():
        raise FileNotFoundError(f"找不到文件夹：{data_dir.resolve()}")

    records = []

    excel_suffixes = {".xlsx", ".xlsm", ".xls"}
    csv_suffixes = {".csv"}
    parquet_suffixes = {".parquet"}

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        base_record = {
            "file_name": file_path.name,
            "relative_path": str(file_path.relative_to(data_dir)),
            "suffix": suffix,
            "parent_folder": str(file_path.parent.relative_to(data_dir)),
        }

        # ========== Excel 文件：每个 sheet 占一行 ==========
        if suffix in excel_suffixes:
            try:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names

                for sheet_name in sheet_names:
                    try:
                        first_raw_values = read_first_row_values(
                            file_path=file_path,
                            suffix=suffix,
                            sheet_name=sheet_name,
                        )

                        record = base_record.copy()
                        record.update({
                            "sheet_name": sheet_name,
                            "first_raw_values": first_raw_values,
                            "read_status": "success",
                            "error_msg": "",
                        })
                        records.append(record)

                    except Exception as e:
                        record = base_record.copy()
                        record.update({
                            "sheet_name": sheet_name,
                            "first_raw_values": [],
                            "read_status": "failed_read_sheet",
                            "error_msg": str(e),
                        })
                        records.append(record)

            except Exception as e:
                record = base_record.copy()
                record.update({
                    "sheet_name": "",
                    "first_raw_values": [],
                    "read_status": "failed_read_excel",
                    "error_msg": str(e),
                })
                records.append(record)

        # ========== CSV 文件：每个文件占一行 ==========
        elif suffix in csv_suffixes:
            try:
                first_raw_values = read_first_row_values(
                    file_path=file_path,
                    suffix=suffix,
                    sheet_name=None,
                )

                record = base_record.copy()
                record.update({
                    "sheet_name": "__csv__",
                    "first_raw_values": first_raw_values,
                    "read_status": "success",
                    "error_msg": "",
                })
                records.append(record)

            except Exception as e:
                record = base_record.copy()
                record.update({
                    "sheet_name": "__csv__",
                    "first_raw_values": [],
                    "read_status": "failed_read_csv",
                    "error_msg": str(e),
                })
                records.append(record)

        # ========== Parquet 文件：每个文件占一行 ==========
        elif suffix in parquet_suffixes:
            try:
                first_raw_values = read_first_row_values(
                    file_path=file_path,
                    suffix=suffix,
                    sheet_name=None,
                )

                record = base_record.copy()
                record.update({
                    "sheet_name": "__parquet__",
                    "first_raw_values": first_raw_values,
                    "read_status": "success",
                    "error_msg": "",
                })
                records.append(record)

            except Exception as e:
                record = base_record.copy()
                record.update({
                    "sheet_name": "__parquet__",
                    "first_raw_values": [],
                    "read_status": "failed_read_parquet",
                    "error_msg": str(e),
                })
                records.append(record)

        # ========== 其他文件：保留文件信息，不读取内容 ==========
        else:
            record = base_record.copy()
            record.update({
                "sheet_name": "",
                "first_raw_values": [],
                "read_status": "not_table_file",
                "error_msg": "",
            })
            records.append(record)

    file_df = pd.DataFrame(records)

    if not file_df.empty:
        file_df = file_df.sort_values(
            ["relative_path", "sheet_name"]
        ).reset_index(drop=True)

    return file_df


# ========== 4. 执行扫描 ==========
file_df = scan_files_with_sheets_and_first_row(DATA_DIR)

print(f"共扫描到 {file_df['relative_path'].nunique()} 个文件")
print(f"共生成 {len(file_df)} 行记录")
print(file_df.head(30))


# ========== 5. 保存结果 ==========
output_path = SCRIPT_DIR / "A_data_file_inventory_with_sheets.xlsx"
file_df.to_excel(output_path, index=False)

print(f"文件清单已保存到：{output_path.resolve()}")