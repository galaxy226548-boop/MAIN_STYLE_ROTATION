"""汇总单因子回测和 IC 结果，用于批量排序筛选。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

try:
    import matplotlib
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    if LOCAL_PYTHON.exists() and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv_mktp").resolve():
        os_args = [str(LOCAL_PYTHON), *sys.argv]
        os.execv(str(LOCAL_PYTHON), os_args)
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter


def find_chinese_font() -> FontProperties | None:
    """寻找并注册本机可用的中文字体，避免图例中文乱码。"""

    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            return FontProperties(fname=font_path)
    return None


CHINESE_FONT_PROP = find_chinese_font()
if CHINESE_FONT_PROP is not None:
    plt.rcParams["font.family"] = [CHINESE_FONT_PROP.get_name()]
plt.rcParams["font.sans-serif"] = [
    CHINESE_FONT_PROP.get_name() if CHINESE_FONT_PROP is not None else "DejaVu Sans",
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Heiti TC",
    "Songti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


REFERENCE_DIR = PROJECT_ROOT / "B_factors" / "reference"
BACKTEST_ROOT = PROJECT_ROOT / "E_backtesting" / "Result"
IC_ROOT = PROJECT_ROOT / "D_analysis" / "IC_output"

DROP_METADATA_COLS = [
    "数据路径",
    "字段名",
    "数据频率",
    "template",
    "default_params",
    "input_fields",
    "input_transform",
]
BACKTEST_METRIC_COLS = [
    "trade_count",
    "ann_ret_long_abs",
    "max_dd_abs",
    "sharpe_abs",
    "sharpe_excess",
    "calmar_ratio",
    "monthly_win_rate",
    "payoff_ratio",
    "expectancy",
]
IC_PROB_COLS = [f"pos_ic_prob_track{i}" for i in range(5)]
SAMPLE_CHOICES = ("ins", "oos", "all")
DEFAULT_SAMPLE = "ins"
STATUS_COLS = ["source_sheet", "backtest_status", "ic_status"]
PERCENT_COLS = ["ann_ret_long_abs", "max_dd_abs", "monthly_win_rate", "expectancy", "pos_ic_prob_mean"]
TWO_DECIMAL_COLS = ["sharpe_abs", "sharpe_excess", "calmar_ratio", "payoff_ratio"]
CHART_METRICS = ["sharpe_abs", "ann_ret_long_abs", "monthly_win_rate", "expectancy"]
METHOD_ORDER = ["level", "trendA", "trendB", "trend_acce"]
METHOD_COLORS = {
    "level": "#4472A9",
    "trendA": "#F79646",
    "trendB": "#A6A6A6",
    "trend_acce": "#E8380D",
}
CATEGORY_COL = "子类"
CATEGORY_COLORS = ["#4472A9", "#E8380D", "#7F7F7F", "#F79646", "#A6A6A6", "#BDBDBD"]
HIGHLIGHT_TOP_N = 2
METRIC_LABELS = {
    "sharpe_abs": "Sharpe Abs",
    "ann_ret_long_abs": "Annual Return Long Abs",
    "monthly_win_rate": "Monthly Win Rate",
    "expectancy": "Expectancy",
}
SUMMARY_SHEET = "summary"
IC_SHEET = "IC analysis"
SUMMARY_HEADER_ROW = 1
SUMMARY_VALUE_ROW = 124
SUMMARY_FULL_PERIOD_LABEL = "全区间"
STD_SUMMARY_LABEL = "std_summary"
SUMMARY_FILE_SUFFIX = "_summary.xlsx"
IC_FILE_SUFFIX = "_IC_analysis.xlsx"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="按 reference JSON 汇总单因子回测 summary 和 IC 正概率均值。"
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="回测结果子目录，例如 Result/I_laboratory、I_laboratory 或 E_backtesting/Result/I_laboratory。",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="可选输出 Excel 路径；默认写入回测目录下的 <name>_single_factors_sorting.xlsx。",
    )
    parser.add_argument(
        "--sample",
        choices=SAMPLE_CHOICES,
        default=DEFAULT_SAMPLE,
        help="选择回测样本段目录，默认 ins。可选：ins、oos、all。",
    )
    return parser.parse_args()


def resolve_backtest_dir(input_path: Path) -> Path:
    """把用户输入的目录格式统一解析到 E_backtesting/Result 下。"""

    if input_path.is_absolute():
        return input_path

    parts = input_path.parts
    if len(parts) >= 2 and parts[0] == "Result":
        return PROJECT_ROOT / "E_backtesting" / input_path
    if len(parts) >= 3 and parts[0] == "E_backtesting" and parts[1] == "Result":
        return PROJECT_ROOT / input_path
    return BACKTEST_ROOT / input_path


def split_group_and_sample_dir(input_path: Path, sample: str) -> tuple[str, Path]:
    """解析因子组目录和实际样本段回测目录。"""

    resolved_dir = resolve_backtest_dir(input_path).resolve()
    group_dir = resolved_dir
    if resolved_dir.name in SAMPLE_CHOICES:
        group_dir = resolved_dir.parent

    group_name = group_dir.name
    sample_dir = group_dir / sample
    if not sample_dir.exists():
        raise FileNotFoundError(f"Cannot find sample backtest dir: {sample_dir}")
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample backtest path is not a directory: {sample_dir}")

    return group_name, sample_dir


def resolve_ic_dir(name: str, sample: str) -> Path:
    """优先使用样本段 IC 目录，不存在时回退到总 IC 目录。"""

    base_dir = IC_ROOT / name
    sample_dir = base_dir / sample
    if sample_dir.exists():
        return sample_dir
    return base_dir


def read_factor_records(reference_path: Path) -> pd.DataFrame:
    """读取 reference JSON，并合并所有 sheet records。"""

    if not reference_path.exists():
        raise FileNotFoundError(f"找不到 reference JSON: {reference_path}")

    with reference_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[dict[str, Any]] = []
    for sheet_name, sheet_info in payload.get("sheets", {}).items():
        for record in sheet_info.get("records", []):
            out = dict(record)
            out.setdefault("source_sheet", sheet_name)
            records.append(out)

    if not records:
        raise ValueError(f"reference JSON 内没有可用 records: {reference_path}")

    factor_df = pd.DataFrame(records)
    if "编号" not in factor_df.columns:
        raise KeyError(f"reference JSON 缺少必需列: 编号 ({reference_path})")

    factor_df = factor_df.drop(columns=[col for col in DROP_METADATA_COLS if col in factor_df.columns])
    return factor_df.sort_values("编号", kind="stable").reset_index(drop=True)


def factor_folder_pattern(factor_id: str) -> re.Pattern[str]:
    """生成回测文件夹命名规则的匹配表达式。"""

    return re.compile(rf"^\d+_mw\d+(?:\.\d+)?_{re.escape(factor_id)}$")


def find_factor_backtest_dir(backtest_dir: Path, factor_id: str) -> tuple[Path | None, str]:
    """根据 factor_id 在回测目录中寻找唯一的因子结果文件夹。"""

    if not backtest_dir.exists():
        return None, f"missing backtest dir: {backtest_dir}"

    pattern = factor_folder_pattern(factor_id)
    matches = [path for path in backtest_dir.iterdir() if path.is_dir() and pattern.match(path.name)]
    if not matches:
        return None, "missing factor backtest folder"
    if len(matches) > 1:
        names = ", ".join(path.name for path in sorted(matches))
        return None, f"multiple factor backtest folders: {names}"
    return matches[0], "ok"


def find_summary_file(factor_dir: Path) -> tuple[Path | None, str]:
    """在单个因子回测文件夹中寻找唯一 summary Excel。"""

    paths = [
        path
        for path in factor_dir.glob(f"*{SUMMARY_FILE_SUFFIX}")
        if path.is_file() and not path.name.startswith("~$")
    ]
    if not paths:
        return None, "missing summary xlsx"
    if len(paths) > 1:
        names = ", ".join(path.name for path in sorted(paths))
        return None, f"multiple summary xlsx files: {names}"
    return paths[0], "ok"


def read_backtest_metrics(summary_path: Path) -> tuple[dict[str, Any], str]:
    """从 summary sheet 的 Excel 第 125 行读取指定回测指标。"""

    try:
        summary_df = pd.read_excel(summary_path, sheet_name=SUMMARY_SHEET, header=None)
    except ValueError as exc:
        return {}, f"missing summary sheet: {exc}"
    except Exception as exc:
        return {}, f"read summary failed: {exc}"

    row_info = find_summary_value_and_header_rows(summary_df)
    if row_info is None:
        return {}, "summary full-period row missing"

    value_row, header_row = row_info
    header = summary_df.iloc[header_row]
    value = summary_df.iloc[value_row]
    col_to_value = {
        str(col_name): value.iloc[idx]
        for idx, col_name in enumerate(header)
        if pd.notna(col_name) and idx < len(value)
    }
    missing_cols = [col for col in BACKTEST_METRIC_COLS if col not in col_to_value]
    metrics = {col: col_to_value.get(col, pd.NA) for col in BACKTEST_METRIC_COLS}

    if missing_cols:
        return metrics, f"missing metric columns: {missing_cols}"
    return metrics, "ok"


def find_summary_value_and_header_rows(summary_df: pd.DataFrame) -> tuple[int, int] | None:
    """寻找最后一个 track 的全区间行及其表头行。"""

    first_col = summary_df.iloc[:, 0].map(lambda value: "" if pd.isna(value) else str(value).strip())
    std_rows = first_col[first_col == STD_SUMMARY_LABEL].index
    search_end = int(std_rows[0]) if len(std_rows) > 0 else len(summary_df)
    full_period_rows = first_col.iloc[:search_end][first_col.iloc[:search_end] == SUMMARY_FULL_PERIOD_LABEL].index

    if len(full_period_rows) > 0:
        value_row = int(full_period_rows[-1])
        header_candidates = first_col.iloc[:value_row][first_col.iloc[:value_row] == "period"].index
        if len(header_candidates) == 0:
            return None
        return value_row, int(header_candidates[-1])

    if len(summary_df) > SUMMARY_VALUE_ROW and len(summary_df) > SUMMARY_HEADER_ROW:
        return SUMMARY_VALUE_ROW, SUMMARY_HEADER_ROW

    return None


def collect_backtest_metrics(backtest_dir: Path, factor_id: str) -> dict[str, Any]:
    """汇总单个因子的回测指标和状态。"""

    empty_metrics = {col: pd.NA for col in BACKTEST_METRIC_COLS}
    factor_dir, status = find_factor_backtest_dir(backtest_dir, factor_id)
    if factor_dir is None:
        return {**empty_metrics, "backtest_status": status}

    summary_path, status = find_summary_file(factor_dir)
    if summary_path is None:
        return {**empty_metrics, "backtest_status": status}

    metrics, status = read_backtest_metrics(summary_path)
    return {**empty_metrics, **metrics, "backtest_status": status}


def normalize_factor_row_name(value: Any) -> str:
    """把 IC 表第一列转成便于匹配的字符串。"""

    if pd.isna(value):
        return ""
    return str(value).strip()


def choose_factor_ic_row(ic_df: pd.DataFrame, factor_id: str) -> pd.Series | None:
    """在 IC analysis 表中选择因子本身行，排除 Bucket 或 horizon 变体。"""

    if ic_df.empty:
        return None

    names = ic_df.iloc[:, 0].map(normalize_factor_row_name)
    preferred_names = {f"f_{factor_id}", factor_id}
    exact_mask = names.isin(preferred_names)
    if exact_mask.any():
        return ic_df.loc[exact_mask].iloc[0]

    variant_pattern = re.compile(r"(_Bucket_|_horizon|bucket|horizon)", flags=re.IGNORECASE)
    plain_mask = names.str.startswith(factor_id, na=False) & ~names.str.contains(variant_pattern, na=False)
    if plain_mask.any():
        return ic_df.loc[plain_mask].iloc[0]

    f_plain_mask = names.str.startswith(f"f_{factor_id}", na=False) & ~names.str.contains(variant_pattern, na=False)
    if f_plain_mask.any():
        return ic_df.loc[f_plain_mask].iloc[0]

    return None


def collect_ic_metrics(ic_dir: Path, factor_id: str) -> dict[str, Any]:
    """读取单个因子的 IC 正概率均值。"""

    ic_path = ic_dir / f"{factor_id}{IC_FILE_SUFFIX}"
    if not ic_path.exists():
        return {"pos_ic_prob_mean": pd.NA, "ic_status": "missing IC analysis xlsx"}

    try:
        ic_df = pd.read_excel(ic_path, sheet_name=IC_SHEET)
    except ValueError as exc:
        return {"pos_ic_prob_mean": pd.NA, "ic_status": f"missing IC analysis sheet: {exc}"}
    except Exception as exc:
        return {"pos_ic_prob_mean": pd.NA, "ic_status": f"read IC failed: {exc}"}

    missing_cols = [col for col in IC_PROB_COLS if col not in ic_df.columns]
    if missing_cols:
        return {"pos_ic_prob_mean": pd.NA, "ic_status": f"missing IC columns: {missing_cols}"}

    factor_row = choose_factor_ic_row(ic_df, factor_id)
    if factor_row is None:
        return {"pos_ic_prob_mean": pd.NA, "ic_status": "missing base factor row"}

    values = pd.to_numeric(factor_row[IC_PROB_COLS], errors="coerce")
    return {"pos_ic_prob_mean": values.mean(skipna=True), "ic_status": "ok"}


def build_sorting_table(name: str, backtest_dir: Path, ic_dir: Path) -> pd.DataFrame:
    """构建最终排序汇总表。"""

    reference_path = REFERENCE_DIR / f"{name}.json"
    factor_df = read_factor_records(reference_path)

    rows: list[dict[str, Any]] = []
    for record in factor_df.to_dict("records"):
        factor_id = str(record["编号"])
        backtest_metrics = collect_backtest_metrics(backtest_dir, factor_id)
        ic_metrics = collect_ic_metrics(ic_dir, factor_id)
        rows.append({**record, **backtest_metrics, **ic_metrics})

    result_df = pd.DataFrame(rows)
    return result_df.sort_values("编号", kind="stable").reset_index(drop=True)


def write_output(result_df: pd.DataFrame, output_path: Path) -> None:
    """输出 Excel 文件，并冻结首行方便查看。"""

    output_df = result_df.drop(columns=[col for col in STATUS_COLS if col in result_df.columns])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, sheet_name="single_factors_sorting", index=False)
        workbook = writer.book
        worksheet = writer.sheets["single_factors_sorting"]
        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        two_decimal_fmt = workbook.add_format({"num_format": "0.00"})
        worksheet.freeze_panes(1, 0)
        for idx, col in enumerate(output_df.columns):
            width = min(max(len(str(col)), 12), 40)
            cell_format = None
            if col in PERCENT_COLS:
                cell_format = percent_fmt
            elif col in TWO_DECIMAL_COLS:
                cell_format = two_decimal_fmt
            worksheet.set_column(idx, idx, width, cell_format)


def parse_factor_parts(result_df: pd.DataFrame) -> pd.DataFrame:
    """从编号中拆出处理方法和原数据尾号。"""

    out = result_df.copy()
    parts = out["编号"].astype(str).str.extract(r"^(?P<method_prefix>.+)_(?P<data_no>\d{3})$")
    out["data_no"] = parts["data_no"]
    out["method"] = parts["method_prefix"].str.replace("I_rf_", "", regex=False)
    return out


def metric_pivot(chart_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """把指标整理成原数据尾号乘处理方法的矩阵。"""

    work_df = chart_df.dropna(subset=["data_no", "method"]).copy()
    work_df[metric] = pd.to_numeric(work_df[metric], errors="coerce")
    pivot_df = work_df.pivot_table(index="data_no", columns="method", values=metric, aggfunc="first")
    ordered_cols = [col for col in METHOD_ORDER if col in pivot_df.columns]
    ordered_cols.extend(col for col in pivot_df.columns if col not in ordered_cols)
    return pivot_df.sort_index().loc[:, ordered_cols]


def category_by_data_no(chart_df: pd.DataFrame) -> pd.Series:
    """建立原数据尾号到子类的映射。"""

    if CATEGORY_COL not in chart_df.columns:
        return pd.Series(dtype="object")

    work_df = chart_df.dropna(subset=["data_no", CATEGORY_COL]).copy()
    if work_df.empty:
        return pd.Series(dtype="object")

    category_map = work_df.groupby("data_no")[CATEGORY_COL].agg(lambda values: values.dropna().iloc[0])
    return category_map.astype(str)


def find_category_top_factors(
    rank_df: pd.DataFrame,
    category_map: pd.Series,
    top_n: int = HIGHLIGHT_TOP_N,
) -> dict[str, tuple[str, int]]:
    """按子类寻找四个方法平均排名最好的前 N 个原数据尾号。"""

    if category_map.empty:
        winners = sorted(rank_df.index[(rank_df == 1).any(axis=1)])
        return {data_no: ("Global Top", idx + 1) for idx, data_no in enumerate(winners)}

    highlight_map: dict[str, tuple[str, int]] = {}
    avg_rank = rank_df.mean(axis=1, skipna=True)
    aligned_category = category_map.reindex(rank_df.index)

    for category in sorted(aligned_category.dropna().unique()):
        category_index = aligned_category[aligned_category == category].index
        category_rank = avg_rank.reindex(category_index).dropna().sort_values(kind="stable")
        for order, data_no in enumerate(category_rank.head(top_n).index, start=1):
            highlight_map[str(data_no)] = (str(category), order)

    return highlight_map


def category_color_map(category_map: pd.Series) -> dict[str, str]:
    """给每个子类分配稳定颜色。"""

    categories = sorted(category_map.dropna().astype(str).unique()) if not category_map.empty else ["Global Top"]
    return {category: CATEGORY_COLORS[idx % len(CATEGORY_COLORS)] for idx, category in enumerate(categories)}


def format_chart_axis(ax: plt.Axes, metric: str) -> None:
    """根据指标类型设置坐标轴格式。"""

    if metric in PERCENT_COLS:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.axhline(0, color="#7F7F7F", linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_grouped_bar_chart(pivot_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """输出四个处理方法的分组柱状图。"""

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=160)
    draw_grouped_bar_chart(ax, pivot_df, metric)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_grouped_bar_chart(ax: plt.Axes, pivot_df: pd.DataFrame, metric: str) -> None:
    """在指定画布轴上绘制分组柱状图。"""

    x = np.arange(len(pivot_df.index))
    methods = list(pivot_df.columns)
    bar_width = min(0.18, 0.74 / max(len(methods), 1))

    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            pivot_df[method],
            width=bar_width,
            label=method,
            color=METHOD_COLORS.get(method, "#7F7F7F"),
        )

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Raw Data and Method", fontsize=14, pad=12)
    ax.set_xlabel("Raw Data No.")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=0)
    format_chart_axis(ax, metric)
    ax.legend(
        ncol=max(1, len(methods)),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        prop=CHINESE_FONT_PROP,
    )


def save_rank_line_chart(
    pivot_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    category_map: pd.Series,
) -> None:
    """输出各原数据尾号在四个处理方法下的排名折线图。"""

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=160)
    draw_rank_line_chart(ax, pivot_df, metric, category_map)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_rank_line_chart(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    metric: str,
    category_map: pd.Series,
) -> None:
    """在指定画布轴上绘制排名折线图。"""

    rank_df = pivot_df.rank(axis=0, ascending=False, method="min")
    methods = list(rank_df.columns)
    x = np.arange(len(methods))
    highlight_map = find_category_top_factors(rank_df, category_map)
    color_map = category_color_map(category_map)

    for data_no, row in rank_df.iterrows():
        highlight_info = highlight_map.get(str(data_no))
        is_highlight = highlight_info is not None
        category, order = highlight_info if highlight_info is not None else ("", 0)
        color = color_map.get(category, "#BFBFBF") if is_highlight else "#BFBFBF"
        linewidth = 2.4 if is_highlight else 1.0
        alpha = 0.95 if is_highlight else 0.45
        linestyle = "-" if order == 1 else "--"
        label_text = f"{data_no} #{order}" if is_highlight else str(data_no)
        ax.plot(x, row[methods], marker="o", linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)
        ax.text(x[-1] + 0.05, row[methods[-1]], label_text, va="center", fontsize=8, color=color, alpha=alpha)

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} Rank Across Methods", fontsize=14, pad=12)
    ax.set_xlabel("Method")
    ax.set_ylabel("Rank, 1 = strongest")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_yticks(range(1, len(rank_df.index) + 1))
    ax.invert_yaxis()
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.2, len(methods) - 0.55)
    if highlight_map:
        handles = [
            plt.Line2D([0], [0], color=color, lw=2.4, label=category)
            for category, color in color_map.items()
            if category in {info[0] for info in highlight_map.values()}
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=max(1, len(handles)),
            prop=CHINESE_FONT_PROP,
        )


def save_combined_grouped_bar_chart(pivots: dict[str, pd.DataFrame], output_path: Path) -> None:
    """把四个指标的分组柱状图合并到一张 2x2 画布。"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=160)
    for ax, metric in zip(axes.ravel(), CHART_METRICS):
        draw_grouped_bar_chart(ax, pivots[metric], metric)
        ax.legend_.remove()
        ax.tick_params(axis="x", labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        prop=CHINESE_FONT_PROP,
    )
    fig.suptitle("Single Factor Strength by Raw Data and Method", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_rank_line_chart(
    pivots: dict[str, pd.DataFrame],
    output_path: Path,
    category_map: pd.Series,
) -> None:
    """把四个指标的排名折线图合并到一张 2x2 画布。"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=160)
    for ax, metric in zip(axes.ravel(), CHART_METRICS):
        draw_rank_line_chart(ax, pivots[metric], metric, category_map)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.tick_params(axis="y", labelsize=8)

    color_map = category_color_map(category_map)
    handles = [plt.Line2D([0], [0], color=color, lw=2.4, label=category) for category, color in color_map.items()]
    if handles:
        fig.legend(
            handles=handles,
            ncol=len(handles),
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            prop=CHINESE_FONT_PROP,
        )
    fig.suptitle("Single Factor Rank Across Methods", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_visualizations(result_df: pd.DataFrame, output_dir: Path, name: str) -> list[Path]:
    """为四个核心指标输出柱状图和排名折线图。"""

    chart_df = parse_factor_parts(result_df)
    category_map = category_by_data_no(chart_df)
    output_paths: list[Path] = []
    pivots: dict[str, pd.DataFrame] = {}
    for metric in CHART_METRICS:
        pivot_df = metric_pivot(chart_df, metric)
        if pivot_df.empty:
            continue

        pivots[metric] = pivot_df
        bar_path = output_dir / f"{name}_{metric}_grouped_bar.png"
        rank_path = output_dir / f"{name}_{metric}_rank_line.png"
        save_grouped_bar_chart(pivot_df, metric, bar_path)
        save_rank_line_chart(pivot_df, metric, rank_path, category_map)
        output_paths.extend([bar_path, rank_path])

    if all(metric in pivots for metric in CHART_METRICS):
        combined_bar_path = output_dir / f"{name}_combined_grouped_bar.png"
        combined_rank_path = output_dir / f"{name}_combined_rank_line.png"
        save_combined_grouped_bar_chart(pivots, combined_bar_path)
        save_combined_rank_line_chart(pivots, combined_rank_path, category_map)
        output_paths.extend([combined_bar_path, combined_rank_path])
    return output_paths


def main() -> None:
    """脚本主入口。"""

    args = parse_args()
    try:
        name, sample_backtest_dir = split_group_and_sample_dir(args.result_dir, args.sample)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_prefix = f"{name}_{args.sample}"
    ic_dir = resolve_ic_dir(name, args.sample)
    output_path = (
        args.output_file.resolve()
        if args.output_file is not None
        else sample_backtest_dir / f"{output_prefix}_single_factors_sorting.xlsx"
    )

    result_df = build_sorting_table(name=name, backtest_dir=sample_backtest_dir, ic_dir=ic_dir)
    write_output(result_df, output_path)
    chart_paths = save_visualizations(result_df, output_path.parent, output_prefix)

    backtest_ok = int((result_df["backtest_status"] == "ok").sum())
    ic_ok = int((result_df["ic_status"] == "ok").sum())
    print(f"Sample: {args.sample}")
    print(f"Backtest dir: {sample_backtest_dir}")
    print(f"IC dir: {ic_dir}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(result_df)}")
    print(f"Backtest ok: {backtest_ok}/{len(result_df)}")
    print(f"IC ok: {ic_ok}/{len(result_df)}")
    print(f"Charts: {len(chart_paths)}")
    for chart_path in chart_paths:
        print(f"- {chart_path}")


if __name__ == "__main__":
    main()
