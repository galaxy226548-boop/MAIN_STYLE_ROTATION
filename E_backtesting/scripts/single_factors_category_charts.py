"""按逻辑路径和子类输出单因子分类图表，兼容 C001 这类普通因子编号。"""

from __future__ import annotations

import argparse
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
        os.execv(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv])
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from single_factors_sorting import (  # noqa: E402
    CHINESE_FONT_PROP,
    DEFAULT_SAMPLE,
    METRIC_LABELS,
    PERCENT_COLS,
    SAMPLE_CHOICES,
    build_sorting_table,
    resolve_ic_dir,
    split_group_and_sample_dir,
)


CHART_METRICS = ["sharpe_abs", "monthly_win_rate", "ann_ret_long_abs", "expectancy"]
GROUP_COLS = ["逻辑路径", "子类"]
FACTOR_ID_COL = "编号"
MEASURE_COL = "测度目标"
RAW_DATA_COL = "原数据"
HIGHLIGHT_TOP_N = 2

MEASURE_ORDER = [
    "水平偏移A",
    "水平偏移B",
    "趋势方向A",
    "趋势方向B",
    "趋势方向C",
    "趋势突破A",
    "趋势突破B",
    "趋势加速度",
]
MEASURE_COLORS = {
    "水平偏移A": "#4472A9",
    "水平偏移B": "#79A3D1",
    "趋势方向A": "#F79646",
    "趋势方向B": "#FDBE94",
    "趋势方向C": "#E8380D",
    "趋势突破A": "#7F7F7F",
    "趋势突破B": "#A6A6A6",
    "趋势加速度": "#D9D9D9",
}
FALLBACK_COLORS = ["#4472A9", "#F79646", "#7F7F7F", "#E8380D", "#A6A6A6", "#BDBDBD"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="为单因子排序结果按逻辑路径/子类生成分类柱状图和排名折线图。"
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="回测结果子目录，例如 Result/factor_C、factor_C 或 E_backtesting/Result/factor_C。",
    )
    parser.add_argument(
        "--sample",
        choices=SAMPLE_CHOICES,
        default=DEFAULT_SAMPLE,
        help="选择回测样本段目录，默认 ins。可选：ins、oos、all。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选图片输出目录；默认写入样本回测目录下的 category_charts。",
    )
    return parser.parse_args()


def clean_label(value: Any, default: str = "unknown") -> str:
    """把空值和非字符串值统一成可展示文本。"""

    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def safe_filename_part(value: Any, max_len: int = 48) -> str:
    """清理文件名片段，避免路径分隔符和特殊符号导致保存失败。"""

    text = clean_label(value)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._ ")
    if not text:
        return "unknown"
    return text[:max_len]


def ordered_measures(values: pd.Series) -> list[str]:
    """按固定业务顺序排列测度目标，未知目标稳定排在后面。"""

    existing = [clean_label(value) for value in values.dropna().unique()]
    ordered = [measure for measure in MEASURE_ORDER if measure in existing]
    ordered.extend(sorted(measure for measure in existing if measure not in ordered))
    return ordered


def measure_color_map(measures: list[str]) -> dict[str, str]:
    """给测度目标分配稳定颜色。"""

    color_map: dict[str, str] = {}
    fallback_idx = 0
    for measure in measures:
        if measure in MEASURE_COLORS:
            color_map[measure] = MEASURE_COLORS[measure]
        else:
            color_map[measure] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    return color_map


def metric_pivot(group_df: pd.DataFrame, metric: str, measures: list[str]) -> pd.DataFrame:
    """把一个子类内的指标整理成因子乘测度目标的矩阵。"""

    work_df = group_df[[FACTOR_ID_COL, MEASURE_COL, metric]].copy()
    work_df[FACTOR_ID_COL] = work_df[FACTOR_ID_COL].map(clean_label)
    work_df[MEASURE_COL] = work_df[MEASURE_COL].map(clean_label)
    work_df[metric] = pd.to_numeric(work_df[metric], errors="coerce")
    work_df = work_df.dropna(subset=[metric])
    if work_df.empty:
        return pd.DataFrame()

    pivot_df = work_df.pivot_table(
        index=FACTOR_ID_COL,
        columns=MEASURE_COL,
        values=metric,
        aggfunc="first",
        sort=False,
    )
    factor_order = group_df[FACTOR_ID_COL].map(clean_label).drop_duplicates().tolist()
    method_order = [measure for measure in measures if measure in pivot_df.columns]
    return pivot_df.reindex(factor_order).dropna(how="all").loc[:, method_order]


def factor_display_labels(group_df: pd.DataFrame, factor_ids: pd.Index) -> list[str]:
    """生成横轴标签，优先展示编号，必要时附带短版原数据。"""

    if RAW_DATA_COL not in group_df.columns:
        return [str(factor_id) for factor_id in factor_ids]

    raw_map = (
        group_df.assign(_factor_id=group_df[FACTOR_ID_COL].map(clean_label))
        .drop_duplicates("_factor_id")
        .set_index("_factor_id")[RAW_DATA_COL]
        .to_dict()
    )
    labels: list[str] = []
    for factor_id in factor_ids:
        raw_text = clean_label(raw_map.get(str(factor_id)), default="")
        if raw_text:
            labels.append(f"{factor_id}\n{raw_text[:10]}")
        else:
            labels.append(str(factor_id))
    return labels


def format_metric_axis(ax: plt.Axes, metric: str) -> None:
    """按指标类型设置坐标轴格式和参考线。"""

    if metric in PERCENT_COLS:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.axhline(0, color="#7F7F7F", linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_metric_bar_chart(
    group_df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    metric: str,
    title_prefix: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    """输出一个子类内的因子指标柱状图。"""

    fig_width = max(8.0, min(16.0, 1.25 * len(pivot_df.index) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), dpi=160)
    draw_metric_bar_chart(ax, group_df, pivot_df, metric, title_prefix, color_map)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_metric_bar_chart(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    metric: str,
    title_prefix: str,
    color_map: dict[str, str],
    show_xlabel: bool = True,
) -> None:
    """在指定坐标轴上绘制单个指标的分组柱状图。"""

    x = np.arange(len(pivot_df.index))
    measures = list(pivot_df.columns)
    bar_width = min(0.28, 0.76 / max(len(measures), 1))

    for idx, measure in enumerate(measures):
        offset = (idx - (len(measures) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            pivot_df[measure],
            width=bar_width,
            label=measure,
            color=color_map.get(measure, "#7F7F7F"),
        )

    ax.set_title(f"{title_prefix} - {METRIC_LABELS.get(metric, metric)}", fontsize=13, pad=12)
    ax.set_xlabel("Factor" if show_xlabel else "")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(factor_display_labels(group_df, pivot_df.index), rotation=0, fontsize=8)
    format_metric_axis(ax, metric)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=max(1, min(len(measures), 4)),
        prop=CHINESE_FONT_PROP,
    )


def save_combined_grouped_bar_chart(
    group_df: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    title_prefix: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    """把四个核心指标合并成一张 2x2 分组柱状图。"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=160)
    handles: list[Any] = []
    labels: list[str] = []
    for ax, metric in zip(axes.ravel(), CHART_METRICS):
        draw_metric_bar_chart(ax, group_df, pivots[metric], metric, title_prefix, color_map, show_xlabel=False)
        legend = ax.get_legend()
        if legend is not None:
            current_handles, current_labels = ax.get_legend_handles_labels()
            if not handles:
                handles = current_handles
                labels = current_labels
            legend.remove()
        ax.tick_params(axis="x", labelsize=7)

    if handles:
        fig.legend(
            handles,
            labels,
            ncol=max(1, min(len(labels), 4)),
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            prop=CHINESE_FONT_PROP,
        )
    fig.suptitle(f"{title_prefix} - Grouped Metrics", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_metric_method_rank_line(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    metric: str,
    color_map: dict[str, str],
) -> bool:
    """绘制同一因子在多个测度目标之间的排名线。"""

    rank_base = pivot_df.dropna(thresh=2).copy()
    if rank_base.empty or len(rank_base.columns) < 2:
        return False

    rank_df = rank_base.rank(axis=0, ascending=False, method="min")
    methods = list(rank_df.columns)
    x = np.arange(len(methods))
    avg_rank = rank_df.mean(axis=1, skipna=True).dropna().sort_values(kind="stable")
    highlight_ids = set(avg_rank.head(HIGHLIGHT_TOP_N).index.astype(str))

    for factor_id, row in rank_df.iterrows():
        factor_text = str(factor_id)
        is_highlight = factor_text in highlight_ids
        color = "#4472A9" if is_highlight else "#BFBFBF"
        linewidth = 2.4 if is_highlight else 1.0
        alpha = 0.95 if is_highlight else 0.42
        ax.plot(x, row[methods], marker="o", linewidth=linewidth, color=color, alpha=alpha)
        if is_highlight:
            ax.text(x[-1] + 0.05, row[methods[-1]], factor_text, va="center", fontsize=8, color=color)

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} Rank Across Measure Targets", fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Rank, 1 = strongest")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=8)
    ax.set_yticks(range(1, len(rank_df.index) + 1))
    ax.invert_yaxis()
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.2, len(methods) - 0.55)

    handles = [
        plt.Line2D([0], [0], color=color_map.get(method, "#7F7F7F"), lw=2.4, label=method)
        for method in methods
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=max(1, min(len(handles), 4)),
        prop=CHINESE_FONT_PROP,
    )
    return True


def save_combined_grouped_line_chart(
    pivots: dict[str, pd.DataFrame],
    title_prefix: str,
    output_path: Path,
    color_map: dict[str, str],
) -> bool:
    """把四个核心指标合并成一张 2x2 排名折线图；数据不足时跳过。"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=160)
    drawn_count = 0
    for ax, metric in zip(axes.ravel(), CHART_METRICS):
        drawn = draw_metric_method_rank_line(ax, pivots[metric], metric, color_map)
        if drawn:
            drawn_count += 1
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            ax.axis("off")
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)}: insufficient method overlap", fontsize=12)

    if drawn_count == 0:
        plt.close(fig)
        return False

    methods: list[str] = []
    for pivot_df in pivots.values():
        for method in pivot_df.columns:
            if method not in methods:
                methods.append(str(method))
    handles = [
        plt.Line2D([0], [0], color=color_map.get(method, "#7F7F7F"), lw=2.4, label=method)
        for method in methods
    ]
    if handles:
        fig.legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=max(1, min(len(handles), 4)),
            prop=CHINESE_FONT_PROP,
        )
    fig.suptitle(f"{title_prefix} - Grouped Rank Lines", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def category_top_factors(group_df: pd.DataFrame) -> set[str]:
    """按四个指标平均排名找出子类内表现最好的前两个因子。"""

    rank_source = group_df[[FACTOR_ID_COL, *CHART_METRICS]].copy()
    rank_source[FACTOR_ID_COL] = rank_source[FACTOR_ID_COL].map(clean_label)
    for metric in CHART_METRICS:
        rank_source[metric] = pd.to_numeric(rank_source[metric], errors="coerce")

    metric_ranks = rank_source.set_index(FACTOR_ID_COL)[CHART_METRICS].rank(
        ascending=False,
        method="min",
    )
    avg_rank = metric_ranks.mean(axis=1, skipna=True).dropna().sort_values(kind="stable")
    return set(avg_rank.head(HIGHLIGHT_TOP_N).index.astype(str))


def save_metric_rank_line_chart(
    group_df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    metric: str,
    title_prefix: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    """输出一个子类内各因子的指标排名折线图。"""

    rank_source = group_df[[FACTOR_ID_COL, MEASURE_COL, metric]].copy()
    rank_source[FACTOR_ID_COL] = rank_source[FACTOR_ID_COL].map(clean_label)
    rank_source[MEASURE_COL] = rank_source[MEASURE_COL].map(clean_label)
    rank_source[metric] = pd.to_numeric(rank_source[metric], errors="coerce")
    rank_source = rank_source.dropna(subset=[metric])
    if len(rank_source) < 2:
        return

    factor_order = [factor_id for factor_id in pivot_df.index.astype(str) if factor_id in set(rank_source[FACTOR_ID_COL])]
    rank_source = rank_source.set_index(FACTOR_ID_COL).reindex(factor_order).reset_index()
    rank_source["rank"] = rank_source[metric].rank(ascending=False, method="min")
    measures = [measure for measure in pivot_df.columns if measure in set(rank_source[MEASURE_COL])]
    x = np.arange(len(rank_source))
    highlight_ids = category_top_factors(group_df)

    fig_width = max(8.0, min(16.0, 1.2 * len(rank_source) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8), dpi=160)
    ax.plot(x, rank_source["rank"], color="#BFBFBF", linewidth=1.0, alpha=0.55)

    for idx, row in rank_source.iterrows():
        factor_text = str(row[FACTOR_ID_COL])
        measure = clean_label(row[MEASURE_COL])
        is_highlight = factor_text in highlight_ids
        color = color_map.get(measure, "#7F7F7F")
        size = 78 if is_highlight else 42
        alpha = 0.96 if is_highlight else 0.58
        edgecolor = "#222222" if is_highlight else "none"
        linewidth = 0.9 if is_highlight else 0.0
        ax.scatter(idx, row["rank"], s=size, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, zorder=3)
        if is_highlight:
            ax.text(idx, row["rank"] - 0.18, factor_text, ha="center", va="bottom", fontsize=8, color="#222222")

    ax.set_title(f"{title_prefix} - {METRIC_LABELS.get(metric, metric)} Rank", fontsize=13, pad=12)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Rank, 1 = strongest")
    ax.set_xticks(x)
    ax.set_xticklabels(factor_display_labels(group_df, pd.Index(rank_source[FACTOR_ID_COL])), rotation=0, fontsize=8)
    ax.set_yticks(range(1, len(rank_source) + 1))
    ax.invert_yaxis()
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, len(rank_source) - 0.5)

    handles = [
        plt.Line2D([0], [0], color=color_map.get(measure, "#7F7F7F"), lw=2.4, label=measure)
        for measure in measures
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=max(1, min(len(handles), 4)),
        prop=CHINESE_FONT_PROP,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_category_visualizations(result_df: pd.DataFrame, output_dir: Path, output_prefix: str) -> list[Path]:
    """按逻辑路径和子类生成所有分类图表。"""

    missing_cols = [col for col in [*GROUP_COLS, FACTOR_ID_COL, MEASURE_COL] if col not in result_df.columns]
    if missing_cols:
        raise KeyError(f"结果表缺少绘图必需列: {missing_cols}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    group_iter = result_df.groupby(GROUP_COLS, dropna=False, sort=False)

    for (logic_path, category), group_df in group_iter:
        group_df = group_df.copy()
        measures = ordered_measures(group_df[MEASURE_COL])
        if not measures:
            continue

        color_map = measure_color_map(measures)
        title_prefix = f"{clean_label(logic_path)} / {clean_label(category)}"
        file_prefix = (
            f"{safe_filename_part(output_prefix)}_"
            f"{safe_filename_part(logic_path)}_"
            f"{safe_filename_part(category)}"
        )

        pivots: dict[str, pd.DataFrame] = {}
        for metric in CHART_METRICS:
            pivot_df = metric_pivot(group_df, metric, measures)
            if pivot_df.empty:
                continue

            pivots[metric] = pivot_df

        if all(metric in pivots for metric in CHART_METRICS):
            combined_bar_path = output_dir / f"{file_prefix}_combined_group_bar.png"
            save_combined_grouped_bar_chart(group_df, pivots, title_prefix, combined_bar_path, color_map)
            output_paths.append(combined_bar_path)

            combined_line_path = output_dir / f"{file_prefix}_combined_group_line.png"
            if save_combined_grouped_line_chart(pivots, title_prefix, combined_line_path, color_map):
                output_paths.append(combined_line_path)

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
    output_dir = args.output_dir.resolve() if args.output_dir is not None else sample_backtest_dir / "category_charts"
    ic_dir = resolve_ic_dir(name, args.sample)
    result_df = build_sorting_table(name=name, backtest_dir=sample_backtest_dir, ic_dir=ic_dir)
    chart_paths = save_category_visualizations(result_df, output_dir, output_prefix)

    backtest_ok = int((result_df["backtest_status"] == "ok").sum())
    ic_ok = int((result_df["ic_status"] == "ok").sum())
    print(f"Sample: {args.sample}")
    print(f"Backtest dir: {sample_backtest_dir}")
    print(f"IC dir: {ic_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Rows: {len(result_df)}")
    print(f"Backtest ok: {backtest_ok}/{len(result_df)}")
    print(f"IC ok: {ic_ok}/{len(result_df)}")
    print(f"Charts: {len(chart_paths)}")
    for chart_path in chart_paths:
        print(f"- {chart_path}")


if __name__ == "__main__":
    main()
