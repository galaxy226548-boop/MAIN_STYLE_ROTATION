"""
Factor Correlation Analysis
============================
Computes intra-group and inter-group correlation matrices for 9 factor groups.
Produces heatmaps and a markdown report.

Groups: C, D, F, G, I, L, O, P, V
Data source: B_factors/output/grouping_document/
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "B_factors" / "output" / "grouping_document"
OUTPUT_DIR = PROJECT_ROOT / "B_factors" / "output" / "correlation_analysis"

GROUPS = list("CDFGILOPV")
MISSING_THRESHOLD = 0.30   # drop factor if >30% values missing after alignment
CORR_METHOD_FACTOR = "pearson"
CORR_METHOD_SIGNAL = "spearman"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class AnalysisData(NamedTuple):
    merged_df: pd.DataFrame          # N_dates × N_factors, aligned
    group_map: dict[str, list[str]]  # group -> sorted list of factor_ids
    factor_order: list[str]          # all factor_ids sorted by group
    dropped: list[str]               # factors dropped for missing data


# ---------------------------------------------------------------------------
# 1. Load and merge data
# ---------------------------------------------------------------------------
def load_data(data_type: str) -> AnalysisData:
    """
    data_type: "factor" (mounted_normalized) or "signal" (signal_ls)
    Returns aligned merged DataFrame and group metadata.
    """
    suffix = "mounted_normalized_factors" if data_type == "factor" else "signal_ls"
    frames: list[pd.DataFrame] = []
    group_map: dict[str, list[str]] = {}

    print(f"\n[load] Reading {data_type} parquet files...")
    for g in GROUPS:
        path = DATA_DIR / f"factor_{g}_{suffix}.parquet"
        if not path.exists():
            print(f"  ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_parquet(path)
        if df.empty:
            print(f"  ERROR: file is empty: {path}", file=sys.stderr)
            sys.exit(1)

        # Sort columns for deterministic order
        df = df.sort_index(axis=1)
        group_map[g] = df.columns.tolist()
        frames.append(df)
        print(f"  Group {g}: {df.shape[1]} factors, {df.shape[0]} dates "
              f"({df.index[0].date()} – {df.index[-1].date()})")

    # Align on common date index
    common_idx = frames[0].index
    for df in frames[1:]:
        common_idx = common_idx.intersection(df.index)
    print(f"\n[load] Common date range: {common_idx[0].date()} – {common_idx[-1].date()} "
          f"({len(common_idx)} dates)")

    frames_aligned = [df.loc[common_idx] for df in frames]
    merged = pd.concat(frames_aligned, axis=1)

    # Drop factors with too many missing values
    missing_frac = merged.isna().mean()
    bad = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
    if bad:
        print(f"\n[load] WARNING: dropping {len(bad)} factors with >{MISSING_THRESHOLD:.0%} "
              f"missing values: {bad}")
        merged = merged.drop(columns=bad)
        for g in GROUPS:
            group_map[g] = [c for c in group_map[g] if c not in bad]
    else:
        print(f"[load] All factors pass the {MISSING_THRESHOLD:.0%} missing-value threshold.")

    # Build factor order (grouped)
    factor_order: list[str] = []
    for g in GROUPS:
        factor_order.extend(group_map[g])

    merged = merged[factor_order]
    print(f"[load] Final merged shape: {merged.shape} "
          f"({sum(len(v) for v in group_map.values())} factors)\n")

    return AnalysisData(merged, group_map, factor_order, bad)


# ---------------------------------------------------------------------------
# 2. Compute correlation matrix
# ---------------------------------------------------------------------------
def compute_corr_matrix(merged_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute pairwise correlation. method: 'pearson' or 'spearman'."""
    print(f"[corr] Computing {method} correlation ({merged_df.shape[1]}×{merged_df.shape[1]})...")
    corr = merged_df.corr(method=method)
    print(f"[corr] Done. NaN entries: {corr.isna().sum().sum()}")
    return corr


# ---------------------------------------------------------------------------
# 3. Full N×N heatmap
# ---------------------------------------------------------------------------
def _group_boundaries(group_map: dict[str, list[str]]) -> list[int]:
    """Return cumulative factor counts — i.e., the boundary positions."""
    sizes = [len(group_map[g]) for g in GROUPS]
    boundaries: list[int] = []
    cumsum = 0
    for s in sizes:
        cumsum += s
        boundaries.append(cumsum)
    return boundaries[:-1]   # skip the final boundary (edge of matrix)


def plot_full_heatmap(
    corr: pd.DataFrame,
    group_map: dict[str, list[str]],
    title: str,
    out_path: Path,
) -> None:
    """Plot N×N heatmap with group boundary lines and group labels."""
    n = len(corr)
    # Scale figure size with factor count; cap at 60 inches
    side = min(60, max(24, n * 0.12))
    print(f"[plot] Plotting full {n}×{n} heatmap → {out_path.name} (figure {side:.0f}×{side:.0f} in)")

    fig, ax = plt.subplots(figsize=(side, side))

    sns.heatmap(
        corr.values,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6, "label": "Correlation"},
        linewidths=0,
        rasterized=True,
    )

    # Draw group boundary lines
    boundaries = _group_boundaries(group_map)
    line_kw = dict(color="black", linewidth=0.8, alpha=0.9)
    for b in boundaries:
        ax.axhline(b, **line_kw)
        ax.axvline(b, **line_kw)

    # Annotate group labels at diagonal midpoints
    sizes = [len(group_map[g]) for g in GROUPS]
    starts = [0] + list(np.cumsum(sizes[:-1]))
    for g, start, size in zip(GROUPS, starts, sizes):
        mid = start + size / 2
        ax.text(
            mid, mid, g,
            ha="center", va="center",
            fontsize=max(8, min(20, size * 0.25)),
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", alpha=0.55, boxstyle="round,pad=0.2", edgecolor="none"),
        )

    ax.set_title(title, fontsize=16, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. 9×9 group-level summary heatmap
# ---------------------------------------------------------------------------
def compute_9x9_matrix(corr: pd.DataFrame, group_map: dict[str, list[str]]) -> pd.DataFrame:
    """
    Build 9×9 group-average correlation matrix.
    Diagonal (k==k): mean of within-group pairs (i<j), excludes self-corr.
    Off-diagonal (k1,k2): mean of all cross-group factor pairs.
    """
    print("[corr] Computing 9×9 group summary matrix...")
    result = pd.DataFrame(index=GROUPS, columns=GROUPS, dtype=float)

    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            factors1 = group_map[g1]
            factors2 = group_map[g2]

            if g1 == g2:
                # Within-group: upper triangle only (i<j), no diagonal
                sub = corr.loc[factors1, factors1]
                n = len(factors1)
                if n < 2:
                    result.loc[g1, g2] = float("nan")
                    continue
                vals = sub.values[np.triu_indices(n, k=1)]
                result.loc[g1, g2] = float(np.nanmean(vals))
            else:
                # Cross-group: all pairs
                sub = corr.loc[factors1, factors2]
                result.loc[g1, g2] = float(np.nanmean(sub.values))

    print("[corr] 9×9 matrix done.")
    return result


def plot_9x9_heatmap(
    mat9: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Plot annotated 9×9 group-average heatmap."""
    print(f"[plot] Plotting 9×9 heatmap → {out_path.name}")
    fig, ax = plt.subplots(figsize=(10, 8))

    annot = mat9.astype(float).round(2)
    sns.heatmap(
        mat9.astype(float),
        ax=ax,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Avg Correlation"},
    )

    ax.set_xticklabels(GROUPS, fontsize=13)
    ax.set_yticklabels(GROUPS, fontsize=13, rotation=0)
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------
class GroupStats(NamedTuple):
    group: str
    n_factors: int
    within_mean: float
    within_median: float
    within_std: float


class AnomalyReport(NamedTuple):
    low_homogeneity: list[str]                         # groups with within_mean < 0.3
    high_cross: list[tuple[str, str, float]]           # (g1, g2, corr) with cross > 0.5
    misplaced: list[tuple[str, str, str, float, float]]  # (factor, own_group, best_other, own_corr, other_corr)


def compute_statistics(
    corr: pd.DataFrame,
    group_map: dict[str, list[str]],
    mat9: pd.DataFrame,
) -> tuple[list[GroupStats], float, float, AnomalyReport]:
    """Compute per-group stats, overall totals, and anomaly flags."""
    print("[stats] Computing statistics...")

    group_stats: list[GroupStats] = []
    for g in GROUPS:
        factors = group_map[g]
        n = len(factors)
        sub = corr.loc[factors, factors]
        if n < 2:
            vals = np.array([float("nan")])
        else:
            vals = sub.values[np.triu_indices(n, k=1)]

        group_stats.append(GroupStats(
            group=g,
            n_factors=n,
            within_mean=float(np.nanmean(vals)),
            within_median=float(np.nanmedian(vals)),
            within_std=float(np.nanstd(vals)),
        ))

    # Overall within vs cross
    all_within_vals: list[float] = []
    all_cross_vals: list[float] = []
    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            if i > j:
                continue
            f1, f2 = group_map[g1], group_map[g2]
            if g1 == g2:
                sub = corr.loc[f1, f1].values
                n = len(f1)
                all_within_vals.extend(sub[np.triu_indices(n, k=1)].tolist())
            else:
                all_cross_vals.extend(corr.loc[f1, f2].values.flatten().tolist())

    total_within = float(np.nanmean(all_within_vals))
    total_cross = float(np.nanmean(all_cross_vals))

    # Anomalies
    low_homogeneity = [s.group for s in group_stats if s.within_mean < 0.3]

    high_cross: list[tuple[str, str, float]] = []
    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            if j <= i:
                continue
            v = float(mat9.loc[g1, g2])
            if v > 0.5:
                high_cross.append((g1, g2, v))

    # Misplaced factors: within_corr < best_other_group_corr
    misplaced: list[tuple[str, str, str, float, float]] = []
    for g in GROUPS:
        factors = group_map[g]
        others_in_group = [f for f in factors]
        for fac in factors:
            # within-group mean (excluding self)
            peers = [f for f in factors if f != fac]
            if not peers:
                continue
            own_corr = float(np.nanmean(corr.loc[fac, peers].values))

            # best other-group mean
            best_other_g = None
            best_other_corr = -999.0
            for g2 in GROUPS:
                if g2 == g:
                    continue
                f2 = group_map[g2]
                if not f2:
                    continue
                v = float(np.nanmean(corr.loc[fac, f2].values))
                if v > best_other_corr:
                    best_other_corr = v
                    best_other_g = g2

            if best_other_corr > own_corr:
                misplaced.append((fac, g, best_other_g, own_corr, best_other_corr))

    print(f"[stats] Done. within={total_within:.3f}, cross={total_cross:.3f}, "
          f"low_homogeneity={low_homogeneity}, high_cross pairs={len(high_cross)}, "
          f"misplaced={len(misplaced)}")

    return group_stats, total_within, total_cross, AnomalyReport(low_homogeneity, high_cross, misplaced)


# ---------------------------------------------------------------------------
# 6. Full single-type analysis pipeline
# ---------------------------------------------------------------------------
class AnalysisResult(NamedTuple):
    data_type: str
    analysis_data: AnalysisData
    corr: pd.DataFrame
    mat9: pd.DataFrame
    group_stats: list[GroupStats]
    total_within: float
    total_cross: float
    anomalies: AnomalyReport
    time_range: tuple[str, str]


def run_analysis(data_type: str) -> AnalysisResult:
    """Run complete correlation analysis for one data type."""
    print(f"\n{'='*60}")
    print(f"  Analysis: {data_type.upper()} VALUES")
    print(f"{'='*60}")

    analysis_data = load_data(data_type)
    merged = analysis_data.merged_df
    group_map = analysis_data.group_map

    method = CORR_METHOD_FACTOR if data_type == "factor" else CORR_METHOD_SIGNAL
    corr = compute_corr_matrix(merged, method)

    mat9 = compute_9x9_matrix(corr, group_map)
    group_stats, total_within, total_cross, anomalies = compute_statistics(corr, group_map, mat9)

    label = "Factor Values" if data_type == "factor" else "Signal Values"
    full_title = f"{label} Correlation Heatmap ({method.capitalize()}) — {merged.shape[1]} factors"
    plot_full_heatmap(
        corr,
        group_map,
        full_title,
        OUTPUT_DIR / f"{data_type}_value_heatmap_full.png",
    )

    summary_title = f"{label} — 9×9 Group-Average Correlation"
    plot_9x9_heatmap(mat9, summary_title, OUTPUT_DIR / f"{data_type}_value_heatmap_9x9.png")

    time_range = (
        str(merged.index[0].date()),
        str(merged.index[-1].date()),
    )

    return AnalysisResult(
        data_type=data_type,
        analysis_data=analysis_data,
        corr=corr,
        mat9=mat9,
        group_stats=group_stats,
        total_within=total_within,
        total_cross=total_cross,
        anomalies=anomalies,
        time_range=time_range,
    )


# ---------------------------------------------------------------------------
# 7. Markdown report
# ---------------------------------------------------------------------------
def _md_group_stats_table(stats: list[GroupStats]) -> str:
    rows = ["| 組別 | 因子數 | 組內均值 | 組內中位數 | 組內標準差 |",
            "|------|--------|----------|------------|------------|"]
    for s in stats:
        rows.append(
            f"| {s.group} | {s.n_factors} | {s.within_mean:.3f} | "
            f"{s.within_median:.3f} | {s.within_std:.3f} |"
        )
    return "\n".join(rows)


def _md_9x9_table(mat9: pd.DataFrame) -> str:
    header = "| 組 | " + " | ".join(GROUPS) + " |"
    sep = "|----" + "|------" * 9 + "|"
    rows = [header, sep]
    for g in GROUPS:
        vals = " | ".join(f"{mat9.loc[g, g2]:.2f}" for g2 in GROUPS)
        rows.append(f"| {g} | {vals} |")
    return "\n".join(rows)


def _md_section(result: AnalysisResult, section_num: int, label: str) -> str:
    s = result
    ad = s.analysis_data

    dropped_str = (
        f"無剔除因子。" if not ad.dropped
        else f"已剔除因子（缺失率 >{MISSING_THRESHOLD:.0%}）：{', '.join(ad.dropped)}"
    )

    within_vs_cross_diff = s.total_within - s.total_cross
    if within_vs_cross_diff > 0.15:
        conclusion = "組內相關性顯著高於組間，因子分類具備較好的同質性。"
    elif within_vs_cross_diff > 0.05:
        conclusion = "組內相關性略高於組間，分類基本合理，但部分組別可能需要審查。"
    else:
        conclusion = "組內與組間相關性差距較小，分類同質性不足，建議深入審查各組。"

    low_hom = s.anomalies.low_homogeneity
    high_cross = s.anomalies.high_cross
    misplaced = s.anomalies.misplaced

    low_hom_str = (
        "無。" if not low_hom
        else "、".join(low_hom) + f"（共 {len(low_hom)} 組組內均值 < 0.3）"
    )
    high_cross_str = (
        "無。" if not high_cross
        else "\n".join(f"  - {g1}-{g2}：{v:.3f}" for g1, g2, v in high_cross)
    )
    misplaced_str: str
    if not misplaced:
        misplaced_str = "無疑似分錯組因子。"
    else:
        lines = ["| 因子 | 當前組 | 最高相關別組 | 本組均值 | 別組均值 |",
                 "|------|--------|-------------|----------|----------|"]
        for fac, own_g, best_g, own_c, best_c in misplaced[:50]:  # cap at 50 rows
            lines.append(f"| {fac} | {own_g} | {best_g} | {own_c:.3f} | {best_c:.3f} |")
        if len(misplaced) > 50:
            lines.append(f"| ... | （共 {len(misplaced)} 個，僅顯示前 50） | | | |")
        misplaced_str = "\n".join(lines)

    return f"""## {section_num}. {label}相關性分析

### {section_num}.1 組內 vs 組間總覽

| 指標 | 數值 |
|------|------|
| 總組內平均相關性 | {s.total_within:.3f} |
| 總組間平均相關性 | {s.total_cross:.3f} |
| 差值（組內 − 組間） | {within_vs_cross_diff:.3f} |

**結論**：{conclusion}

### {section_num}.2 各組組內統計

{_md_group_stats_table(s.group_stats)}

### {section_num}.3 9×9 組間相關性矩陣

（對角線為組內平均，上下三角為組間平均）

{_md_9x9_table(s.mat9)}

### {section_num}.4 異常檢測

**組內同質性差（均值 < 0.3）**：{low_hom_str}

**組間高度相似（均值 > 0.5）**：
{high_cross_str if high_cross else "無。"}

**疑似分錯組因子**（與別組平均相關性高於本組）：

{misplaced_str}
"""


def generate_report(signal_result: AnalysisResult, factor_result: AnalysisResult) -> None:
    """Write markdown report combining both analyses."""
    print("\n[report] Generating markdown report...")

    # Data overview section (use signal analysis data as reference)
    ad = signal_result.analysis_data
    group_counts = {g: len(ad.group_map[g]) for g in GROUPS}
    dropped_all = list(set(signal_result.analysis_data.dropped + factor_result.analysis_data.dropped))

    overview_rows = ["| 組別 | 因子數 |", "|------|--------|"]
    for g in GROUPS:
        overview_rows.append(f"| {g} | {group_counts[g]} |")
    overview_rows.append(f"| **合計** | **{sum(group_counts.values())}** |")

    # Section 4: signal vs factor comparison
    comparison_lines: list[str] = []
    anomaly_found = False
    for s_stat in signal_result.group_stats:
        g = s_stat.group
        f_stat_list = [x for x in factor_result.group_stats if x.group == g]
        if not f_stat_list:
            continue
        f_stat = f_stat_list[0]
        if f_stat.within_mean > 0.5 and s_stat.within_mean < 0.3:
            comparison_lines.append(
                f"- **{g}**：因子值組內均值={f_stat.within_mean:.3f}（高），"
                f"信號組內均值={s_stat.within_mean:.3f}（低）。"
                f"同質因子產生了異質信號，建議檢查信號生成規則。"
            )
            anomaly_found = True

    if not anomaly_found:
        comparison_lines.append("無明顯異常：各組因子值與信號值的組內相關性模式基本一致。")

    comparison_str = "\n".join(comparison_lines)

    report = f"""# 因子相關性分析報告

> 生成日期：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
> 相關性方法：因子值用 Pearson；信號值用 Spearman

---

## 一、數據概況

### 各組因子數量

{chr(10).join(overview_rows)}

### 時間範圍

信號值時間範圍：{signal_result.time_range[0]} – {signal_result.time_range[1]}

因子值時間範圍：{factor_result.time_range[0]} – {factor_result.time_range[1]}

### 剔除因子

{"無剔除因子。" if not dropped_all else "剔除因子（缺失率 >" + f"{MISSING_THRESHOLD:.0%}" + "）：" + "、".join(dropped_all)}

---

{_md_section(signal_result, section_num=2, label="信號值（主分析）")}

---

{_md_section(factor_result, section_num=3, label="因子值（輔助）")}

---

## 四、信號 vs 因子值對比

{comparison_str}
"""

    out_path = OUTPUT_DIR / "correlation_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[report] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        signal_result = run_analysis("signal")
        factor_result = run_analysis("factor")

    generate_report(signal_result, factor_result)

    print("\n[done] All outputs written to:", OUTPUT_DIR)
    print("  " + "\n  ".join(str(p) for p in sorted(OUTPUT_DIR.iterdir())))


if __name__ == "__main__":
    main()
