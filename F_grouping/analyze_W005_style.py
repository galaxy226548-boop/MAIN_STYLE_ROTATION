"""
W005 策略分析：成長/價值風格主導 + 策略誤判分析
Usage: python F_grouping/analyze_W005_style.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
GROWTH_XLSX = ROOT / "A_data/data/style_indexes/growth_index.xlsx"
VALUE_XLSX  = ROOT / "A_data/data/style_indexes/value_index.xlsx"
SIGNAL_PQ   = ROOT / "F_grouping/output_COMB/W005_binary_signal_ls.parquet"
RECORD_JSON = ROOT / "F_grouping/output_COMB/W005_binary_factor_grouping_record.json"
OUT_DIR     = ROOT / "F_grouping/output_COMB"

# signal_ls convention for this project:
#  +1 = bullish growth (expect growth > value)
#  -1 = bullish value  (expect value > growth)
SIGNAL_COL   = "W005_binary"
GROWTH_WINS  = +1.0   # signal value that means "predict growth outperforms"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_style_index(path: Path) -> pd.Series:
    """Load close price series, drop footer rows, return daily close."""
    df = pd.read_excel(path)
    df = df[pd.to_numeric(df["close"], errors="coerce").notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].astype(float)


def rolling_relative(g: pd.Series, v: pd.Series, window: int = 60) -> pd.Series:
    """60-day rolling return spread: growth_ret - value_ret."""
    gr = g.pct_change()
    vr = v.pct_change()
    spread = (gr - vr).rolling(window, min_periods=window // 2).mean()
    return spread


def annual_return(price: pd.Series) -> pd.Series:
    """Annual total return from daily close prices."""
    return price.resample("YE").last().pct_change()


# ── load data ─────────────────────────────────────────────────────────────────
growth_px = load_style_index(GROWTH_XLSX)
value_px  = load_style_index(VALUE_XLSX)
sig_df    = pd.read_parquet(SIGNAL_PQ)

with open(RECORD_JSON) as f:
    record = json.load(f)

component_cols = [c for c in sig_df.columns if c != SIGNAL_COL]
print(f"Component signals: {component_cols}")
print(f"Signal date range: {sig_df.index.min().date()} ~ {sig_df.index.max().date()}")
print(f"Strategy signal: {sig_df[SIGNAL_COL].value_counts().to_dict()}")

# Align to overlapping dates
common_dates = sig_df.index.intersection(growth_px.index).intersection(value_px.index)
growth_px = growth_px.reindex(common_dates)
value_px  = value_px.reindex(common_dates)
sig_df    = sig_df.reindex(common_dates)
print(f"Overlapping dates: {common_dates.min().date()} ~ {common_dates.max().date()}, n={len(common_dates)}")


# ── derived series ────────────────────────────────────────────────────────────
# Daily relative return (growth - value)
daily_spread = growth_px.pct_change() - value_px.pct_change()

# 60-day rolling spread (smoothed)
roll_spread = rolling_relative(growth_px, value_px, window=60)

# Cumulative relative performance (growth / value rebased to 1)
g_norm = growth_px / growth_px.iloc[0]
v_norm = value_px  / value_px.iloc[0]
cum_rel = g_norm / v_norm   # >1 means growth has outperformed since start

# Annual returns
g_annual = annual_return(growth_px)
v_annual = annual_return(value_px)

# Annual signal: most-common daily value each year
annual_signal = sig_df[SIGNAL_COL].resample("YE").apply(
    lambda s: s.mode().iloc[0] if len(s.dropna()) > 0 else np.nan
)

# Annual spread: growth_return - value_return
annual_spread = g_annual - v_annual   # >0 → growth dominated that year

# Prediction correctness per day:
# correct if (signal==+1 and growth>value) OR (signal==-1 and value>growth)
sig_aligned = sig_df[SIGNAL_COL]
outcome     = np.sign(daily_spread)   # +1 growth won, -1 value won, 0 tie
correct_day = (sig_aligned == outcome).astype(float)
correct_day[sig_aligned.isna() | (outcome == 0)] = np.nan

# Annual accuracy
annual_acc = correct_day.resample("YE").mean()

# Annual signal is WRONG when annual_spread and annual_signal disagree
annual_actual_winner = np.sign(annual_spread)   # +1 growth dominated, -1 value
annual_pred_winner   = annual_signal             # same convention
annual_correct       = (annual_pred_winner == annual_actual_winner)


# ── component signal contribution in error years ──────────────────────────────
error_years = annual_spread.index[~annual_correct & annual_correct.notna()]

component_alignment = {}
for col in component_cols:
    col_annual = sig_df[col].resample("YE").apply(
        lambda s: s.mode().iloc[0] if len(s.dropna()) > 0 else np.nan
    )
    # +1: component agreed with final signal; -1: disagreed
    alignment = (col_annual == annual_signal).map({True: 1, False: -1})
    component_alignment[col] = alignment

alignment_df = pd.DataFrame(component_alignment)


# ══ FIGURE 1: Style Dominance Overview ═══════════════════════════════════════
fig1, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig1.suptitle("成長 vs 價值 風格主導分析（W005 策略區間）", fontsize=14, fontweight="bold", y=0.98)

# Panel 1: Cumulative relative performance
ax = axes[0]
ax.plot(cum_rel.index, cum_rel.values, color="#e74c3c", linewidth=1.2, label="成長/價值 累計相對表現")
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
ax.fill_between(cum_rel.index, cum_rel.values, 1.0,
                where=cum_rel.values >= 1.0, alpha=0.25, color="#e74c3c", label="成長主導")
ax.fill_between(cum_rel.index, cum_rel.values, 1.0,
                where=cum_rel.values < 1.0, alpha=0.25, color="#3498db", label="價值主導")
ax.set_ylabel("相對指數（基準=1）", fontsize=9)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("累計相對表現：成長指數 / 價值指數（同起點歸一化）", fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Panel 2: 60-day rolling spread
ax = axes[1]
pos_mask = roll_spread >= 0
neg_mask = roll_spread < 0
ax.fill_between(roll_spread.index, roll_spread.values, 0,
                where=pos_mask, alpha=0.5, color="#e74c3c", label="成長主導（60日滾動）")
ax.fill_between(roll_spread.index, roll_spread.values, 0,
                where=neg_mask, alpha=0.5, color="#3498db", label="價值主導（60日滾動）")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("日均超額收益（成長-價值）", fontsize=9)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("60日滾動平均日超額收益（成長 − 價值）", fontsize=10)

# Panel 3: W005_binary signal overlay
ax = axes[2]
ax.step(sig_df.index, sig_df[SIGNAL_COL].fillna(method="ffill").values,
        color="#8e44ad", linewidth=0.8, alpha=0.8, label="W005 信號（+1=看漲成長，-1=看漲價值）")
ax.set_yticks([-1, 1])
ax.set_yticklabels(["−1 看漲價值", "+1 看漲成長"], fontsize=9)
ax.set_ylabel("信號方向", fontsize=9)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("W005 複合信號方向", fontsize=10)
ax.set_xlabel("日期", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = OUT_DIR / "W005_style_dominance.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")


# ══ FIGURE 2: Annual Hit Rate & Error Analysis ════════════════════════════════
years = annual_spread.index.year
n_years = len(years)

fig2, axes = plt.subplots(3, 1, figsize=(14, 12))
fig2.suptitle("W005 策略年度準確率與誤判分析", fontsize=14, fontweight="bold", y=0.98)

# Panel 1: Annual spread (actual) + signal prediction bar
ax = axes[0]
bar_width = 0.35
x = np.arange(n_years)
bars_actual = ax.bar(x - bar_width/2, annual_spread.values * 100,
                     bar_width, label="實際年超額（成長-價值）%",
                     color=["#e74c3c" if v > 0 else "#3498db" for v in annual_spread.values],
                     alpha=0.8)
# Mark signal direction as arrow/dot
for i, (yr, sig) in enumerate(zip(years, annual_signal.values)):
    if np.isnan(sig):
        continue
    marker = "▲" if sig == 1 else "▼"
    color  = "#e74c3c" if sig == 1 else "#3498db"
    ax.text(i + bar_width/2, 0, marker, ha="center", va="bottom" if sig == 1 else "top",
            fontsize=12, color=color)

ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontsize=8)
ax.set_ylabel("年超額收益率（%）", fontsize=9)
ax.set_title("年度：實際超額收益（柱）+ 策略預測方向（▲成長 ▼價值）", fontsize=10)
ax.legend(fontsize=8)

# Panel 2: Annual accuracy + highlight error years
ax = axes[1]
acc_values = annual_acc.values
acc_colors = ["#27ae60" if v >= 0.5 else "#e74c3c" for v in acc_values]
bars = ax.bar(x, acc_values * 100, color=acc_colors, alpha=0.8)
ax.axhline(50, color="gray", linewidth=1, linestyle="--", label="50% 基準線")
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45, fontsize=8)
ax.set_ylabel("日勝率（%）", fontsize=9)
ax.set_ylim(0, 100)
ax.set_title("年度日勝率（綠=勝>50%，紅=誤判為主）", fontsize=10)
ax.legend(fontsize=8)

# Label error years
for i, (yr, v) in enumerate(zip(years, acc_values)):
    if v < 0.5:
        ax.text(i, v * 100 + 1, f"{yr}\n{v*100:.0f}%", ha="center", va="bottom",
                fontsize=7, color="#c0392b", fontweight="bold")
    else:
        ax.text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center", va="bottom",
                fontsize=6, color="#27ae60")

# Panel 3: Component signal heatmap in error years
ax = axes[2]
if len(error_years) > 0:
    err_yr_labels = [str(y.year) for y in error_years]
    heat_data = alignment_df.loc[error_years].T   # components × error_years
    im = ax.imshow(heat_data.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(err_yr_labels)))
    ax.set_xticklabels(err_yr_labels, fontsize=9)
    ax.set_yticks(range(len(component_cols)))
    ax.set_yticklabels(component_cols, fontsize=9)
    ax.set_title("誤判年份中各子信號方向（藍=與策略一致/推向錯誤，紅=逆策略方向）", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6, label="+1=與W005一致  −1=逆W005")

    # Annotate cells with agreement/disagreement
    for yi, col in enumerate(component_cols):
        for xi, yr in enumerate(error_years):
            val = heat_data.iloc[yi, xi]
            if pd.notna(val):
                ax.text(xi, yi, "↑" if val == 1 else "↓", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
else:
    ax.text(0.5, 0.5, "無誤判年份", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("誤判年份中各子信號方向", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = OUT_DIR / "W005_annual_hitrate_error.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")


# ══ FIGURE 3: Error-year deep dive ═══════════════════════════════════════════
if len(error_years) > 0:
    # For each error year, show component signal timeline + actual spread
    n_err = len(error_years)
    fig3, axes3 = plt.subplots(n_err, 1, figsize=(14, 4 * n_err + 1), sharex=False)
    if n_err == 1:
        axes3 = [axes3]
    fig3.suptitle("誤判年份詳細分析：子信號 vs 實際超額走勢", fontsize=13, fontweight="bold")

    for ax, err_yr in zip(axes3, error_years):
        yr = err_yr.year
        mask = sig_df.index.year == yr
        sub_sig = sig_df[mask]
        sub_spread = daily_spread[mask].cumsum() * 100

        ax2 = ax.twinx()
        ax2.fill_between(sub_spread.index, sub_spread.values, 0,
                         where=sub_spread.values >= 0, alpha=0.15, color="#e74c3c", label="成長跑贏（累計）")
        ax2.fill_between(sub_spread.index, sub_spread.values, 0,
                         where=sub_spread.values < 0, alpha=0.15, color="#3498db", label="價值跑贏（累計）")
        ax2.set_ylabel("累計超額（成長-價值, %）", fontsize=8)

        colors_map = {c: plt.cm.tab10(i) for i, c in enumerate(component_cols)}
        for col in component_cols:
            s = sub_sig[col].fillna(method="ffill")
            ax.step(s.index, s.values, linewidth=1.2, label=col,
                    color=colors_map[col], alpha=0.75)

        # W005 final signal
        ax.step(sub_sig.index, sub_sig[SIGNAL_COL].fillna(method="ffill").values,
                linewidth=2.5, color="black", linestyle="--", label=f"{SIGNAL_COL}（最終）")

        ax.set_yticks([-1, 1])
        ax.set_yticklabels(["−1", "+1"], fontsize=8)
        ax.set_ylabel("信號方向", fontsize=9)
        ax.set_title(f"{yr} 年誤判分析（策略 vs 各子信號）", fontsize=10)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax2.legend(loc="upper right", fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out3 = OUT_DIR / "W005_error_year_detail.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")


# ══ PRINT SUMMARY ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W005 策略摘要")
print("=" * 60)
print(f"信號覆蓋：{component_cols}")
print(f"\n年度結果（'actual' = 成長-價值年超額，'signal' = 策略年方向，'hit' = 日勝率>50%）：")
summary = pd.DataFrame({
    "actual_spread%": (annual_spread * 100).round(1),
    "actual_winner": annual_actual_winner.map({1.0: "成長", -1.0: "價值", 0.0: "平手"}),
    "W005_signal": annual_signal.map({1.0: "看漲成長", -1.0: "看漲價值"}),
    "correct": annual_correct,
    "day_hit%": (annual_acc * 100).round(1),
})
summary.index = summary.index.year
print(summary.to_string())

print(f"\n誤判年份：{[y.year for y in error_years]}")
if len(error_years) > 0:
    print("\n各誤判年份中，與策略最一致（最應反思）的子信號：")
    for yr in error_years:
        row = alignment_df.loc[yr]
        agree_sigs = row[row == 1].index.tolist()
        disagree_sigs = row[row == -1].index.tolist()
        print(f"  {yr.year}: 贊同策略（引向錯誤）= {agree_sigs} | 提出異議 = {disagree_sigs}")

print("\n" + "=" * 60)
print("建議探索方向：")
suggestions = [
    "1. 換手率分析：W005 信號切換頻率 vs 實際風格切換頻率 是否匹配？",
    "2. 信號領先/滯後測試：各子信號對未來 N 日超額收益的 IC 值",
    "3. 市場狀態分層：在熊市/牛市/震蕩市中，W005 勝率是否分化？",
    "4. 子信號去除實驗：移除哪個子信號後年度整體勝率提升？",
    "5. 宏觀週期疊加：對比中債10Y收益率、PMI 轉折點與誤判年份的重疊",
    "6. 滯後效果：成長/價值輪動通常存在慣性，考慮對信號做1~5日平滑",
]
for s in suggestions:
    print(f"  {s}")
