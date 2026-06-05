import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# 让 Python 能找到 src/mapping_loader.py（与本文件同级的 src/ 目录）
sys.path.insert(0, str(Path(__file__).parent))
from src.mapping_loader import load_mapping, build_file_coverage

# ── 默认路径常量 ──────────────────────────────────────────────────────────────
DEFAULT_RAW_DIR   = r"/Users/chloezh/Projects/MAIN_STYLE_ROTATION/A_data/data"
DEFAULT_CLEAN_DIR   = r"/Users/chloezh/Projects/MAIN_STYLE_ROTATION/A_data/prepared_data"
DEFAULT_MAPPING_JSON = r"/Users/chloezh/Projects/MAIN_STYLE_ROTATION/A_data/reference/data_inventory_A.json"

st.set_page_config(page_title="清洗覆盖审计", page_icon="🗂️", layout="wide")
st.title("🗂️ Parquet 清洗覆盖审计")

# ── session_state 初始化 ──────────────────────────────────────────────────────
if "scanned" not in st.session_state:
    st.session_state.scanned = False
if "raw_dir_saved" not in st.session_state:
    st.session_state.raw_dir_saved = ""
if "cleaned_dir_saved" not in st.session_state:
    st.session_state.cleaned_dir_saved = ""
if "mapping_json_saved" not in st.session_state:
    st.session_state.mapping_json_saved = ""

# ── 侧边栏 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("目录配置")
    raw_dir_input = st.text_input(
        "raw_dir（原始数据目录）", placeholder="/data/raw",
        value=st.session_state.raw_dir_saved or DEFAULT_RAW_DIR,
    )
    cleaned_dir_input = st.text_input(
        "cleaned_dir（清洗数据目录）", placeholder="/data/cleaned",
        value=st.session_state.cleaned_dir_saved or DEFAULT_CLEAN_DIR,
    )
    mapping_json_input = st.text_input(
        "mapping JSON（可选）", placeholder="/path/to/mapping.json",
        value=st.session_state.mapping_json_saved or DEFAULT_MAPPING_JSON,
    )
    if st.button("🔍 扫描", use_container_width=True):
        st.session_state.scanned            = True
        st.session_state.raw_dir_saved      = raw_dir_input
        st.session_state.cleaned_dir_saved  = cleaned_dir_input
        st.session_state.mapping_json_saved = mapping_json_input

# ── 未扫描时提示 ──────────────────────────────────────────────────────────────
if not st.session_state.scanned:
    st.info("请在左侧填写两个目录路径，然后点击「扫描」。")
    st.stop()

# ── 使用 session_state 中保存的路径 ──────────────────────────────────────────
raw_str     = st.session_state.raw_dir_saved.strip()
cleaned_str = st.session_state.cleaned_dir_saved.strip()

raw_dir     = Path(raw_str).expanduser()     if raw_str     else None
cleaned_dir = Path(cleaned_str).expanduser() if cleaned_str else None

errors = []
if not raw_dir or not raw_dir.exists():
    errors.append(f"❌ raw_dir 不存在或未填写：`{raw_str}`")
if not cleaned_dir or not cleaned_dir.exists():
    errors.append(f"❌ cleaned_dir 不存在或未填写：`{cleaned_str}`")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ── 扫描文件 ──────────────────────────────────────────────────────────────────
SCAN_SUFFIXES = {".parquet", ".xlsx"}
raw_map     = {p.name: p for p in raw_dir.rglob("*") if p.suffix in SCAN_SUFFIXES}
cleaned_map = {p.name: p for p in cleaned_dir.rglob("*") if p.suffix in SCAN_SUFFIXES}
all_files   = raw_map.keys() | cleaned_map.keys()

# ── 加载 mapping（可选）──────────────────────────────────────────────────────
mapping      = None
mapping_note = None
mapping_str  = st.session_state.mapping_json_saved.strip()
if mapping_str:
    mapping, mapping_err = load_mapping(mapping_str)
    if mapping_err:
        st.warning(f"⚠️ mapping JSON 读取失败，已回退到同名匹配：{mapping_err}")
    else:
        mapping_note = f"✅ 已加载 mapping JSON，共 {len(mapping)} 条记录"

if mapping_note:
    st.info(mapping_note)

# ── 覆盖表 ────────────────────────────────────────────────────────────────────
rows = build_file_coverage(raw_map, cleaned_map, mapping)
df   = pd.DataFrame(rows, columns=[
    "file_name", "raw_exists", "cleaned_exists", "status",
    "match_method", "mapped_raw_files", "mapping_status",
])

# ── 统计指标 ──────────────────────────────────────────────────────────────────
n_raw       = len(raw_map)
n_cleaned   = len(cleaned_map)
n_uncleaned = int((df["status"] == "⏳ 未清洗").sum())
n_orphan    = int((df["status"] == "👻 孤儿清洗文件").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw 文件数",    n_raw)
c2.metric("Cleaned 文件数", n_cleaned)
c3.metric("未清洗",        n_uncleaned, delta=f"-{n_uncleaned}" if n_uncleaned else None, delta_color="inverse")
c4.metric("孤儿清洗文件",  n_orphan,    delta=f"{n_orphan}"     if n_orphan    else None, delta_color="inverse")

st.divider()

# ── 过滤器 + 覆盖表 ───────────────────────────────────────────────────────────
status_options  = ["全部"] + sorted(df["status"].unique().tolist())
selected_status = st.selectbox("按状态筛选", status_options)
display_df      = df if selected_status == "全部" else df[df["status"] == selected_status]

st.subheader(f"覆盖表（共 {len(display_df)} 条）")
st.dataframe(
    display_df, use_container_width=True, hide_index=True,
    column_config={
        "file_name":        st.column_config.TextColumn("文件名（cleaned）", width="large"),
        "raw_exists":       st.column_config.CheckboxColumn("raw 存在"),
        "cleaned_exists":   st.column_config.CheckboxColumn("cleaned 存在"),
        "status":           st.column_config.TextColumn("状态",          width="medium"),
        "match_method":     st.column_config.TextColumn("匹配方式",       width="small"),
        "mapped_raw_files": st.column_config.TextColumn("对应 raw 文件",  width="large"),
        "mapping_status":   st.column_config.TextColumn("mapping 状态",   width="medium"),
    },
)

st.divider()

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def build_profile(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in data.columns:
        s      = data[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        records.append({
            "column":  col,
            "dtype":   str(s.dtype),
            "null_%":  round(s.isna().mean() * 100, 2),
            "min":     round(float(s.min()),  4) if is_num and s.notna().any() else None,
            "max":     round(float(s.max()),  4) if is_num and s.notna().any() else None,
            "mean":    round(float(s.mean()), 4) if is_num and s.notna().any() else None,
        })
    return pd.DataFrame(records)

def show_profile(data: pd.DataFrame, label: str):
    st.markdown(f"**{label}** — {data.shape[0]:,} 行 × {data.shape[1]} 列")
    prof = build_profile(data)
    st.dataframe(
        prof, use_container_width=True, hide_index=True,
        column_config={
            "column": st.column_config.TextColumn("字段名"),
            "dtype":  st.column_config.TextColumn("类型"),
            "null_%": st.column_config.ProgressColumn("缺失率 %", min_value=0, max_value=100, format="%.2f%%"),
            "min":    st.column_config.NumberColumn("min",  format="%.4f"),
            "max":    st.column_config.NumberColumn("max",  format="%.4f"),
            "mean":   st.column_config.NumberColumn("mean", format="%.4f"),
        },
    )

def show_profile_diff(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame):
    raw_prof     = build_profile(raw_data).set_index("column")
    cleaned_prof = build_profile(cleaned_data).set_index("column")
    all_cols     = raw_prof.index.union(cleaned_prof.index)
    rows = []
    for col in all_cols:
        r = raw_prof.loc[col]     if col in raw_prof.index     else {}
        c = cleaned_prof.loc[col] if col in cleaned_prof.index else {}
        rows.append({
            "字段名":     col,
            "raw dtype":  r.get("dtype",  "—"),
            "cln dtype":  c.get("dtype",  "—"),
            "raw null %": r.get("null_%", None),
            "cln null %": c.get("null_%", None),
            "raw min":    r.get("min",    None),
            "cln min":    c.get("min",    None),
            "raw max":    r.get("max",    None),
            "cln max":    c.get("max",    None),
            "raw mean":   r.get("mean",   None),
            "cln mean":   c.get("mean",   None),
        })
    diff_df = pd.DataFrame(rows)
    st.dataframe(
        diff_df, use_container_width=True, hide_index=True,
        column_config={
            "字段名":      st.column_config.TextColumn("字段名",    width="medium"),
            "raw dtype":  st.column_config.TextColumn("raw 类型"),
            "cln dtype":  st.column_config.TextColumn("cln 类型"),
            "raw null %": st.column_config.ProgressColumn("raw 缺失%", min_value=0, max_value=100, format="%.2f%%"),
            "cln null %": st.column_config.ProgressColumn("cln 缺失%", min_value=0, max_value=100, format="%.2f%%"),
            "raw min":    st.column_config.NumberColumn("raw min",  format="%.4f"),
            "cln min":    st.column_config.NumberColumn("cln min",  format="%.4f"),
            "raw max":    st.column_config.NumberColumn("raw max",  format="%.4f"),
            "cln max":    st.column_config.NumberColumn("cln max",  format="%.4f"),
            "raw mean":   st.column_config.NumberColumn("raw mean", format="%.4f"),
            "cln mean":   st.column_config.NumberColumn("cln mean", format="%.4f"),
        },
    )

# ── 文件选择 & 预览 ───────────────────────────────────────────────────────────
st.subheader("📄 文件预览 & Profile")

# selectbox 的候选列表用 df 的 file_name（与覆盖表一致）
preview_options = ["（请选择）"] + sorted(df["file_name"].tolist())
selected_file = st.selectbox("选择要检视的文件", preview_options)

if not selected_file or selected_file == "（请选择）":
    st.stop()

file_row  = df[df["file_name"] == selected_file].iloc[0]
has_raw   = bool(file_row["raw_exists"])
has_clean = bool(file_row["cleaned_exists"])

# ── 定位 raw 文件真实路径（mapping 模式下 raw 文件名可能与 file_name 不同）──
def find_raw_path(file_row, raw_map) -> Path | None:
    """优先用 mapped_raw_files 的第一个文件名，回退到 file_name 同名查找。"""
    mapped = file_row.get("mapped_raw_files", "")
    if mapped and mapped != "—":
        first = mapped.split(",")[0].strip()
        if first in raw_map:
            return raw_map[first]
    # 同名兜底
    name = file_row["file_name"]
    return raw_map.get(name)

def find_cleaned_path(file_row, cleaned_map) -> Path | None:
    name = file_row["file_name"]
    return cleaned_map.get(name)

# ── 根据后缀选择读取方式 ──────────────────────────────────────────────────────
def safe_read(path: Path):
    try:
        if path.suffix == ".xlsx":
            return pd.read_excel(path), None
        else:
            return pd.read_parquet(path), None
    except Exception as e:
        return None, str(e)

raw_path     = find_raw_path(file_row, raw_map)     if has_raw   else None
cleaned_path = find_cleaned_path(file_row, cleaned_map) if has_clean else None

raw_data,     raw_err   = safe_read(raw_path)     if raw_path     else (None, "找不到对应的 raw 文件路径")
cleaned_data, clean_err = safe_read(cleaned_path) if cleaned_path else (None, "找不到对应的 cleaned 文件路径")

# ── 动态 Tab ──────────────────────────────────────────────────────────────────
tab_labels = []
if has_raw:               tab_labels.append("Raw 预览")
if has_clean:             tab_labels.append("Cleaned 预览")
if has_raw:               tab_labels.append("Raw Profile")
if has_clean:             tab_labels.append("Cleaned Profile")
if has_raw and has_clean: tab_labels.append("Profile 对比")

tabs = st.tabs(tab_labels)
idx  = 0

if has_raw:
    with tabs[idx]:
        if raw_err: st.error(f"读取失败：{raw_err}")
        else:
            st.caption(f"{raw_data.shape[0]:,} 行 × {raw_data.shape[1]} 列，展示前 50 行")
            st.dataframe(raw_data.head(50), use_container_width=True)
    idx += 1

if has_clean:
    with tabs[idx]:
        if clean_err: st.error(f"读取失败：{clean_err}")
        else:
            st.caption(f"{cleaned_data.shape[0]:,} 行 × {cleaned_data.shape[1]} 列，展示前 50 行")
            st.dataframe(cleaned_data.head(50), use_container_width=True)
    idx += 1

if has_raw:
    with tabs[idx]:
        if raw_err: st.error(f"读取失败：{raw_err}")
        else: show_profile(raw_data, "Raw")
    idx += 1

if has_clean:
    with tabs[idx]:
        if clean_err: st.error(f"读取失败：{clean_err}")
        else: show_profile(cleaned_data, "Cleaned")
    idx += 1

if has_raw and has_clean:
    with tabs[idx]:
        if raw_err or clean_err:
            if raw_err:   st.error(f"raw 读取失败：{raw_err}")
            if clean_err: st.error(f"cleaned 读取失败：{clean_err}")
        else:
            rc, cc = raw_data.shape, cleaned_data.shape
            m1, m2 = st.columns(2)
            m1.metric("Raw 行数",     f"{rc[0]:,}")
            m2.metric("Cleaned 行数", f"{cc[0]:,}", delta=f"{cc[0]-rc[0]:+,}", delta_color="normal")
            n1, n2 = st.columns(2)
            n1.metric("Raw 列数",     rc[1])
            n2.metric("Cleaned 列数", cc[1], delta=f"{cc[1]-rc[1]:+}", delta_color="normal")
            st.markdown("#### 逐字段对比")
            show_profile_diff(raw_data, cleaned_data)