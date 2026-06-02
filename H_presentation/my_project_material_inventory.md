# 项目材料清单 — 成长/价值风格轮动

> 生成日期：2026-05-28｜用于 PPT 大纲规划

---

## 1. 项目背景：为什么做这个课题

从周报和研报阅读笔记可以还原：

- **宏观逻辑**：成长/价值风格轮动是资本市场中被广泛记录的现象（利率、流动性、盈利周期均能驱动切换），但市场上多数研究停留在"观察描述"层面，缺乏系统化、可落地的量化信号体系。
- **实践动机**：该课题参考了广发、兴业、华创、招商等多家券商的方法框架，但并非照搬，而是自主构建了一套覆盖宏观、流动性、估值、量价等多类因子的信号体系。
- **延伸应用**：构建的信号（特别是 W008）已被接入宏观风险预算/风险平价模型，作为权益资产内部的结构性 alpha 来源，说明研究有直接的资产配置应用场景。
- **Fund_Analysis 旁证**：通过对主动权益型基金做 RBSA 风格敞口分析，从外部视角佐证了成长/价值风格确实存在、且具有周期性切换特征。

---

## 2. 研究目标：要解决什么问题

1. **主问题**：能否用可观测的宏观/量价/估值指标，提前判断 A 股成长风格与价值风格之间的强弱切换方向？
2. **子问题**：
   - 哪些因子（单独或组合）具有稳定的预测信号（IC稳定、月度胜率 > 50%）？
   - 如何将因子信号转化为多空仓位（大盘 vs 小盘 / 价值 vs 成长）并评估净值表现？
   - 多因子投票组合是否优于单因子？
   - 组合信号能否增厚资产配置模型中的权益收益？

---

## 3. 已完成工作（按模块）

### 3.1 数据与因子库建设（B_factors）

- 建立了覆盖 **9 大类别** 的因子库，总量超过 **473 个**（来自 IC_score 行数），含 C/D/F/G/I/L/O/P/V 类别，以及 ZHAO 系列（初始测试用因子）。
- 代码架构：`因子矩阵生成与批量挂载.py`（不可改动的核心脚本）→ `build_factor_matrix.py`（主入口）→ 12 个 `priceFactors*.py`、`profitFactors*.py`、`flationNgrowth_factors.py`、`flow_factors.py`、`overseaFactors.py` 等模块。
- 针对每类因子定义了专用的挂载规则：**state 因子**用 shift(1) 信息可用逻辑，**event 因子**仅挂到事件触发后的首个候选日。
- 因子 metadata 管理：`factor_done.json`（登记完成因子）、`factor_generated.json`（生成成功登记）。

### 3.2 IC 分析（D_analysis）

- 对 474 个 因子/信号/组合版本 全量跑了 IC 分析，输出 `*_IC_analysis.xlsx` + `*_rolling_IC.png`。
- IC 覆盖层级：
  - W003 全因子矩阵（按类别：G/I/P/V/W）
  - W004 全因子矩阵（按类别：D/F/G/I/L/P/V）
  - W004_factor_generator（逐因子：D001/D002/F001/G003-G006/I003/L003-L006/O001-O002/P002/V003-V004）
  - factor_V 系列（约 170 个，V071～V210，含 rolling IC png）
- `IC_score.xlsx`：对所有因子按 4 个维度打分（非正IC概率、正IC概率、Pearson/Rank IC方向），满分 4 分；
  - **得分 ≥ 3 的因子共 46 个**，含 C033/C034、F014/F023/F024、G001/G026/G027/G055/G056、I001/I002/I006/I007、P021/P035、V001/V128/V155/V167/V168/V201 及 W003-W008 系列组合。
- 负 IC 检查：`nega_checked.md`（已通过的因子，约 200+）+ `nega_doubt.md`（待核查因子）。

### 3.3 信号构建与组合演进

| 版本 | 说明 | 最优月胜率 |
|------|------|------------|
| W003 | 初代信号（多因子分类投票，多个子信号：G/I/P/V/W） | 约 50-53% |
| W004 | 扩大因子库（含 D/F/L 类），含 factor_generator 子信号 | 约 49-52% |
| W005_binary | 分类 binary 投票，第一代 | **55.21%** |
| W006_binary | 优化组合逻辑 | **55.31%** |
| W007_binary | 再优化 | 54.06% |
| **W008_binary** | **目前最优组合**（binary 分类投票，IC得分3/3） | **55.42%** |
| W008_signal_binary | 信号+binary 联合版 | 55.42% |
| W009_binary | 新加入因子后的测试 | 51.77%（下降，说明某些新因子降噪） |
| W010_binary | 进一步调整 | 53.96% |
| W00C_binary | 对照组/特定配置 | 49-52% |

**逻辑断裂注意**：W009 胜率明显低于 W008，说明并非因子越多越好——这是答辩可以主动解释的点。

### 3.4 回测系统（E_backtesting）

- 固定回测框架：给定信号 → 转成 growth/value 目标仓位 → 每 50 个交易日 rebalance → 输出净值/summary。
- 已跑回测总计：全因子回测 **494 个** `*_summary.xlsx`（含 factor_V 的 170 个、W00x 系列、W003/W004 各子类）。
- 每个回测输出：`combo_nav.png`、`combo_nav_ratio.png`（净值比值图）、逐 track 净值图、`summary.xlsx`（含分年度胜率、期望收益、超额指标）。
- W008_binary（最优）全区间指标：月胜率 **55.42%**，期间胜率 **52.2%**，盈亏比 1.057，期望收益 +0.0024/月，年化多头绝对收益 9.96%，年化超额收益 6.49%，最大回撤约 40%。

### 3.5 因子分组分析（F_grouping）

- 对 C/D/F/G/I/L/O/P/V 九个类别内部做了相关性分析，已生成 9 张热力图（`group_*_correlation_heatmaps.png`）。
- 有边际加入测试（`marginal_test/`）：测试每个因子加入组合后的边际效果。

### 3.6 资产配置模型嵌入（Uses_in_risk_parity_and_budget_model）

- 将 W008 信号作为权益资产内部的动态权重调整输入，嵌入风险预算/风险平价模型。
- 输出图表（`C_output/figures/W008/`）：
  - `01_dynamic_risk_budget.png`：动态风险预算分配
  - `01b_dynamic_target_weights.png`：动态目标权重
  - `02_unlevered_nav_month_vs_rw.png`：加权月频 vs 滚动窗口净值对比
  - `03/04_month_weights_line/stacked_bar.png`：权重变化
  - `05-07_levered_nav.png`：含杠杆的净值曲线对比
- 单指标单资产事件检验（`C_output/Results_20260223_235650/`）：对 HS300、HSI、SP500、NHCI、CBA02001、COMEX 六类资产，测试了大量宏观指标的事件驱动有效性。
- `B_analysis/Asset performance/`：六类资产的净值、回撤、最大回撤图（HS300/HSI/SP500/NHCI/COMEX/CBA02001）。

### 3.7 基金风格分析（Fund_Analysis）

- 用 RBSA（收益率基风格分析）对主动权益型基金做风格敞口拆解。
- 输出（`D_analysis/output/RBSA/`）：
  - `style_exposure_dynamics_scatter.png`：风格敞口动态散点
  - `cumulative_return_comparison.png` / `cumulative_return_comparison_from_2018.png`：成长 vs 价值累计收益对比
  - `annual_return_diff.png`：年度收益差
  - `annual_return_lines.png` / `annualized_return_lines.png`
  - `cumulative_stable_vs_wrong_5050.png`：稳定基金 vs 固定 5050 策略

---

## 4. 数据与资产

| 维度 | 内容 |
|------|------|
| 标的资产 | A 股成长/价值风格指数（大小盘代理），具体对应标的待在 PPT 中明确标注 |
| 对比基准 | 等权配置（50% 成长 + 50% 价值），每 2 个月 rebalance |
| 数据频率 | **月频**（因子信号为月频；回测调仓约每 50 交易日） |
| 样本区间 | **2010—2026**（含 2026 年 Q1，共约 16 年） |
| 宏观数据来源 | Wind（SHIBOR、DR007、信用利差、中短票、国开债、CPI、PMI、中长期贷款余额等） |
| 资产数据 | CSMAR（A股/全球指数）、Wind（沪深300、南华商品、中债企业债财富指数、恒生指数、SPX等） |
| 因子类别 | 流动性（L）、盈利（F/P）、成长（G）、估值（V）、量价（C/D）、债券（L/I）、事件/状态 |

---

## 5. 方法框架

```
原始数据（宏观/价量/估值）
    ↓
因子生成（B_factors，月频挂载，state/event 两类挂载逻辑）
    ↓
IC 分析 + 负 IC 检查（D_analysis，Rolling IC / IC打分）
    ↓
信号构建（binary 投票 / 分类投票，bar 阈值敏感性测试）
    ↓
仓位生成（C_positions，signal_ls → growth/value 目标仓位）
    ↓
固定回测（E_backtesting，50交易日 rebalance，基准=等权）
    ↓
多因子组合迭代（F_grouping 边际加入 + W00x 系列演进）
    ↓
资产配置嵌入（Uses_in_risk_parity，W008 → 权益子项动态权重）
```

**信号有效性评价标准**（来自周报研究综述）：
- 胜率（月度 > 50%，优秀 > 55%）
- 期望收益 > 0，盈亏比 > 1
- 触发次数 > 15
- 成长/价值波段分别胜率 > 50%（这一条 W008 未能满足，需主动解释）
- 超额夏普 > 0

---

## 6. 已有结果（哪些可以直接进 PPT）

### ✅ 可直接用的图/数据

1. **W008_binary 全区间 summary**（分年度指标表，2010—2026）：月胜率55.4%、期间胜率52.2%、盈亏比1.057、期望收益0.0024/月、全区间超额年化6.49%。
2. **W008_binary combo_nav.png**：主策略净值曲线（vs 基准）。
3. **W008_binary combo_nav_ratio.png**：相对净值比值图（清晰体现成长/价值切换节奏）。
4. **W005-W010 胜率对比表**（自制，从文件名中的 `mw` 数字提取）：体现组合演进的优化轨迹。
5. **IC_score.xlsx 得分分布**：474 个因子打分，46 个满足高分（≥3），说明筛选严格。
6. **Factor category 相关性热力图**（F_grouping，9 张）：体现组合设计基础。
7. **Fund_Analysis RBSA 图表**（4—5 张）：佐证成长/价值风格的现实意义。
8. **Uses_in_risk_parity W008 figures**（7 张）：展示信号在资产配置中的增厚效果。
9. **Rolling IC 图**（W003/W004 系列各类别）：体现因子稳定性。

### ⚠️ 需要加工才能用

- factor_V 系列单因子回测 summary：数量多（170个），需整理成汇总表而非逐一展示。
- nega_checked.md：因子筛选过程文档，可提炼成"筛选流程图"而非直接引用。

---

## 7. 可用图表清单

| 文件路径 | 图表内容 | 建议放在哪页 |
|----------|----------|--------------|
| `E_backtesting/Result/W008_binary/6_mw55.42_W008_binary/W008_binary_rebalance_50_combo_nav.png` | W008最优组合净值曲线（策略 vs 基准） | 核心结果页 |
| `E_backtesting/Result/W008_binary/6_mw55.42_W008_binary/W008_binary_rebalance_50_combo_nav_ratio.png` | 成长/价值净值比值 + 策略仓位 | 核心结果页 |
| `E_backtesting/Result/W008_binary/6_mw55.42_W008_binary/track_{0-4}_nav.png` | 各子轨净值曲线 | 组合分析/附录 |
| `D_analysis/IC_output/W004_factor/W004_rolling_IC.png` | W004 全因子矩阵 Rolling IC | 因子有效性页 |
| `D_analysis/IC_output/W004_factor/W004_{D/F/G/I/L/P/V}_rolling_IC.png` | 各类别 Rolling IC | 因子分类展示页 |
| `D_analysis/IC_output/W003_factor/W003_rolling_IC.png` | W003 滚动IC（早期版本对比） | 迭代演进页 |
| `F_grouping/reference/grouping_correlations/group_{C/D/F/G/I/L/O/P/V}_correlation_heatmaps.png` | 各类别内部因子相关性热力图（9张） | 因子构建/组合设计页 |
| `Fund_Analysis/D_analysis/output/RBSA/cumulative_return_comparison.png` | 主动基金中成长 vs 价值累计收益对比 | 问题动机/背景页 |
| `Fund_Analysis/D_analysis/output/RBSA/style_exposure_dynamics_scatter.png` | 主动基金风格敞口动态演变 | 问题动机/背景页 |
| `Fund_Analysis/D_analysis/output/RBSA/annual_return_diff.png` | 成长/价值年度收益差 | 风格分化描述页 |
| `Fund_Analysis/D_analysis/output/RBSA/annual_return_lines_stable.png` | 稳定风格基金年化收益线 | 风格分化描述页 |
| `Uses_in_risk_parity_and_budget_model/C_output/figures/W008/01_dynamic_risk_budget.png` | W008 驱动的动态风险预算分配 | 应用场景/增厚页 |
| `Uses_in_risk_parity_and_budget_model/C_output/figures/W008/02_unlevered_nav_month_vs_rw.png` | 嵌入信号后无杠杆净值 vs 基准 | 应用场景/增厚页 |
| `Uses_in_risk_parity_and_budget_model/C_output/figures/W008/04_month_weights_stacked_bar.png` | 月度资产权重堆叠图（含成长/价值切换） | 应用场景/增厚页 |
| `Uses_in_risk_parity_and_budget_model/C_output/figures/W008/07_levered_nav_month_vs_rw.png` | 含杠杆月频 vs 滚动窗口净值对比 | 应用场景/增厚页 |
| `Uses_in_risk_parity_and_budget_model/B_analysis/Asset performance/HS300 performance/01_NAV.png` | 沪深300净值曲线（资产标的说明） | 数据与标的页 |

---

## 8. 工作量体现

以下内容能证明做了大量系统性探索：

1. **因子规模**：自建因子库 473+ 个，分9大类别，每个因子均定义了 factor_id、方向、bar 阈值、频率、挂载规则；含 event 和 state 两类工程逻辑。
2. **IC全量分析**：479 个 IC 分析文件（xlsx + png），含分年度滚动IC，对全部因子系统筛查，并写了负IC专项检查（nega_checker）和得分体系（IC_score）。
3. **回测规模**：494 个 `*_summary.xlsx` 回测结果，5928 张净值图，覆盖从单因子到组合版本的全路径。
4. **迭代演进**：从 W003→W004→W005→W006→W007→W008→W009→W010，记录了组合优化的完整版本历程（含 W009 胜率下降的逆例）。
5. **工程管道**：构建了完整的 pipeline（run_factor_pipeline.py / G_engine），支持一键从信号到回测到IC分析。
6. **交叉验证**：同时做了 Fund_Analysis（RBSA）+ Uses_in_risk_parity（风险预算模型），形成"因子建设 → 信号回测 → 资产配置"的完整链条。
7. **文献研究**：研读并整理了广发、兴业、华创、招商、国金、国泰君安等多家券商风格轮动报告（含详细笔记）。

---

## 9. 创新点与可包装亮点

（不夸张，基于实际产出）

1. **系统性 binary 投票机制**：区别于直接连续信号合成，采用分类 binary 投票，单因子信号贡献可解释，组合对异常值更鲁棒。W005→W008 的演进本身就是一个可讲的设计过程。

2. **跨类别信号筛选 + 边际测试**：不是把所有因子无脑加入，而是先做相关性分析（F_grouping），再做边际加入效果评估（marginal_test），有组合设计的结构性考量。

3. **双向验证**：既用自建信号做前向回测（E_backtesting），也用 RBSA 对真实基金做外部验证（Fund_Analysis），两条路印证风格轮动的真实存在性。

4. **资产配置嵌入闭环**：风格轮动信号（W008）被直接用于权益子项的动态权重调整，完成了"研究 → 应用"的闭合，说明结果有实际价值而非纯回测。

5. **负 IC 专项管理机制**：为 200+ 因子做了系统性逆向检验（nega_checker），有明确的"问题因子清单"和"通过清单"，体现工程严谨性。

---

## 10. 当前缺口

| 缺口类型 | 具体说明 | 严重程度 |
|----------|----------|----------|
| **逻辑缺口** | W009 胜率明显低于 W008，但未有书面解释为何加入某些因子后表现下降 | ⚠️ 需主动说明 |
| **指标缺口** | W008 的"成长/价值波段分别胜率 > 50%"未达标（value_regime_win_rate 仅约 42%），这是内置的弱点 | ⚠️ 需准备答辩口径 |
| **对照缺口** | 没有明确的"等权合成因子"或"纯单因子"对照组与 W008 的量化对比 | 中 |
| **图缺口** | W005/W006/W007/W008 各版本的净值曲线汇总对比图（4条线在一张图）尚未生成 | 高（答辩视觉效果） |
| **结论缺口** | "哪类因子贡献了主要胜率"尚无明确的因子贡献分解报告（只有总分，没有贡献分解） | 中 |
| **时间段缺口** | 分市场周期（如2015牛市、2018下跌、2020-2021成长行情）的专项表现分析缺少 | 中 |
| **解释缺口** | Uses_in_risk_parity 中 W008 嵌入后的净值改善幅度（超额收益）尚未被明确提炼成一句话结论 | 高（这是最强应用落地证据） |
| **标的缺口** | 全文对"成长指数 = XXX 指数，价值指数 = XXX 指数"没有明确标注 | ⚠️ 必须补清楚 |

---

## 11. 建议补充内容（按优先级）

### P0（必须，答辩前完成）

1. **明确成长/价值指数标的**：确认用的是哪两个指数（如国证成长 vs 国证价值，或 CS成长 vs CS价值），补充到背景页。
2. **版本胜率对比图**：将 W003→W010 各版本的月胜率（55.21% → 55.31% → 54.06% → 55.42%...）画成折线图，体现组合迭代过程。
3. **W008 嵌入资产配置的净值增厚结论**：从 Uses_in_risk_parity 的 `07_levered_nav_month_vs_rw.png` 中提炼出具体数字（例如 "嵌入W008信号后，权益模块年化收益提升 X%"）。
4. **W008 value_regime 胜率低的解释**：准备一段话，说明为何价值波段胜率偏低（或该指标是否存在计算口径问题）。

### P1（建议完成，提升答辩质量）

5. **因子贡献分解**：列出 W008 中哪几类因子（L/G/V/C...）贡献最大，可以做简单的单类别 binary 版本对比。
6. **分市场状态分析**：选 2-3 个代表性市场环境（如2021年成长行情、2022年价值行情），展示 W008 的表现是否符合逻辑。
7. **W009 下降原因说明**：说明加入了哪些因子导致胜率下降，是类别重复还是信号互相抵消。

### P2（锦上添花）

8. **RBSA 分析更深入的呈现**：用 stable_fund_stats.csv 提炼"多少比例的主动基金存在显著的风格暴露"，支撑"风格轮动有现实意义"的论点。
9. **因子类别说明表**：一张 3 列表（类别 | 代表因子 | 经济逻辑），概括 9 类因子的逻辑基础。

---

## 附：材料矛盾 / 需要核查的问题

1. **IC_score 中 I001/I002/V001/W003 等因子出现多次重复行**（见得分≥3列表），推测是同一因子在不同信号版本中被重复登记，答辩前需确认是否属于重复计数，不要在 PPT 中把总量说高。
2. **exported_nav 目录下仅有 7 个文件**（csv/parquet/xlsx，均为 W008_signal_binary），与其他版本没有导出对应数据，需确认这是否是最终对外的标准格式。
3. **W00C_binary 的含义未明**：从文件名无法判断其因子组合逻辑，如果答辩被问到需提前补记录。
4. **Uses_in_risk_parity 中单指标单资产事件检验**（C_output/Results_20260223_235650）是大量 png，但没有汇总结论文件，与主线风格轮动的关系需理清（是预研还是已整合？）。
