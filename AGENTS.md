# AGENTS.md

## 项目概述
这是一个基于 Python 的风格轮动研究/回测项目。它构建因子值和多空信号，将信号转换为成长/价值目标位，运行固定回测，计算 IC 诊断，并可选择对因子/信号进行分组。

主要本地环境：
- Python：3.12.4，通过 `.venv_mktp/bin/python` 安装
- 本地虚拟环境中已发现的依赖项：`pandas`、`numpy`、`pyarrow`、`openpyxl`、`xlsxwriter`、`matplotlib`、`scipy`、`statsmodels`、`seaborn`、`tqdm`
- 仓库中没有 `requirements.txt`、`pyproject.toml` 或 lockfile 文件。依赖项的安装/复现尚待确认。

## 目录结构

- `A_data/`：原始数据、预处理数据、参考数据和输出数据。大型/本地数据区域；许多路径已被 .gitignore 忽略。
- `B_factors/`：因子生成。
- `scripts/`：因子脚本和共享因子工具。
- `reference/`：因子元数据和工程规则。
- `output/`：生成的因子矩阵和信号文件。
- `C_positions/`：将 `signal_ls` 矩阵转换为每个因子的目标位置文件；包含基准位置参考。
- `D_analysis/`：IC 分析、IC 评分、负 IC 检查和生成的分析输出。
- `E_backtesting/`：修复回测引擎和结果输出。
- `F_grouping/`：将因子/信号组合成分组策略。
- `G_engine/`：编排和迁移/检查工具，特别是 `run_factor_pipeline.py`。
- `SY_Baseline/`：用作参考和导入源的旧版基线脚本/配置/notebook。
- `SY_Reference/`：项目文档和数据/模块引用。

## Common Commands

Activate the existing local environment:

```bash
source .venv_mktp/bin/activate
```

Run the factor-to-position/backtest/IC pipeline:

```bash
python G_engine/run_factor_pipeline.py \
  --signal-path 'B_factors/output/W004_factor_generator_signal_ls.parquet' \
  --factor-path 'B_factors/output/W004_factor_generator_mounted_normalized_factors.parquet'
```

Generate the current factor matrix outputs:

```bash
python B_factors/scripts/build_factor_matrix.py
```

Generate per-factor position Excel files from a signal matrix:

```bash
python C_positions/scripts/generate_factor_positions.py \
  --signal-path B_factors/output/zhao_signal_ls.parquet \
  --output-dir C_positions/output/factor_positions
```

Run one backtest from a position file:

```bash
python E_backtesting/scripts/backtesting.py \
  --position-file C_positions/output/factor_positions/ZHAO01_position.xlsx \
  --output-root E_backtesting/Result
```

Run IC negative checker:

```bash
.venv_mktp/bin/python D_analysis/scripts/IC_nega_checker.py
```

Quality commands:

- Syntax check for touched Python files: `python -m py_compile <files>`
- Test command: 待确认. No pytest/unittest test suite or test config was found.
- Lint command: 待确认. No ruff/flake8/eslint config was found.
- Typecheck command: 待确认. No mypy/pyright config was found.
- Build command: 待确认. This repo appears script-driven rather than package-build-driven.

## 业务流程
1. `A_data` 准备可重用的数据表，特别是 `A_data/prepared_data` 目录下的文件。
2. `B_factors` 生成原始因子源帧，将其挂载/归一化到交易日期，并创建 `signal_ls` 矩阵。
3. `C_positions` 将 `signal_ls` 映射到增长/价值目标权重。
4. `E_backtesting` 对目标仓位文件运行稳定的回测框架。
5. `D_analysis` 对已挂载的归一化因子运行 IC 分析和诊断。
6. `F_grouping` 将多个单因子或信号组合成复合策略。
7. `G_engine/run_factor_pipeline.py` 协调因子/信号文件对的仓位生成、回测和 IC 分析。

## 代码风格和命名
- Python 脚本使用“from __future__ import 注解”、“pathlib.Path”和“argparse”作为 CLI 入口点。
- 常量通常是大写的，例如`PROJECT_ROOT`、`DEFAULT_SIGNAL_PATH`、`OUTPUT_DIR`。
- DataFrame 变量通常以 `_df` 结尾；系列通常以“_series”结尾。
- 因子输出约定：
 - 安装标准化因子：`*_mounted_normalized_factors.parquet`加上匹配的`.xlsx`
 - 信号：`*_signal_ls.parquet`加上匹配的`.xlsx`
 - 每个因素的位置：`{factor_id}_position.xlsx`
- 因子矩阵列应使用“factor_id”，而不是显示名称或“代码”。
- 更喜欢将新的研究特定模块添加为英文“B_factors/scripts/paper_*.py”文件，然后通过“B_factors/scripts/build_factor_matrix.py”连接它们。
- 保持“B_factors/scripts/factor_utils.py”中的常见更改向后兼容。
- 要加中文的注释，每个函数都最好加一下注释

## Factor Rules To Preserve

The strongest project-specific rules are documented in `B_factors/reference/factor_generation_rules.md`.

- Do not delete, rename, or rewrite `B_factors/scripts/因子矩阵生成与批量挂载.py`.
- Do not refresh unrelated `zhao_*` outputs when implementing a new paper/factor.
- Do not use old raw paths as the primary source if a prepared source exists; prefer `A_data/prepared_data`.
- Before using a data source, check `A_data/reference/data_inventory_A.json` when available, then map to prepared data.
- Do not silently `ffill`, expand holding states, or fill signal frequency unless the user explicitly confirms it.
- `state` factors should follow the existing mount `shift(1)` availability logic.
- `event` factors should only mount to the first candidate date after the event date per `track_id`; non-event dates remain `NaN`.
- For event signals, `NaN` means no event; `0` only means an explicitly neutral triggered event.
- If formula, data source, factor window, `factor_id`, `bar`, or direction is uncertain, report it as uncertain instead of fabricating a reproduced factor.

## Generated / Sensitive Areas

Gitignored or generated/local areas include:

- `.venv_mktp/`, `__pycache__/`, `.ipynb_checkpoints/`
- `A_data/`
- `B_factors/input/`, `B_factors/output/`
- `C_positions/output/`
- `D_analysis/IC_output/`
- `E_backtesting/Result/`
- `F_grouping/input_COMB/`, `F_grouping/output_COMB/`
- generic `input/`, `output/`, `input_COMB/`, `output_COMB/` folders

Treat these as data/output areas. Do not delete, overwrite, or bulk-regenerate them unless the user explicitly asks for that exact operation.

## Safe Work Habits

- Do not modify business code when the task is documentation-only.
- Before editing files, check `git status --short`; this repo may contain user edits.
- Avoid destructive commands such as `rm`, `git reset --hard`, `git checkout --`, or bulk output cleanup unless explicitly requested.
- When changing factor logic, validate with the relevant script plus `python -m py_compile` on touched Python files.
- For pipeline runs that produce many files, name output directories/files narrowly so unrelated historical outputs are preserved.


## Additional Agent Notes

- `build_factor_matrix.py` 是“运行当前已接入因子模块”的主入口。

- `initial_factors.py` 当前表示：从 `factor_done.json` 中筛出的 `category == null`、且可从 `SY_Baseline/factor_done.py` 稳定复现的因子；其中已由 W004 覆盖的重复因子已经移除，不应再用新编号重复生成。

- 已确认由 W004 覆盖、应排除在 `initial_factors` 外的重复因子：
  `D001/L23`、`D002/L005`、`F001/L07`、`G003/L02`、`G004/L52`、`L003/L28`、`L004/L35_2`、`O002/L89`、`V003/ZHAO02`、`V004/V79`。

- 遇到 `factor_id` 冲突时，先判断是不是已有模块已经覆盖的同一个因子。不要默认改大编号；如果业务上是重复因子，优先从新增模块中移除。

- `factor_done.json` 是生成和登记流程依赖的 metadata。修改其中的 `factor_id` 可能影响已有模块的 metadata 查找；改号前必须检查现有输出模块是否仍依赖旧编号。

- `factor_generated.json` 是成功生成因子的登记文件。单独生成某个模块时，要使用该模块自己的 `_generated_output_prefix`；如果因子从模块中移除，也要避免留下对应的陈旧登记记录。

- `initial_factors` 可以单独运行现有工具链生成输出，不需要接回 `build_factor_matrix.py`。单独输出前缀使用 `initial_factors`。

- 本项目中“生成成功”至少应检查：输出列使用 `factor_id`；parquet/xlsx 文件存在；`factor_generated.json` 已更新；每个因子有非空数量、首个有效日期、末个有效日期；必要时确认不包含已排除的 W004 重复编号。

- 对生成输出要谨慎：近期任务有意生成/清理过 `B_factors/output/initial_factors_*`，并更新过 `factor_generated.json`。不要批量清理输出目录。

- 如果恢复或调整 `build_factor_matrix.py`，应保留当前 W004 行为，除非用户明确要求修改 W004。W004 应继续使用既有 `factor_id`。

- 文档类任务默认只给建议内容；只有用户明确要求保存时，才写入本地文件。

- 用中文写注释

----快捷指令包----
# gfgraph
當我在 Codex prompt 中提到 gfgraph 時，請自動套用以下繪圖規範：

畫圖的參考配色：
RGB(68,114,169), RGB(79,129,189), RGB(189,205,229),
RGB(232,56,13), RGB(247,150,70), RGB(253,190,148),
RGB(127,127,127), RGB(166,166,166), RGB(217,217,217)。

選擇顏色時，不要機械按順序使用，而要根據圖表目標選擇：
- 若是成長 vs 價值、策略 vs 基準、兩類對比，優先使用藍色系 vs 橙色系；
- 若是不同參數、不同窗口、不同閾值、不同風險預算比例，優先使用同一色系深淺變化；
- 若是突出核心結論，核心線條用主藍或主橙紅，輔助線用灰色或淺色；
- 零線、均值線、基準線、參考區間優先使用灰色系；
- 如需補充顏色，需保持商務、克制、適合實習答辯 PPT，避免螢光色、彩虹色和過度飽和色。

圖表需要適合直接放入 PPT：標題、坐標軸、圖例清晰；時間序列圖日期刻度合理；圖例不得遮擋數據；優先保證可讀性和結論表達。