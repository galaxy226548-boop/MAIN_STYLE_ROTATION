# AGENTS.md

## Project Overview

This is a Python style-rotation research/backtesting project. It builds factor values and long/short signals, converts signals to growth/value target positions, runs fixed backtests, calculates IC diagnostics, and optionally groups factors/signals.

Primary local environment:

- Python: 3.12.4 via `.venv_mktp/bin/python`
- Observed dependencies in the local venv: `pandas`, `numpy`, `pyarrow`, `openpyxl`, `xlsxwriter`, `matplotlib`, `scipy`, `statsmodels`, `seaborn`, `tqdm`
- There is no `requirements.txt`, `pyproject.toml`, or lockfile in the repo. Dependency installation/reproduction is 待确认.

## Directory Map

- `A_data/`: raw, prepared, reference, and output data. Large/local data area; many paths are gitignored.
- `B_factors/`: factor generation.
  - `scripts/`: factor scripts and shared factor utilities.
  - `reference/`: factor metadata and engineering rules.
  - `output/`: generated factor matrices and signal files.
- `C_positions/`: converts `signal_ls` matrices into per-factor target-position files; contains benchmark position references.
- `D_analysis/`: IC analysis, IC scoring, negative-IC checks, and generated analysis outputs.
- `E_backtesting/`: fixed backtest engine and result outputs.
- `F_grouping/`: combines factors/signals into grouped strategies.
- `G_engine/`: orchestration and migration/checking utilities, especially `run_factor_pipeline.py`.
- `SY_Baseline/`: legacy baseline scripts/configs/notebooks used as references and import sources.
- `SY_Reference/`: project docs and data/module references.

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

## Business Flow

1. `A_data` prepares reusable market/data tables, especially files under `A_data/prepared_data`.
2. `B_factors` generates raw factor source frames, mounts/normalizes them to market dates, and creates `signal_ls` matrices.
3. `C_positions` maps `signal_ls` to growth/value target weights.
4. `E_backtesting` runs the stable backtest framework on target-position files.
5. `D_analysis` runs IC analysis and diagnostics for mounted normalized factors.
6. `F_grouping` groups multiple single factors or signals into composite strategies.
7. `G_engine/run_factor_pipeline.py` orchestrates position generation, backtesting, and IC analysis for a factor/signal file pair.

## Code Style And Naming

- Python scripts use `from __future__ import annotations`, `pathlib.Path`, and `argparse` for CLI entry points.
- Constants are usually uppercase, e.g. `PROJECT_ROOT`, `DEFAULT_SIGNAL_PATH`, `OUTPUT_DIR`.
- DataFrame variables usually end in `_df`; series often end in `_series`.
- Factor output conventions:
  - mounted normalized factors: `*_mounted_normalized_factors.parquet` plus matching `.xlsx`
  - signals: `*_signal_ls.parquet` plus matching `.xlsx`
  - per-factor positions: `{factor_id}_position.xlsx`
- Factor matrix columns should use `factor_id`, not display names or `code`.
- Prefer adding new research-specific modules as English `B_factors/scripts/paper_*.py` files, then wiring them through `B_factors/scripts/build_factor_matrix.py`.
- Keep common changes in `B_factors/scripts/factor_utils.py` backward-compatible.

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