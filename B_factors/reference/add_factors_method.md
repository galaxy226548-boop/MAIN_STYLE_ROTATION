# 新增因子实现 SOP

本文档沉淀本次新增 `flow_factors.py` 过程中确认的执行规则。后续从计划表、研报整理表或 JSON 元数据中批量新增因子时，应按以下 SOP 执行。

## 1. 开工前先读现有代码

1. 先阅读同目录已有脚本，确认是否有可复用逻辑，尤其是：
   - `B_factors/scripts/factor_utils.py`
   - 已有因子生成脚本，例如 `group_W004_factor_generator.py`、`initial_factors.py`
   - 当前主入口 `build_factor_matrix.py`
2. 优先复用现有数据读取、因子注册、挂载、信号生成、输出和登记工具，不重复造轮子。
3. 除非明确要求，不要把新因子模块接入 `build_factor_matrix.py`，避免改变现有主入口行为。

## 2. 判断哪些因子需要实现

1. 对计划表中的每条记录，先检查 `docu` 和 `data_field`。
2. `docu` 或 `data_field` 为 `unknown`、`todo`、空值时，默认跳过，不强行实现。
3. 跳过项应明确记录原因。例如本次 `D024` 的来源和字段均为 `unknown`，因此不实现。
4. 对剩余有明确来源的因子，应尽量全部实现；不能实现时必须说明具体数据或公式阻塞点。

## 3. 元数据口径处理

1. 如果 `paper_id` 为空值，写入登记前统一补为 `DIY`。
2. 输出文件前缀应与本批脚本一致。例如 `flow_factors.py` 对应：
   - `flow_factors_mounted_normalized_factors.parquet`
   - `flow_factors_mounted_normalized_factors.xlsx`
   - `flow_factors_signal_ls.parquet`
   - `flow_factors_signal_ls.xlsx`
3. `factor_generated.json` 必须同步更新，并保留已有其他 `_generated_output_prefix` 的记录。
4. 新增或覆盖登记时，只覆盖本批 `_generated_output_prefix` 对应记录。

## 4. 数据读取与替换规则

1. 优先使用 `A_data/prepared_data` 下的 prepared 数据源。
2. 宽表数据直接按列读取，例如 `macro_monthly.parquet`、`rate_daily.parquet`、`flow_daily.parquet`、`FactorC_daily.parquet`。
3. `macro.parquet` 这类长表应按国家、指标名称关键字和值字段读取，不要当作宽表直接按列取。
4. 如果计划中的字段在本地 prepared 数据中不存在，但存在明确、经确认的替代口径，可以替换；替换必须满足经济含义和单位可比。
5. 本次确认的替换规则：
   - `新成立基金份额:偏股混合型基金` 替换为 `新成立基金份额:混合型基金:偏股混合型基金`。
   - 需要美国 1 年期国债收益率时，整条利差中涉及的中美国债收益率都替换成 10 年期，保持期限和单位可比。
   - `中债企业债到期收益率(AAA):5年` 缺失时，替换为 `中债中短期票据到期收益率(AAA):5年`；与其计算的另一项若是国债 5 年期，可保留，以维持信用利差含义。
6. 所有替换后的真实口径必须写入 `factor_generated.json`：
   - `data_field` 写实际使用字段。
   - `notes` 或 `calc_method` 写明替换关系。
   - 保留原始字段口径，便于追溯。

## 5. 公式实现规则

1. 不要直接 `eval` 计划表中的 `condition` 字符串。
2. 对公式建立白名单实现，按 `factor_id` 或稳定公式模式显式编码。
3. 对明显笔误可以按意图修正，但应保持实现可读、可追溯。例如函数名拼写错误、变量名漏 `_1`、括号多写等。
4. 日频 rolling 计算前要注意非交易日空值；如果原始表包含非交易日空行，应先对参与 rolling 的序列或利差 `dropna()`，避免窗口永远凑不满。

## 6. 信号生成规则

1. `signal_rule` 中 `>0 => growth; <0 => value` 的含义是：
   - 因子值 `> 0` 时，`signal_ls = 1`
   - 因子值 `< 0` 时，`signal_ls = -1`
   - 因子值 `== 0` 时，`signal_ls = 0`
2. 不要把事件型因子预先改造成 `±1` 信号。
3. 事件型因子应先保留公式算出的事件因子值，再由统一的 `>0/<0/==0` 规则生成 `signal_ls`。
4. 事件型挂载仍沿用现有逻辑：仅挂载到事件日期之后每个 `track_id` 的第一个候选交易日；非事件日期保持 `NaN`。
5. `state` 因子仍沿用现有挂载逻辑：按可得性向后填充，并使用现有 `shift(1)` 风格的可用性处理。

## 7. 输出与登记验证

1. 实现后必须运行语法检查：
   ```bash
   .venv_mktp/bin/python -m py_compile B_factors/scripts/<new_script>.py
   ```
2. 必须运行生成脚本：
   ```bash
   .venv_mktp/bin/python B_factors/scripts/<new_script>.py
   ```
3. 验证输出文件存在，包括 parquet 和 xlsx。
4. 验证输出列名全部使用 `factor_id`，不能使用展示名、`code` 或中文因子名。
5. 验证每个已实现因子都有非空数量、首个有效日期、末个有效日期。
6. 验证 `factor_generated.json`：
   - 本批记录数量正确。
   - 空 `paper_id` 已补为 `DIY`。
   - 替换过的数据口径已写入实际 `data_field` 和说明。
   - 既有其他批次登记未被破坏。

## 8. 安全边界

1. 不删除、重命名或重写历史脚本。
2. 不批量清理输出目录。
3. 不刷新无关 `zhao_*`、`W004_*` 或其他历史输出。
4. 修改前后都检查 `git status --short`，识别并避开用户已有改动。
5. 若任务只是新增独立脚本，不要顺手重构公共工具，除非确有必要且保持向后兼容。
