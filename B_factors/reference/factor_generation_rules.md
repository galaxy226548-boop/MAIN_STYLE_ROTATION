# Factor Generation Engineering Rules

本文档记录新增研报因子复现时的工程边界和验证规则。目标是让后续实现像 harness 一样可控：只改允许改的面，保持旧产物和旧脚本稳定，所有行为都有可验证输出。

## 不可修改范围

- 严禁删除、重命名或改写 `B_factors/scripts/因子矩阵生成与批量挂载.py`。
- 严禁为了让新因子跑通而回退、覆盖或清空既有数据文件、旧输出、参考 JSON 的无关记录。
- 严禁在没有明确指令时对因子值做 `ffill`、持有态展开或频率补齐。
- 严禁把不确定公式、缺数据源或未确认窗口的因子伪造成已复现因子。
- 严禁依赖旧 raw 路径作为主数据源；应优先使用 `A_data/prepared_data` 下的 prepared parquet/xlsx。

## 可修改范围

- 可以新增按研报拆分的英文脚本模块，例如 `B_factors/scripts/paper_*.py`。
- 可以在 `B_factors/scripts/build_factor_matrix.py` 中接入本次明确要求的研报模块和输出流程。
- 可以在 `B_factors/scripts/factor_utils.py` 中新增通用、向后兼容的工具函数，但不得破坏旧调用默认行为。
- 可以在 `B_factors/reference/record_all.json` 中补充本次研报相关的 `factor_id`、`signal_type`、prepared 数据源说明；不得批量改动无关 paper。
- 可以新增检查用输出文件，但输出命名必须按用户确认的约定生成。

## 因子列名与 Metadata

- 生成矩阵列名必须使用 `factor_id`，不得使用 `code`。
- `record_all.json` 中同一 `paper_id + factor_id` 必须能唯一匹配 metadata。
- 若 sheet2 派生因子缺 `factor_id`，必须先与用户确认命名策略，再写回 JSON 并生成。
- 若 `bar` 缺失，只有在用户允许默认值时才可使用默认，并应生成 missing bar 说明文件。

## State 与 Event 规则

- `state` 因子：挂载时只允许使用既有框架的 `shift(1)` 逻辑，表示下一期可用；不得自动 `ffill`。
- `event` 因子：应对齐 `SY_Baseline/风格轮动信号检验法 copy_副本.py` 中 `merge_factor_to_market(..., factor_type="event")` 的逻辑。
- event 挂载含义：
  - 原始事件发生在 `data_df` 某日期。
  - 对每条 `track_id`，只挂到事件日期之后的第一个候选日期。
  - 无事件日期保持 `NaN`。
  - event 的 `signal_ls` 无事件时必须保持 `NaN`，不能写成 `0`。
  - `0` 只能表示明确触发了中性事件。
- 月频宏观信号若被定义为 event，不应展开成完整日频持有态。

## 数据源规则

- 数据源必须先查 `A_data/reference/data_inventory_A.json`，再映射到 `A_data/prepared_data`。
- prepared 文件优先级：parquet/xlsx 优先；不新增外部依赖。
- 若 prepared 文件或字段不明确，必须报告缺失文件/字段和 inventory 查找结果，并向用户确认。
- 成功确认的新数据映射，应尽量补回 `record_all.json` 的本研报记录。

## 输出规则

- 本次研报任务只输出用户指定的新研报文件，不刷新无关 `zhao_*` 输出。
- parquet 与 xlsx 成对输出，xlsx 内容应与 parquet 等价，仅作检查用途。
- 没有 missing bar 时，不生成 `*_missing_bar_defaults.md`。
- 输出后必须打印或报告每个新增因子的非空数量、首个有效日期、末个有效日期。

## 验证门槛

- 必须运行主入口，例如 `python B_factors/scripts/build_factor_matrix.py`。
- 必须验证输出列完整包含目标 `factor_id`。
- 必须验证 event 因子没有被无指令展开或 forward fill。
- 必须运行语法检查，例如 `python -m py_compile` 覆盖新增/修改脚本。

## 变更纪律

- 修改 JSON 时必须精准定位到本次 `paper_id`，避免宽匹配污染其他 paper。
- 修改公共工具时必须保持默认行为向后兼容。
- 对任何自动推断、频率转换、方向翻转和阈值转换，都要能从研报规则或用户确认中追溯。
- 如果发现前一步实现与用户约束冲突，应立即撤回该部分改动并重新验证输出。
