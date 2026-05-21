# Financial Statement Factor Building Rules

本文档用于记录盈利、财报、成分股、行业映射类因子的构建规则。若新增因子的 `factor_id` 以 `P` 开头，且涉及利润、ROE、财报科目、成分股权重或行业映射，应先阅读本文档，再开始实现。

## 指数层面与成分股重建

盈利类因子通常有两种可行口径：

- 直接使用 `IndexStatement.xlsx` 等指数层面财报汇总数据。
- 使用成分股权重和个股财报数据重建指数层面指标。

如果指数层面数据存在大量 `0`、缺失或明显污染，不应静默使用。应切换到成分股重建，或向用户报告不可复现原因。发生 fallback 时，应在 plan JSON 或 generated records 的 `notes` 中说明 fallback 数据源和计算口径。

## 日期口径

财报类因子必须显式区分以下日期：

- 报告期：通常为 `Date` 或 `Accper`。
- 可得日：通常为 `PubDate`。
- 成分股权重日期：通常为 `TRADE_DT`。

涉及财报发布的事件型因子，应优先使用 `PubDate` 作为事件日。报告期只表示财报所属期间，不应直接当作可交易可得日期。使用成分股权重时，应明确采用「报告期当日或之前最近一期权重」等规则。

### 清洗推导的 PubDate

原始财报 Excel 不一定包含正式财报发布日期。例如原始 `IS_AexcluST1F.xlsx` 只有 `Accper` 和 `DeclareDate`，其中 `DeclareDate` 是差错更正披露日期，不是正式财报发布日期，不能作为 event 因子的可得日。

清洗后的 prepared parquet 可以包含清洗流程新增的 `PubDate`。例如 `Ashare_profit.parquet` 中的 `PubDate` 是根据 `Accper` 推导出的可得日，而不是原始 Excel 自带字段。因子脚本应优先使用 prepared parquet 中已经清洗好的 `PubDate`，不要在因子脚本中重复实现日期推导，也不要用 `DeclareDate` 替代正式发布日期。

若某个原始财报值表缺少 `PubDate`，应从对应 prepared 表按 `stock_code + Accper` 合并清洗后的 `PubDate`。若合并后缺失比例异常，应报告数据清洗或对齐问题，而不是在因子脚本中自行推导。

## 成分股权重

涉及指数成分股的因子，必须确认以下字段：

- 指数代码字段，例如 `S_INFO_WINDCODE`。
- 成分股代码字段，例如 `S_CON_WINDCODE`。
- 权重字段，例如 `I_WEIGHT`。
- 权重日期字段，例如 `TRADE_DT`。

若研报或用户要求指数成分股口径，默认应使用成分股权重加权计算，而不是中位数或简单平均。只有在明确要求时，才使用中位数、等权或其他聚合方式。

## 行业占比加权

行业占比加权类盈利因子通常需要三层数据：

- 个股财报指标，例如 `IS_AexcluST1F.xlsx` 的 `B002000101`。
- 个股行业映射，例如 `Ashare_profit.parquet` 中的 `Stkcd -> Indcd`。
- 风格指数成分股权重，例如 `AIndexHS300FreeWeight.parquet`。

推荐流程：

1. 在股票层面计算盈利变化，例如 TTM、TTM 同比、TTM 同比环比。
2. 根据 `Indcd` 聚合为行业景气度。
3. 根据风格指数成分股及权重计算各行业在指数中的占比。
4. 用行业占比对行业景气度加权，得到风格指数景气度。
5. 按研报或用户确认的公式计算最终因子值，例如 `growth_score - value_score`。

`Stkcd`、`S_CON_WINDCODE`、`S_INFO_WINDCODE` 等代码字段应先标准化后再合并。六位股票代码、带交易所后缀代码、整数型代码不能直接混用。

## 数据质量与 Fallback

实现前应检查关键序列的数据质量，尤其是：

- 缺失值比例。
- 异常 `0` 值比例。
- 首末有效日期。
- 是否能与成分股权重、行业映射按代码和日期合并。

若指数层面数据污染严重，可以使用成分股重建作为 fallback。fallback 需要满足：

- fallback 数据源已在本地确认可读。
- fallback 计算口径与原因子含义一致。
- fallback 行为写入 `notes`，避免后续误以为完全使用了原始指数层面字段。

## Factor ID 与登记

若 plan 中的新 `factor_id` 和既有 `factor_done.json` 或 `factor_generated.json` 冲突，不得直接生成。应先判断：

- 是否是同一因子的重复覆盖。
- 是否是不同因子误用同一编号。
- 是否需要给新增因子改号。

改号后，脚本列名、plan JSON、输出文件、`factor_generated.json` 登记记录必须全部一致。

## 验证要求

完成后至少检查：

- 输出列名使用 `factor_id`。
- parquet/xlsx 输出存在。
- 每个因子有非空数量、首个有效日期、末个有效日期。
- `factor_generated.json` 中 `_generated_output_prefix`、`factor_id`、`signal_type` 与本次脚本一致。
- 若有 fallback，日志或 notes 能说明 fallback 原因。

## 常见文件与使用环节

以下文件是在盈利/财报类因子中常见、且已经在 `profitFactors.py` 中验证过的本地数据源。

| 文件 | 常用字段 | 使用环节 |
| --- | --- | --- |
| `B_factors/reference/working_multiple_factors_plan.json` | `factor_id`, `signal_type`, `condition`, `docu`, `data_field`, `notes` | 作为脚本选择因子、构造 metadata、写入 `factor_generated.json` 的来源。实现前应同步修正口径说明。 |
| `A_data/prepared_data/IndexStatement.xlsx` | `Indexcd`, `Date`, `PubDate`, `ROE`, `归母净利润`, `营业收入`, `归属母公司股东的权益` | 指数层面财报汇总。适合直接计算指数 ROE、净利润 TTM、同比增速差；使用前必须检查缺失与异常 0 值。 |
| `A_data/prepared_data/AIndexHS300FreeWeight.parquet` | `S_INFO_WINDCODE`, `S_CON_WINDCODE`, `TRADE_DT`, `I_WEIGHT` | 指数成分股和权重。用于按权重重建指数层面指标、计算行业占比、计算一致预期或 ROE 的加权均值。 |
| `A_data/prepared_data/Con_np_yoy_roll.feather` | `TRADE_DT`, `S_INFO_WINDCODE`, `Con_np_yoy_roll` | 个股一致预期净利润增速。通常与指数成分股权重按 `TRADE_DT + stock_code` 合并后加权。 |
| `A_data/prepared_data/Ashare_profit.parquet` | `Stkcd`, `Accper`, `Typrep`, `Indcd`, `PubDate`, `F050102B` | 个股盈利能力与行业映射。`Stkcd` 是六位股票代码，`Indcd` 是行业编号，`F050102B` 可用于成分股加权 ROE fallback；`PubDate` 是清洗阶段根据 `Accper` 推导出的可得日。 |
| `A_data/data/update/IS_AexcluST1F.xlsx` | `Stkcd`, `Accper`, `Typrep`, `DeclareDate`, `B002000101` | 原始个股归母净利润来源。`B002000101` 可用于股票层面 TTM、TTM 同比、TTM 同比环比；原始文件不含正式 `PubDate`，`DeclareDate` 是差错更正披露日期，不得当作财报正式发布日期。 |
| `A_data/prepared_data/macro_monthly.parquet` | 例如 `中国:利润总额:规模以上工业企业:累计同比` | 宏观盈利序列。可作为单序列因子来源；若被定义为 event，应按事件规则挂载，不自动展开持有态。 |
| `B_factors/output/factor_generated.json` | `_generated_output_prefix`, `_generated_at`, `factor_id`, `signal_type` | 生成登记文件。脚本运行后应检查新增记录是否使用正确输出前缀和 `factor_id`，避免旧编号残留。 |

使用大表时优先只读必要列，并尽量按目标指数或目标字段过滤。不要为了确认结构全量扫描成分股权重表。

## 常見指數和對應的指數代碼
沪深300：000300 / 399300 / 000300.SH / 399300.SZ
中證A100：000903
中证500：000905 / 399905 / 000905.SH / 399905.SZ
中證1000:000852
中證2000：932000
中证全指：000985 / 399985 / 000985.SH / 399985.SZ
创业板指：399006 / 399006.SZ
国证成长：399370 / 399370.SZ / 399370.XSHE
国证价值：399371 / 399371.SZ / 399371.XSHE
Wind全A：881001 / 881001.WI
上證50：000016
巨潮大盤指數：399314
巨潮小盤指數：399316
申萬大盤指數：801811
申萬小盤指數：801813


## 可复用函数接口

以下函数来自 `B_factors/scripts/profitFactors.py`，后续若多次复用，可考虑抽到 `factor_utils.py` 或新建 `statement_factor_utils.py`。函数名可调整，但输入/输出约定建议保留。

### 代码标准化

`_index_code_key(value: object) -> str`

- 输入：指数代码，可以是 `399370.SZ`、`399370`、`000300.SH` 等。
- 输出：六位指数代码字符串，例如 `399370`、`000300`。
- 用途：统一 `IndexStatement.xlsx`、`AIndexHS300FreeWeight.parquet` 中不同格式的指数代码。

`_stock_code_key(value: object) -> str`

- 输入：股票代码，可以是 `000001.SZ`、`1`、`000001` 等。
- 输出：六位股票代码字符串，例如 `000001`。
- 用途：统一 `Stkcd`、`S_CON_WINDCODE`、`S_INFO_WINDCODE` 后再合并。

### 指数财报读取

`_load_index_statement(columns: list[str]) -> pd.DataFrame`

- 输入：需要从 `IndexStatement.xlsx` 读取的财报字段列表，例如 `["ROE", "归母净利润"]`。
- 输出：包含 `Indexcd`、`Date`、`PubDate`、所需字段、标准化 `index_code` 的 DataFrame。
- 用途：统一读取指数层面财报，并保证报告期和可得日都存在。

`_index_statement_report_frame(index_code: str, value_col: str) -> pd.DataFrame`

- 输入：指数代码和一个财报字段，例如 `("000300.SH", "ROE")`。
- 输出：按报告期排序的 DataFrame，列为 `Date`、`PubDate`、`value_col`。
- 用途：计算单个指数的 ROE、归母净利润 TTM、同比变化等。

### 财报时间序列转换

`_quarterly_ttm(cumulative: pd.Series) -> pd.Series`

- 输入：以报告期为 index 的季度累计值序列。
- 输出：TTM 序列。
- 用途：将累计口径财报科目转为滚动四季度口径。

`_ttm_yoy(cumulative: pd.Series) -> pd.Series`

- 输入：以报告期为 index 的季度累计值序列。
- 输出：TTM 同比增速序列。
- 用途：计算净利润、收入等 TTM 同比。

`_calc_stock_ttm_yoy(profit_df: pd.DataFrame) -> pd.DataFrame`

- 输入：股票层面净利润表，至少包含 `stock_code`、`Accper`、`PubDate`、`net_profit`。
- 输出：股票层面的 TTM 同比表，列为 `stock_code`、`Accper`、`PubDate`、`value`。
- 用途：成分股加权重建指数净利润同比。

`_calc_stock_ttm_yoy_change(profit_df: pd.DataFrame) -> pd.DataFrame`

- 输入：股票层面净利润表，至少包含 `stock_code`、`Accper`、`PubDate`、`net_profit`。
- 输出：股票层面的 TTM 同比环比变化表，列为 `stock_code`、`Accper`、`PubDate`、`value`。
- 用途：行业景气度类因子，例如先算股票盈利改善，再按行业聚合。

### 成分股权重与加权

`_read_index_weights(target_index_codes: list[str] | tuple[str, ...]) -> pd.DataFrame`

- 输入：目标指数代码列表，例如 `["000905.SH", "000300.SH"]`。
- 输出：目标指数的成分股权重 DataFrame，包含 `index_code`、`stock_code`、`TRADE_DT`、`I_WEIGHT`。
- 用途：读取并标准化 `AIndexHS300FreeWeight.parquet`，优先只取目标指数。

`_weighted_average(group: pd.DataFrame, value_col: str, weight_col: str = "I_WEIGHT") -> float`

- 输入：包含数值列和权重列的 DataFrame 分组。
- 输出：一个加权均值浮点数；无有效权重时返回 `NaN`。
- 用途：成分股加权、行业权重加权都可复用。

`_weighted_stock_metric_by_index(metric_df: pd.DataFrame, target_index_code: str) -> pd.DataFrame`

- 输入：股票层面指标表，至少包含 `stock_code`、`Accper`、`PubDate`、`value`；以及目标指数代码。
- 输出：指数层面加权指标表，列为 `Accper`、`PubDate`、`value`。
- 用途：用成分股权重重建指数 ROE、净利润同比、一致预期净利润增速等。

### 个股财报与行业映射

`_load_ashare_profit_metric(value_col: str) -> pd.DataFrame`

- 输入：`Ashare_profit.parquet` 中的指标字段名，例如 `F050102B`。
- 输出：股票层面指标表，列为 `stock_code`、`Accper`、`PubDate`、`value`。
- 用途：读取个股 ROE 等盈利能力指标，用于成分股加权 fallback。

`_load_stock_industry_map() -> pd.DataFrame`

- 输入：无显式参数，读取 `Ashare_profit.parquet`。
- 输出：股票行业映射表，列为 `stock_code`、`Accper`、`Indcd`。
- 用途：将股票层面盈利变化映射到行业，再计算行业景气度。

`_load_parent_net_profit() -> pd.DataFrame`

- 输入：无显式参数，读取原始 `IS_AexcluST1F.xlsx` 的归母净利润 `B002000101`，并从 prepared `Ashare_profit.parquet` 按 `stock_code + Accper` 合并清洗后的 `PubDate`。
- 输出：股票层面归母净利润表，列为 `stock_code`、`Accper`、`PubDate`、`net_profit`。
- 用途：计算股票 TTM、TTM 同比、TTM 同比环比。读取大 Excel 时建议在脚本内缓存，避免同一次运行重复读取；若 `PubDate` 合并缺失比例异常，应报错或报告清洗对齐问题。

### 模块集成

`generate_profitFactors_factors(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]`

- 输入：项目默认 `data_df`，主要用于提供输出初始 index。
- 输出：`factor_source_df` 和从 plan JSON 选出的 records。
- 用途：供单脚本 main 和 `build_factor_matrix.py` 接入。其他财报因子模块可仿照这个接口。

`metadata_from_profitFactors_records(records: list[dict[str, object]]) -> dict[str, dict[str, object]]`

- 输入：从 plan JSON 选出的因子 records。
- 输出：以 `factor_id` 为 key 的 metadata，包含 `signal_type`、`bar`、`factor` 等。
- 用途：传给 `mount_factor_source_frame` 和 `build_threshold_signal_ls_df`。
