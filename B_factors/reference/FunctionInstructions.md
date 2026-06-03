# overseaFactors 因子值生成函数说明

本文档只解释 `B_factors/scripts/overseaFactors.py` 中 `_calc_oversea_factor(factor_id: str) -> pd.Series` 下面会用到的函数和常见操作。阅读顺序按“生成一个原始因子值”时通常发生的流程排列：读取数据、整理时间序列、计算中间指标、把指标转成因子值。

## 0. 总体理解

`_calc_oversea_factor()` 的职责是：给定一个 `factor_id`，返回这个因子的原始时间序列 `pd.Series`。

它不负责以下事情：

- 不负责把因子对齐到最终市场交易日历；
- 不负责因子标准化；
- 不负责生成最终 `signal_ls`；
- 不负责保存 parquet/xlsx。

这些后续步骤在 `generate_overseaFactors_factors()`、`mount_factor_source_frame()`、`build_threshold_signal_ls_df()` 和 `save_factor_outputs()` 中完成。

所以你在 `_calc_oversea_factor()` 里新增代码时，重点是保证返回一个：

- index 是日期；
- value 是数值；
- 缺失值用 `NaN`；
- 方向和阈值口径已经在原始因子层面处理好的 `pd.Series`。

## 1. 数据读取环节

### `read_prepared_series(table_name, column_name)`

来源：`factor_utils.py`

用途：从 `A_data/prepared_data` 里的 prepared 表中读取一个字段，直接返回一条按日期索引的数值序列。

主要功能：

- 内部先调用 `load_prepared_table(table_name)` 读取整张表；
- 自动寻找日期索引或常见日期列：`date`、`日期`、`TRADE_DT`、`Trddt`；
- 把目标列转成数值；
- 日期排序；
- 同一天重复记录保留最后一条。

适合场景：

- prepared 表已经是标准日频/月频表；
- 你只需要其中一个字段；
- 表里有可识别日期列或已经用日期作为 index。

在本文件中的例子：

- `read_prepared_series(EXCHANGE_TABLE, HKD_SPOT_COL)`
- `read_prepared_series(OVERSEA_DAILY_TABLE, SOX_COL)`
- `read_prepared_series(OVERSEA_MONTHLY_TABLE, KOREA_SEMI_EXPORT_COL)`

### `_load_price_file_close(file_name, name)`

来源：`overseaFactors.py`

用途：专门读取“行情统计”类文件里的收盘价。

主要功能：

- 调用 `load_prepared_table(file_name)` 读取整张表；
- 要求表里必须有 `交易日期` 和 `收盘价` 两列；
- 使用 `_clean_date_series()` 清理交易日期；
- 使用 `_as_float_series()` 生成数值时间序列；
- 使用 `_positive_series()` 过滤非正价格。

适合场景：

- 文件是类似 `VIX.GI-行情统计-20260509.xlsx`、`SPX.GI-行情统计-20260519.xlsx` 这种行情统计文件；
- 你明确要用 `收盘价`；
- 日期列名是 `交易日期`，价格列名是 `收盘价`。

在本文件中的例子：

- O011：读取 VIX 收盘价；
- O017：读取 NDX 和 SPX 收盘价；
- O019：读取 SPX 收盘价。

### `load_prepared_table(table_name)`

来源：`factor_utils.py`

用途：读取 prepared 表的底层通用函数，返回 `pd.DataFrame`。

主要功能：

- 从 `A_data/prepared_data` 查找文件；
- 支持 `.parquet`、`.xlsx`、`.xls`；
- 带缓存，同一张表重复读取时不会每次都重新从磁盘加载；
- 返回时会 `.copy()`，避免调用方误改缓存里的原始表。

在 `_calc_oversea_factor()` 中通常不直接用它，而是通过：

- `read_prepared_series()` 读取普通字段；
- `_load_price_file_close()` 读取行情统计收盘价。

## 2. 日期和数值整理环节

### `_clean_date_series(series)`

来源：`overseaFactors.py`

用途：把日期列转成标准日期。

主要功能：

- `pd.to_datetime(..., errors="coerce")`：无法识别的日期变成 `NaT`；
- `.dt.normalize()`：把日期时间统一到当天 00:00:00。

通常不在 `_calc_oversea_factor()` 里直接调用，而是在 `_load_price_file_close()` 里面配套使用。

### `_as_float_series(series, index, name)`

来源：`overseaFactors.py`

用途：把一列数值和一列日期组装成干净的 `float64` 时间序列。

主要功能：

- 把数值列转成 numeric，无法转换的变成 `NaN`；
- 用传入的日期作为 index；
- 删除日期缺失的行；
- 按日期排序；
- 同一天重复记录保留最后一条；
- 输出类型为 `float64`。

通常不在 `_calc_oversea_factor()` 里直接调用，而是在 `_load_price_file_close()` 里面配套使用。

### `_positive_series(series)`

来源：`overseaFactors.py`

用途：保留正数，非正数改成 `NaN`。

适合场景：

- 价格；
- 汇率；
- 指数点位；
- 其他理论上必须大于 0 的序列。

在本文件中的例子：

- O012：港币即期汇率；
- O014：USDCNH；
- O016：港币即期汇率；
- `_load_price_file_close()` 内部也会调用它清理收盘价。

### `_clean_positive_limited_ffill_series(series, limit=5)`

来源：`overseaFactors.py`

用途：先保留正数，再做短期限前值填充。

主要功能：

- 非正数改成 `NaN`；
- 按日期升序排序；
- 最多向前填充 `limit` 天，默认 5 天。

适合场景：

- 海外日频市场数据偶尔有少量缺口；
- 你希望补短缺口，但不希望把长期停更的数据一直延续下去。

在本文件中的例子：

- O018-O024：SOX 费城半导体指数。

与 `_positive_series()` 的区别：

- `_positive_series()` 只清理非正数，不填充缺失；
- `_clean_positive_limited_ffill_series()` 会额外短期 `ffill`，更适合指数数据少量缺口场景。

### `_complete_month_end_values(series)`

来源：`overseaFactors.py`

用途：从一个序列中取每个完整月份的月末值，并剔除尚未结束的当前月。

主要功能：

- 删除缺失值；
- 按月分组，每个月只保留最后一条；
- 如果最后一条属于当前未结束月份，则删除这个月的数据。

适合场景：

- 月末事件因子；
- 不想把当前月尚未完整的数据当成正式月末值。

在本文件中的例子：

- O013：美元兑港元 1M 掉期点月末事件信号。

## 3. 基础收益率和差值计算

这些不是封装函数，而是 `_calc_oversea_factor()` 中常见的 pandas 写法。

### `series / series.shift(n) - 1`

用途：计算 n 期收益率或变化率。

在本文件中的例子：

- O011：`vix_close / vix_close.shift(20) - 1`，VIX 20 日收益率；
- O014：`usdcnh / usdcnh.shift(20) - 1`，USDCNH 20 日变化；
- O018：`sox / sox.shift(20) - 1`，SOX 20 日收益率；
- O025：`exports / exports.shift(12) - 1`，韩国半导体出口同比。

注意：

- 日频 `shift(20)` 通常近似 20 个交易日；
- 月频 `shift(12)` 通常表示同比；
- 该操作不会自动检查频率，需要你自己确认数据频率。

### `a - b`

用途：计算超额收益、趋势加速度或偏离值。

在本文件中的例子：

- O017：NDX 20 日收益率减 SPX 20 日收益率；
- O019：SOX 20 日收益率减 SPX 20 日收益率；
- O021：20 日趋势斜率减 120 日趋势斜率；
- O025：出口同比减过去均值。

## 4. 滚动标准化和突破打分环节

### `calc_rolling_zscore(series, window, min_periods=None)`

来源：`factor_utils.py`

用途：计算滚动 z-score。

计算逻辑：

```python
(series - rolling_mean) / rolling_std
```

参数含义：

- `window`：滚动窗口长度；
- `min_periods`：至少有多少个有效样本才开始计算。若不传，默认为 `window // 2`。

在本文件中的例子：

- O011：对 VIX 20 日收益率计算 255 日滚动 z-score，`min_periods=120`。

### `_rolling_std_breakout(series, window, min_periods, std_multiplier=1.0)`

来源：`overseaFactors.py`

用途：计算“超过滚动均值若干倍标准差之后的突破强度”。

计算逻辑可以理解为：

1. 计算滚动均值 `rolling_mean`；
2. 计算滚动标准差 `rolling_std`；
3. 计算当前值偏离均值的距离 `deviation = value - rolling_mean`；
4. 只有当 `abs(deviation)` 超过 `std_multiplier * rolling_std` 时，保留超出的部分；
5. 再除以 `rolling_std`，得到标准差单位下的突破强度。

输出方向：

- 当前值显著高于均值：正数；
- 当前值显著低于均值：负数；
- 未突破阈值：0；
- 样本不足或标准差不可用：`NaN`。

在本文件中的例子：

- O013：港元掉期点月末值，约一年窗口，1 倍标准差突破；
- O015：BDI，120 日窗口，2 倍标准差突破，再乘 `-1`；
- O016：估算港元掉期点，约一年窗口，1 倍标准差突破；
- O025 的最后一步也手写了类似逻辑。

### `_safe_divide(numerator, denominator)`

来源：`overseaFactors.py`

用途：安全除法，避免分母为 0 时产生 `inf`。

主要功能：

- 把分母里的 `0.0` 替换为 `NaN`；
- 再做普通除法。

在本文件中的例子：

- `_rolling_std_breakout()` 内部；
- O022：计算趋势 z-score；
- O025：把超过 1 倍标准差的偏离除以标准差。

## 5. 阈值型方向打分环节

### `_signed_excess_score(value, threshold)`

来源：`overseaFactors.py`

用途：只保留超过阈值的部分，并保留原方向。

计算逻辑：

```python
sign(value) * max(abs(value) - threshold, 0)
```

例子：

- `value = 0.08`，`threshold = 0.05`，输出 `+0.03`；
- `value = -0.08`，`threshold = 0.05`，输出 `-0.03`；
- `value = 0.02`，`threshold = 0.05`，输出 `0`。

在本文件中的例子：

- O014：USDCNH 20 日变化超过 1.5% 后计入，再乘 `-1`；
- O017：NDX 相对 SPX 20 日超额收益超过 3% 后计入；
- O018：SOX 20 日收益率超过 5% 后计入；
- O019：SOX 相对 SPX 20 日超额收益超过 5% 后计入。

与 `_rolling_std_breakout()` 的区别：

- `_signed_excess_score()` 使用固定阈值，例如 1.5%、3%、5%；
- `_rolling_std_breakout()` 使用滚动均值和滚动标准差形成动态阈值。

## 6. 趋势类特征环节

O020-O024 都基于 SOX 价格趋势，因此会集中用到下面几个函数。

### `_rolling_log_slope(series, window, min_periods)`

来源：`overseaFactors.py`

用途：对价格取 log 后，在滚动窗口内拟合时间趋势斜率。

直观理解：

- 价格沿时间稳定上涨，斜率为正；
- 价格沿时间稳定下跌，斜率为负；
- 趋势越陡，绝对值越大。

主要步骤：

1. 删除缺失值并按日期排序；
2. 对正价格取 `log`；
3. 在滚动窗口内用时间序号作为 x、log 价格作为 y；
4. 用简单线性回归公式计算斜率。

在本文件中的例子：

- O020：60 日趋势斜率；
- O021：20 日斜率减 120 日斜率；
- O023：60 日趋势斜率；
- O024：20/60/120 日趋势斜率。

### `_rolling_time_corr(series, window, min_periods)`

来源：`overseaFactors.py`

用途：计算序列与时间序号的滚动相关性，用来衡量趋势是否线性、稳定。

直观理解：

- 接近 `+1`：随时间稳定上行；
- 接近 `-1`：随时间稳定下行；
- 接近 `0`：趋势不明显或波动杂乱。

在本文件中的例子：

- O023：计算 60 日 `corr_60d`，再用 `corr_60d ** 2` 作为趋势斜率的质量权重。

### `_rolling_rank(series, window, min_periods)`

来源：`overseaFactors.py`

用途：计算当前值在过去一段窗口里的百分位排名。

输出范围：

- 原始排名是 `0` 到 `1` 附近；
- 本文件中常用 `rank * 2 - 1` 映射到约 `[-1, 1]`。

在本文件中的例子：

- O020：60 日趋势斜率做 1250 日滚动排名；
- O021：趋势加速度做 1250 日滚动排名；
- O022：趋势 z-score 做 1250 日滚动排名；
- O023：R2 加权斜率做 1250 日滚动排名；
- O024：20/60/120 日斜率分别排名后取平均。

注意：

- `window=1250` 大致对应 5 年交易日；
- `min_periods=500` 表示至少约 2 年样本才开始输出排名；
- 这个函数会先 `.dropna()`，所以只在有效样本序列上滚动。

## 7. 常见 pandas/numpy 操作

### `.rolling(window, min_periods).mean()` / `.std()`

用途：滚动均值、滚动标准差。

在本文件中的例子：

- O012：港币即期汇率 20 日均值；
- O022：log SOX 的 250 日均值和标准差；
- O025：韩国半导体出口同比的 36 月均值和标准差。

### `.dropna()`

用途：删除缺失值。

常见使用位置：

- 在进入滚动趋势、滚动突破前，确保窗口计算只基于有效样本；
- 在两个序列相减后，删除两边对齐产生的缺失。

注意：

- `.dropna()` 会让输出只保留有效日期；
- 如果你希望保留完整日期索引并让无效日期继续为 `NaN`，不要过早 `.dropna()`。

### `.sort_index(ascending=True)`

用途：确保时间序列按日期从早到晚排列。

滚动计算、`shift()`、月末取值都依赖正确的日期顺序。

### `np.sign(x)`

用途：取方向。

- 正数返回 `+1`；
- 负数返回 `-1`；
- 0 返回 `0`。

常和 `abs(x)`、阈值扣减配合，用来保留方向。

### `np.log(series.where(series > 0))`

用途：只对正数价格取 log。

在本文件中的例子：

- O020-O024 的 SOX 趋势类因子。

注意：

- 非正数会先变成 `NaN`；
- 价格趋势用 log 后，斜率更接近连续收益率意义。

## 8. 按因子快速查函数

| 因子 | 主要读取函数 | 主要计算函数/操作 | 输出含义 |
| --- | --- | --- | --- |
| O011 | `_load_price_file_close()` | 20 日收益率、`calc_rolling_zscore()`、手写超过 1 倍 z-score 部分 | VIX 风险冲击，方向取反 |
| O012 | `read_prepared_series()`、`_positive_series()` | 20 日均值、固定区间 7.77/7.83 偏离 | 港币联系汇率区间压力 |
| O013 | `read_prepared_series()`、`_complete_month_end_values()` | `_rolling_std_breakout()` | 港元掉期点月末突破 |
| O014 | `read_prepared_series()`、`_positive_series()` | 20 日变化、`_signed_excess_score()` | USDCNH 贬值/升值压力，方向取反 |
| O015 | `read_prepared_series()` | `_rolling_std_breakout()` | BDI 景气突破，方向取反 |
| O016 | `read_prepared_series()`、`_positive_series()` | 理论掉期点、`_rolling_std_breakout()` | 利差隐含港元掉期压力 |
| O017 | `_load_price_file_close()` | 20 日超额收益、`_signed_excess_score()` | 纳指相对标普强弱 |
| O018 | `read_prepared_series()`、`_clean_positive_limited_ffill_series()` | 20 日收益率、`_signed_excess_score()` | SOX 绝对强弱 |
| O019 | `read_prepared_series()`、`_clean_positive_limited_ffill_series()`、`_load_price_file_close()` | 20 日超额收益、`_signed_excess_score()` | SOX 相对标普强弱 |
| O020 | `read_prepared_series()`、`_clean_positive_limited_ffill_series()` | `_rolling_log_slope()`、`_rolling_rank()` | SOX 60 日趋势排名 |
| O021 | 同 O020 | 短斜率 - 长斜率、`_rolling_rank()` | SOX 趋势加速度排名 |
| O022 | 同 O020 | 250 日趋势 z-score、`_safe_divide()`、`_rolling_rank()` | SOX 长趋势位置排名 |
| O023 | 同 O020 | `_rolling_log_slope()`、`_rolling_time_corr()`、`_rolling_rank()` | R2 加权 SOX 趋势排名 |
| O024 | 同 O020 | 20/60/120 日 `_rolling_log_slope()`、`_rolling_rank()` | SOX 多窗口趋势综合排名 |
| O025 | `read_prepared_series()` | 同比、36 月均值/标准差、`_safe_divide()` | 韩国半导体出口同比突破 |

## 9. 新增因子时的实用顺序

如果你要在 `_calc_oversea_factor()` 里新增一个因子，可以按下面顺序写：

1. 先确定数据源。
   - 普通 prepared 表字段：优先用 `read_prepared_series()`；
   - 行情统计收盘价文件：优先用 `_load_price_file_close()`。

2. 再做必要清洗。
   - 价格/汇率必须为正：用 `_positive_series()`；
   - 日频指数有短缺口：考虑 `_clean_positive_limited_ffill_series()`；
   - 月末事件：考虑 `_complete_month_end_values()`。

3. 再计算中间指标。
   - n 日/月变化：`series / series.shift(n) - 1`；
   - 超额收益：`a_return - b_return`；
   - 滚动均值/标准差：`.rolling(...).mean()` / `.std()`；
   - 趋势斜率：`_rolling_log_slope()`。

4. 最后转成因子值。
   - 固定阈值：`_signed_excess_score()`；
   - 动态标准差阈值：`_rolling_std_breakout()`；
   - 历史相对位置：`_rolling_rank() * 2 - 1`；
   - 需要安全除法：`_safe_divide()`。

5. 明确方向。
   - 如果经济含义和成长/价值方向相反，可以在最后乘 `-1`；
   - 不确定方向时，不要硬编，先在 notes 或注释里写明待确认。
