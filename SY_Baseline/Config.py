from pathlib import Path
import pandas as pd

class Config:
    """
    项目的全局配置文件。
    作用：
    1. 统一管理路径
    2. 统一管理研究口径
    3. 避免参数散落在 notebook 各处
    """

    # 1. 路径配置
    DATA_DIR = Path("data")
    STYLE_INDEX_DIR = DATA_DIR / "style_indexes"

    PATH_GROWTH = STYLE_INDEX_DIR / "growth_index.xlsx"
    PATH_VALUE = STYLE_INDEX_DIR / "value_index.xlsx"

    # 2. 目标变量与基础列名
    TARGET_COMPARE = "target_close_ratio" # 连续变量：成长指数收盘价 / 价值指数收盘价
    TARGET_CONT = "target_return_diff"   # 连续变量：未来一期成长 - 价值
    TARGET_BIN = "target_label"          # 二分类标签：成长是否跑赢价值
    TRACK_COL = "track_id"               # 五轨道编号列

    # 3. 市场与数据基础设置
    TARGET_COLS = ["Open", "High", "Low", "Close"]
    Tunnels = 5          # 周频五轨道
    MONTH_DAYS = 20      # 近似一个月的交易日数量
    ANNUAL_TRADING_DAYS = 252      # 每年按 252 个交易日处理 (计算年化收益率时是按照后续的annual_days变量代表的自然日计算的)

    # 收益率与成交口径
    RETURN_TYPE = "log"  # "log" 或 "simple"
    DEAL_TYPE = "close"   # "open" 或 "close"

    # 4. 因子相关设置
    # 这里放正式因子名（不带 _raw）
    FEATURE_LIST = []

    # 5. Bucket / Horizon 检验参数
    # Bucket Decay：未来第 k 周“单周收益差”
    BUCKET_WEEKS = [2, 3, 4, 6, 8]
    # Horizon IC：未来 h 周“累计收益差”
    HORIZON_WEEKS = [2, 3, 4, 6, 8]

    BUCKET_PREFIX = "bucket_target_"
    HORIZON_PREFIX = "horizon_target_"

    # Newey-West 滞后阶数默认值
    NW_LAG_DEFAULT = 1

    # 样本数太少时，不做统计检验
    MIN_VALID_SAMPLE = 10

    # 分样本设置（按 trade_date）
    INS_START = pd.Timestamp("2010-03-01")
    INS_END   = pd.Timestamp("2022-12-31")

    OOS_START = pd.Timestamp("2023-01-01")
    OOS_END   = pd.Timestamp("2026-03-13")

    ALL_START = min(INS_START, OOS_START)
    ALL_END = max(INS_END, OOS_END)

    # 7. 信号回测参数
    TRANS_FEE = 2/1000          # 单边手续费
    LOCK = True                     # True=锁仓；False=不锁仓
    CHARGE_INITIAL_TRADE = True     # 第一笔开仓是否收费
    RF = 0.015                      # 年化无风险利率
    ANNUAL_DAYS = 365.25            # 年化换算天数（自然日）

    # 8. pandas 显示设置
    DISPLAY_MAX_COLUMNS = 200
    DISPLAY_WIDTH = 200
    DISPLAY_MAX_ROWS = 50

    @staticmethod
    def apply_pandas_display():
        """
        在 notebook 里调用这个函数，统一设置 DataFrame 显示格式。
        """
        pd.set_option("display.max_columns", Config.DISPLAY_MAX_COLUMNS)
        pd.set_option("display.width", Config.DISPLAY_WIDTH)
        pd.set_option("display.max_rows", Config.DISPLAY_MAX_ROWS)
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    @staticmethod
    def make_output_dirs():
        """
        如果输出目录不存在，就自动创建。
        """
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
