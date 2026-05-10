"""Single-factor regression configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SingleFactorConfig:
    raw_factor_col: str
    factor_name: str
    bar: int
    signal_type: str
    benchmark_mode: str
    benchmark_rebalance_months: int
    trans_fee: float
    charge_initial_trade: bool
    output_dir: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OW_F001_CN10Y_Q3Y = SingleFactorConfig(
    raw_factor_col="OW_F001_CN10Y_Q3Y_raw",
    factor_name="OW_F001_CN10Y_Q3Y",
    bar=0,
    signal_type="state",
    benchmark_mode="rebalance_50",
    benchmark_rebalance_months=2,
    trans_fee=0.002,
    charge_initial_trade=True,
    output_dir=PROJECT_ROOT / "output_regression" / "OW_F001_CN10Y_Q3Y_refactor_tmp",
)
