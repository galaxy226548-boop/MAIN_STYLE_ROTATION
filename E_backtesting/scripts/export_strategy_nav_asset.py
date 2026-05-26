"""Export a style-rotation strategy NAV as an asset price series.

The output ``asset_input`` table uses the same long format as ALL_ASSETS.xlsx:
Indexcd, Trddt, Clsidx.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYTHON = PROJECT_ROOT / ".venv_mktp" / "bin" / "python"
SY_BASELINE_DIR = PROJECT_ROOT / "SY_Baseline"
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

try:
    import pandas as pd
except ModuleNotFoundError:
    if LOCAL_PYTHON.exists() and Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv_mktp").resolve():
        os.execv(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv])
    raise

for path in (PROJECT_ROOT, SY_BASELINE_DIR, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Config import Config  # noqa: E402
import backtesting as bt  # noqa: E402


DEFAULT_SIGNAL_PATH = PROJECT_ROOT / "F_grouping" / "output_COMB" / "W008_signal_binary_signal_ls.parquet"
DEFAULT_FACTOR_COL = "W008_signal_binary"
DEFAULT_MARKET_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "market_df.parquet"
DEFAULT_DATA_PATH = PROJECT_ROOT / "A_data" / "prepared_data" / "data_df.parquet"
DEFAULT_BENCHMARK_POSITION_PATH = PROJECT_ROOT / "C_positions" / "reference" / "benchmark_positions.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "E_backtesting" / "Result" / "exported_nav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a signal strategy NAV as a synthetic asset price series."
    )
    parser.add_argument("--signal-path", type=Path, default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--factor-col", default=DEFAULT_FACTOR_COL)
    parser.add_argument("--market-path", type=Path, default=DEFAULT_MARKET_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--benchmark-position-path", type=Path, default=DEFAULT_BENCHMARK_POSITION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--asset-code",
        default=None,
        help="Indexcd value for the exported asset. Defaults to --factor-col.",
    )
    parser.add_argument(
        "--price-base",
        type=float,
        default=1000.0,
        help="Starting Clsidx level for the synthetic asset price series.",
    )
    parser.add_argument(
        "--trans-fee",
        type=float,
        default=None,
        help="Internal style-rotation one-way fee. Defaults to Config.TRANS_FEE; use 0 for a gross NAV.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_signal_position(signal_path: Path, factor_col: str) -> pd.DataFrame:
    if not signal_path.exists():
        raise FileNotFoundError(f"Cannot find signal parquet: {signal_path}")

    signal_df = pd.read_parquet(signal_path).sort_index()
    signal_df.index = pd.to_datetime(signal_df.index)
    if factor_col not in signal_df.columns:
        raise KeyError(f"{factor_col!r} not found in signal columns: {list(signal_df.columns)}")

    signal_ls = pd.to_numeric(signal_df[factor_col], errors="coerce")
    position_df = pd.DataFrame(index=signal_df.index)
    position_df["signal_ls"] = signal_ls
    position_df["target_weight_g"] = 0.5 + 0.5 * signal_ls
    position_df["target_weight_v"] = 0.5 - 0.5 * signal_ls
    bt.validate_position_df(position_df)
    return position_df


def load_market_inputs(
    market_path: Path,
    data_path: Path,
    benchmark_position_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (market_path, data_path, benchmark_position_path):
        if not path.exists():
            raise FileNotFoundError(f"Cannot find input file: {path}")

    market_df = pd.read_parquet(market_path).sort_index()
    data_df = pd.read_parquet(data_path).sort_index()
    benchmark_position_df = pd.read_parquet(benchmark_position_path).sort_index()
    for df in (market_df, data_df, benchmark_position_df):
        df.index = pd.to_datetime(df.index)
    return market_df, data_df, benchmark_position_df


def build_asset_input_df(strategy_combo_nav: pd.Series, asset_code: str, price_base: float) -> pd.DataFrame:
    nav = strategy_combo_nav.dropna().sort_index()
    if nav.empty:
        raise ValueError("strategy_combo_nav is empty")
    if nav.iloc[0] <= 0:
        raise ValueError(f"first NAV must be positive, got {nav.iloc[0]}")

    price_series = nav / nav.iloc[0] * price_base
    return pd.DataFrame(
        {
            "Indexcd": asset_code,
            "Trddt": price_series.index,
            "Clsidx": price_series.values,
        }
    )


def export_outputs(
    output_dir: Path,
    factor_col: str,
    asset_code: str,
    strategy_combo_nav: pd.Series,
    benchmark_combo_nav: pd.Series,
    strategy_track_nav_df: pd.DataFrame,
    asset_input_df: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = asset_code

    paths = {
        "strategy_combo_nav_csv": output_dir / f"{prefix}_strategy_combo_nav.csv",
        "strategy_combo_nav_parquet": output_dir / f"{prefix}_strategy_combo_nav.parquet",
        "benchmark_combo_nav_csv": output_dir / f"{prefix}_benchmark_combo_nav.csv",
        "strategy_track_nav_csv": output_dir / f"{prefix}_strategy_track_nav.csv",
        "asset_input_csv": output_dir / f"{prefix}_asset_input.csv",
        "asset_input_parquet": output_dir / f"{prefix}_asset_input.parquet",
        "asset_input_xlsx": output_dir / f"{prefix}_asset_input.xlsx",
    }

    strategy_combo_nav.rename(factor_col).to_frame().to_csv(paths["strategy_combo_nav_csv"])
    strategy_combo_nav.rename(factor_col).to_frame().to_parquet(paths["strategy_combo_nav_parquet"])
    benchmark_combo_nav.rename("benchmark_combo_nav").to_frame().to_csv(paths["benchmark_combo_nav_csv"])
    strategy_track_nav_df.to_csv(paths["strategy_track_nav_csv"])
    asset_input_df.to_csv(paths["asset_input_csv"], index=False)
    asset_input_df.to_parquet(paths["asset_input_parquet"], index=False)
    asset_input_df.to_excel(paths["asset_input_xlsx"], index=False)
    return paths


def main() -> None:
    args = parse_args()

    signal_path = resolve_project_path(args.signal_path)
    market_path = resolve_project_path(args.market_path)
    data_path = resolve_project_path(args.data_path)
    benchmark_position_path = resolve_project_path(args.benchmark_position_path)
    output_dir = resolve_project_path(args.output_dir)
    factor_col = str(args.factor_col)
    asset_code = str(args.asset_code or factor_col)
    trans_fee = float(Config.TRANS_FEE if args.trans_fee is None else args.trans_fee)

    position_df = read_signal_position(signal_path=signal_path, factor_col=factor_col)
    market_df, data_df, benchmark_position_df = load_market_inputs(
        market_path=market_path,
        data_path=data_path,
        benchmark_position_path=benchmark_position_path,
    )

    work_df = bt.build_work_df(market_df=market_df, position_df=position_df, factor_col=factor_col)
    track_list = sorted(work_df[bt.TRACK_COL].dropna().astype(int).unique())
    if not track_list:
        raise ValueError("No valid track_id values found in work_df")

    strategy_target_df = bt.build_strategy_target_df(work_df=work_df, factor_col=factor_col)
    benchmark_target_df = bt.build_benchmark_target_df(
        work_df=work_df,
        benchmark_position_df=benchmark_position_df,
    )
    strategy_period_df, benchmark_period_df = bt.run_period_backtests(
        strategy_target_df=strategy_target_df,
        benchmark_target_df=benchmark_target_df,
        track_list=track_list,
        trans_fee=trans_fee,
        charge_initial_trade=bool(Config.CHARGE_INITIAL_TRADE),
    )

    daily_asset_ret_df = bt.build_daily_asset_ret_df(data_df=data_df)
    (
        _strategy_track_nav_dict,
        _benchmark_track_nav_dict,
        strategy_track_nav_df,
        _benchmark_track_nav_df,
        strategy_combo_nav,
        benchmark_combo_nav,
    ) = bt.build_nav_outputs(
        strategy_period_df=strategy_period_df,
        benchmark_period_df=benchmark_period_df,
        daily_asset_ret_df=daily_asset_ret_df,
        track_list=track_list,
    )

    asset_input_df = build_asset_input_df(
        strategy_combo_nav=strategy_combo_nav,
        asset_code=asset_code,
        price_base=float(args.price_base),
    )
    paths = export_outputs(
        output_dir=output_dir,
        factor_col=factor_col,
        asset_code=asset_code,
        strategy_combo_nav=strategy_combo_nav,
        benchmark_combo_nav=benchmark_combo_nav,
        strategy_track_nav_df=strategy_track_nav_df,
        asset_input_df=asset_input_df,
    )

    print(f"factor_col: {factor_col}")
    print(f"asset_code: {asset_code}")
    print(f"trans_fee: {trans_fee}")
    print(f"date range: {asset_input_df['Trddt'].min()} -> {asset_input_df['Trddt'].max()}")
    print(f"rows: {len(asset_input_df)}")
    print(f"start Clsidx: {asset_input_df['Clsidx'].iloc[0]:.6f}")
    print(f"end Clsidx: {asset_input_df['Clsidx'].iloc[-1]:.6f}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
