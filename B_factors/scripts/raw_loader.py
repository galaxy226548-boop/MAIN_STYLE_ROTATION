"""Raw factor loader for regression scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_raw_factor(data_df: pd.DataFrame, raw_factor_col: str, project_root: Path) -> pd.DataFrame:
    if raw_factor_col in data_df.columns:
        return data_df

    raw_factor_path = project_root / "ow_factors_raw.parquet"
    if not raw_factor_path.exists():
        raise KeyError(
            f"{raw_factor_col} 不在 data_df.columns 中，且未找到 {raw_factor_path}；"
            "本脚本不会临时生成因子。"
        )

    raw_factor_df = pd.read_parquet(raw_factor_path)
    raw_factor_df.index = pd.to_datetime(raw_factor_df.index)
    raw_factor_df = raw_factor_df.sort_index()
    if raw_factor_col not in raw_factor_df.columns:
        raise KeyError(
            f"{raw_factor_col} 不在 data_df.columns 中，且 {raw_factor_path} 中也没有该列；"
            "本脚本不会临时生成因子。"
        )

    data_df = data_df.copy()
    data_df[raw_factor_col] = raw_factor_df[raw_factor_col].reindex(data_df.index)
    print(f"loaded {raw_factor_col} from: {raw_factor_path}")
    return data_df
