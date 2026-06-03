"""Build normalized factor matrix, mounted factors, and long/short signals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from factor_utils import (
    load_default_data,
    load_record_all_factor_metadata_with_records,
)
from factor_pipeline_runner import run_factor_output_pipeline
from paper_odds_win_style_rotation import (
    FACTOR_IDS as PAPER_FACTOR_IDS,
    PAPER_ID,
    generate_paper_odds_win_style_rotation_factor_source_frame,
)
from overseaFactors import (
    FACTOR_IDS as OVERSEA_FACTOR_IDS,
    generate_overseaFactors_factors,
    metadata_from_overseaFactors_records,
)
from priceFactors1 import (
    FACTOR_IDS as PRICE_FACTOR_IDS,
    generate_priceFactors1_factors,
    metadata_from_priceFactors1_records,
)
from I_laboratory import (
    FACTOR_IDS as I_LABORATORY_FACTOR_IDS,
    generate_I_laboratory_factors,
    metadata_from_I_laboratory_records,
)


def generate_factor_source_frame(data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    try:
        overseaFactors_factor_source_df, overseaFactors_records = generate_overseaFactors_factors(data_df)
    except ValueError as exc:
        if "working_multiple_factors_plan.json missing implemented records" not in str(exc):
            raise
        print(f"Skipping overseaFactors factors: {exc}")
        overseaFactors_factor_source_df = pd.DataFrame(index=pd.to_datetime(data_df.index))
        overseaFactors_records = []
    priceFactors1_factor_source_df, priceFactors1_records = generate_priceFactors1_factors(data_df)
    I_laboratory_factor_source_df, I_laboratory_records = generate_I_laboratory_factors(data_df)
    factor_frames = [
        generate_paper_odds_win_style_rotation_factor_source_frame(data_df),
        overseaFactors_factor_source_df,
        priceFactors1_factor_source_df,
        I_laboratory_factor_source_df,
    ]
    factor_source_df = pd.concat(factor_frames, axis=1, sort=False)
    duplicated_cols = factor_source_df.columns[factor_source_df.columns.duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"factor source columns duplicated: {duplicated_cols}")
    return (
        factor_source_df,
        overseaFactors_records + priceFactors1_records + I_laboratory_records,
    )


def main() -> None:
    data_df, _market_df = load_default_data()

    factor_source_df, generated_records = generate_factor_source_frame(data_df)
    factor_metadata, missing_bar_defaults, selected_records = load_record_all_factor_metadata_with_records(
        PAPER_ID,
        PAPER_FACTOR_IDS,
    )
    overseaFactors_records = [
        record for record in generated_records if str(record.get("factor_id") or "") in OVERSEA_FACTOR_IDS
    ]
    priceFactors1_records = [
        record for record in generated_records if str(record.get("factor_id") or "") in PRICE_FACTOR_IDS
    ]
    I_laboratory_records = [
        record for record in generated_records if str(record.get("factor_id") or "") in I_LABORATORY_FACTOR_IDS
    ]
    factor_metadata.update(metadata_from_overseaFactors_records(overseaFactors_records))
    factor_metadata.update(metadata_from_priceFactors1_records(priceFactors1_records))
    factor_metadata.update(metadata_from_I_laboratory_records(I_laboratory_records))
    selected_records.extend(generated_records)

    run_factor_output_pipeline(
        output_prefix=PAPER_ID,
        factor_source_df=factor_source_df,
        metadata=factor_metadata,
        selected_records=selected_records,
        missing_bar_defaults=missing_bar_defaults,
        write_empty_missing_bar_file=False,
    )


if __name__ == "__main__":
    main()
