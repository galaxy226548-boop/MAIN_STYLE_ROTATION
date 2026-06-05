"""Shared sample-window helpers for IC and backtest scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SUPPORTED_SAMPLES = ("all", "ins", "oos", "custom")


@dataclass(frozen=True)
class SampleWindow:
    """样本区间描述；默认日期以后只需要在 SY_Baseline/Config.py 中修改。"""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def start_text(self) -> str:
        return self.start.strftime("%Y-%m-%d")

    @property
    def end_text(self) -> str:
        return self.end.strftime("%Y-%m-%d")


def parse_date(value: str | pd.Timestamp | None, arg_name: str) -> pd.Timestamp | None:
    """把命令行日期转成 Timestamp，空值保留为 None。"""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{arg_name} is not a valid date: {value!r}")
    return timestamp.normalize()


def _default_window_from_config(config: Any, sample: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if sample == "all":
        return pd.Timestamp(config.ALL_START), pd.Timestamp(config.ALL_END)
    if sample == "ins":
        return pd.Timestamp(config.INS_START), pd.Timestamp(config.INS_END)
    if sample == "oos":
        return pd.Timestamp(config.OOS_START), pd.Timestamp(config.OOS_END)
    raise ValueError(f"Unknown default sample: {sample}")


def resolve_sample_window(
    config: Any,
    sample: str = "all",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> SampleWindow:
    """解析样本区间；custom 必须显式传入起止日期。"""
    sample_name = str(sample).strip().lower()
    if sample_name not in SUPPORTED_SAMPLES:
        supported = ", ".join(SUPPORTED_SAMPLES)
        raise ValueError(f"Unknown sample {sample!r}; supported samples: {supported}")

    parsed_start = parse_date(start_date, "--start-date")
    parsed_end = parse_date(end_date, "--end-date")

    if sample_name == "custom":
        if parsed_start is None or parsed_end is None:
            raise ValueError("--sample custom requires both --start-date and --end-date")
        start = parsed_start
        end = parsed_end
    else:
        start, end = _default_window_from_config(config, sample_name)
        if parsed_start is not None:
            start = parsed_start
        if parsed_end is not None:
            end = parsed_end

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError(f"sample start date {start.date()} is later than end date {end.date()}")

    return SampleWindow(name=sample_name, start=start, end=end)


def sample_mask(index: pd.Index, window: SampleWindow) -> pd.Series:
    """按样本窗口生成布尔掩码。"""
    date_index = pd.to_datetime(index)
    return (date_index >= window.start) & (date_index <= window.end)


def sample_cli_kwargs(window: SampleWindow) -> list[str]:
    """生成传给下游脚本的样本参数。"""
    args = ["--sample", window.name]
    if window.name == "custom":
        args.extend(["--start-date", window.start_text, "--end-date", window.end_text])
    return args
