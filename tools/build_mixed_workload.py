#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a mixed-workload dag_jsonl file from public AIPerf datasets.

Usage:
    uv run python tools/build_mixed_workload.py --config data/config.mix.yml
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from ruamel.yaml import YAML


@dataclass(slots=True)
class SourceConfig:
    loader: str
    weight: int
    chat_delay_ms: tuple[int, int] | None = None
    num_traces: int | None = None


@dataclass(slots=True)
class MixConfig:
    sources: dict[str, SourceConfig]
    total_conversations: int
    out_file: Path
    tokenizer: str


def parse_config(path: Path) -> MixConfig:
    yaml = YAML(pure=True)
    with open(path) as f:
        data: dict[str, Any] = yaml.load(f)
    sources: dict[str, SourceConfig] = {}
    total_weight = 0
    for name, src in data.get("sources", {}).items():
        delay = src.get("chat_delay_ms")
        sources[name] = SourceConfig(
            loader=src["loader"],
            weight=src["weight"],
            chat_delay_ms=tuple(delay) if delay else None,
            num_traces=src.get("num_traces"),
        )
        total_weight += src["weight"]
    if total_weight != 100:
        raise ValueError(
            f"Source weights must sum to 100, got {total_weight}"
        )
    return MixConfig(
        sources=sources,
        total_conversations=data["total_conversations"],
        out_file=Path(data["out_file"]),
        tokenizer=data.get("tokenizer", "builtin"),
    )
