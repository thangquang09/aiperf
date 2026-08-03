# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from tools.build_mixed_workload import MixConfig, parse_config


def test_parse_config_valid(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(
        """
sources:
  chat:
    loader: sharegpt
    weight: 30
    chat_delay_ms: [500, 3000]
  rag:
    loader: speed_bench_rag
    weight: 30
    chat_delay_ms: [500, 3000]
  agentic:
    loader: semianalysis_cc_traces_weka_062126_256k
    weight: 40
    num_traces: 2
total_conversations: 30
out_file: data/merged_workload.dag.jsonl
tokenizer: builtin
"""
    )
    cfg = parse_config(cfg_file)
    assert cfg.total_conversations == 30
    assert cfg.tokenizer == "builtin"
    assert cfg.sources["chat"].weight == 30
    assert cfg.sources["agentic"].num_traces == 2


def test_parse_config_weights_must_sum_100(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(
        """
sources:
  chat: { loader: sharegpt, weight: 30 }
  rag: { loader: speed_bench_rag, weight: 30 }
total_conversations: 30
out_file: out.jsonl
tokenizer: builtin
"""
    )
    with pytest.raises(ValueError, match=r"weights.*100"):
        parse_config(cfg_file)
