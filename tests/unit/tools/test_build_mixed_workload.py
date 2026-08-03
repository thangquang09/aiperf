# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import Conversation
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.prerequisites import TurnPrerequisite
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.dataset.loader.dag_jsonl_models import DagConversation
from tools.build_mixed_workload import (
    build_mixed_workload,
    conversation_to_dag_dict,
    load_source,
    MixConfig,
    parse_config,
    SourceConfig,
    write_and_validate,
)


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


def _single_turn_conv(
    sid: str, content: str, max_tokens: int = 64, delay: float = 0.0
) -> Conversation:
    return Conversation(
        session_id=sid,
        turns=[
            Turn(
                raw_messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                delay=delay,
            )
        ],
    )


def test_serialize_single_turn_sharegpt_shape() -> None:
    conv = _single_turn_conv(
        "sharegpt-0001", "Hello, who are you?", max_tokens=32, delay=1500
    )
    d = conversation_to_dag_dict(conv, sid_prefix="sharegpt", is_root=True)
    DagConversation.model_validate(d)
    assert d["session_id"] == "sharegpt-sharegpt-0001"
    assert d["turns"][0]["messages"] == [{"role": "user", "content": "Hello, who are you?"}]
    assert d["turns"][0]["max_tokens"] == 32
    assert d["turns"][0]["delay"] == 1500


def test_serialize_spawn_with_join_at() -> None:
    conv = Conversation(
        session_id="root",
        turns=[
            Turn(
                raw_messages=[{"role": "user", "content": "plan"}],
                branch_ids=["root:0"],
            ),
            Turn(
                raw_messages=[{"role": "user", "content": "after"}],
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
                ],
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["subagent_a"],
                mode=ConversationBranchMode.SPAWN,
                is_background=False,
            )
        ],
    )
    d = conversation_to_dag_dict(conv, sid_prefix="weka", is_root=True)
    DagConversation.model_validate(d)
    assert d["turns"][0]["spawns"] == [{"children": ["weka-subagent_a"], "join_at": 1}]


def test_serialize_demotes_system_on_non_root_turn() -> None:
    conv = Conversation(
        session_id="forkchild",
        turns=[
            Turn(
                raw_messages=[{"role": "system", "content": "you are X"}],
            ),
        ],
    )
    d = conversation_to_dag_dict(conv, sid_prefix="weka", is_root=False)
    DagConversation.model_validate(d)
    assert d["turns"][0]["messages"][0]["role"] == "user"


def test_serialize_falls_back_to_texts_when_no_raw_messages() -> None:
    conv = Conversation(
        session_id="sharegpt-0001",
        turns=[
            Turn(
                texts=[Text(contents=["Hello, who are you?"])],
                max_tokens=32,
            )
        ],
    )
    d = conversation_to_dag_dict(conv, sid_prefix="sharegpt", is_root=True)
    DagConversation.model_validate(d)
    assert d["turns"][0]["messages"] == [{"role": "user", "content": "Hello, who are you?"}]
    assert d["turns"][0]["max_tokens"] == 32


class _FakeComposer:
    def __init__(self, config, tokenizer, **kw):
        self._config = config
        self._tokenizer = tokenizer

    async def create_dataset_async(self) -> list[Conversation]:
        return [
            _single_turn_conv("0001", "fake prompt", max_tokens=16)
            for _ in range(3)
        ]


def _asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_load_source_uses_injected_composer() -> None:
    tok = Tokenizer.from_pretrained("builtin")
    src = SourceConfig(loader="sharegpt", weight=30, chat_delay_ms=(500, 3000))
    convs = _asyncio_run(
        load_source(src, tok, ["mock-model"], composer_factory=_FakeComposer)
    )
    assert len(convs) == 3
    assert all(c.turns[0].delay >= 500 for c in convs)


def test_write_and_validate_round_trips(tmp_path: Path) -> None:
    lines = [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "spawns": ["child"],
                }
            ],
        },
        {
            "session_id": "child",
            "turns": [
                {"messages": [{"role": "user", "content": "sub"}], "max_tokens": 8}
            ],
        },
    ]
    out = tmp_path / "merged.dag.jsonl"
    write_and_validate(lines, out)
    loaded = DagJsonlLoader(out).load()
    assert {c.session_id for c in loaded} == {"root", "child"}


def test_write_and_validate_rejects_bad_file(tmp_path: Path) -> None:
    out = tmp_path / "bad.dag.jsonl"
    with pytest.raises(Exception):
        write_and_validate([{"session_id": "x", "turns": []}], out)
    assert not out.exists()


async def _fake_source_loader(src, tokenizer, model_names, **kw):
    counts = {
        "sharegpt": 9,
        "speed_bench_rag": 9,
        "semianalysis_cc_traces_weka_062126_256k": 4,
    }
    n = counts[src.loader]
    return [_single_turn_conv(f"{src.loader}-{i}", f"p{i}") for i in range(n)]


def test_build_mixed_workload_end_to_end(tmp_path: Path) -> None:
    tok = Tokenizer.from_pretrained("builtin")
    cfg = MixConfig(
        sources={
            "chat": SourceConfig(loader="sharegpt", weight=30),
            "rag": SourceConfig(loader="speed_bench_rag", weight=30),
            "agentic": SourceConfig(
                loader="semianalysis_cc_traces_weka_062126_256k", weight=40, num_traces=2
            ),
        },
        total_conversations=30,
        out_file=tmp_path / "merged.dag.jsonl",
        tokenizer="builtin",
    )
    out = _asyncio_run(
        build_mixed_workload(cfg, tok, ["mock-model"], source_loader=_fake_source_loader)
    )
    loaded = DagJsonlLoader(out).load()
    assert len(loaded) >= 22  # 9+9+4
