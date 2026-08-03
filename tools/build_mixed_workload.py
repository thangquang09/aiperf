#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a mixed-workload dag_jsonl file from public AIPerf datasets.

Usage:
    uv run python tools/build_mixed_workload.py --config data/config.mix.yml
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from ruamel.yaml import YAML

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.public import PublicDatasetComposer


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


def conversation_to_dag_dict(
    conv: Conversation, sid_prefix: str, is_root: bool
) -> dict[str, Any]:
    """Serialize a Conversation to a dag_jsonl line dict.

    Branches on the conversation are mapped to per-turn forks/spawns by
    matching branch_id against each turn's branch_ids. SPAWN_JOIN prereqs
    on a later turn define join_at for object-form spawns.
    """
    branch_by_id = {b.branch_id: b for b in conv.branches}
    turns_out: list[dict[str, Any]] = []
    for idx, turn in enumerate(conv.turns):
        messages: list[dict[str, Any]] = list(turn.raw_messages or [])
        if not is_root:
            messages = [
                {**m, "role": "user"}
                if isinstance(m, dict) and m.get("role") == "system"
                else m
                for m in messages
            ]
        out_turn: dict[str, Any] = {"messages": messages}
        if turn.max_tokens is not None:
            out_turn["max_tokens"] = turn.max_tokens
        if turn.delay is not None:
            out_turn["delay"] = turn.delay
        forks: list[str] = []
        spawns: list[Any] = []
        for bid in turn.branch_ids:
            branch = branch_by_id.get(bid)
            if branch is None:
                continue
            children = [f"{sid_prefix}-{c}" for c in branch.child_conversation_ids]
            if branch.mode == ConversationBranchMode.FORK:
                forks.extend(children)
            elif branch.mode == ConversationBranchMode.SPAWN:
                join_at = _find_join_at(conv, bid, idx)
                if join_at is not None:
                    spawns.append({"children": children, "join_at": join_at})
                else:
                    spawns.extend(children)
        if forks:
            out_turn["forks"] = forks
        if spawns:
            out_turn["spawns"] = spawns
        turns_out.append(out_turn)
    return {
        "session_id": f"{sid_prefix}-{conv.session_id}",
        "turns": turns_out,
    }


def _find_join_at(
    conv: Conversation, branch_id: str, after_idx: int
) -> int | None:
    """Return the turn index carrying a SPAWN_JOIN prereq for branch_id, else None."""
    for k, turn in enumerate(conv.turns):
        if k <= after_idx:
            continue
        for pre in turn.prerequisites:
            if pre.kind == PrerequisiteKind.SPAWN_JOIN and pre.branch_id == branch_id:
                return k
    return None


def build_user_config(
    public_dataset: str,
    num_dataset_entries: int | None,
    model_names: list[str],
) -> UserConfig:
    """Build a minimal UserConfig sufficient for PublicDatasetComposer.

    ``endpoint.type`` is ``chat`` (a registered EndpointType that tokenizes
    input and produces tokens, so the tokenizer validators pass). The brief's
    ``"openai"`` is not a registered EndpointType and is rejected by the enum.
    ``endpoint.urls`` defaults to ``["localhost:8000"]`` so no URL is needed.
    """
    data: dict[str, Any] = {
        "endpoint": {
            "type": "chat",
            "model_names": model_names,
        },
        "input": {
            "public_dataset": public_dataset,
        },
        "tokenizer": {"name": "builtin"},
    }
    if num_dataset_entries is not None:
        data["input"]["conversation"] = {"num_dataset_entries": num_dataset_entries}
    return UserConfig(**data)


async def load_source(
    src: SourceConfig,
    tokenizer: Tokenizer,
    model_names: list[str],
    *,
    composer_factory: type = PublicDatasetComposer,
) -> list[Conversation]:
    """Load one source via PublicDatasetComposer, applying chat_delay if set."""
    num_entries = src.num_traces if src.num_traces is not None else None
    config = build_user_config(src.loader, num_entries, model_names)
    composer = composer_factory(config=config, tokenizer=tokenizer)
    conversations = await composer.create_dataset_async()
    if src.chat_delay_ms is not None:
        lo, hi = src.chat_delay_ms
        for conv in conversations:
            for turn in conv.turns:
                turn.delay = random.randint(lo, hi)
    return conversations
