#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a mixed-workload dag_jsonl file from public AIPerf datasets.

Usage:
    uv run python tools/build_mixed_workload.py --config data/config.mix.yml
"""
from __future__ import annotations

import argparse
import asyncio
import orjson
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
from aiperf.common.models import Conversation, DatasetMetadata
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


@dataclass(slots=True)
class SourceConfig:
    loader: str
    weight: int | None = None
    chat_delay_ms: tuple[int, int] | None = None
    num_traces: int | None = None
    max_sessions: int | None = None


@dataclass(slots=True)
class MixConfig:
    sources: dict[str, SourceConfig]
    out_file: Path
    tokenizer: str
    total_conversations: int | None = None


def parse_config(path: Path) -> MixConfig:
    yaml = YAML(pure=True)
    with open(path) as f:
        data: dict[str, Any] = yaml.load(f)
    sources: dict[str, SourceConfig] = {}
    total_weight = 0
    has_weight = False
    for name, src in data.get("sources", {}).items():
        delay = src.get("chat_delay_ms")
        weight = src.get("weight")
        if weight is not None:
            has_weight = True
            total_weight += weight
        sources[name] = SourceConfig(
            loader=src["loader"],
            weight=weight,
            chat_delay_ms=tuple(delay) if delay else None,
            num_traces=src.get("num_traces"),
            max_sessions=src.get("max_sessions"),
        )
    total_conversations = data.get("total_conversations")
    if has_weight and total_weight != 100:
        raise ValueError(
            f"Source weights must sum to 100, got {total_weight}"
        )
    if has_weight and total_conversations is None:
        raise ValueError(
            "total_conversations is required when weight-based slicing is used"
        )
    return MixConfig(
        sources=sources,
        total_conversations=total_conversations,
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
        raw: list[dict[str, Any]] = list(turn.raw_messages) if turn.raw_messages else []
        if not raw and turn.texts:
            joined = "\n".join(c for t in turn.texts for c in t.contents if c)
            raw = [{"role": "user", "content": joined}] if joined else []
        if not raw:
            raw = [{"role": "user", "content": " "}]
        messages = raw
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


def write_and_validate(lines: list[dict[str, Any]], out_path: Path) -> None:
    """Write JSONL, then re-validate with DagJsonlLoader as a format barrier.

    On validation failure the partial file is deleted and an exception raised.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for line in lines:
            f.write(orjson.dumps(line))
            f.write(b"\n")
    try:
        conversations = DagJsonlLoader(out_path).load()
        validate_for_orchestrator_v1(
            DatasetMetadata(
                conversations=[c.to_metadata() for c in conversations],
                sampling_strategy=DatasetSamplingStrategy.RANDOM,
            )
        )
    except Exception:
        out_path.unlink(missing_ok=True)
        raise


def _slice_count(weight: int, total: int) -> int:
    return max(1, round(total * weight / 100))


def _conv_input_tokens(conv: Conversation, tokenizer: Tokenizer) -> int:
    """Estimate total input tokens (ISL) across all turns of a conversation."""
    total = 0
    for turn in conv.turns:
        msgs = list(turn.raw_messages) if turn.raw_messages else []
        if not msgs and turn.texts:
            joined = "\n".join(
                c for t in turn.texts for c in t.contents if c
            )
            msgs = [{"role": "user", "content": joined}] if joined else []
        total += sum(
            len(tokenizer.encode(m.get("content", "")))
            for m in msgs
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
    return total


def _report_distribution(
    conversations: list[tuple[Conversation, str]],
    tokenizer: Tokenizer,
) -> None:
    """Print per-source token/session/request distribution to stderr."""
    import collections

    stats: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"sessions": 0, "turns": 0, "tokens": 0}
    )
    for conv, name in conversations:
        s = stats[name]
        s["sessions"] += 1
        s["turns"] += len(conv.turns)
        s["tokens"] += _conv_input_tokens(conv, tokenizer)
    grand = sum(s["tokens"] for s in stats.values())
    print(
        f"\n{'source':<12} {'sessions':>10} {'requests':>10} "
        f"{'input_tokens':>16} {'token%':>7}",
        file=sys.stderr,
    )
    print("-" * 57, file=sys.stderr)
    for name, s in stats.items():
        pct = f"{s['tokens'] * 100 / grand:.1f}%" if grand else "-"
        print(
            f"{name:<12} {s['sessions']:>10,} {s['turns']:>10,} "
            f"{s['tokens']:>16,} {pct:>7}",
            file=sys.stderr,
        )
    print("-" * 57, file=sys.stderr)
    print(
        f"{'total':<12} {sum(s['sessions'] for s in stats.values()):>10,} "
        f"{sum(s['turns'] for s in stats.values()):>10,} "
        f"{grand:>16,} {'100.0%':>7}",
        file=sys.stderr,
    )


async def build_mixed_workload(
    cfg: MixConfig,
    tokenizer: Tokenizer,
    model_names: list[str],
    *,
    source_loader=load_source,
) -> Path:
    """Load all sources, serialize, and write the merged dag_jsonl file.

    When ``total_conversations`` and per-source ``weight`` are set, sessions
    are sliced by weight (legacy session-count mode). Otherwise each source
    is loaded in full (or capped by ``num_traces`` / ``max_sessions``).
    """
    all_conversations: list[tuple[Conversation, str]] = []
    for name, src in cfg.sources.items():
        convs = await source_loader(src, tokenizer, model_names)
        if cfg.total_conversations is not None and src.weight is not None:
            convs = convs[: _slice_count(src.weight, cfg.total_conversations)]
        elif src.max_sessions is not None:
            convs = convs[: src.max_sessions]
        all_conversations.extend((c, name) for c in convs)
    _report_distribution(all_conversations, tokenizer)
    referenced: set[str] = set()
    for c, _ in all_conversations:
        for b in c.branches:
            referenced.update(b.child_conversation_ids)
    lines: list[dict[str, Any]] = []
    for conv, name in all_conversations:
        is_root = conv.session_id not in referenced
        lines.append(conversation_to_dag_dict(conv, sid_prefix=name, is_root=is_root))
    write_and_validate(lines, cfg.out_file)
    return cfg.out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed-workload dag_jsonl file.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--model", default="mock-model", help="Model name(s)")
    parser.add_argument("--total-conversations", type=int, default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--weka-num-traces", type=int, default=None)
    parser.add_argument("--max-chat-sessions", type=int, default=None)
    args = parser.parse_args()

    cfg = parse_config(Path(args.config))
    if args.total_conversations is not None:
        cfg.total_conversations = args.total_conversations
    if args.out_file is not None:
        cfg.out_file = Path(args.out_file)
    if args.weka_num_traces is not None:
        if "agentic" not in cfg.sources:
            raise SystemExit("--weka-num-traces requires an 'agentic' source in config")
        cfg.sources["agentic"].num_traces = args.weka_num_traces
    if args.max_chat_sessions is not None:
        if "chat" not in cfg.sources:
            raise SystemExit("--max-chat-sessions requires a 'chat' source in config")
        cfg.sources["chat"].max_sessions = args.max_chat_sessions

    tokenizer = Tokenizer.from_pretrained(cfg.tokenizer)
    model_names = [m.strip() for m in args.model.split(",")]
    out = asyncio.run(build_mixed_workload(cfg, tokenizer, model_names))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
