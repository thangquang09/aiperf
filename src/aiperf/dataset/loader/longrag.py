# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class LongRAGLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for ``TIGER-Lab/LongRAG``.

    LongRAG provides long-context retrieval-augmented generation inputs:
    each row pairs a short ``query`` with a pre-concatenated ``context``
    of retrieved Wikipedia passages (~20K-30K tokens). The loader assembles
    a single-turn prompt from the two fields via ``prompt_template`` so the
    benchmark exercises heavy prefill and KV-cache pressure typical of
    production long-context RAG traffic.

    Splits: ``subset_100`` (100 rows), ``subset_1000`` (1000), ``full``.
    Subsets: ``nq`` (NaturalQuestions), ``hotpot_qa`` (HotpotQA).

    Example plugins.yaml entry::

        longrag_nq:
          class: aiperf.dataset.loader.longrag:LongRAGLoader
          metadata:
            hf_dataset_name: TIGER-Lab/LongRAG
            hf_split: subset_100
            hf_subset: nq
    """

    def __init__(
        self,
        user_config: UserConfig,
        prompt_template: str = (
            "Context:\n{context}\n\nQuestion: {query}\n\n"
            "Answer the question based on the context above."
        ),
        **kwargs: Any,
    ) -> None:
        self.prompt_template = prompt_template
        super().__init__(user_config=user_config, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each LongRAG row into a single-turn Conversation."""
        self.info(
            f"Validating LongRAG dataset and constructing single-turn "
            f"long-context conversations"
        )
        dataset = data["dataset"]
        conversations: list[Conversation] = []
        skipped = 0
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            context = row.get("context")
            query = row.get("query")
            if not context or not query:
                skipped += 1
                continue

            prompt = self.prompt_template.format(
                context=str(context), query=str(query)
            )
            if not prompt.strip():
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(texts=[Text(contents=[prompt])]),
                    ],
                )
            )

        self.debug(
            lambda: (
                f"Converted {len(conversations)} LongRAG rows"
                f" (skipped {skipped} with missing context/query)"
            )
        )
        return conversations
