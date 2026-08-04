# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.sharegpt import ShareGPTLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class ShareGPTMultiTurnLoader(ShareGPTLoader):
    """ShareGPT loader that preserves the full multi-turn conversation.

    Unlike :class:`ShareGPTLoader`, which keeps only the first user/assistant
    exchange as a single-turn request, this loader emits one :class:`Turn`
    per user/assistant pair so the benchmark replays the growing context
    across turns (exercising prefix-cache and prefill under realistic
    multi-turn chat pressure).
    """

    tag = "ShareGPTMultiTurn"

    async def convert_to_conversations(
        self, dataset: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each ShareGPT row into a multi-turn Conversation.

        Each human/assistant message pair becomes one ``Turn`` whose
        ``raw_messages`` carries only the user message; the assistant reply's
        token length is used as ``max_tokens``. Filters out rows with fewer
        than 2 messages (no complete exchange).
        """
        self.info(
            f"Validating {self.tag} dataset and constructing multi-turn conversations"
        )
        filtered: list[Conversation] = []
        skipped = 0
        for entry in dataset:
            messages = entry.get("conversations", [])
            turns = self._build_turns(messages)
            if not turns:
                skipped += 1
                continue
            filtered.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=turns,
                )
            )
        self.debug(
            lambda: (
                f"Filtered to {len(filtered)} multi-turn conversations "
                f"(skipped {skipped} with no complete exchange)"
            )
        )
        return filtered

    def _build_turns(self, messages: list[dict[str, Any]]) -> list[Turn]:
        """Build one Turn per human/assistant pair, skipping invalid entries."""
        turns: list[Turn] = []
        i = 0
        while i + 1 < len(messages):
            user_msg = messages[i]
            asst_msg = messages[i + 1]
            if (
                not isinstance(user_msg, dict)
                or not isinstance(asst_msg, dict)
                or user_msg.get("from") != "human"
                or asst_msg.get("from") not in ("gpt", "chatgpt", "model")
            ):
                i += 1
                continue
            prompt = str(user_msg.get("value", "")).strip()
            completion = str(asst_msg.get("value", "")).strip()
            if not prompt or not completion:
                i += 2
                continue
            prompt_len = len(self.tokenizer.encode(prompt))
            completion_len = len(self.tokenizer.encode(completion))
            if not self.is_valid_sequence(
                prompt_len=prompt_len,
                output_len=completion_len,
                skip_min_output_len_check=self.output_tokens_mean is not None,
            ):
                i += 2
                continue
            turns.append(
                Turn(
                    model=self._select_model_name(),
                    raw_messages=[{"role": "user", "content": prompt}],
                    max_tokens=completion_len,
                )
            )
            i += 2
        return turns

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.RANDOM
