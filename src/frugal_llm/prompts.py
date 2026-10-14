"""Prompt generation and JSONL loading."""

from __future__ import annotations

import json
import random
from pathlib import Path

# Short, common English words: almost all are a single token in mainstream tokenizers,
# so word count is a fair proxy for prompt length without shipping a tokenizer.
_WORDS = (
    "the of and to in is it you that he was for on are as with his they at be this "
    "have from or one had by word but not what all were we when your can said there "
    "use an each which she do how their if will up other about out many then them "
    "these so some her would make like him into time has look two more write go see "
    "number no way could people my than first water been call who oil its now find "
    "long down day did get come made may part over new sound take only little work "
    "know place year live me back give most very after thing our just name good "
    "sentence man think say great where help through much before line right too mean "
    "old any same tell boy follow came want show also around form three small set put "
    "end does another well large must big even such because turn here why ask went men "
    "read need land different home us move try kind hand picture again change off play "
    "spell air away animal house point page letter mother answer found study still learn"
).split()

_INSTRUCTION = "Continue the following text for as long as you can:"
# Rough allowance for the instruction plus the chat template the server wraps it in.
_OVERHEAD_TOKENS = 20


class GeneratedPrompts:
    """Deterministic synthetic prompts of approximately ``prompt_tokens`` tokens.

    Each index gets different text from its very first words, so no two requests
    share a prefix and server-side prefix caching cannot flatter TTFT.
    """

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self._n_words = max(prompt_tokens - _OVERHEAD_TOKENS, 1)

    def get(self, index: int) -> str:
        rng = random.Random(index)
        words = " ".join(rng.choice(_WORDS) for _ in range(self._n_words))
        return f"{index}: {words}\n\n{_INSTRUCTION}"


class FilePrompts:
    """Prompts from a JSONL file of ``{"prompt": "..."}`` lines, used in rotation."""

    def __init__(self, prompts: list[str], path: Path | None = None) -> None:
        if not prompts:
            raise ValueError(f"no prompts found in {path}")
        self.prompts = prompts
        self.path = path

    @classmethod
    def load(cls, path: Path) -> FilePrompts:
        prompts = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
                prompt = record.get("prompt") if isinstance(record, dict) else None
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f'{path}:{lineno}: expected {{"prompt": "<non-empty>"}}')
                prompts.append(prompt)
        return cls(prompts, path)

    def get(self, index: int) -> str:
        return self.prompts[index % len(self.prompts)]
