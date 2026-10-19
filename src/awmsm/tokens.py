"""Token counting with the base-model tokenizer (§6.3 S12).

``TokenCounter`` uses the training base model's tokenizer (``configs/base.yaml`` ``train_base_model``,
default ``meta-llama/Llama-3.1-8B``) from the local Hugging Face cache when ``transformers`` is installed.
Otherwise it falls back to an approximation (1.33 tokens per word) and says so in ``method``, so a corpus
manifest always records how its token budget was counted.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
WORDS_TO_TOKENS = 1.33   # fallback only; English prose with a Llama-3 tokenizer is roughly 1.3 tokens/word


class TokenCounter:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B"):
        self.model_id = model_id
        self._tok = None
        self.method = f"approx {WORDS_TO_TOKENS} tokens/word (no local tokenizer for {model_id})"
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            self._tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self.method = f"tokenizer {model_id}"
        except Exception as e:  # noqa: BLE001 - transformers missing or model not cached
            log.warning("TokenCounter: %s; using %s tokens/word", type(e).__name__, WORDS_TO_TOKENS)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok(text, add_special_tokens=False)["input_ids"])
        return round(len(text.split()) * WORDS_TO_TOKENS)
