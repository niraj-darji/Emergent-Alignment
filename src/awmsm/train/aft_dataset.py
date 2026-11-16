"""Dataset loader and dynamic padding collator for Stage 2 Alignment Fine-Tuning (AFT).

Reference: AW_MSM_Methodology.md (§7.3, §8.4) and AFT_training_blueprint.md (§3, §5).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Template
from torch.utils.data import Dataset


class AFTDataset(Dataset):
    """Dataset for Stage 2 AFT with assistant-turn loss masking."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any,
        chat_template_path: Path,
        max_seq_len: int = 4096,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(chat_template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

        self.encoded_items = self._prepare_data()

    def _prepare_data(self) -> List[Dict[str, List[int]]]:
        """Pre-encodes conversations and generates assistant-only loss masks."""
        items = []

        for sample in self.samples:
            messages = sample.get("messages", [])
            if not messages:
                continue

            # Tokenize turn-by-turn to construct exact loss masks
            input_ids: List[int] = []
            labels: List[int] = []

            # We iterate through turns, rendering prefixes and responses
            for idx, msg in enumerate(messages):
                role = msg.get("role")
                content = msg.get("content", "")

                if role in ("system", "user"):
                    # Format user or system turn
                    rendered_turn = self.template.render(
                        messages=[msg], add_generation_prompt=False
                    )
                    turn_tokens = self.tokenizer.encode(
                        rendered_turn, add_special_tokens=False
                    )
                    input_ids.extend(turn_tokens)
                    # User and system tokens are masked with -100
                    labels.extend([-100] * len(turn_tokens))

                elif role == "assistant":
                    # Assistant turn prefix: '<|assistant|>\n'
                    prefix_str = "<|assistant|>\n"
                    prefix_tokens = self.tokenizer.encode(
                        prefix_str, add_special_tokens=False
                    )

                    # Assistant turn content + termination: '{content}<|end_of_text|>\n'
                    eos_text = "<|end_of_text|>\n"
                    content_str = f"{content}{eos_text}"
                    content_tokens = self.tokenizer.encode(
                        content_str, add_special_tokens=False
                    )

                    # Header prefix is masked out
                    input_ids.extend(prefix_tokens)
                    labels.extend([-100] * len(prefix_tokens))

                    # Response content and EOS token are supervised
                    input_ids.extend(content_tokens)
                    labels.extend(content_tokens)

            # Ensure truncation to max_seq_len
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]

            # Only retain samples that have at least one active supervised token
            active_tokens = sum(1 for label in labels if label != -100)
            if active_tokens > 0:
                items.append({
                    "id": sample.get("id", f"sample_{len(items)}"),
                    "input_ids": input_ids,
                    "labels": labels,
                })

        return items

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encoded_items[idx]


class AFTCollator:
    """Collator with dynamic batch padding for input_ids, attention_mask, and labels."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of > 0:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            ids = item["input_ids"]
            lbls = item["labels"]
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
            batch_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def get_aft_datasets(
    config: Dict[str, Any], tokenizer: Any
) -> Tuple[AFTDataset, AFTDataset, AFTCollator]:
    """Loads, splits, and prepares train and evaluation datasets for Stage 2 AFT."""
    project_root = Path(__file__).resolve().parents[3]

    data_cfg = config.get("data", {})
    sft_path_str = data_cfg.get("sft_path", "data/aft/general_sft_10k.jsonl")
    sft_path = Path(sft_path_str)
    if not sft_path.is_absolute():
        sft_path = project_root / sft_path

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {sft_path}")

    chat_template_str = config.get("model", {}).get(
        "chat_template", "configs/chat_template.jinja"
    )
    chat_template_path = Path(chat_template_str)
    if not chat_template_path.is_absolute():
        chat_template_path = project_root / chat_template_path

    if not chat_template_path.exists():
        raise FileNotFoundError(f"Chat template not found at {chat_template_path}")

    heldout_frac = data_cfg.get("heldout_frac", 0.02)
    split_seed = data_cfg.get("split_seed", 0)
    max_seq_len = data_cfg.get("seq_len", 4096)

    # Load raw records
    all_samples = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    # Deterministic document/sample-level split
    train_samples = []
    eval_samples = []

    for sample in sorted(all_samples, key=lambda x: str(x.get("id", ""))):
        sid = str(sample.get("id", ""))
        hash_val = int(
            hashlib.sha256(f"{split_seed}:{sid}".encode("utf-8")).hexdigest(), 16
        )
        frac = (hash_val % 100000) / 100000.0

        if frac < heldout_frac:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    train_dataset = AFTDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        chat_template_path=chat_template_path,
        max_seq_len=max_seq_len,
    )

    eval_dataset = AFTDataset(
        samples=eval_samples,
        tokenizer=tokenizer,
        chat_template_path=chat_template_path,
        max_seq_len=max_seq_len,
    )

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collator = AFTCollator(pad_token_id=pad_token_id)

    print(
        f"[AFT Dataset] Total: {len(all_samples)} | Train: {len(train_dataset)} | Eval: {len(eval_dataset)}"
    )

    return train_dataset, eval_dataset, collator

