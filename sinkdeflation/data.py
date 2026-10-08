"""WikiText-2 windows and a teacher-forced loss, as used in the paper."""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


def wikitext_windows(tokenizer, split: str, max_tokens: int, seq_len: int, bos: bool = False, max_lines: int = None) -> torch.Tensor:
    """Non-overlapping windows of `seq_len` tokens of WikiText-2 raw (train or test), shape (windows, seq_len).

    With bos=True every window starts with the tokenizer's BOS token followed by seq_len - 1 text tokens.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    lines = ds["text"][:max_lines] if max_lines else ds["text"]
    ids = tokenizer("\n\n".join(lines), return_tensors="pt", add_special_tokens=False).input_ids[0]
    body = seq_len - 1 if bos else seq_len
    n = min(max_tokens // seq_len, ids.numel() // body)
    windows = ids[: n * body].view(n, body)
    if bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("bos=True but the tokenizer has no BOS token")
        windows = torch.cat([torch.full((n, 1), tokenizer.bos_token_id, dtype=windows.dtype), windows], dim=1)
    return windows


def batches_of(windows: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    return [windows[i:i + batch_size] for i in range(0, windows.shape[0], batch_size)]


@torch.no_grad()
def evaluate_nll(model: nn.Module, windows: torch.Tensor, batch_size: int = 2) -> float:
    """Mean next-token cross-entropy in nats per token under teacher forcing; perplexity is exp of this."""
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for ids in batches_of(windows, batch_size):
        ids = ids.to(device)
        logits = model(ids).logits[:, :-1].float()
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
        count += ids[:, 1:].numel()
    return total / count
