"""Apply the factorizations to the linear layers of a transformer."""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .factor import factors_for, rank_for_removal
from .stats import LayerStats

BLOCK_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
KV_TYPES = ("k_proj", "v_proj")


def find_linear_layers(model: nn.Module, types: Sequence[str] = BLOCK_TYPES) -> "OrderedDict[str, nn.Linear]":
    """nn.Linear modules inside the transformer blocks (names containing 'layers.<i>.') whose name ends with one of `types`."""
    found: "OrderedDict[str, nn.Linear]" = OrderedDict()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and re.search(r"(^|\.)layers\.\d+\.", name) and name.rsplit(".", 1)[-1] in types:
            found[name] = mod
    return found


class LowRankLinear(nn.Module):
    """y = A (B x) + b, the factored replacement of an nn.Linear; stores k (m + n) weights."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        kw = dict(device=A.device, dtype=A.dtype)
        self.down = nn.Linear(B.shape[1], B.shape[0], bias=False, **kw)        # n -> k
        self.up = nn.Linear(A.shape[1], A.shape[0], bias=bias is not None, **kw)  # k -> m
        with torch.no_grad():
            self.down.weight.copy_(B)
            self.up.weight.copy_(A)
            if bias is not None:
                self.up.bias.copy_(bias)

    @property
    def rank(self) -> int:
        return self.down.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _replace_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new)


@torch.no_grad()
def compress_model(model: nn.Module, stats: Mapping[str, LayerStats], ratio: float, recipe: str = "deflate",
                   lam: float = 0.3, damp: float = 1e-3, apply_to: Sequence[str] = KV_TYPES,
                   factor_device: Optional[str] = None, materialize: bool = True) -> Dict[str, int]:
    """Truncate every layer in `stats` so that `ratio` of its parameters are removed.

    Layers whose type is in `apply_to` use `recipe` ('standard', 'reweight' or 'deflate'); all other layers use
    the standard statistic. With materialize=True the dense weight is overwritten by A @ B (what the paper
    evaluates; storage is accounted as k (m + n)); with materialize=False the module is replaced by a
    LowRankLinear that actually holds the two factors. Returns {layer name: rank}.
    """
    ranks: Dict[str, int] = {}
    for name, st in stats.items():
        mod = model.get_submodule(name)
        W = mod.weight.detach()
        dev = torch.device(factor_device) if factor_device else W.device
        k = rank_for_removal(W.shape[0], W.shape[1], ratio)
        rec = recipe if name.rsplit(".", 1)[-1] in apply_to else "standard"
        A, B = factors_for(W.to(dev), st.C0, st.C_rest, st.X0, st.seq_len, k, rec, lam, damp)
        if materialize:
            mod.weight.copy_((A @ B).to(mod.weight.dtype).to(mod.weight.device))
        else:
            _replace_module(model, name, LowRankLinear(A.to(W.dtype).to(W.device), B.to(W.dtype).to(W.device), mod.bias))
        ranks[name] = k
    return ranks


def parameter_fraction(ranks: Mapping[str, int], dense_total: int, shapes: Mapping[str, tuple]) -> float:
    """Fraction of the dense model's parameters a compressed model stores, from the per-layer ranks and dense shapes."""
    saved = sum(m * n - k * (m + n) for name, k in ranks.items() for m, n in [shapes[name]])
    return (dense_total - saved) / dense_total


def layer_shapes(modules: Mapping[str, nn.Linear]) -> Dict[str, tuple]:
    return {name: tuple(mod.weight.shape) for name, mod in modules.items()}
