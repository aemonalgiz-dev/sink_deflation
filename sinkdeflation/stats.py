"""Collect the first-position and remaining-position input statistics of linear layers with forward pre-hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
from torch import nn

from .factor import reweighted_statistic, standard_statistic


@dataclass
class LayerStats:
    """Input statistics of one linear layer over a calibration set of windows of length seq_len."""
    C0: torch.Tensor       # (n, n) mean of x_0 x_0^T over windows
    C_rest: torch.Tensor   # (n, n) mean of x_t x_t^T over positions 1..T-1 of all windows
    X0: torch.Tensor       # (windows, n) the first-position inputs themselves
    seq_len: int

    def standard(self) -> torch.Tensor:
        return standard_statistic(self.C0, self.C_rest, self.seq_len)

    def reweighted(self, lam: float) -> torch.Tensor:
        return reweighted_statistic(self.C0, self.C_rest, lam)


@torch.no_grad()
def collect_stats(model: nn.Module, modules: Mapping[str, nn.Linear], batches: Iterable[torch.Tensor],
                  budget_gb: float = 8.0, dtype: torch.dtype = torch.float32) -> Dict[str, LayerStats]:
    """Run the calibration batches through `model` and accumulate LayerStats for every module in `modules`.

    batches must be re-iterable (a list of input tensors of shape (batch, T)); modules are processed in
    groups whose n x n accumulators fit in `budget_gb` of device memory, one pass over the batches per group.
    """
    device = next(model.parameters()).device
    size_gb = lambda mod: mod.in_features ** 2 * 4 / 2 ** 30
    pending = sorted(modules.items(), key=lambda kv: size_gb(kv[1]), reverse=True)
    out: Dict[str, LayerStats] = {}
    while pending:
        group, used = [], 0.0
        while pending and (not group or used + size_gb(pending[0][1]) <= budget_gb):
            used += size_gb(pending[0][1])
            group.append(pending.pop(0))
        acc = {name: torch.zeros(mod.in_features, mod.in_features, device=device, dtype=dtype) for name, mod in group}
        count = {name: 0 for name, _ in group}
        first = {name: [] for name, _ in group}
        seq_len = {}

        def make_hook(name):
            def hook(mod, inp):
                x = inp[0].detach().to(dtype)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                seq_len[name] = x.shape[1]
                first[name].append(x[:, 0].cpu())
                xr = x[:, 1:].reshape(-1, x.shape[-1])
                acc[name].addmm_(xr.T, xr)
                count[name] += xr.shape[0]
            return hook

        handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in group]
        try:
            for ids in batches:
                model(ids.to(device))
        finally:
            for h in handles:
                h.remove()
        for name, _ in group:
            X0 = torch.cat(first[name])
            out[name] = LayerStats(C0=X0.T @ X0 / X0.shape[0], C_rest=(acc[name] / count[name]).cpu(),
                                   X0=X0, seq_len=seq_len[name])
        del acc
    return out
