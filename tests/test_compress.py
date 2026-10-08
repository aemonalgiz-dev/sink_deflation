"""End-to-end test on a toy transformer-shaped model: statistics by hooks, compression, factored modules."""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from sinkdeflation import (LowRankLinear, collect_stats, compress_model, find_linear_layers, layer_shapes, parameter_fraction,
                           rank_for_removal, sink_direction)

torch.manual_seed(0)
VOCAB, D, H = 50, 32, 48


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, p, nn.Linear(D, D, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(D, H, bias=False)
        self.mlp.up_proj = nn.Linear(D, H, bias=False)
        self.mlp.down_proj = nn.Linear(H, D, bias=False)

    def forward(self, x):
        a = self.self_attn
        x = x + a.o_proj(torch.tanh(a.q_proj(x)) * a.k_proj(x) + a.v_proj(x))
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))


class Toy(nn.Module):
    """Two blocks over an embedding; token 0 at position 0 plays the sink."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(D, VOCAB)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.layers:
            x = block(x)
        return self.head(x)


def _batches():
    out = []
    for _ in range(3):
        ids = torch.randint(1, VOCAB, (4, 16))
        ids[:, 0] = 0
        out.append(ids)
    return out


def test_statistics_and_compression_end_to_end():
    model = Toy().eval()
    modules = find_linear_layers(model)
    assert len(modules) == 14 and all(".layers." in n for n in modules)
    stats = collect_stats(model, modules, _batches(), budget_gb=1e-5)  # tiny budget forces several groups and passes
    st = stats["layers.0.self_attn.k_proj"]
    assert st.X0.shape == (12, D) and st.seq_len == 16
    assert torch.allclose(st.C_rest, st.C_rest.T, atol=1e-5)

    dense = copy.deepcopy(model)
    W_before = dense.get_submodule("layers.1.self_attn.k_proj").weight.detach().clone()
    ranks = compress_model(model, stats, ratio=0.4, recipe="deflate")
    assert set(ranks) == set(modules)
    assert ranks["layers.0.mlp.down_proj"] == rank_for_removal(D, H, 0.4)

    # deflation keeps every key projection exact on its sink direction
    u = sink_direction(stats["layers.1.self_attn.k_proj"].X0)[:, 0].float()
    W_after = model.get_submodule("layers.1.self_attn.k_proj").weight.detach()
    assert torch.allclose(W_after @ u, W_before @ u, atol=1e-4)

    ids = _batches()[0]
    out = model(ids)
    assert torch.isfinite(out).all()
    shapes = layer_shapes(find_linear_layers(dense))
    frac = parameter_fraction(ranks, sum(p.numel() for p in dense.parameters()), shapes)
    assert 0.5 < frac < 1.0


def test_low_rank_modules_match_materialized_weights():
    model = Toy().eval()
    modules = find_linear_layers(model)
    stats = collect_stats(model, modules, _batches())
    ids = _batches()[0]
    dense_params = sum(p.numel() for p in model.parameters())

    materialized = copy.deepcopy(model)
    compress_model(materialized, stats, ratio=0.4, recipe="reweight", lam=0.3)
    factored = copy.deepcopy(model)
    ranks = compress_model(factored, stats, ratio=0.4, recipe="reweight", lam=0.3, materialize=False)

    mod = factored.get_submodule("layers.0.self_attn.v_proj")
    assert isinstance(mod, LowRankLinear) and mod.rank == ranks["layers.0.self_attn.v_proj"]
    assert torch.allclose(materialized(ids), factored(ids), atol=1e-4)
    stored = sum(p.numel() for p in factored.parameters())
    assert stored < dense_params
    assert stored == dense_params - sum(m * n - k * (m + n) for name, k in ranks.items() for m, n in [tuple(modules[name].weight.shape)])
