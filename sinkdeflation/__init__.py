"""sinkdeflation: sink-aware calibration statistics for activation-whitened SVD compression of language models.

Quick start (any Hugging Face causal LM):

    from sinkdeflation import find_linear_layers, collect_stats, compress_model
    modules = find_linear_layers(model)                 # q, k, v, o, gate, up, down projections of every block
    stats = collect_stats(model, modules, calib_batches)  # one calibration pass; keeps position 0 apart
    ranks = compress_model(model, stats, ratio=0.2, recipe="deflate")   # or "reweight" / "standard"
"""
from .compress import BLOCK_TYPES, KV_TYPES, LowRankLinear, compress_model, find_linear_layers, layer_shapes, parameter_fraction
from .data import batches_of, evaluate_nll, wikitext_windows
from .factor import (RECIPES, damped_cholesky, deflated_factors, factors_for, rank_for_removal, reweighted_statistic,
                     sink_direction, standard_statistic, whitened_factors)
from .stats import LayerStats, collect_stats

__version__ = "0.1.0"
__all__ = [
    "BLOCK_TYPES", "KV_TYPES", "RECIPES", "LayerStats", "LowRankLinear", "batches_of", "collect_stats", "compress_model",
    "damped_cholesky", "deflated_factors", "evaluate_nll", "factors_for", "find_linear_layers", "layer_shapes",
    "parameter_fraction", "rank_for_removal", "reweighted_statistic", "sink_direction", "standard_statistic",
    "whitened_factors", "wikitext_windows",
]
