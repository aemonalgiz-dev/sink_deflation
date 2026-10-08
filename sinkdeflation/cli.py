"""Command line: compress a Hugging Face causal LM and report the WikiText-2 loss increase.

    python -m sinkdeflation --model Qwen/Qwen3-1.7B --ratio 0.2 --recipe deflate
    python -m sinkdeflation --model meta-llama/Llama-3.1-8B --ratio 0.2 --recipe reweight --lam 0.3 --bos --save-dir out
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from .compress import KV_TYPES, compress_model, find_linear_layers, layer_shapes, parameter_fraction
from .data import batches_of, evaluate_nll, wikitext_windows
from .factor import RECIPES
from .stats import collect_stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--ratio", type=float, required=True, help="fraction of each block linear layer's parameters to remove")
    ap.add_argument("--recipe", choices=RECIPES, default="deflate", help="statistic for the key and value projections")
    ap.add_argument("--lam", type=float, default=0.3, help="first-position weight for --recipe reweight")
    ap.add_argument("--calib-tokens", type=int, default=65536)
    ap.add_argument("--eval-tokens", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--bos", action="store_true", help="start every window with the BOS token (Llama, Gemma, TinyLlama, MiniCPM)")
    ap.add_argument("--damp", type=float, default=1e-3)
    ap.add_argument("--budget-gb", type=float, default=8.0, help="device memory for the statistics accumulators")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--factor-device", default=None, help="device for the SVDs, e.g. cuda:1 when the model fills cuda:0")
    ap.add_argument("--low-rank", action="store_true", help="keep the two factors as modules instead of writing dense weights")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()

    test = wikitext_windows(tok, "test", args.eval_tokens, args.seq_len, args.bos)
    dense_nll = evaluate_nll(model, test, args.batch_size)
    print(f"{args.model}: dense loss {dense_nll:.4f} nats/token, ppl {math.exp(dense_nll):.2f}", flush=True)

    modules = find_linear_layers(model)
    shapes = layer_shapes(modules)
    dense_total = sum(p.numel() for p in model.parameters())
    calib = batches_of(wikitext_windows(tok, "train", args.calib_tokens, args.seq_len, args.bos, max_lines=20000), args.batch_size)
    t0 = time.time()
    stats = collect_stats(model, modules, calib, budget_gb=args.budget_gb)
    print(f"statistics for {len(stats)} matrices in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    ranks = compress_model(model, stats, args.ratio, args.recipe, args.lam, args.damp, KV_TYPES,
                           args.factor_device, materialize=not args.low_rank)
    nll = evaluate_nll(model, test, args.batch_size)
    frac = parameter_fraction(ranks, dense_total, shapes)
    print(f"{args.recipe} at {args.ratio:.0%} removed: loss {nll:.4f} (+{nll - dense_nll:.4f}), ppl {math.exp(nll):.2f}, "
          f"model x{frac:.3f} parameters, {time.time() - t0:.0f}s", flush=True)

    if args.save_dir:
        if args.low_rank:
            torch.save({"state_dict": model.state_dict(), "ranks": ranks, "ratio": args.ratio, "recipe": args.recipe}, f"{args.save_dir}.pt")
            print(f"saved factored state dict to {args.save_dir}.pt")
        else:
            model.save_pretrained(args.save_dir)
            tok.save_pretrained(args.save_dir)
            print(f"saved to {args.save_dir}")


if __name__ == "__main__":
    main()
