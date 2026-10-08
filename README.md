# sinkdeflation

Sink-aware calibration statistics for activation-whitened SVD compression of language models.

Whitened SVD truncation (SVD-LLM style) picks the rank-k factors of each weight matrix that minimize the output error
under the calibration input covariance. That covariance gives every token position the same weight, so the first
position of a window of length T gets weight 1/T. In the key and value projections the first position is the attention
sink: every query reads it, and it carries most of the gradient energy of those projections. This library corrects the
statistic of the key and value projections in two ways, at the same storage and the same inference cost:

| recipe | statistic for k_proj and v_proj | parameter |
|---|---|---|
| `standard` | ordinary covariance, position 0 at weight 1/T | none |
| `reweight` | `lam * C_0 + (1 - lam) * C_rest`, position 0 at weight lam | lam = 0.3 |
| `deflate` | ordinary covariance, factors constrained to be exact on the top eigenvector of `C_0` | none |

`C_0` is the mean of `x_0 x_0^T` over calibration windows and `C_rest` the mean over the other positions. All other
matrices always use the standard statistic. Deflation is the recommended recipe: it is optimal among rank-k
factorizations that reproduce the sink exactly, and it needs no tuning.

## Install

```bash
pip install -e .            # library only (needs torch)
pip install -e ".[hf,test]" # plus transformers/datasets for the CLI and pytest for the tests
pytest                      # synthetic tests, CPU, a few seconds
```

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sinkdeflation import find_linear_layers, collect_stats, compress_model, wikitext_windows, batches_of, evaluate_nll

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()

modules = find_linear_layers(model)                                   # q, k, v, o, gate, up, down of every block
calib = batches_of(wikitext_windows(tok, "train", 65536, 1024), 2)    # 64 windows of 1,024 tokens
stats = collect_stats(model, modules, calib)                          # one pass; position 0 kept apart

ranks = compress_model(model, stats, ratio=0.2, recipe="deflate")     # 20% of every block matrix's parameters removed
test = wikitext_windows(tok, "test", 32768, 1024)
print(evaluate_nll(model, test))                                      # nats per token; exp() for perplexity
```

`compress_model(..., materialize=False)` replaces each layer by a `LowRankLinear` holding the two factors, which
realizes the memory saving; the default writes the dense product back, which is what the paper evaluates.
Pass `recipe="reweight", lam=0.3` for the mixture, or `recipe="standard"` for the baseline. For models whose
tokenizer prepends a BOS token (Llama, Gemma, TinyLlama, MiniCPM) build the windows with `bos=True`.

The same run from the command line:

```bash
python -m sinkdeflation --model Qwen/Qwen3-1.7B --ratio 0.2 --recipe deflate
python -m sinkdeflation --model meta-llama/Llama-3.1-8B --ratio 0.2 --recipe reweight --lam 0.3 --bos --save-dir llama-0.2
```

Working with the factors directly:

```python
from sinkdeflation import whitened_factors, deflated_factors, standard_statistic, rank_for_removal
st = stats["model.layers.8.self_attn.k_proj"]
W = model.get_submodule("model.layers.8.self_attn.k_proj").weight
k = rank_for_removal(*W.shape, 0.4)
A, B = deflated_factors(W, standard_statistic(st.C0, st.C_rest, st.seq_len), st.X0, k)   # W ~ A @ B, exact on the sink
```

## What to expect

Across ten models from six families (0.5B to 32B) at 20% and 40% of block parameters removed, deflation lowered the
WikiText-2 loss increase over the dense model in 18 of 20 cases, by 18% on average, and raised seven-task zero-shot
accuracy significantly on Qwen3-4B, Qwen3-32B and Llama-3.1-8B (by 1.1 to 3.3 points), with no change on Qwen3-1.7B and
Gemma-2-27B and a 1.4-point drop on Qwen3-8B at 40%. Loss and accuracy effects are only weakly related; judge a
compressed model on accuracy. Details and the composition with KFAC-SVD and LEMS are in the paper.

## Layout

```
sinkdeflation/factor.py    rank rule, statistics, whitened and deflated factorizations (float64)
sinkdeflation/stats.py     forward pre-hooks that accumulate C_0, C_rest and X0 per layer within a memory budget
sinkdeflation/compress.py  find block linear layers, apply a recipe, LowRankLinear, parameter accounting
sinkdeflation/data.py      WikiText-2 windows and the teacher-forced loss
sinkdeflation/cli.py       python -m sinkdeflation ...
tests/                     synthetic tests: exactness, optimality, hooks, factored modules
```

## License

MIT.
