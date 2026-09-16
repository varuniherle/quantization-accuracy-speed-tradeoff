# Quantization Accuracy/Speed Tradeoff — Zero-Shot Classification

A hands-on study of what happens when you quantize `facebook/bart-large-mnli`, a zero-shot text classification model, from FP32 down to INT8 — measuring accuracy, latency, throughput, and model size at each step, entirely on CPU.

## Why

Quantization is usually sold as a free lunch: smaller model, faster inference, tiny accuracy cost. This project tests that claim directly instead of assuming it, on a real (if small) benchmark, and finds that **the type of quantization matters far more than whether you quantize at all** — naive dynamic quantization actively broke this model, while a more targeted approach preserved accuracy entirely.

## Setup

- **Model:** `facebook/bart-large-mnli` (zero-shot classification via NLI entailment)
- **Task:** 4-way topic classification — `world news`, `sports`, `business`, `science and technology`
- **Dataset:** 40 hand-labeled examples, 10 per category (see `dataset.py`)
- **Hardware:** CPU only, no GPU
- **Framework:** PyTorch 2.12, Transformers 4.57, torchao

## Results

| Model | Accuracy | Mean Latency | Throughput | Size |
|---|---|---|---|---|
| FP32 (baseline) | 85.0% | 854.9 ms | 1.17 ex/sec | 1554 MB |
| Dynamic INT8 (weights + activations) | **35.0%** ⚠️ | 2078.4 ms | 0.48 ex/sec | 543 MB |
| Weight-only INT8 (weights only) | **87.5%** ✅ | 1596.4 ms | 0.63 ex/sec | 544 MB |
| Weight-only INT8, batched (size 16) | 87.5% | — | **2.09 ex/sec** | 544 MB |

## Key finding

Full dynamic INT8 quantization (quantizing both weights *and* activations, with activation ranges computed fresh on every forward pass) **collapsed accuracy to near-random** and made inference *slower*, not faster, on this model. Restricting quantization to weights only — leaving activations in FP32 — recovered full accuracy while still cutting model size by ~65%.

The likely cause: transformer activations tend to have a small number of high-magnitude outlier channels. Per-tensor dynamic quantization has to stretch its scale to cover those outliers, which compresses the resolution available for the rest of the (smaller-magnitude, more informative) activation values. Across BART's 24 stacked encoder/decoder layers, that per-layer precision loss compounds badly by the time it reaches the final classification decision.

A diagnostic experiment confirmed the damage happens inside the encoder/decoder backbone itself, not at the final classification head — quantizing only the backbone (head left FP32) was just as broken as quantizing everything.

## What's in this repo

| File | Purpose |
|---|---|
| `dataset.py` | Shared 40-example labeled dataset used across all runs |
| `baseline_benchmark.ipynb` | Loads FP32 model, explains the NLI/entailment mechanism, measures baseline metrics |
| `quantize_and_benchmark.ipynb` | Applies dynamic INT8 quantization, compares against baseline, diagnoses the accuracy collapse |
| `all_quantized_models.ipynb` | Full comparison: FP32 vs. dynamic INT8 vs. weight-only INT8, plus batching experiments |
| `quantization_notes.md` | Written notes on quantization theory — the math, scale/zero-point, what's quantized (`W` vs `b`) and why |

## Running it

```bash
pip install torch transformers torchao
```

Run the notebooks in order: `baseline_benchmark.ipynb` → `quantize_and_benchmark.ipynb` → `all_quantized_models.ipynb`. Each saves its results to JSON so later notebooks can load and compare against earlier runs.

## Caveats

- 40 examples is a small test set — good for fast iteration and clear qualitative signal, not for tight statistical confidence. Treat accuracy percentages as directional, not precise.
- All benchmarks ran on CPU without AVX-512 VNNI support, which affects whether INT8 kernels actually outperform FP32 — results may differ meaningfully on other hardware.
- Label categories overlap semantically in places (e.g. macroeconomic news vs. "world news"), which is a dataset design factor, not a quantization effect — see `quantization_notes.md` for the per-class error breakdown from the baseline run.


