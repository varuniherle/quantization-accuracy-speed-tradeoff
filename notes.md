# Quantization Notes — Zero-Shot Classification (bart-large-mnli)

## 1. The core formula

Quantization maps a real-valued (FP32) number to a low-precision integer using a **linear mapping**:

```
q = round(r / scale + zero_point)          # FP32 -> INT8
r ≈ scale × (q - zero_point)               # INT8 -> FP32 (dequantize)
```

- `r` = the real (FP32) value
- `q` = the quantized (INT8) value, restricted to a small range (e.g. -128 to 127)
- `scale` = a positive float; how much real-number distance one integer "step" represents
- `zero_point` = the integer that represents real value `0.0` exactly

This is **affine quantization**: a straight-line (scale + shift) mapping from a large continuous range down to a small fixed set of integers. Every INT8 weight is really just "some integer, times a per-tensor scale, shifted by a zero point."

## 2. What are we actually quantizing? (`y = wx + c`)

Every linear layer in the model computes `y = Wx + b` (matrix form of `y = wx + c`):

- **`W` (weights)** — quantized. This is the large parameter matrix, and it's what all "INT8 quantization" in our experiments actually targets. It's fixed after training, so it can be quantized once, offline, with no runtime cost.
- **`x` (activations/inputs)** — quantized *only* in "dynamic quantization" mode, and only at runtime (see Section 5). Left as FP32 in "weight-only" mode.
- **`b` (bias)** — **never quantized** in either method we tested. Biases are added *after* the matmul, are tiny in number compared to `W` (one value per output neuron vs. thousands of weights), and are numerically sensitive since they directly shift the output — so PyTorch and torchao both keep bias in FP32 by default. Quantizing it would save almost no memory and risks hurting accuracy for free.

**Bottom line: quantization is fundamentally about shrinking `W`. `b` is left alone, and `x` is the variable that determines *which kind* of quantization you're doing.**

## 3. Why do this at all?

- **Size**: INT8 is 1/4 the bytes of FP32. Our FP32 model was 1554 MB; our INT8 weight-quantized models landed around 543–546 MB — roughly a 65% reduction (not the full 75%/4x, because embeddings and layer norms stay FP32, only `nn.Linear` weights got converted).
- **Speed (in theory)**: CPUs have fast vectorized INT8 instructions, so INT8 matrix multiplies *can* be faster than FP32 ones.
- **Memory bandwidth**: smaller weights means less data movement between RAM and CPU cache — often a bigger real-world win than the raw compute speedup.

Our experiments showed size reduction reliably, but **speed did not follow the "quantization = faster" assumption** — see Section 6.

## 4. Scale and zero-point, concretely

For a weight tensor `W` with values ranging from `min(W)` to `max(W)`:

```
scale = (max(W) - min(W)) / 255          # spread the FP32 range across 256 INT8 levels
zero_point = round(-min(W) / scale)      # the INT8 value that represents 0.0
```

Two important consequences we saw play out:

- **Per-tensor scale = one scale for the *whole* weight matrix.** If most values are small but a few are large outliers, the scale is stretched to cover the outliers, and the "normal" values get compressed into a tiny sliver of the 256 available integers — losing precision exactly where most of the signal is.
- **Dynamic activation quantization computes a *new* scale/zero-point for every single forward pass**, based on that batch's observed min/max. This adds real per-call overhead (compute the range, quantize, run INT8 matmul, dequantize) — which is exactly why our "Dynamic INT8" run got *slower*, not faster, than FP32.

## 5. Batching and speed

Per-example (batch size 1) latency measures how fast a single request can be answered — this matters for latency-sensitive services. **Throughput** (examples/sec) is a different question: how much total work can the machine do per second, and batching directly targets this by letting the CPU do more useful work per weight-load and better use vectorized instructions.

Our batching experiment on the weight-only INT8 model showed throughput scaling up with batch size:

| Batch size | Throughput (examples/sec) |
|---|---|
| 4  | 1.46 |
| 8  | 1.82 |
| 16 | 2.09 |

This is a separate lever from quantization entirely — you can (and should) batch a FP32 model too. We combined it with quantization here to see the compounding effect on throughput specifically, since quantization's speed story on CPU is easily muddied unless you also account for how many examples move through the model at once.

## 6. Types of quantization we tested (5-line summaries)

### FP32 baseline (reference point)
No quantization — all weights, activations, and biases in 32-bit float. Accuracy: **85.0%**. Mean latency: **854.9 ms**. Throughput: **1.17 ex/sec**. Size: **1554 MB**. This is what everything else gets compared against.

### Dynamic INT8 (`torch.quantization.quantize_dynamic`, weights + activations)
Weights quantized offline to INT8; activations quantized *on-the-fly* per forward pass using a freshly observed range each time, then immediately dequantized after the matmul. Accuracy collapsed to **35.0%** (barely above the 25% random-guess baseline for 4 classes), and it got *slower* (**2078 ms**, worse than FP32) rather than faster. The per-call activation quantize/dequantize overhead combined with per-tensor scale compressing outlier-heavy transformer activations badly hurt this deep 24-layer encoder-decoder model.

### Weight-only INT8 (`torchao.quantization.Int8WeightOnlyConfig`)
Only the weight matrix `W` is quantized to INT8; activations `x` stay in FP32 throughout — no runtime activation quantization step at all. Accuracy actually **held at 87.5%** (matching or slightly exceeding FP32, within the noise of a 40-example test set), and latency (**1596 ms**) was meaningfully better than dynamic INT8, though still slower than FP32 per-example. Size dropped by the same ~65% as dynamic INT8. This isolated the real problem: it wasn't "quantizing weights" that broke accuracy — it was **quantizing activations dynamically** on this particular architecture.

### Partial backbone-only quantization (diagnostic experiment, dynamic INT8)
Applied the same dynamic (weight+activation) INT8 quantization as above, but only to the encoder/decoder backbone, leaving the final classification head in FP32. Accuracy was **still ~30%**, just as broken as full dynamic quantization. This ruled out "the small classification head is the sensitive part" — the damage was happening earlier, inside the deep backbone's activation quantization, not at the final decision layer.

### Batched weight-only INT8
Same weight-only INT8 model as above, but examples processed in batches instead of one at a time. Accuracy unchanged (**87.5%**, since batching doesn't change *what's computed*, only how many examples are computed together) but throughput scaled up from 0.63 ex/sec (batch=1 baseline) to **2.09 ex/sec at batch=16** — demonstrating that batching and quantization are independent, stackable levers for performance.

## Key takeaway

**"Quantize the model" is not one technique — it's a family of choices, and which one you use matters more than *whether* you quantize.** For this model (`bart-large-mnli`, a deep 24-layer encoder-decoder with a classification head), naive full dynamic quantization was actively harmful on both accuracy *and* speed. Restricting quantization to weights only, and leaving activations in FP32, preserved accuracy and still delivered the memory win — showing that transformer activations (with their outlier-heavy channels) are often the fragile part of the pipeline, not the weights.
