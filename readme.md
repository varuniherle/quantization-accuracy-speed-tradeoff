# Model Quantization & Inference Optimization

A hands-on experiment exploring **model quantization, memory reduction, accuracy impact, and inference performance** using `facebook/bart-large-mnli` for zero-shot classification.

## Model

**Model:** `facebook/bart-large-mnli`

**Task:** Zero-shot text classification

**Dataset:** 40 examples across 4 categories:

- World News
- Sports
- Business
- Science & Technology

---

## Quantization Basics

A neural network layer can be represented as:

```text
y = Wx + b
```

Where:

- `W` = weights
- `x` = input / activation
- `b` = bias
- `y` = output

Quantization can be applied to the **weights (`W`)**, activations (`x`), or both.

### Symmetric Quantization

```text
q = round(x / scale)

x ≈ q × scale
```

### Asymmetric Quantization

```text
q = round(x / scale) + zero_point

x ≈ (q - zero_point) × scale
```

### Scale

**Scale = spacing.**

It determines how much real-valued information is represented by one integer step.

### Zero Point

**Zero point = shift.**

It determines which integer value represents real zero.

---

## Quantization Methods Tested

### 1. Full Dynamic INT8

Quantized supported `Linear` layers using dynamic INT8 quantization.

```text
Weights      → INT8
Activations  → dynamically quantized
```

**Result:** Large reduction in model size, but significant accuracy degradation.

---

### 2. INT8 + FP32 Classification Head

The transformer layers were quantized while attempting to keep the final NLI classification head in FP32.

```text
Transformer layers → INT8
Classification head → FP32
```

**Result:** Accuracy did not recover, showing that keeping only the final head in FP32 was not sufficient.

---

### 3. Weight-Only INT8

Only the model weights were quantized.

```text
Weights      → INT8
Activations  → FP32
```

This was considerably less aggressive and preserved the model's predictive behavior.

**Result:** Accuracy was maintained while significantly reducing model size.

---

## Results

| Model | Accuracy | Model Size | Mean Latency | Throughput |
|---|---:|---:|---:|---:|
| FP32 | 85.0% | 1554 MB | 855 ms | 1.17 ex/s |
| Dynamic INT8 | 35.0% | 543 MB | 2078 ms | 0.48 ex/s |
| INT8 + FP32 Head | 30.0% | — | 2031 ms | — |
| Weight-only INT8 | **87.5%** | **544 MB** | 1596 ms | 0.63 ex/s |
| Weight-only INT8 + Batch | **87.5%** | **544 MB** | — | **1.90 ex/s** |

---

