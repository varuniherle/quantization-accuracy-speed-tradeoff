# Model Quantization

## 1. What is Quantization?

Quantization is the process of representing model values using **lower numerical precision**.

For example:

```text
FP32 → INT8
32 bits → 8 bits
```

The main goals are:

- Reduce model memory/size
- Reduce memory bandwidth
- Potentially improve inference speed
- Make models cheaper to deploy

The trade-off is that lower precision introduces **quantization error**, which can affect model accuracy.

---

## 2. What are we actually quantizing?

A neural network layer can be represented as:

```text
y = Wx + b
```

where:

- `W` = **weights**
- `x` = input/activation
- `b` = bias
- `y` = output/activation

Quantization can be applied to different parts:

```text
Weights       → quantize W
Activations   → quantize x
Bias          → usually kept at higher precision
```

So when we say **weight quantization**, we primarily mean:

```text
FP32 weights → INT8 weights
```

For example:

```text
W = 0.52
```

may become:

```text
W → INT8 representation
```

The original value is approximately reconstructed when needed.

---

## 3. Basic Quantization Formula

### Symmetric Quantization

The simplest form is:

```text
q = round(x / scale)
```

and during dequantization:

```text
x ≈ q × scale
```

Where:

- `x` = original FP32 value
- `q` = quantized integer
- `scale` = conversion factor between floating-point and integer values

For INT8, the range is approximately:

```text
-127 to +127
```

### Example

Suppose:

```text
x = 0.5
scale = 1 / 127
```

Then:

```text
q = round(0.5 / (1/127))
  = round(63.5)
  ≈ 64
```

To recover the approximate value:

```text
x ≈ 64 × (1/127)
  ≈ 0.504
```

The difference between `0.5` and `0.504` is the **quantization error**.

---

## 4. What is Scale?

Think of **scale as the spacing** between the original floating-point values represented by consecutive integers.

```text
FP32 values:

... -0.5  -0.25   0   0.25   0.5 ...

                 ↓ quantize

INT8 values:

...   -2    -1    0    1    2 ...
```

The scale tells us:

> "How much real-world floating-point value does one integer step represent?"

So:

```text
scale = spacing
```

Smaller scale → finer representation.

Larger scale → coarser representation.

---

## 5. What is Zero Point?

For **asymmetric quantization**, we use:

```text
q = round(x / scale) + zero_point
```

and:

```text
x ≈ (q - zero_point) × scale
```

The **zero point shifts the integer representation** so that the real value `0` maps to a particular integer.

Think:

```text
Scale      = spacing
Zero point = shift
```

For symmetric quantization, zero point is commonly `0`.

For asymmetric quantization, it can be a non-zero integer.

---

# 6. Why did we quantize?

Our original BART-large-MNLI model was approximately:

```text
FP32 model
~1554 MB
```

We wanted to investigate whether we could make the model:

```text
Smaller
+
Memory efficient
+
Fast enough
+
Without significantly hurting accuracy
```

We therefore tested different quantization approaches and benchmarked:

- Accuracy
- Model size
- Mean latency
- Median latency
- P95 latency
- Throughput

---

# 7. Quantization Methods We Tested

## 7.1 Full Dynamic INT8 Quantization

We used:

```python
torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```

This quantized supported `Linear` layers and dynamically handled activation quantization during inference.

Conceptually:

```text
Weights      → INT8
Activations  → dynamically quantized
```

### Result

```text
Accuracy: 35%
Model size: ~543 MB
Latency: ~2078 ms
```

Although the model became much smaller, accuracy dropped dramatically.

This showed us that **aggressively quantizing both weights and activations can damage the model's numerical behavior**.

---

## 7.2 INT8 with the Classification Head Kept in FP32

We then tried keeping the NLI classification head in FP32 while quantizing the rest of the model.

Conceptually:

```text
Transformer layers → INT8
Classification head → FP32
```

### Result

```text
Accuracy: 30%
Latency: ~2031 ms
```

This did **not** recover accuracy.

Therefore, simply keeping the final classification head in FP32 was not enough for this model.

---

## 7.3 Weight-Only INT8 Quantization

Next, we used weight-only quantization.

Conceptually:

```text
Weights      → INT8
Activations  → FP32
```

This was much less aggressive than full INT8 quantization.

### Result

```text
Accuracy: 87.5%
Model size: ~544 MB
Latency: ~1596 ms
```

Compared with FP32:

```text
FP32:
Accuracy = 85%
Size = 1554 MB
Latency = 855 ms

Weight-only INT8:
Accuracy = 87.5%
Size = 544 MB
Latency = 1596 ms
```

The important observation was:

```text
Size ↓ ~65%
Accuracy ≈ preserved
Latency ↑
```

So **smaller does not automatically mean faster**.

The actual inference runtime and hardware kernels also matter.

---

# 8. Batching for Speed

After weight-only INT8, we tried **batching**.

Instead of sending one text at a time:

```text
Text 1 → Model
Text 2 → Model
Text 3 → Model
...
```

we processed multiple texts together:

```text
Text 1 ─┐
Text 2  │
Text 3  ├──→ Model → predictions
...     │
Text 8 ─┘
```

With:

```python
batch_size = 8
```

we processed 8 input texts together.

### Why batching helps

Batching reduces repeated execution overhead and allows the hardware to process multiple inputs together.

Our results:

```text
Without batching:
0.63 examples/sec

With batching:
1.90 examples/sec
```

Accuracy remained:

```text
87.5%
```

So throughput improved by approximately **3×**.

Important distinction:

> **Batching primarily improves throughput, not necessarily the latency of one individual request.**

---

# 9. Our Overall Experiment

```text
                    Accuracy     Size       Latency

FP32                  85%       1554 MB      855 ms

Full INT8             35%        543 MB      2078 ms

INT8 + FP32 head      30%          —         2031 ms

Weight-only INT8     87.5%        544 MB     1596 ms

Weight-only INT8
+ batching           87.5%        544 MB       — 
                                              1.90 ex/sec
```

---

# 10. Key Takeaways

### Quantization

> Quantization reduces the numerical precision used to represent model values.

### Weight quantization

> `W` in `y = Wx + b` is converted from higher precision such as FP32 to lower precision such as INT8.

### Scale

> Scale determines the spacing between real floating-point values represented by integer values.

### Zero Point

> Zero point shifts the integer representation so that real zero can be represented correctly, especially in asymmetric quantization.

### Weight-only quantization

> Quantizes weights while keeping activations in floating point, which can preserve accuracy better than aggressively quantizing both weights and activations.

### Batching

> Processes multiple inputs together to improve throughput.

### Most important lesson

> **Quantization does not automatically make a model faster.**

Model size, numerical precision, CPU/GPU hardware, inference kernels, runtime, and batching all affect actual inference performance.

---

