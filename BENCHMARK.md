## Benchmarks of `GradCraft` Autograd Engine

All benchmarks used single-precision floating-point (`FP32`). PyTorch baselines were evaluated in eager-mode with `FlashAttention` disabled.

### Hardware & Environment
* **OS:** Windows 10
* **GPUs:** NVIDIA GeForce RTX 3090 (24GB VRAM), NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
* **CPU:** AMD Ryzen 5 5600 (6-core / 12-thread)

---

### 1. Fused `CUDA` Kernels

Comparison of unoptimized chains of primitive operations vs fused `CUDA` kernels doing forward and backward passes on an RTX 3090.

| Operation | Naive Execution (ms) | Fused Kernel (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **`RMSNorm`** (Fwd + Bwd) | 167.63 ms | 1.75 ms | **95.7x** |
| **`SwiGLU`** (Fwd + Bwd) | 3.95 ms | 1.74 ms | **2.27x** |
| **`Causal Softmax`** (Fwd + Bwd) | 1033.29 ms | 11.06 ms | **93.4x** |
| **`Softmax Cross-Entropy`** (Fwd + Bwd) | 1020.01 ms | 9.36 ms | **108.9x** |
| **`AdamW Step`** | 53.97 ms | 0.67 ms | **80.5x** |

*Note: The speedup is **MASSIVE**. That's because a primitive `RMSNorm` `.realize()` dispatches, 9 (**NINE!!**) kernels, while a fast one fires up exactly **ONE** kernel. It reads once, not **NINE** times.
---

### 2. `GradCraft` vs `PyTorch` Eager (`FP32`, No `FlashAttention`) @ 180M Model

Training throughput and VRAM High-Water-Mark on an RTX 3090 with sequence length $T=1024$

| Batch Size ($B$) | `PyTorch` (tok/s) | `PyTorch` HWM | `GradCraft` (tok/s) | `GradCraft` HWM | % of PyTorch speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$B=1$** | 8,783 | 3.46 GB | 3,732 | 5.10 GB | **42,3%** |
| **$B=2$** | 9,198 | 6.16 GB | 4,248 | 9.52 GB | **46.2%** |
| **$B=4$** | 9,727 | 11.63 GB | 4,449 | 18.37 GB | **45.7%** |

**Takeaway:**
* `GradCraft` achieves **~45-50% of `PyTorch` eager-mode throughput** under `FP32` conditions and without `FlashAttention`.
* `GradCraft` got **_smoked_**.

---

### 3. VRAM Scaling (8GB vs 24GB VRAM) @ 180M Model

| GPU | Batch Size ($B$) | Throughput (tok/s) | Peak VRAM (HWM) | OK? |
| :--- | :--- | :--- | :--- | :--- |
| **RTX 3070 Ti (8GB)** | $B=1$ | 2,959 | 5.10 GB | OK |
| | $B=2$ | 90 | 9.52 GB | **PCIe Thrashing** (Spilled to Host) |
| | $B=4$ | — | — | **OOM** |
| **RTX 3090 (24GB)** | $B=1$ | 3,732 | 5.10 GB | OK |
| | $B=2$ | 4,248 | 9.52 GB | OK |
| | $B=4$ | 4,449 | 18.37 GB | OK |

*Note: At $B=2$ on the RTX 3070 Ti, allocation exceeds VRAM limits, causing `CUDA` to use system RAM and making throughput crash.*

---

### 4. CPU vs. GPU Compute Scaling @ 180M Model

Execution performance comparing `CPU` vs. `CUDA` dispatch ($B=4, T=128$).

| Processor | Throughput (tok/s) | Speedup Factor |
| :--- |:--- | :--- |
| **Ryzen 5 5600** | 6 | 1.0x |
| **RTX 3090** | 3,150 | **525x** |

*Note: I forgot adding OpenMP to the CPU. But who even trains on the CPU? Either way - if we utilized all threads and optimistically gained a 12x speedup, the GPU would still be **33x faster**. In reality it is even more.*
