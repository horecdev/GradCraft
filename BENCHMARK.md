## Benchmarks of GradCraft Autograd Engine

All benchmarks used single-precision floating-point (`FP32`). PyTorch baselines were evaluated in eager-mode with `FlashAttention` disabled.

### Hardware
* **GPUs:** NVIDIA GeForce RTX 3090 (24GB VRAM), NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
* **CPU:** AMD Ryzen 5 5600 (6-core / 12-thread)

---

### 1. Fused CUDA Kernels

Comparison of unoptimized chains of primitive operations vs fused CUDA kernels executing forward and backward passes on an RTX 3090 ($T=1024$).

| Operation | Naive Execution (ms) | Fused Kernel (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **RMSNorm** (Fwd + Bwd) | 167.63 ms | 1.75 ms | **95.7x** |
| **SwiGLU** (Fwd + Bwd) | 3.95 ms | 1.74 ms | **2.27x** |
| **Causal Softmax** (Fwd + Bwd) | 1033.29 ms | 11.06 ms | **93.4x** |
| **Softmax Cross-Entropy** (Fwd + Bwd) | 1020.01 ms | 9.36 ms | **108.9x** |
| **AdamW Step** | 53.97 ms | 0.67 ms | **80.5x** |

*Note: Fused kernels eliminate intermediate global memory reads/writes by performing all operations within a single launch, instead of firing up a separate kernel for every primitive operation.*

---

### 2. GradCraft vs PyTorch Eager (FP32, No FlashAttention)

Training throughput and VRAM High-Water-Mark on an RTX 3090 with sequence length $T=1024$.

| Batch Size ($B$) | PyTorch (tok/s) | PyTorch VRAM | Gradcraft (tok/s) | Gradcraft HWM | % of PyTorch speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$B=1$** | 9,386 | 3.28 GB | 4,189 | 4.27 GB | **44.6%** |
| **$B=2$** | 9,908 | 5.73 GB | 4,521 | 7.88 GB | **45.6%** |
| **$B=4$** | 9,260 | 10.68 GB | 4,639 | 15.11 GB | **50.1%** |
| **$B=5$** | 8,605 | 13.18 GB | 4,830 | 18.73 GB | **56.1%** |

*Note: PyTorch baselines were evaluated in eager-mode with FlashAttention disabled. GradCraft uses a custom hand-written CUDA kernel for Causal Softmax in SDPA. This difference contributed to its performance against PyTorch eager.*
*Takeaway*
* Gradcraft achieves **~50% of PyTorch eager-mode throughput** under FP32 conditions and without FlashAttention.

---

### 3. VRAM Scaling (8GB vs 24GB VRAM) @ 180M Model

| GPU | Batch Size ($B$) | Throughput (tok/s) | Peak VRAM (HWM) | OK? |
| :--- | :--- | :--- | :--- | :--- |
| **RTX 3070 Ti (8GB)** | $B=1$ | 2,959 | 5.10 GB | OK |
| | $B=2$ | 90 | 9.58 GB | **PCIe Thrashing** (Spilled to Host) |
| | $B=4$ | — | — | **OOM** |
| **RTX 3090 (24GB)** | $B=1$ | 4,189 | 4.27 GB | OK |
| | $B=2$ | 4,521 | 7.88 GB | OK |
| | $B=4$ | 4,639 | 15.11 GB | OK |
| | $B=5$ | 4,830 | 18.73 GB | OK |

*Note: At $B=2$ on the RTX 3070 Ti, allocation exceeds VRAM limits, causing CUDA to use system RAM and giving a giant drop in throughput.*

---

### 4. CPU vs. GPU Compute Scaling @ 180M Model

Execution performance comparing host execution vs. device dispatch ($B=4, T=128$).

| Processor | Throughput (tok/s) | Speedup Factor |
| :--- |:--- | :--- |
| **Ryzen 5 5600** | 7 | 1.0x |
| **RTX 3090** | 3,900 | **557x** |

*Note: If we utilized all threads and optimistically gained a 12x speedup, the GPU would still be **46x faster**. In reality it would be even faster.*