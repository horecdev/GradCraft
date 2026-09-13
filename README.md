# GradCraft Autograd Engine

A custom, 12,000-line C++ Deep Learning framework built entirely from scratch. It doesn't rely on any other framework.

`GradCraft` consists of an autograd engine, memory pools (for both `CPU` and `CUDA`), tons of math and algos (tons is an understatement honestly), and handwritten kernels. 

It reaches **45-50%** eager `PyTorch` speed while training a 180 000 000 param GPT under the same conditions (no `FlashAttention`, `fp32`).  
More benchmarks are in [BENCHMARK.md](BENCHMARK.md)

## Proof
To prove the math holds up I trained a 180M param LLM called **MALLMOC** (LLM + MALLOC = MALLMOC) in `GradCraft` on an RTX 3090.

`PROMPT:`
```cpp
// Here is a function to reverse a string:
std::string reverse_string(const std::string& s) {
```

`PROMPT + ANSWER:`
```cpp

```

As you can see, the clanker correctly reversed the string. He is only pre-trained, so you cannot prompt him directly unfortunately.

## Under the Hood

Building `GradCraft` was brutal but rewarding. It's built for hardware efficiency. More in-detail architecture is in [ARCHITECTURE.md](ARCHITECTURE.md)

Really shortly:
* GradCraft is lazily-evaluated. This means you first build the graph, and then call `.realize()`.
* After doing so, you can call `.backward()` to get all gradients computed (if `Tensors` require them).
* Core classes are `Storage`, `TensorState`, `Tensor` and `Node` to build the graph and dispatch calculations.
* It works both on the `CPU` and `CUDA`.
* There are custom memory pools so that no repeated `cudaMalloc` or `cudaFree` is called during a training loop.
* Dispatchers route math to either `CPU` or `CUDA` so you can switch devices with just `.to(Device)`.
* Fused handwritten `CUDA` kernels to speed up bottlenecks (speedups from 2x to 100x)
* DL frontend such as `Optimizer`, `Module<T>` or `Parameter<T>`.
* Checkpointing training runs
* A batched BPE tokenizer with multi-threading
* ... and many more.

## Documentation

As mentioned, there are two more files.
* [ARCHITECTURE.md](ARCHITECTURE.md): A technical deepdive into how the architecture parts interact (memory, core classes)
* [BENCHMARK.md](BENCHMARK.md): How fused kernels obliterate naive ones, comparison to PyTorch, etc.

## Presequities

* **Operating System:** Windows (MSVC host compiler required for NVCC)
* **Compiler:** Visual Studio 2022 (v17.5+) with C++23 support enabled
* **CUDA Toolkit:** 12.0+ (Tested on RTX 3090 / Compute Capability 8.6)
* **Dependencies:** `vcpkg` package manager with `openblas` and `openmp` installed

Install via vcpkg
```cmd
vcpkg install openblas openmp
```

## Quick Start
To build `train` and `tokenizer`:
1. Configure with CMake:
Open the Developer Command Prompt for VS 2022 and run:

```cmd
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/your/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

2. Build the engine and executables
```cmd
cmake --build . --config Release --target train
cmake --build . --config Release --target tokenizer
```

3. Run it!!!
```cmd
.\Release\train.exe
```
