1. Ryzen 5 5600 vs. RTX 3070 Ti - 174M GPT (fwd + bwd)
# REPEAT ON COMPILER OPTIMIZED!!!!!!!!
Single-thread Ryzen 5 5600 @ B=4, T=128: 7 tok/s
RTX 3070 Ti @ B=4, T=128: 2116-2160 tok/s
2. RTX 3070 Ti vs. RTX 3090
3. Microbenchmarks for RMSNorm, SDPA, SoftmaxCEL
4. RTX 3090 vs PyTorch (same model, pytorch without flashattention, fp32)

2. 
RTX 3070 Ti
B=1 T=1024 - 3376 tok/s, HWM: 4.27GB
B=2, T=1024 - 66 tok/s, HWM: 7.88GB
B=4, T=1024 - OOM
B=6, T=1024 - OOM