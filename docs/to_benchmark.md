1. CPU vs GPU on a T=128. Small model, big batch size
2. Fused kernel micro-benchmarks (RMSNorm, SoftmaxCEL, fast SDPA) - RTX3070Ti
3. 3090 vs 3070Ti in fast training at B=1, B=2, B=4, B=6 (show RAM spill)
4. Lock pytoch in fp32 and disable FlashAttention. Compare TRAIN with B=1, B=6
If youre in range of 70%, its absolute GOLD