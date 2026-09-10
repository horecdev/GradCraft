1. Ryzen 5 5600 vs RTX 3090 B=4, T=128
RTX3090 @ B=4, T=128 - 3900 tok/s
Ryzen 5 5600 @ B=4, T=128 - 7 tok/s (single threaded)

2. RTX 3070 Ti vs RTX 3090 in training
RTX 3070 Ti
B=1 T=1024 - 3376 tok/s, HWM: 4.27GB
B=2, T=1024 - 66 tok/s, HWM: 7.88GB (spilled to OS)
B=4, T=1024 - OOM
B=6, T=1024 - OOM
RTX 3090
B=1 T=1024 - 4189 tok, HWM: 4,27GB
B=2, T=1024 - 4521 tok/s, HWM: 7.88GB
B=4, T=1024 - 4639 tok/s, HWM: 15.11GB
B=5, T=1024 - 4830 tok/s, HWM: 18.73GB

3. Microbenchmarks @ RTX 3090
RMSNorm Naive (Fwd+Bwd): 167.632 ms
RMSNorm Fast (Fwd+Bwd): 1.75059 ms
SCEL Naive (Fwd+Bwd): 1020.01 ms
SCEL Fast (Fwd+Bwd): 9.35781 ms
AdamW Step Naive: 53.9721 ms
AdamW Step Fast: 0.672704 ms
SwiGLU Naive (Fwd+Bwd): 3.9531 ms
SwiGLU Fast (Fwd+Bwd): 1.73768 ms
Causal Softmax Naive (Fwd+Bwd): 1033.29 ms
Causal Softmax Fast (Fwd+Bwd): 11.056 ms

4. Pytorch vs Gradcraft on RTX 3090 (pytorch in fp32, no flashattention)
PyTorch @ RTX 3090
B=1, T=1024 - 9386 tok/s, HWM: 3.28GB
B=2, T=1024 - 9908 tok/s, HWM: 5.73GB
B=4, T=1024 - 9260 tok/s, HWM: 10.68GB
B=5, T=1024 - 8605 tok/s, HWM: 13.18GB
Gradcraft @ RTX 3090
B=1 T=1024 - 4189 tok, HWM: 4.27GB
B=2, T=1024 - 4521 tok/s, HWM: 7.88GB
B=4, T=1024 - 4639 tok/s, HWM: 15.11GB
B=5, T=1024 - 4830 tok/s, HWM: 18.73GB