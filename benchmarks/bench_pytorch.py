import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Strict Precision Controls (Force Standard FP32)
# =====================================================================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# =====================================================================
# 2. Model Architecture Components
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ManualCausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Explicit N^2 attention matrix without FlashAttention
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Additive lower-triangular causal mask (-inf for upper triangle)
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        
        probs = F.softmax(scores, dim=-1)
        out = probs @ v
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = ManualCausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim, eps)
        self.mlp = SwiGLUMLP(dim, hidden_dim=4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PyTorchGPT(nn.Module):
    def __init__(self, vocab_size=32768, seq_len=1024, dim=768, n_heads=12, n_layers=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return self.lm_head(x)

# =====================================================================
# 3. Benchmark Driver
# =====================================================================
def run_benchmark(batch_size: int = 1, seq_len: int = 1024, warmup_steps: int = 10, bench_steps: int = 50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- PyTorch FP32 Baseline on {torch.cuda.get_device_name(0)} ---")
    print(f"Config: B={batch_size}, T={seq_len}, Model=174M")

    model = PyTorchGPT(vocab_size=32768, seq_len=seq_len, dim=768, n_heads=12, n_layers=16).to(device)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    Y = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    # Warmup
    for _ in range(warmup_steps):
        logits = model(X)
        loss = loss_fn(logits.view(-1, 32768), Y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measured Run
    start_time = time.perf_counter()
    for _ in range(bench_steps):
        logits = model(X)
        loss = loss_fn(logits.view(-1, 32768), Y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time_s = end_time - start_time
    avg_ms_per_step = (total_time_s / bench_steps) * 1000.0
    total_tokens = batch_size * seq_len * bench_steps
    tok_per_sec = total_tokens / total_time_s
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"Avg Time per Step: {avg_ms_per_step:.2f} ms")
    print(f"Throughput:        {int(tok_per_sec)} tok/s")
    print(f"Peak VRAM (HWM):   {peak_vram_gb:.2f} GB\n")

if __name__ == "__main__":
    #run_benchmark(batch_size=1, seq_len=1024)
    run_benchmark(batch_size=1, seq_len=1024)