import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Mathematical Hyperparameters / 수학적 하이퍼파라미터
batch_size = 1 # Keep it 1 for 4GB RAM / 4GB RAM 유지를 위해 1로 고정
block_size = 64 # Context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
# Odyssey Config for 1B-Monster (v0.5.0)
# 이 설정은 약 12억(1.2B) 개의 파라미터를 생성하며, SSD 매핑 없이는 4GB RAM에서 구동이 불가능합니다.
# This config generates ~1.2B parameters, which cannot run on 4GB RAM without SSD-mapping.
n_embd = 2048 
n_head = 16   
n_layer = 24  
dropout = 0.1
# Total Params: ~1,250,000,000 (1.25B)

import numpy as np
import os

class MmapLinear(nn.Module):
    """
    Linear layer that reads weights from an SSD-mapped file to save RAM.
    SSD 매핑된 파일에서 가중치를 읽어 RAM을 절약하는 선형 레이어.
    """
    def __init__(self, in_features, out_features, mmap_path=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mmap_path = mmap_path
        
        if mmap_path and os.path.exists(mmap_path):
            self.weight = nn.Parameter(torch.from_numpy(np.memmap(mmap_path, dtype='float32', mode='r', shape=(out_features, in_features))), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA Adapters (The only thing that will be in RAM and trained)
        # LoRA 어댑터 (RAM에 상주하며 학습되는 유일한 부분)
        self.lora_rank = 4
        self.lora_A = nn.Parameter(torch.randn(in_features, self.lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.lora_rank, out_features))

    def forward(self, x):
        # Base weight (Frozen/Offloaded) + LoRA path (Active/RAM)
        res = F.linear(x, self.weight, self.bias)
        res += (x @ self.lora_A) @ self.lora_B
        return res

# 1. RMSNorm: Mathematically simpler and faster than LayerNorm / LayerNorm보다 수학적으로 단순하고 빠른 RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 2. RoPE (Rotary Positional Embedding): Better relative positioning / 더 나은 상대적 위치 표현력을 위한 RoPE
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(3))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 3. Optimized Attention with SDPA / SDPA를 사용한 최적화된 어텐션
class CausalSelfAttention(nn.Module):
    def __init__(self, block_idx):
        super().__init__()
        assert n_embd % n_head == 0
        path_attn = f"data/weights/block_{block_idx}_attn_c_attn.bin"
        path_proj = f"data/weights/block_{block_idx}_attn_c_proj.bin"
        
        self.c_attn = MmapLinear(n_embd, 3 * n_embd, mmap_path=path_attn)
        self.c_proj = MmapLinear(n_embd, n_embd, mmap_path=path_proj)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply RoPE / RoPE 적용
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Scaled Dot-Product Attention / 고밀도 수학적 어텐션 연산
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, block_idx):
        super().__init__()
        path_fc = f"data/weights/block_{block_idx}_mlp_c_fc.bin"
        path_proj = f"data/weights/block_{block_idx}_mlp_c_proj.bin"
        
        self.c_fc = MmapLinear(n_embd, 4 * n_embd, mmap_path=path_fc)
        self.c_proj = MmapLinear(4 * n_embd, n_embd, mmap_path=path_proj)
        self.nonlin = nn.SiLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, block_idx):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(block_idx)
        self.rms_2 = RMSNorm(n_embd)
        self.mlp = MLP(block_idx)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.rms_1(x), freqs_cis)
        x = x + self.mlp(self.rms_2(x))
        return x

class NanoSLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(i) for i in range(n_layer)])
        self.rms_f = RMSNorm(n_embd)
        self.lm_head = MmapLinear(n_embd, vocab_size, mmap_path="data/weights/lm_head.bin")
        
        # Precompute RoPE frequencies / RoPE 주파수 사전 계산
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, block_size * 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        
        # Pass freqs_cis to blocks / 블록에 RoPE 주파수 전달
        freqs_cis = self.freqs_cis[:T].to(idx.device)
        
        for block in self.blocks:
            x = block(x, freqs_cis)
            
        x = self.rms_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
