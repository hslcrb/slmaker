import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# General Hyperparameters / 일반 하이퍼파라미터
batch_size = 1 # Keep it 1 for 4GB RAM / 4GB RAM 유지를 위해 1로 고정
block_size = 64 # Context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
dropout = 0.1

# Model Lineup Selection / 모델 라인업 선택
# 'Monster': 4.5M Lite (RAM-only target)
# 'Odyssey': 1.2B Pro (SSD-mapped, LoRA target)
MODEL_TYPE = 'Odyssey' 

if MODEL_TYPE == 'Monster':
    # 🚀 Monster Config (v0.3.0/v1.0-Lite)
    # 4.5M parameters, extremely fast on any CPU.
    n_embd = 384
    n_head = 6
    n_layer = 6
else:
    # 🌌 Odyssey Config (v1.0-Pro)
    # 1.2B parameters, utilizes SSD-mapping and RoPE/RMSNorm.
    n_embd = 2048 
    n_head = 16   
    n_layer = 24  

import numpy as np
import os

class MmapLinear(nn.Module):
    """
    Linear layer that reads weights from an SSD-mapped file to save RAM.
    Supports RAM caching for high-frequency layers and prefetching hints.
    """
    def __init__(self, in_features, out_features, mmap_path=None, cache_to_ram=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mmap_path = mmap_path
        
        if mmap_path and os.path.exists(mmap_path):
            mmap_data = np.memmap(mmap_path, dtype='float32', mode='r', shape=(out_features, in_features))
            if cache_to_ram:
                # Force load into RAM for performance / 성능을 위해 RAM으로 강제 로드
                self.weight = nn.Parameter(torch.from_numpy(np.array(mmap_data)), requires_grad=False)
            else:
                self.weight = nn.Parameter(torch.from_numpy(mmap_data), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA Adapters - Scaled for v0.6.0 Odyssey Propulsion
        self.lora_rank = 16 # Increased from 4
        self.lora_A = nn.Parameter(torch.randn(in_features, self.lora_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.lora_rank, out_features))

    def prefetch(self):
        """Hint the OS to load the weight shard into page cache / OS에 페이지 캐싱 힌트 제공"""
        if hasattr(self.weight.data, 'storage') and hasattr(self.weight.data.storage(), 'filename'):
            # This is a bit low-level, a simple sum or touch often works as a prefetch
            _ = self.weight.data[0, 0].item()

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
    # freqs_cis: (T, hs//2) -> (1, 1, T, hs//2) for broadcasting / 브로드캐스팅을 위해 (1, 1, T, hs//2)로 변환
    freqs_cis = freqs_cis.view(1, 1, xq_.size(2), xq_.size(3))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 3. Optimized Attention with SDPA & KV Cache / SDPA 및 KV 캐시가 적용된 최적화된 어텐션
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
        
        # KV Cache containers / KV 캐시 컨테이너
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis, use_cache=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply RoPE / RoPE 적용
        q, k = apply_rotary_emb(q, k, freqs_cis)

        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k = k.detach()
            self.cache_v = v.detach()

        # Scaled Dot-Product Attention / 고밀도 수학적 어텐션 연산
        y = F.scaled_dot_product_attention(q, k, v, is_causal=not use_cache)
        
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

    def forward(self, x, freqs_cis, use_cache=False):
        x = x + self.attn(self.rms_1(x), freqs_cis, use_cache=use_cache)
        x = x + self.mlp(self.rms_2(x))
        return x

class NanoSLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(i) for i in range(n_layer)])
        self.rms_f = RMSNorm(n_embd)
        # RAM Caching for LM Head (High usage) / 고부하 LM 헤드를 위해 RAM 캐싱 적용
        self.lm_head = MmapLinear(n_embd, vocab_size, mmap_path="data/weights/lm_head.bin", cache_to_ram=True)
        
        # Precompute RoPE frequencies / RoPE 주파수 사전 계산
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, block_size * 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _reset_cache(self):
        for block in self.blocks:
            block.attn.cache_k = None
            block.attn.cache_v = None

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.shape
        # If use_cache, we only process the last token / 캐시 사용 시 마지막 토큰만 처리
        if use_cache:
            # We need the correct position for RoPE / RoPE를 위한 정확한 위치 정보 필요
            # Assuming cache is populated up to curr_pos
            curr_pos = self.blocks[0].attn.cache_k.size(2) if self.blocks[0].attn.cache_k is not None else 0
            x = self.token_embedding_table(idx[:, -1:])
            freqs_cis = self.freqs_cis[curr_pos : curr_pos + 1].to(idx.device)
        else:
            x = self.token_embedding_table(idx)
            freqs_cis = self.freqs_cis[:T].to(idx.device)
        
        for i, block in enumerate(self.blocks):
            # Propulsion: Prefetch next block weights / 추진력: 다음 블록 미리 불러오기
            if i + 1 < len(self.blocks):
                next_block = self.blocks[i+1]
                next_block.attn.c_attn.prefetch()
                next_block.attn.c_proj.prefetch()
                next_block.mlp.c_fc.prefetch()
                next_block.mlp.c_proj.prefetch()
            
            x = block(x, freqs_cis, use_cache=use_cache)
            
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
        self._reset_cache()
        # Prefill cache with prompt / 프롬프트로 캐시 채우기
        logits, _ = self(idx, use_cache=True) 
        
        for _ in range(max_new_tokens):
            logits, _ = self(idx, use_cache=True)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == 256: # EOT
                break
        self._reset_cache()
        return idx
