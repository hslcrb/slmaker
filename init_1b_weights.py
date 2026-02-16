import torch
import numpy as np
import os
# Default values if model.py cannot be imported or values are missing
n_embd = 2048
n_layer = 24
n_head = 16
vocab_size = 257 # Fixed for Byte-level + EOT

print(f"Universal Vocab Size: {vocab_size}")

# Create data/weights directory if not exists
os.makedirs('data/weights', exist_ok=True)

def create_mmap_weight(name, shape):
    path = f'data/weights/{name}.bin'
    # Always overwrite for clean upgrade / 클린 업그레이드를 위해 항상 덮어쓰기
    if os.path.exists(path):
        os.remove(path)
    
    print(f"Creating {path} with shape {shape} (~{np.prod(shape)*4/1e6:.2f} MB)")
    # Initialize on disk to avoid RAM usage
    mmap = np.memmap(path, dtype='float32', mode='w+', shape=shape)
    # Fill with small random values sector by sector if possible, 
    # but for simplicity, we use zeros or very small random
    mmap[:] = np.random.randn(*shape).astype('float32') * 0.02
    mmap.flush()
    return path

if __name__ == "__main__":
    # We need to scan the model and create files for all MmapLinear layers
    # For simplicity, let's manually define the needed tensors based on the 1B config
    
    # 1. LM Head
    create_mmap_weight('lm_head', (vocab_size, n_embd))
    
    # 2. Blocks
    for i in range(n_layer):
        # Attention c_attn: 3 * n_embd
        create_mmap_weight(f'block_{i}_attn_c_attn', (3 * n_embd, n_embd))
        # Attention c_proj: n_embd
        create_mmap_weight(f'block_{i}_attn_c_proj', (n_embd, n_embd))
        # MLP c_fc: 4 * n_embd
        create_mmap_weight(f'block_{i}_mlp_c_fc', (4 * n_embd, n_embd))
        # MLP c_proj: n_embd
        create_mmap_weight(f'block_{i}_mlp_c_proj', (n_embd, 4 * n_embd))

    print("All weight files created on SSD. 1B Model is ready to Odyssey.")
