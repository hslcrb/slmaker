import torch
import numpy as np
import os
# Default values if model.py cannot be imported or values are missing
n_embd = 2048
n_layer = 24
n_head = 16
vocab_size = 80 # Default for character-level TinyStories

# Try to get from data
try:
    with open('data/tinystories.txt', 'r', encoding='utf-8') as f:
        text = f.read(100000) # Check first 100k chars for vocab
    from tokenizer import Tokenizer
    t = Tokenizer(text)
    vocab_size = t.vocab_size
    print(f"Detected Vocab Size: {vocab_size}")
except Exception as e:
    print(f"Using default vocab_size: {vocab_size} due to {e}")

# Create data/weights directory if not exists
os.makedirs('data/weights', exist_ok=True)

def create_mmap_weight(name, shape):
    path = f'data/weights/{name}.bin'
    if os.path.exists(path):
        print(f"Skipping {path} (already exists)")
        return path
    
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
