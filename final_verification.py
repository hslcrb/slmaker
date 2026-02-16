import torch
from tokenizer import Tokenizer
from model import NanoSLM, n_embd, n_head, n_layer
import os

def verify_system():
    print("🌌 Starting Odyssey Verification System...")
    
    # 1. Tokenizer Check
    t = Tokenizer()
    test_str = "KOREAN: 안녕하세요 | CODE: def solve(): pass | ENGLISH: Hello"
    encoded = t.encode(test_str)
    decoded = t.decode(encoded)
    print(f"Tokenizer: {'PASS' if test_str == decoded else 'FAIL'}")
    
    # 2. Weights Check
    weights_path = "data/weights/lm_head.bin"
    if os.path.exists(weights_path):
        size = os.path.getsize(weights_path)
        print(f"Weights (LM Head): {size} bytes ({'PASS' if size > 0 else 'FAIL'})")
    else:
        print("Weights (LM Head): MISSING (FAIL)")

    # 3. Model Architecture Check
    try:
        model = NanoSLM(t.vocab_size)
        print(f"Model: {n_layer} layers, {n_head} heads, {n_embd} embed dim (PASS)")
        
        # 4. KV Cache Check
        test_idx = torch.zeros((1, 1), dtype=torch.long)
        logits, _ = model.forward(test_idx, use_cache=True)
        print(f"KV Cache Forward: PASS (Logits shape: {logits.shape})")
        
    except Exception as e:
        print(f"Model/KV Cache Error: {e}")

if __name__ == "__main__":
    verify_system()
