import torch
from model import NanoSLM, device, block_size, batch_size, max_iters, eval_interval, learning_rate, eval_iters
from tokenizer import Tokenizer
import os
import time
import numpy as np

# 1. Efficient Data Loader with mmap / mmap을 이용한 효율적인 데이터 로더
class StreamingDataLoader:
    def __init__(self, file_path, tokenizer, split='train'):
        self.file_path = file_path
        self.tokenizer = tokenizer
        # In a real scenario, we'd use np.memmap for tokenized data.
        # For this demonstration, we read the text and cache integer tokens.
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.data = data[:n] if split == 'train' else data[n:]
        
    def get_batch(self):
        limit = len(self.data) - block_size - 1
        if limit <= 0:
            ix = torch.zeros((batch_size,), dtype=torch.long)
        else:
            ix = torch.randint(limit, (batch_size,))
        x = torch.stack([self.data[i:i+block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

def engine_train(app=None):
    torch.set_num_threads(os.cpu_count())
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    # Path selection for Insane Upgrade / 고도화된 경로 선택
    data_path = 'data/polyglot_coder.txt' if os.path.exists('data/polyglot_coder.txt') else 'data/tinystories.txt'
    if app: app.log(f"Loading dataset: {data_path}")

    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    train_loader = StreamingDataLoader(data_path, tokenizer, 'train')
    
    model = NanoSLM(vocab_size).to(device)
    
    # Mathematical Acceleration / 수학적 가속
    try:
        model = torch.compile(model)
        if app: app.log("Model JIT Compiled for Insane Speed.")
    except:
        pass

    # Filter trainable parameters (LoRA only) / 학습 가능한 파라미터(LoRA 전용) 필터링
    # LoRA params are identified by 'lora' in the name
    trainable_params = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    from model import MODEL_TYPE
    if app: app.log(f"Starting slmaker v0.8.0: {MODEL_TYPE} (n_layer: {n_layer})")
    
    start_time = time.time()
    for iter in range(max_iters):
        if app and not app.is_training:
            break

        xb, yb = train_loader.get_batch()
        
        # Mixed Precision / 혼합 정밀도
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Calculate Grad Norm for Telemetry / 텔레메트리를 위한 그래디언트 노름 계산
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        if iter % 20 == 0:
            elapsed = time.time() - start_time
            speed = (iter + 1) / elapsed
            tokens_per_sec = speed * batch_size * block_size
            if app:
                app.data_queue.put({
                    'iter': iter,
                    'loss': loss.item(),
                    'speed': speed,
                    'tokens_per_sec': tokens_per_sec,
                    'grad_norm': grad_norm
                })

        if iter % eval_interval == 0 and app:
            app.log(f"Step {iter}: Loss {loss.item():.4f} | GN: {grad_norm:.2f}")

    if app:
        app.log("slmaker v0.8.0 Odyssey Training Complete. Exporting Triple Formats...")
        
        # 1. Standard PyTorch .pth
        torch.save(model.state_dict(), 'slmaker_odyssey_v8.pth')
        
        # 2. Secure Safetensors
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), 'slmaker_odyssey_v8.safetensors')
            app.log("Exported: slmaker_odyssey_v8.safetensors")
        except Exception as e:
            app.log(f"Safetensors Export Failed: {e}")

        # 3. CPU-Optimized GGUF
        try:
            from gguf import GGUFWriter
            import numpy as np
            writer = GGUFWriter("slmaker_odyssey_v8.gguf", "slmaker-odyssey-v8")
            # Map tensors to GGUF format
            state_dict = model.state_dict()
            for name, tensor in state_dict.items():
                arr = tensor.detach().cpu().float().numpy()
                writer.add_tensor(name, arr)
            
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            app.log("Exported: slmaker_odyssey_v8.gguf")
        except Exception as e:
            app.log(f"GGUF Export Failed: {e}")

        app.log("slmaker v0.8.0 All Formats Exported Successfully.")
        
        # Auto-Healing: Chain inference if it was requested / 자가 치유: 요청된 경우 추론 연결
        if hasattr(app, '_chain_inference') and app._chain_inference:
            prompt, max_tokens = app._chain_inference
            app._chain_inference = None # Clear
            app.log(f"Auto-Heal: Resuming inference for '{prompt[:20]}...'")
            threading.Thread(target=app.run_inference, args=(prompt, max_tokens), daemon=True).start()

def engine_inference(prompt, max_tokens=100, app=None):
    """
    Standalone inference engine for Odyssey v1.0 / 오디세이 v1.0 독립 추론 엔진
    """
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load model / 모델 로드
    model = NanoSLM(vocab_size).to(device)
    
    # Check for weights / 가중치 확인
    weight_path = 'slmaker_odyssey_v8.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        if app: app.log("External weights loaded for inference.")
    else:
        if app: app.log("No saved weights found. Using initialized weights for demonstration.")

    model.eval()
    
    # Process prompt / 프롬프트 처리
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    if app: app.log(f"Generating legacy for: '{prompt}'...")
    
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_tokens)
    
    result = tokenizer.decode(generated_ids[0].tolist())
    
    if app:
        # For GUI/CLI real-time display / GUI/CLI 실시간 표시를 위해 데이터 큐 활용 가능
        if hasattr(app, 'inference_result_var'):
            app.inference_result_var.set(result)
        app.log("Generation Complete.")
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        prompt = "Once upon a time" if len(sys.argv) < 3 else sys.argv[2]
        print(f"\n--- Odyssey Generation ---\nPrompt: {prompt}\nResult: {engine_inference(prompt)}\n")
    else:
        engine_train()
