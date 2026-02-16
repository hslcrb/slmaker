import torch
from model import NanoSLM, device, block_size, batch_size, max_iters, eval_interval, learning_rate, eval_iters
from tokenizer import Tokenizer
import os
import time

# Engine wrapper for GUI / GUI를 위한 엔진 래퍼
def engine_train(gui_app=None):
    # Extreme CPU Acceleration / 극한의 CPU 가속 설정
    torch.set_num_threads(os.cpu_count())
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    
    # 1. Data Loading / 데이터 로딩
    data_path = 'data/sample.txt'
    if not os.path.exists(data_path):
        if gui_app: gui_app.log("Error: data/sample.txt not found!")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Tokenizer Initialization / 토크나이저 초기화
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    if gui_app: gui_app.log(f"Vocab size: {vocab_size}")

    # 3. Model Initialization / 모델 초기화
    model = NanoSLM(vocab_size).to(device)
    
    # Mathematical Acceleration: Model Compilation / 수학적 가속: 모델 컴파일
    try:
        if gui_app: gui_app.log("Compiling model for insane speed... / 미친 속도를 위해 모델 컴파일 중...")
        # Optimize for CPU backend / CPU 백엔드 최적화
        model = torch.compile(model, backend='inductor')
        if gui_app: gui_app.log("Model compiled successfully. / 모델 컴파일 완료.")
    except Exception as e:
        if gui_app: gui_app.log(f"Compilation skipped (Not supported): {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Split data / 데이터 분할
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        limit = len(data_split) - block_size - 1
        if limit <= 0:
            ix = torch.zeros((batch_size,), dtype=torch.long)
        else:
            ix = torch.randint(limit, (batch_size,))
        
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # 4. Training Loop / 학습 루프
    if gui_app: gui_app.log("Starting training with Extreme Math Optimization...")
    
    start_time = time.time()
    for iter in range(max_iters):
        if gui_app and not gui_app.is_training:
            break

        # Grad Accumulation Simulation / 그래디언트 누적 시뮬레이션 (수학적 가속)
        xb, yb = get_batch('train')
        
        # Use automated mixed precision / 혼합 정밀도 사용 (CPU에서도 bfloat16 가능 시)
        # Note: Some older CPUs might not support this well, but high-end software should try.
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Update GUI every 20 iterations to reduce overhead / 오버헤드 감소를 위해 20회마다 업데이트
        if iter % 20 == 0:
            elapsed = time.time() - start_time
            speed = (iter + 1) / elapsed
            if gui_app:
                gui_app.data_queue.put({
                    'iter': iter,
                    'loss': loss.item(),
                    'speed': speed
                })

        if iter % eval_interval == 0 and gui_app:
            gui_app.log(f"Step {iter}: Loss {loss.item():.4f}")

    if gui_app:
        gui_app.log("Training session complete. / 학습 세션 종료.")
        torch.save(model.state_dict(), 'nano_slm.pth')
        gui_app.log("Model saved to nano_slm.pth")

if __name__ == "__main__":
    engine_train()
