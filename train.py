import torch
from model import NanoSLM, device, block_size, batch_size, max_iters, eval_interval, learning_rate, eval_iters
from tokenizer import Tokenizer
import os

# 1. Data Loading / 데이터 로딩
data_path = 'data/sample.txt'
if not os.path.exists(data_path):
    print("Error: data/sample.txt not found!")
    exit()

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Tokenizer Initialization / 토크나이저 초기화
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# 3. Model Initialization / 모델 초기화
model = NanoSLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Split data into train and validation / 학습 및 검증 데이터 분할
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    # Safety check: if data is too small for block_size, we need a different approach
    limit = len(data_split) - block_size - 1
    if limit <= 0:
        # If dataset is too small, just loop or pad (simple fallback)
        ix = torch.zeros((batch_size,), dtype=torch.long)
    else:
        ix = torch.randint(limit, (batch_size,))
    
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        # Reduce eval_iters if dataset is too small
        num_iters = min(eval_iters, len(train_data if split == 'train' else val_data) // batch_size)
        num_iters = max(1, num_iters) # At least 1
        losses = torch.zeros(num_iters)
        for k in range(num_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 4. Training Loop / 학습 루프
print("Starting training on CPU... / CPU 학습 시작...")
for iter in range(max_iters):
    # Evaluation / 모델 평가
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Training step / 학습 단계
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final generation test / 최종 생성 테스트
print("\n--- Final Generation Test / 최종 생성 테스트 ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with character index 0
generated = model.generate(context, max_new_tokens=100)[0].tolist()
print(tokenizer.decode(generated))

# Save the model / 모델 저장
torch.save(model.state_dict(), 'nano_slm.pth')
print("\nModel saved to nano_slm.pth / 모델이 nano_slm.pth에 저장되었습니다.")
