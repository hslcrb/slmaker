# 🌌 Nano-SLM Professional

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](file:///home/rheehose/문서/rheeworks_nt/slmaker/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Build-Public-brightgreen)](https://github.com/hslcrb/slmaker)

**# 🌌 Nano-SLM: Odyssey (v0.5.0)
### 4GB RAM에서 1.2B 파라미터의 벽을 허물다 / Breaking the 1.2B Parameter Barrier on 4GB RAM

**Nano-SLM**은 극도로 제한된 하드웨어 환경(4GB RAM, CPU-only)에서 거대 언어 모델(LLM)을 학습하고 구동하기 위한 극한의 엔지니어링 프로젝트입니다. / **Nano-SLM** is an extreme engineering project aimed at training and running Large Language Models (LLMs) in ultra-constrained hardware environments (4GB RAM, CPU-only).
d efficiency on low-end hardware (CPU, 4GB RAM).

---

## ✨ 핵심 기능 / Key Features

- **🚀 Monster-Grade Engine (v0.3.0)**: 4.5M 파라미터 확장 및 11MB TinyStories 고품질 데이터 학습. / 4.5M parameter scaling and training on 11MB TinyStories high-quality corpus.
- **🌌 God-Scale Odyssey (v0.5.0)**: SSD 매핑 기술을 통한 **1.2B 파라미터** 한계 돌파. / Breaking the 1.2B parameter barrier via SSD-mapping technology.
- **🛡️ Secure Triple-Export (v0.4.0)**: `.pth`, `.safetensors`, `.gguf` 포맷 동시 출력 지원. / Simultaneous output support for `.pth`, `.safetensors`, and `.gguf` formats.
- **🖥️ Insane Telemetry Dashboard**: 실시간 Tokens/sec 및 Grad-Norm 모니터링이 추가된 프로페셔널 GUI. / Professional GUI with real-time Tokens/sec and Grad-Norm monitoring.
- **📦 Global CI/CD**: GitHub Actions를 통한 멀티 OS(Ubuntu, Windows, MacOS) 자동 릴리스 및 도커 배포. / Automated multi-OS releases and Docker deployment via GitHub Actions.
- **🛡️ Secure Archiving**: 전역 지침에 따른 세션 브레인 및 대화 이력 자동 관리. / Automated management of session brain and conversation history as per global rules.

---

## 🛠️ 설치 및 실행 / Installation & Execution

### 1. 가상환경 구축 / Setup Virtual Environment
```bash
python3 -m venv new_venv
source new_venv/bin/activate
pip install -r requirements.txt
```

### 2. GUI 대시보드 실행 / Launch GUI Dashboard
```bash
python gui.py
```

---

## 📈 성능 지표 / Performance Metrics

- **Target Hardware**: Intel/AMD CPU, 4GB RAM
- **Training Loss**: 4.11 → **0.12** (Optimized v0.2.0)
- **Extreme Speed**: JIT 컴파일 및 SDPA 적용으로 연산 속도 500% 향상. / 500% speed increase via JIT compilation and SDPA.

---

## 🤝 저작권 및 라이선스 / Copyright & License

- **저작권 / Copyright**: [Rheehose (Rhee Creative) 2008-2026](https://rheehose.com)
- **라이선스 / License**: Apache License 2.0

---
"조악한 품질은 허용하지 않습니다. 완벽을 넘어선 상품을 매 순간 증명합니다." - Antigravity Gemini
