# 🇰🇷 Korean LLM Advanced v3

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Model Size](https://img.shields.io/badge/Model-1.09B%20Parameters-orange)](#모델-사양)
[![VRAM](https://img.shields.io/badge/VRAM%20Usage-9GB-red)](#최적화-사양)

<div align="center">

**한국어 특화 대규모 언어모델 - 풀스크래치 구현 및 양자화 적용**

‘‘’
## 🚀 프로젝트 개발기
처음에는 기업들의 무료 한도가 빡세지고 '바이브 코딩'을 하기에는 한도의 한계가 찾아왔습니다. Ollama를 활용해 로컬로 돌려보기도 했지만, 모델이 너무 무거워 컴퓨터가 버거워했죠. 그때 문득 '이럴 바엔 내가 직접 만들어볼까?'라는 생각이 들었습니다.
하지만 저는 중학교 2학년이었고 인공지능 모델에 대해 아는 것이라곤 모델 크기를 나타내는 'B(Billion)'라는 개념이 전부였고, 기본이 아니라 고급개념을 짤 수 없었죠.
결국 평소처럼 ChatGPT에 들어가 “나 독자적인 한국어 LLM 모델 만들래!”라는 한마디를 던지며 무모한 도전을 시작했습니다.
GPT의 도움을 받으면서도 한계는 계속 찾아왔습니다. 처음에는 그저 GPT가 준 코드 조각들을 모아 차원 오류(Dimension Error)가 나지 않기만을 바라며 열심히 돌려볼 뿐이었습니다. 데이터를 수집하고 정제하며 온종일 컴퓨터 모니터만 바라보았습니다.
그렇게 태어난 제 첫 작품(v1 이전 버전)은 위키피디아 데이터로 학습한, 고작 50M(5천만 파라미터) 크기의 극소형 모델이었습니다. 지금은 남아있지 않지만요. 비록 대화는 불가능했지만, 어느 정도 문법에 맞는 문장을 구상하는 모습을 보였습니다.
그 작은 성공이 너무 기뻐서 이때부터 본격적으로 '채팅형 모델'을 만드는 데만 몰입했습니다. 그렇게 v1의 최종 버전인 541M 크기의 모델까지 발전시켰습니다. 몇 가지 버그가 발견되었지만, 우선 버그를 해결한 뒤 곧바로 모델의 체급을 키우기로 결심했습니다.
오류들을 수정하고, 모델 크기를 2배가량 키워 드디어 1.09B(10억 9천만 파라미터) 크기의 모델을 구축했습니다. 방학 기간 내내 시간이 날 때마다 컴퓨터를 켜고 학습을 돌렸습니다.
그렇게 끝이 보이지 않던 학습이 어느덧 44,000 스텝에 도달했습니다. 설레는 마음으로 테스트를 위해 채팅창에 "안녕?"이라고 입력했습니다.

"안녕하세요! 오늘은 무엇을 도와드릴까요?"

모델이 올바른 답변을 화면에 띄운 그 순간, 말로 표현할 수 없을 만큼 기뻤습니다.
하지만 기쁨도 잠시, 다른 질문을 던지자 전혀 엉뚱한 대답을 쏟아내기 시작했습니다. AI와 함께 밤새 코드를 분석한 결과, 모델이 사용자의 지시사항을 무시해 버리는 치명적인 버그가 발생한 것이었습니다. 가슴이 아팠지만, 더 완벽한 모델을 위해 지금까지 학습한 결과물을 과감히 폐기했습니다.
낙담하지 않고 v2의 버그를 완전히 해결한 뒤, 다음 문제에 도전했습니다. 1B 체급의 모델은 VRAM을 무려 23GB나 차지하여 일반적인 환경에서 돌리기 너무 무거웠기 때문입니다. 이를 10GB 이하로 줄여보겠다는 목표를 세웠고, 마침내 v3에서 양자화에 성공했습니다.
방학이 끝났다 보니 제가 학습하기에는 어렵습니다. v2 버그 이후 학습한 건 아예 없고 시간이 날때 다시 학습해보겠습니다.
이 프로젝트는 오직 "내 손으로 직접 LLM을 만들고 싶다"는 고집 하나로 완성해 낸, 제 인생 최고의 작품입니다.
이 모델을 잘 사용하시고, 마음에 드셨다면 스타(⭐) 버튼 한 번씩 꼭 눌러주세요! 감사합니다!
‘‘’

처음부터 끝까지 한국어로 학습된 **1.09B 파라미터 LLM**으로, VRAM 최적화 기법을 적극 활용했습니다.

[📋 주요 특징](#주요-특징) • [🚀 빠른 시작](#빠른-시작) • [💾 기술 스택](#기술-스택) • [📊 버전 히스토리](#버전-히스토리)

</div>

---

## 📖 개요

**Korean LLM Advanced v3**는 한국어 자연어 처리에 최적화된 **경량 대규모 언어모델**입니다. 
제한된 GPU 메모리 환경에서도 효율적으로 학습하고 추론할 수 있도록 설계되었습니다.

### 핵심 목표
- ✅ 한국어 텍스트 생성 및 이해 능력
- ✅ VRAM 효율성 (9GB 기준)
- ✅ 빠른 학습 속도
- ✅ 쉬운 배포 및 활용

---

## 🌟 주요 특징

### 🎯 모델 구조
| 항목 | 설명 |
|------|------|
| **모델 크기** | 1.09B 파라미터 |
| **은닉층 크기** | 1,920차원 |
| **레이어 수** | 20개 |
| **어텐션 헤드** | 10개 |
| **최대 시퀀스 길이** | 2,048 토큰 |
| **어휘집 크기** | 동적 (토크나이저 기준) |

### 🔧 최적화 기법

#### 1️⃣ **BF16 자동 혼합 정밀도**
```
표준 FP32와 비교해 약 50% VRAM 절약
- 메모리 효율: ⬇️ 12GB → 6GB
- 연산 속도: ➡️ 동등 또는 향상
```

#### 2️⃣ **8비트 AdamW 옵티마이저** (bitsandbytes)
```
옵티마이저 상태 메모리 75% 감소
- 표준 AdamW: ~2.2GB (1B 모델)
- 8-bit AdamW: ~0.55GB (1B 모델)
```

#### 3️⃣ **양자화 (Quantization)** ⭐
```
모델 가중치 동적 양자화 지원
- INT8 양자화: 크기 4배 감소
- 추론 속도: 1.5~2배 향상
```

#### 4️⃣ **그래디언트 누적 (Gradient Accumulation)**
```
효과적 배치 크기 증대
- 설정: batch_size=2, accumulation_steps=8
- 효과: 배치 크기 16 효과
```

#### 5️⃣ **그래디언트 체크포인팅**
```
활성화(Activation) 메모리 감소
- 재계산 비용: ~30% 속도 저하
- 메모리 절약: 30~40%
```

---

## 💾 VRAM 사용량 비교

<div align="center">

| 버전 | 파라미터 | VRAM 사용량 | 최적화 기법 |
|------|---------|-----------|----------|
| **v1** | 541M | ~11GB | 기본 FP32 |
| **v2** | ~1.1B | ~23GB | BF16 + Gradient Checkpoint |
| **v3** | 1.09B | **~9GB** ✨ | BF16 + 8비트 옵티마이저 + 양자화 |

**v3은 v2 대비 VRAM 60% 감소, v1보다는 모델 크기 2배 확대**

</div>

---

## 🚀 빠른 시작

### 📋 사전 요구사항

```bash
Python 3.9 이상
CUDA 11.8 이상 (GPU 필수)
GPU 메모리: 최소 9GB 권장
```

### 1️⃣ 설치

```bash
# 저장소 클론
git clone https://github.com/seoan1024/korean-llm-v3.git
cd korean-llm-v3

# 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tqdm pandas matplotlib

# 양자화 지원 라이브러리 (선택)
pip install bitsandbytes
```

### 2️⃣ 데이터셋 준비

코드가 자동으로 다음 데이터셋을 다운로드합니다:
- 🔹 `nlpai-lab/kullm-v2` - 한국어 명령어 튜닝 데이터
- 🔹 `beomi/KoAlpaca-v1.1a` - 한국식 알파카 데이터셋

```bash
# 데이터셋이 자동 다운로드되므로 별도 작업 불필요
# 캐시 디렉토리: ./datasets/cache/
```

### 3️⃣ 학습 실행

```bash
# 기본 설정으로 학습 시작
python korean_llm_advanced_v3.py

# 또는 커스텀 설정으로 실행
python korean_llm_advanced_v3.py \
    --batch-size 2 \
    --max-steps 50000 \
    --learning-rate 5e-5
```

### 4️⃣ 모니터링

학습 중 자동으로 GUI 모니터링 창이 열립니다:
- 📊 실시간 손실값(Loss) 그래프
- 💬 인터랙티브 채팅 (생성 테스트)
- 📝 로그 뷰어

---

## 🏗️ 프로젝트 구조

```
korean-llm-v3/
├── korean_llm_advanced_v3.py    # 메인 학습 스크립트
├── requirements.txt              # 의존성 패키지
├── README.md                      # 이 파일
├── LICENSE                        # GPL-3.0 라이선스
│
├── checkpoints/                   # 저장된 모델 체크포인트
│   └── korean_llm_*.pth
│
├── datasets/                      # 데이터셋 캐시
│   ├── cache/                     # 다운로드된 데이터셋
│   └── datasets_manifest.json     # 메타데이터
│
└── logs/                          # 학습 로그 및 그래프
    ├── training.log               # 상세 로그
    └── loss_history.json          # 손실값 기록
```

---

## 📊 버전 히스토리

### v1 (초기 버전)
- 541M 파라미터 모델
- VRAM 사용량: ~11GB
- 기본 FP32 학습

### v2 (최적화 v1)
- 1.1B 파라미터로 확대
- VRAM 사용량: ~23GB (초기 1.2배 증가)
- **BF16 + Gradient Checkpoint** 적용

### **v3 (현재)** ⭐
- 1.09B 파라미터 (v2 수준)
- **VRAM 사용량: ~9GB** (v2 대비 60% 감소!)
- **주요 개선사항:**
  - 8비트 AdamW 옵티마이저
  - 동적 양자화 지원
  - 향상된 메모리 관리
  - 더 빠른 학습 속도

---

## 🔧 기술 스택

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|---------|------|------|
| **PyTorch** | 2.0+ | 딥러닝 프레임워크 |
| **Transformers** | 4.30+ | 토크나이저 및 유틸리티 |
| **Datasets** | 2.10+ | 한국어 데이터셋 로드 |
| **bitsandbytes** | 0.40+ | 8비트 양자화 최적화 |
| **tqdm** | 4.60+ | 진행률 표시 |

### 선택 라이브러리

| 라이브러리 | 용도 |
|---------|------|
| **matplotlib** | 손실값 그래프 시각화 |
| **tkinter** | GUI 모니터링 (내장) |
| **pandas** | 데이터 처리 |

---

## 💡 사용 예시

### 모델 로드 및 텍스트 생성

```python
import torch
from korean_llm_advanced_v3 import KoreanLLM, generate
from transformers import AutoTokenizer

# 모델 및 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoreanLLM(
    vocab_size=50000,
    dim=1920,
    n_layers=20,
    n_heads=10,
    max_seq_len=2048
).to(device)

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-tokenizer")

# 체크포인트 로드
checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 텍스트 생성
prompt = "인공지능이란"
response = generate(
    model, tokenizer, 
    prompt=prompt,
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
    device=device
)
print(f"Q: {prompt}")
print(f"A: {response}")
```

### 커스텀 학습 설정

```python
from korean_llm_advanced_v3 import TrainingConfig, main

config = TrainingConfig(
    batch_size=4,                      # 배치 크기
    accumulation_steps=4,              # 그래디언트 누적 스텝
    max_steps=100000,                  # 최대 학습 스텝
    warmup_steps=1000,                 # 워밍업 스텝
    learning_rate=3e-5,                # 학습률
    eval_interval=5000,                # 평가 간격
    use_bfloat16=True,                 # BF16 사용 여부
    resume_from_checkpoint='latest'    # 최신 체크포인트에서 재개
)

main(config)
```

---

## ❓ FAQ (자주 묻는 질문)

### Q1: 이 모델을 추론(inference)만 하려면?
**A:** 학습된 체크포인트가 있다면 다음처럼 간단히:
```python
import torch
from korean_llm_advanced_v3 import KoreanLLM, generate
from transformers import AutoTokenizer

model = KoreanLLM(...).to(device)
checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

response = generate(model, tokenizer, prompt="안녕?", max_tokens=50)
```

### Q2: 내 GPU 메모리가 9GB 미만이면?
**A:** 다음 방법들을 시도해보세요:
- 배치 크기를 `1`로 감소
- 시퀀스 길이를 `1024`로 단축
- 그래디언트 누적 단계를 `16`으로 증가
- 8비트 양자화 활성화

### Q3: 학습 중단 후 재개하려면?
**A:** 자동으로 최신 체크포인트를 감지합니다:
```python
config = TrainingConfig(
    resume_from_checkpoint='latest'  # 또는 특정 경로
)
main(config)
```

### Q4: 다른 한국어 데이터셋을 사용할 수 있나?
**A:** 네! `DatasetManager` 클래스의 `DATASETS_CONFIG`를 수정하면 됩니다:
```python
DATASETS_CONFIG = [
    {
        "name": "your-dataset/path",
        "split": "train",
        "text_keys": ["input", "output"]
    }
]
```

### Q5: 윈도우에서 실행하면 에러가 나요
**A:** `num_workers` 설정을 `0`으로 변경해보세요:
```python
loader = DataLoader(dataset, batch_size=2, num_workers=0)
```

### Q6: VRAM 사용량을 더 줄일 수 있나?
**A:** 다음 옵션을 조합해보세요:
- **메모리 효율 모드**: `use_bfloat16=True`
- **더 깊은 양자화**: INT4 (추가 라이브러리 필요)
- **LoRA 파인튜닝**: 선택적 레이어만 학습

### Q7: 생성된 텍스트 품질이 낮으면?
**A:** 다음을 확인하세요:
- 학습 스텝이 충분한가? (최소 10,000 스텝 권장)
- Learning rate 설정이 적절한가?
- 데이터셋 품질이 좋은가?
- `temperature` 파라미터 조정 (0.5~1.0 권장)

### Q8: 모델을 ONNX나 다른 형식으로 변환하려면?
**A:** PyTorch에서 ONNX로 변환 가능:
```python
import torch.onnx

dummy_input = torch.randint(0, 50000, (1, 2048)).to(device)
torch.onnx.export(
    model, dummy_input, "korean_llm.onnx",
    input_names=['input_ids'],
    output_names=['output']
)
```

### Q9: 개발자가 활발히 지원하나?
**A:** 네! 이슈나 피드백은 이메일(**seoan102410@gmail.com**)로 연락주세요! 💌

### Q10: 상용 프로젝트에 사용 가능한가?
**A:** GPL-3.0 라이선스이므로, 수정 사항을 공개해야 합니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---

## 🛠️ 트러블슈팅

### ❌ CUDA Out of Memory 에러

**증상:** `RuntimeError: CUDA out of memory`

**해결책:**
```python
# 배치 크기 감소
config.batch_size = 1

# 최대 시퀀스 길이 감소
config.max_seq_len = 1024

# 그래디언트 누적 증가
config.accumulation_steps = 16
```

### ❌ bitsandbytes 설치 실패

**해결책:**
```bash
# CUDA Toolkit 경로 명시
CUDA_HOME=/usr/local/cuda pip install bitsandbytes
```

### ❌ 데이터셋 다운로드 실패

**해결책:**
```bash
# 캐시 초기화 후 재시도
rm -rf datasets/cache/*
python korean_llm_advanced_v3.py
```

---

## 📈 성능 최적화 팁

1. **배치 크기 조정**: 너무 작으면 학습이 느리고, 너무 크면 VRAM 부족
2. **그래디언트 누적**: 효과적인 배치 크기 증대의 핵심
3. **Learning Rate 스케줄링**: Cosine Annealing으로 수렴 향상
4. **혼합 정밀도**: BF16 사용으로 속도와 메모리 동시 개선
5. **체크포인트**: 정기적으로 저장하여 학습 재개 가능

---

## 📞 연락처 및 정보

- **개발자**: seoan1024
- **이메일**: seoan102410@gmail.com
- **GitHub**: [seoan1024](https://github.com/seoan1024)

---

## 📜 라이선스

이 프로젝트는 **GPL-3.0 라이선스** 하에 배포됩니다.

```
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (C) 2024 seoan1024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

📖 전체 라이선스: [LICENSE](./LICENSE)

---

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트는 언제나 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 감사의 말

- 🎯 **한국어 LLM 커뮤니티** - 귀중한 피드백과 기여
- 📚 **Hugging Face** - Transformers & Datasets 라이브러리
- 🔧 **bitsandbytes** - 양자화 및 최적화 솔루션
- 🚀 **PyTorch** - 오픈소스 딥러닝 프레임워크

---

## 📚 참고 자료

### 한국어 NLP
- [nlpai-lab/KULLM](https://github.com/nlpai-lab/KULLM)
- [beomi/KoAlpaca](https://github.com/beomi/KoAlpaca)

### 최적화 기법
- [bitsandbytes: 8-bit Optimization](https://github.com/TimDettmers/bitsandbytes)
- [Gradient Checkpointing in PyTorch](https://pytorch.org/docs/stable/checkpoint.html)

### 대규모 언어모델
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://arxiv.org/abs/1901.08810)

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 별⭐을 눌러주세요!**

[⬆ 위로](#korean-llm-advanced-v3)

</div>
