# 🛡️ SNS 혐오·차별 발언 자동 탐지 및 등급 분류 시스템

> 2026 홍익대학교 IT시스템종합설계 · 지도교수 임종태 교수님

한 문장을 입력하면 **혐오 종류(성별·인종/출신·연령·기타)** 와 **심각도(정상·공격적·약한 혐오·강한 혐오)** 를 동시에 판별하는 한국어 혐오 표현 탐지 시스템입니다.
수업에서 배운 **ANN · CNN · RNN** 을 직접 구현해 비교하고, 앙상블과 사전학습 모델(KLUE-RoBERTa)까지 비교한 뒤 Streamlit 웹 데모로 완성했습니다.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/KLUE--BERT-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)

```
입력: "여자가 무슨 운전이야"  →  카테고리: 성별 혐오  ·  심각도: 혐오(강)
```

## 📌 핵심 결과

| 모델 | 카테고리 평균 F1 | 심각도 F1 | **종합 F1** |
|------|:---:|:---:|:---:|
| ANN | 0.559 | 0.563 | 0.561 |
| TextCNN | 0.517 | 0.566 | 0.541 |
| BiLSTM + Attention | 0.577 | 0.565 | 0.571 |
| **Weighted Ensemble** (ANN+CNN+BiLSTM) | – | – | **0.582** |
| KLUE-RoBERTa (참고용 비교) | – | – | 0.620 |

- 직접 구현한 세 모델 중 **BiLSTM** 이 가장 우수 (단어 순서·문맥 반영)
- 가중 투표 앙상블로 **4개 카테고리와 심각도 모든 항목에서 일관되게 향상** (0.571 → 0.582)
- 최신 사전학습 모델과의 격차는 약 3.8%p — 직접 구현한 모델로도 경쟁력 있는 수준 확인

## 🗂️ 데이터

- **K-HATERS** (한국어 뉴스 댓글 192,158건, Park et al., EMNLP 2023 Findings)
- 심각도 4단계 + 타깃 카테고리를 함께 제공하는 데이터셋이라 멀티태스크 학습에 사용
- 자동 라벨링으로 데이터를 늘리는 방안(K-MHaS, UnSmile)도 검토했지만, 부정확한 라벨이 오류를 전파하는 문제 때문에 **사람이 직접 라벨링한 K-HATERS만** 사용

**전처리**: NFC 정규화 → 익명화 토큰·URL·멘션·HTML 제거 → 자모 반복 축소(ㅋㅋㅋㅋ → ㅋㅋ) → 공백 정규화

## 🧠 모델 설계

```
입력 텍스트
   ↓
KLUE-BERT 사전학습 임베딩 (768 → 256차원 투영)
   ↓
[ ANN | TextCNN | BiLSTM + Self-Attention ]   ← 비교 대상 인코더
   ↓
멀티태스크 출력: 카테고리 4개 (sigmoid, multi-label) + 심각도 4단계 (softmax)
```

- **멀티태스크 라벨 설계**: 처음엔 카테고리×심각도 12클래스 단일 분류로 설계했지만, "여자 노인이 운전하면…"처럼 여러 혐오가 겹치는 문장을 표현할 수 없어 **카테고리별 이진 분류 + 심각도 분류**로 재설계
- **ANN**: Mean + Max Pooling → MLP
- **TextCNN**: 커널 2·3·4·5 Conv1D + BatchNorm
- **BiLSTM**: 2층 양방향 LSTM + Multi-head Self-Attention
- **학습**: 8:1:1 층화 분할, BCEWithLogits(pos_weight) + CrossEntropy(class weight, label smoothing), AdamW (임베딩 1e-5 / 나머지 1e-4), Cosine + Warmup, Early Stopping

## 💻 웹 데모 (Streamlit)

- 문장을 입력하면 심각도 4단계 확률과 카테고리별 혐오 확률을 바로 표시
- 사이드바에서 **앙상블 / 단일 모델(ANN·TextCNN·BiLSTM)** 선택
- **OpenAI API 연동 (gpt-4o-mini)**: 분류는 직접 학습한 모델이 맡고, GPT는 모델이 하기 어려운 **문제 표현 위치 추출 · 판단 근거 · 순화 표현 제안**을 담당하도록 역할 분담 (GPT의 카테고리·심각도 판단은 참고용으로 함께 표시)
  - 출력 형식을 `CATEGORY / SEVERITY / HATE_PHRASES / REASON / SOFTENED` 키로 고정해 응답을 안정적으로 파싱
  - 추출된 문제 표현은 원문에서 빨간색으로 강조하고, 순화 문장은 별도로 제시
  - 혐오 표현이 없으면 `NONE`을 반환하도록 해 정상 문장의 오탐 방지

### 실행 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- 학습된 가중치 `model_ANN_best.pt`, `model_TextCNN_best.pt`, `model_BiLSTM_best.pt` 를 같은 폴더 또는 `output/` 폴더에 두어야 합니다. (용량 문제로 저장소에는 포함하지 않았습니다)
- OpenAI API 키는 실행 후 사이드바에 입력합니다. 키를 넣지 않아도 분류 기능은 동작합니다.

## ⚠️ 한계와 향후 과제

- **데이터 불균형**: 연령(1%)·성별(2%) 혐오 데이터가 적어 해당 카테고리 F1이 낮음 (연령 0.26~0.37)
- **약한 혐오(L1_hate)** 는 공격적 표현·강한 혐오와 경계가 모호해 모든 모델이 어려워함
- 향후: 소수 클래스 데이터 증강, 다양한 출처의 데이터 추가, 사전학습 모델 본격 도입

## 📄 문서

- [최종 보고서 (PDF)](docs/report.pdf)
- [발표 자료 (PDF)](docs/presentation.pdf)

## 참고문헌

1. Park, S. et al., *K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings*, Findings of EMNLP 2023.
2. Park, S. et al., *KLUE: Korean Language Understanding Evaluation*, NeurIPS 2021 Datasets and Benchmarks.
3. Kim, Y., *Convolutional Neural Networks for Sentence Classification*, EMNLP 2014.
