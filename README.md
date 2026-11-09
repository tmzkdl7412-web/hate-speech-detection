# SNS 혐오·차별 발언 탐지 및 심각도 분류

2026 홍익대학교 IT시스템종합설계 프로젝트 (지도교수 임종태 교수님)

문장 하나를 넣으면 누구를 겨냥한 혐오인지(성별, 인종·출신, 연령, 기타)와 얼마나 심한지(정상, 공격적, 약한 혐오, 강한 혐오)를 함께 분류합니다. 수업에서 배운 ANN, CNN, RNN을 직접 구현해 비교했고, 세 모델을 묶은 앙상블과 사전학습 모델 KLUE-RoBERTa 결과도 같이 확인했습니다. 학습한 모델은 Streamlit 데모로 바로 써 볼 수 있게 만들었습니다.

```
입력: 여자가 무슨 운전이야
결과: 카테고리 성별 / 심각도 강한 혐오
```

## 결과

| 모델 | 카테고리 평균 F1 | 심각도 F1 | 종합 F1 |
| --- | :---: | :---: | :---: |
| ANN | 0.559 | 0.563 | 0.561 |
| TextCNN | 0.517 | 0.566 | 0.541 |
| BiLSTM + Attention | 0.577 | 0.565 | 0.571 |
| Weighted Ensemble (ANN + CNN + BiLSTM) | - | - | **0.582** |
| KLUE-RoBERTa (비교용) | - | - | 0.620 |

- 직접 구현한 세 모델 중에서는 단어 순서와 문맥을 반영하는 BiLSTM이 가장 좋았습니다.
- 세 모델의 예측 확률을 검증셋 성능 기준으로 가중 평균한 앙상블은 0.582로, 네 개 카테고리와 심각도 모두에서 BiLSTM 단일 모델보다 높게 나왔습니다.
- 사전학습 모델인 KLUE-RoBERTa와의 차이는 0.038이었습니다.

## 데이터

- K-HATERS: 한국어 뉴스 댓글 192,158건 (Park et al., Findings of EMNLP 2023). 심각도와 대상 카테고리 라벨이 같이 있어서 두 가지를 동시에 학습할 수 있습니다.
- 데이터를 늘리려고 K-HATERS로 학습한 분류기로 K-MHaS에 심각도 라벨을 자동으로 붙이는 방법도 생각했습니다. 하지만 자동 라벨의 오류가 그대로 학습에 들어가는 문제가 있어서, 사람이 직접 라벨링한 K-HATERS만 사용했습니다.
- 전처리: NFC 정규화, 익명화 토큰·URL·멘션·HTML 태그 제거, 자모 반복 줄이기(ㅋㅋㅋㅋ → ㅋㅋ), 공백 정리

## 모델

세 모델을 공정하게 비교하려고 입력과 출력 구조는 같게 두고 가운데 인코더만 바꿨습니다.

```
입력 텍스트
→ KLUE-BERT 사전학습 임베딩 (768차원 → 256차원)
→ 인코더: ANN / TextCNN / BiLSTM + Self-Attention
→ 출력: 카테고리 4개(각각 이진 분류) + 심각도 4단계
```

- 라벨 구성: 처음에는 카테고리와 심각도를 곱한 12개 클래스 중 하나로 분류했습니다. 그런데 "여자 노인이 운전하면 사고 난다"처럼 성별과 연령 혐오가 함께 들어 있는 문장을 하나로만 표현할 수 있어서, 카테고리마다 해당 여부를 따로 판단하고 심각도는 별도로 분류하는 구조로 바꿨습니다.
- ANN: Mean + Max Pooling 후 MLP
- TextCNN: 커널 크기 2, 3, 4, 5의 Conv1D + BatchNorm
- BiLSTM: 2층 양방향 LSTM + Multi-head Self-Attention
- 학습: 8:1:1 층화 분할, 손실은 BCEWithLogits(pos_weight) + CrossEntropy(class weight, label smoothing), AdamW(임베딩 1e-5, 나머지 1e-4), Cosine 스케줄 + Warmup, Early Stopping

## 웹 데모 (Streamlit)

- 문장을 입력하면 심각도 4단계 확률과 카테고리별 혐오 확률을 보여줍니다.
- 사이드바에서 앙상블 또는 단일 모델(ANN, TextCNN, BiLSTM)을 고를 수 있습니다.
- 분류는 직접 학습한 모델이 하고, OpenAI API(gpt-4o-mini)는 모델이 하기 어려운 일을 맡깁니다. 문장에서 문제가 되는 표현을 찾아 빨간색으로 표시하고, 판단 근거와 순화한 문장을 보여줍니다. GPT가 내린 카테고리·심각도 판단은 참고용으로만 같이 표시합니다.
- GPT 응답은 `CATEGORY / SEVERITY / HATE_PHRASES / REASON / SOFTENED` 형식으로 고정해서 파싱하고, 혐오 표현이 없으면 `NONE`을 돌려받도록 했습니다.

### 실행 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- 학습된 가중치 파일(`model_ANN_best.pt`, `model_TextCNN_best.pt`, `model_BiLSTM_best.pt`)을 같은 폴더나 `output/` 폴더에 넣어야 합니다. 용량이 커서 저장소에는 올리지 않았습니다.
- OpenAI API 키는 실행 후 사이드바에 입력합니다. 키가 없어도 분류는 됩니다.

## 한계

- 연령(1%)과 성별(2%) 혐오 데이터가 적어서 해당 카테고리 F1이 낮습니다. 연령은 모델에 따라 0.26~0.37 수준입니다.
- 약한 혐오는 공격적 표현이나 강한 혐오와 경계가 애매해서 모든 모델이 어려워했습니다.
- 소수 클래스 데이터를 늘리고, 다른 출처의 데이터를 더하고, 사전학습 모델을 본격적으로 써 보는 것이 다음 과제입니다.

## 문서

- [최종 보고서 (PDF)](docs/report.pdf)
- [발표 자료 (PDF)](docs/presentation.pdf)

## 참고문헌

1. Park, S. et al., *K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings*, Findings of EMNLP 2023.
2. Park, S. et al., *KLUE: Korean Language Understanding Evaluation*, NeurIPS 2021 Datasets and Benchmarks.
3. Kim, Y., *Convolutional Neural Networks for Sentence Classification*, EMNLP 2014.
