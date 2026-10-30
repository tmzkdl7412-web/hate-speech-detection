"""
SNS 혐오·차별 발언 자동 탐지 및 등급 분류 시스템 — Streamlit 데모
홍익대학교 졸업 프로젝트 | 임종태 교수

실행 방법:
    pip install streamlit torch transformers openai
    streamlit run streamlit_app.py

필요 파일 (같은 폴더 또는 output/ 안):
    model_ANN_best.pt
    model_TextCNN_best.pt
    model_BiLSTM_best.pt
"""

import os
import re
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="혐오 발언 탐지 시스템",
    layout="centered",
)

# 모델 파일 경로 (output 폴더 우선, 없으면 현재 폴더)
def find_model(name):
    for path in [f"output/model_{name}_best.pt", f"model_{name}_best.pt"]:
        if os.path.exists(path):
            return path
    return f"model_{name}_best.pt"

PRETRAINED_NAME = "klue/bert-base"
MAX_LEN    = 96
HIDDEN_DIM = 256
LABEL_COLS = ["gender", "race", "age", "other"]
LABEL_KR   = {"gender": "성별", "race": "인종/출신", "age": "연령", "other": "기타"}
SEVERITY_NAMES = ["정상", "공격적", "혐오(약)", "혐오(강)"]
SEVERITY_COLORS = ["#1D9E75", "#EF9F27", "#D85A30", "#E24B4A"]

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ──────────────────────────────────────────────────────────────
# 1. 전처리
# ──────────────────────────────────────────────────────────────
ANON_RE    = re.compile(r"#@[^#]+#")
URL_RE     = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HTML_RE    = re.compile(r"<[^>]+>")
SPACE_RE   = re.compile(r"\s{2,}")


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = ANON_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = re.sub(r"([\u3130-\u318F])\1{2,}", r"\1\1", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# 2. 모델 정의 (학습 때와 동일)
# ──────────────────────────────────────────────────────────────
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cat_heads = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            ) for col in LABEL_COLS
        })
        self.sev_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 4)
        )

    def forward(self, x):
        out = {col: head(x).squeeze(-1) for col, head in self.cat_heads.items()}
        out["severity"] = self.sev_head(x)
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = pretrained_embedding
        self.proj = nn.Linear(pretrained_embedding.embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.proj(x)
        x = self.layer_norm(x)
        return self.dropout(x)


class ANNModel(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim=HIDDEN_DIM, dropout=0.4):
        super().__init__()
        self.embedding = EmbeddingLayer(pretrained_embedding, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        mean_emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        masked_emb = emb.masked_fill(mask == 0, -1e9)
        max_emb, _ = masked_emb.max(dim=1)
        pooled = torch.cat([mean_emb, max_emb], dim=-1)
        return self.head(self.mlp(pooled))


class TextCNNModel(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim=HIDDEN_DIM,
                 num_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.4):
        super().__init__()
        self.embedding = EmbeddingLayer(pretrained_embedding, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, num_filters, k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
            ) for k in kernel_sizes
        ])
        out_dim = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids).transpose(1, 2)
        outs = []
        for conv in self.convs:
            h = conv(emb)
            h = F.max_pool1d(h, h.size(-1)).squeeze(-1)
            outs.append(h)
        h = torch.cat(outs, dim=1)
        h = self.dropout(h)
        return self.head(self.fc(h))


class BiLSTMModel(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim=HIDDEN_DIM,
                 lstm_hidden=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.embedding = EmbeddingLayer(pretrained_embedding, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=num_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        lstm_out_dim = lstm_hidden * 2
        self.attention = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=4,
                                                dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = MultiTaskHead(hidden_dim)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        key_padding_mask = (attention_mask == 0)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out,
                                     key_padding_mask=key_padding_mask)
        attn_out = self.layer_norm(lstm_out + attn_out)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(self.fc(pooled))


# ──────────────────────────────────────────────────────────────
# 3. 모델 로드 (캐싱)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
    bert = AutoModel.from_pretrained(PRETRAINED_NAME)
    pretrained_embedding = bert.embeddings.word_embeddings

    models = {}
    for name, cls in [("ANN", ANNModel), ("TextCNN", TextCNNModel), ("BiLSTM", BiLSTMModel)]:
        m = cls(pretrained_embedding).to(DEVICE)
        m.load_state_dict(torch.load(find_model(name), map_location=DEVICE))
        m.eval()
        models[name] = m
    return tokenizer, models


# 앙상블 가중치 (학습 시 산정된 값)
ENSEMBLE_WEIGHTS = {"ANN": 0.3337, "TextCNN": 0.3308, "BiLSTM": 0.3355}


@torch.no_grad()
def predict(text, tokenizer, models, use_ensemble=True, single_model="BiLSTM"):
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    if use_ensemble:
        cat_probs = {col: 0.0 for col in LABEL_COLS}
        sev_probs = None
        for name, w in ENSEMBLE_WEIGHTS.items():
            logits = models[name](ids, mask)
            for col in LABEL_COLS:
                cat_probs[col] += w * torch.sigmoid(logits[col]).item()
            sp = F.softmax(logits["severity"], dim=-1).cpu().numpy()[0]
            sev_probs = w * sp if sev_probs is None else sev_probs + w * sp
    else:
        logits = models[single_model](ids, mask)
        cat_probs = {col: torch.sigmoid(logits[col]).item() for col in LABEL_COLS}
        sev_probs = F.softmax(logits["severity"], dim=-1).cpu().numpy()[0]

    return cat_probs, sev_probs


# ──────────────────────────────────────────────────────────────
# 4. OpenAI — 혐오 표현 위치 + 순화
# ──────────────────────────────────────────────────────────────
def parse_gpt_result(result):
    """GPT 응답을 CATEGORY / SEVERITY / HATE_PHRASES / REASON / SOFTENED로 분리"""
    parsed = {
        "CATEGORY": "none",
        "SEVERITY": "0",
        "HATE_PHRASES": "NONE",
        "REASON": "",
        "SOFTENED": "",
    }

    for line in result.splitlines():
        line = line.strip()
        for key in parsed.keys():
            if line.startswith(key + ":"):
                parsed[key] = line.replace(key + ":", "", 1).strip()

    hate_raw = parsed["HATE_PHRASES"]
    if hate_raw.upper() in ["NONE", "없음", ""]:
        hate_phrases = []
    else:
        hate_phrases = [x.strip() for x in hate_raw.split(",") if x.strip()]

    return (
        parsed["CATEGORY"],
        parsed["SEVERITY"],
        hate_phrases,
        parsed["REASON"],
        parsed["SOFTENED"],
    )


def analyze_with_gpt(text, api_key):
    """OpenAI GPT로 혐오 표현 위치, 카테고리, 심각도, 근거, 순화 표현 생성"""
    if not api_key:
        return None, None, [], None, None, "OpenAI API 키를 입력하세요."

    client = OpenAI(api_key=api_key)

    prompt = f"""
당신은 한국어 SNS 혐오·차별 발언 분석 전문가입니다.

입력 문장:
{text}

분석 목표:
1. 혐오 카테고리 분류
2. 심각도 분류
3. 문제 표현 추출
4. 판단 근거 설명
5. 순화 문장 생성

카테고리:
gender : 성별 혐오
region : 지역 혐오
race : 인종/국적 혐오
age : 연령 혐오
other : 기타 혐오
none : 혐오 아님

심각도:
0 = 정상
1 = 경미
2 = 보통
3 = 심각

주의:
- HATE_PHRASES는 원문에 실제로 존재하는 표현만 추출하세요.
- 여러 개면 쉼표로 구분하세요.
- 혐오 표현이 없으면 HATE_PHRASES는 NONE으로 출력하세요.
- 반드시 아래 형식만 출력하세요.

CATEGORY:
SEVERITY:
HATE_PHRASES:
REASON:
SOFTENED:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "한국어 혐오표현 분석 전문가"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
        )

        result = response.choices[0].message.content
        category, severity, hate_phrases, reason, softened = parse_gpt_result(result)

        return category, severity, hate_phrases, reason, softened, None

    except Exception as e:
        return None, None, [], None, None, f"GPT API 오류: {e}"


def highlight_text(text, hate_phrases):
    """혐오 표현을 빨간색으로 하이라이팅한 HTML 반환"""
    highlighted = text
    for phrase in hate_phrases:
        if phrase and phrase in highlighted:
            highlighted = highlighted.replace(
                phrase,
                f'<span style="background-color:#FCEBEB;color:#A32D2D;'
                f'font-weight:bold;padding:1px 3px;border-radius:3px;">{phrase}</span>'
            )
    return highlighted


# ──────────────────────────────────────────────────────────────
# 5. UI
# ──────────────────────────────────────────────────────────────
st.title("SNS 혐오·차별 발언 탐지 시스템")
st.caption("홍익대학교 졸업 프로젝트 | 지도교수: 임종태교수님 ")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    use_ensemble = st.toggle("앙상블 사용 (ANN+CNN+BiLSTM)", value=True)
    single_model = st.selectbox("단일 모델 선택", ["BiLSTM", "ANN", "TextCNN"],
                                 disabled=use_ensemble)
    st.divider()
    st.subheader("OpenAI 입력")
    api_key = st.text_input("API Key", type="password",
                            help="혐오 표현 하이라이팅 + 순화 기능에 필요")
    st.divider()
    st.caption(f"Device: {DEVICE}")

# 모델 로드
try:
    tokenizer, models = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.info("model_ANN_best.pt, model_TextCNN_best.pt, model_BiLSTM_best.pt 파일이 "
            "같은 폴더 또는 output/ 안에 있는지 확인하세요.")
    models_loaded = False

# 입력
st.subheader("문장 입력")
text_input = st.text_area(
    "분석할 문장을 입력하세요",
    placeholder="예: 여자가 무슨 운전이야",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔍 분석", type="primary", use_container_width=True)

# 분석 실행
if analyze_btn and text_input.strip() and models_loaded:
    cat_probs, sev_probs = predict(
        text_input, tokenizer, models,
        use_ensemble=use_ensemble, single_model=single_model
    )

    st.divider()

    # ── 심각도 ──
    sev_idx = int(sev_probs.argmax())
    st.subheader("심각도 판정")
    cols = st.columns(4)
    for i, (name, color) in enumerate(zip(SEVERITY_NAMES, SEVERITY_COLORS)):
        with cols[i]:
            is_pred = (i == sev_idx)
            st.markdown(
                f'<div style="text-align:center;padding:12px;border-radius:8px;'
                f'background-color:{"%s22" % color if is_pred else "transparent"};'
                f'border:2px solid {color if is_pred else "#ddd"};">'
                f'<div style="font-size:13px;color:#666;">{name}</div>'
                f'<div style="font-size:20px;font-weight:bold;color:{color};">'
                f'{sev_probs[i]*100:.0f}%</div></div>',
                unsafe_allow_html=True
            )

    # ── 카테고리 ──
    st.subheader("혐오 카테고리")
    cat_cols = st.columns(4)
    for i, col in enumerate(LABEL_COLS):
        with cat_cols[i]:
            prob = cat_probs[col]
            is_hate = prob > 0.5
            st.metric(
                LABEL_KR[col],
                f"{prob*100:.0f}%",
                "혐오 감지" if is_hate else "정상",
                delta_color="inverse" if is_hate else "off"
            )

    # ── OpenAI: 카테고리 + 심각도 + 하이라이팅 + 근거 + 순화 ──
    if api_key:
        st.divider()
        st.subheader("🔴 GPT 기반 혐오 표현 상세 분석")

        with st.spinner("GPT가 분석 중..."):
            category, severity, hate_phrases, reason, softened, err = analyze_with_gpt(
                text_input,
                api_key
            )

        if err:
            st.warning(err)
        else:
            CATEGORY_KR = {
                "gender": "성별 혐오",
                "region": "지역 혐오",
                "race": "인종/국적 혐오",
                "age": "연령 혐오",
                "other": "기타 혐오",
                "none": "혐오 아님",
            }

            SEVERITY_KR = {
                "0": "0단계 정상",
                "1": "1단계 경미",
                "2": "2단계 보통",
                "3": "3단계 심각",
            }

            # 여기서 st.metric 추가
            m1, m2 = st.columns(2)
            with m1:
                st.metric("카테고리", CATEGORY_KR.get(category, category))
            with m2:
                st.metric("심각도", SEVERITY_KR.get(str(severity), str(severity)))

            if hate_phrases:
                st.markdown("**원문 (혐오 표현 강조):**")
                st.markdown(
                    f'<div style="padding:12px;background:#f8f8f8;border-radius:8px;'
                    f'font-size:16px;">{highlight_text(text_input, hate_phrases)}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("**문제 표현:**")
                st.warning(", ".join(hate_phrases))
            else:
                st.info("혐오 표현이 발견되지 않았습니다.")

            if reason:
                st.markdown("**판단 근거:**")
                st.info(reason)

            if softened:
                st.markdown("**순화 표현 제안:**")
                st.success(softened)
    else:
        st.divider()
        st.info("💡 사이드바에 OpenAI API 키를 입력하면 혐오 표현 하이라이팅, 판단 근거, 순화 표현을 볼 수 있습니다.")

elif analyze_btn and not text_input.strip():
    st.warning("문장을 입력하세요.")

# 예시
st.divider()
st.caption("**예시 문장:** 여자가 무슨 운전이야 · 틀딱들 일 안하고 뭐하냐 · 오늘 날씨 좋다")
