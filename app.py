"""
🎭 SentimentScope
Model  : Logistic Regression + TF-IDF (scikit-learn)
UI     : Gradio
Deploy : Hugging Face Spaces
"""

import gradio as gr
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ─────────────────────────────────────────────
# 1.  Training Data
# ─────────────────────────────────────────────

TRAIN_DATA = [
    # POSITIVE
    ("I absolutely love this product it is amazing", "positive"),
    ("Best purchase I have ever made highly recommend", "positive"),
    ("Fantastic quality and super fast delivery", "positive"),
    ("I am so happy with this exceeded all expectations", "positive"),
    ("Outstanding service and wonderful experience", "positive"),
    ("This is brilliant works perfectly every time", "positive"),
    ("Great value for money very satisfied customer", "positive"),
    ("Wonderful product my family loves it so much", "positive"),
    ("Excellent could not be happier with this purchase", "positive"),
    ("Super impressed delivery was quick and item is perfect", "positive"),
    ("Totally worth it amazing features and great build", "positive"),
    ("Incredible experience from start to finish", "positive"),
    ("Very happy with my order thank you so much", "positive"),
    ("Top quality product will definitely buy again", "positive"),
    ("Loved every bit of it simply outstanding", "positive"),
    ("Five stars absolutely recommend to everyone", "positive"),
    ("Perfect gift recipient was thrilled", "positive"),
    ("Smooth and easy process very pleased", "positive"),
    ("Works like a charm very well made", "positive"),
    ("Superb craftsmanship and attention to detail", "positive"),

    # NEGATIVE
    ("This is terrible complete waste of money", "negative"),
    ("Worst product I have ever bought totally broken", "negative"),
    ("Horrible quality fell apart after one day", "negative"),
    ("Very disappointed does not work as described", "negative"),
    ("Awful experience customer service was useless", "negative"),
    ("Total garbage do not buy this product", "negative"),
    ("Extremely poor quality very unhappy customer", "negative"),
    ("Broken on arrival absolute disaster", "negative"),
    ("Do not waste your money on this trash", "negative"),
    ("Terrible terrible terrible worst ever", "negative"),
    ("Complete scam nothing like advertised", "negative"),
    ("Arrived damaged and support refused to help", "negative"),
    ("Dreadful experience from start to finish", "negative"),
    ("Zero stars if I could absolutely dreadful", "negative"),
    ("Cheaply made rubbish that stopped working immediately", "negative"),
    ("Regret buying this total disappointment", "negative"),
    ("Never again this product is beyond awful", "negative"),
    ("Broken useless and a huge waste of cash", "negative"),
    ("Poor packaging item arrived completely smashed", "negative"),
    ("Disgusting quality fell apart immediately", "negative"),

    # NEUTRAL
    ("The product arrived on time nothing special", "neutral"),
    ("It is okay does what it says on the box", "neutral"),
    ("Average quality neither good nor bad", "neutral"),
    ("Decent enough for the price I paid", "neutral"),
    ("Works fine no complaints but nothing exciting", "neutral"),
    ("It is alright meets basic expectations only", "neutral"),
    ("Standard product does the job adequately", "neutral"),
    ("Not bad not great just somewhere in the middle", "neutral"),
    ("Mediocre at best could be better honestly", "neutral"),
    ("Received the item seems okay so far I guess", "neutral"),
    ("Fairly ordinary nothing to write home about", "neutral"),
    ("Does what it is supposed to nothing more", "neutral"),
    ("Middling quality acceptable for everyday use", "neutral"),
    ("It is fine I have no strong feelings either way", "neutral"),
    ("Reasonable product at a reasonable price point", "neutral"),
    ("So so meets expectations but does not exceed them", "neutral"),
    ("Neither impressed nor disappointed with this purchase", "neutral"),
    ("Just okay would not strongly recommend or avoid", "neutral"),
    ("Passable quality serves its basic purpose well", "neutral"),
    ("Average in every way completely unremarkable", "neutral"),
]

MODEL_PATH = "model.pkl"

# ─────────────────────────────────────────────
# 2.  Train / Load Model
# ─────────────────────────────────────────────

def train_and_save():
    texts  = [t for t, _ in TRAIN_DATA]
    labels = [l for _, l in TRAIN_DATA]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
        )),
    ])
    pipe.fit(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    return pipe, texts, labels


def load_or_train():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            pipe = pickle.load(f)
        texts  = [t for t, _ in TRAIN_DATA]
        labels = [l for _, l in TRAIN_DATA]
        return pipe, texts, labels
    return train_and_save()


print("Initialising model...")
model, tr_texts, tr_labels = load_or_train()

preds = model.predict(tr_texts)
METRICS = {
    "accuracy":  round(accuracy_score(tr_labels, preds) * 100, 1),
    "precision": round(precision_score(tr_labels, preds,
                       average="weighted", zero_division=0) * 100, 1),
    "recall":    round(recall_score(tr_labels, preds,
                       average="weighted", zero_division=0) * 100, 1),
    "f1":        round(f1_score(tr_labels, preds,
                       average="weighted", zero_division=0) * 100, 1),
}
print(f"Model ready. Metrics: {METRICS}")

EMOJI  = {"positive": "😊", "neutral": "😐", "negative": "😠"}
COLOR  = {"positive": "#10B981", "neutral": "#F59E0B", "negative": "#FF4B6E"}

EXAMPLES = [
    ["I absolutely love this product it is amazing"],
    ["This is the worst experience of my life so disappointed"],
    ["The package arrived on time nothing special really"],
    ["Outstanding quality and superb customer service"],
    ["It is okay does what it says on the box"],
    ["Complete waste of money broken on arrival"],
    ["Decent product for the price no complaints"],
    ["Totally blown away best purchase ever made"],
]

# ─────────────────────────────────────────────
# 3.  Prediction helpers
# ─────────────────────────────────────────────

def predict(text: str):
    if not text or not text.strip():
        return (
            gr.update(value="⚠️ Please enter some text.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    text = text.strip()
    proba   = model.predict_proba([text])[0]
    classes = model.classes_
    scores  = dict(zip(classes, proba))

    top   = max(scores, key=scores.get)
    score = scores[top]
    emoji = EMOJI[top]
    color = COLOR[top]

    bars = ""
    for lbl in ["positive", "neutral", "negative"]:
        pct = round(scores[lbl] * 100, 1)
        bars += f"""
        <div style="margin:10px 0">
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;margin-bottom:4px">
            <span style="font-weight:600;color:#e2e8f0">
              {EMOJI[lbl]} {lbl.capitalize()}
            </span>
            <span style="color:{COLOR[lbl]};font-weight:700">{pct}%</span>
          </div>
          <div style="background:#1e293b;border-radius:999px;height:10px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{COLOR[lbl]};
                        border-radius:999px"></div>
          </div>
        </div>"""

    card = f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);
                border:1px solid {color}55;border-radius:16px;
                padding:24px 28px;box-shadow:0 0 28px {color}22;
                font-family:sans-serif">
      <div style="text-align:center;margin-bottom:16px">
        <div style="font-size:52px">{emoji}</div>
        <div style="font-size:28px;font-weight:800;color:{color};
                    letter-spacing:-0.5px">{top.capitalize()}</div>
        <div style="color:#94a3b8;font-size:13px;margin-top:4px">
          Confidence:
          <strong style="color:{color}">{round(score*100,1)}%</strong>
        </div>
      </div>
      <hr style="border:none;border-top:1px solid #334155;margin:12px 0">
      <div style="font-size:11px;color:#64748b;font-weight:700;
                  text-transform:uppercase;letter-spacing:.08em;
                  margin-bottom:10px">Score Breakdown</div>
      {bars}
    </div>"""

    label_scores = {
        f"{EMOJI[c]} {c.capitalize()}": round(scores[c], 4)
        for c in ["positive", "neutral", "negative"]
    }

    meta = (
        f"**Text:** _{text[:120]}{'...' if len(text)>120 else ''}_\n\n"
        f"**Model:** `Logistic Regression + TF-IDF`\n\n"
        f"**Result:** {emoji} {top.capitalize()} ({round(score*100,1)}%)"
    )

    return (
        gr.update(value=card, visible=True),
        gr.update(value=label_scores, visible=True),
        gr.update(value=meta, visible=True),
    )


def batch_predict(raw: str):
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if not lines:
        return "⚠️ Enter one sentence per line."

    md = "| # | Text | Sentiment | Confidence |\n|---|------|-----------|------------|\n"
    for i, line in enumerate(lines[:20], 1):
        proba   = model.predict_proba([line])[0]
        scores  = dict(zip(model.classes_, proba))
        top     = max(scores, key=scores.get)
        conf    = round(scores[top] * 100, 1)
        short   = line[:55] + ("..." if len(line) > 55 else "")
        md += f"| {i} | {short} | {EMOJI[top]} {top.capitalize()} | {conf}% |\n"
    return md

# ─────────────────────────────────────────────
# 4.  Gradio UI
# ─────────────────────────────────────────────

CSS = """
body, .gradio-container {
  background: #020617 !important;
  font-family: 'Segoe UI', sans-serif !important;
  color: #e2e8f0 !important;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }

textarea, input[type=text] {
  background: #0f172a !important;
  border: 1px solid #334155 !important;
  border-radius: 12px !important;
  color: #e2e8f0 !important;
  font-size: .95rem !important;
}
textarea:focus, input:focus {
  border-color: #818cf8 !important;
  box-shadow: 0 0 0 3px rgba(129,140,248,.15) !important;
}
button.primary {
  background: linear-gradient(135deg,#6366f1,#818cf8) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important; font-weight: 700 !important;
}
button.secondary {
  background: #1e293b !important; color: #64748b !important;
  border: 1px solid #334155 !important; border-radius: 10px !important;
}
.tab-nav button {
  background: transparent !important; color: #64748b !important;
  border: none !important; font-weight: 600 !important;
  border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
  color: #818cf8 !important;
  border-bottom: 2px solid #818cf8 !important;
}
label span {
  color: #64748b !important; font-size: .72rem !important;
  font-weight: 700 !important; text-transform: uppercase !important;
  letter-spacing: .08em !important;
}
"""

with gr.Blocks(css=CSS, title="SentimentScope") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:48px 20px 24px">
      <h1 style="font-size:2.8rem;font-weight:800;letter-spacing:-2px;
                 background:linear-gradient(135deg,#818cf8,#34d399,#f472b6);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin-bottom:10px">🎭 SentimentScope</h1>
      <p style="color:#64748b;font-size:.98rem;max-width:480px;margin:0 auto">
        Real-time sentiment analysis with
        <strong style="color:#e2e8f0">Logistic Regression + TF-IDF</strong>
        — no GPU, no heavy downloads.
      </p>
    </div>""")

    gr.HTML("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;
                padding:0 16px 24px">
      <span style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                   padding:5px 14px;color:#94a3b8;font-size:11px;font-family:monospace">
        🤖 Logistic Regression</span>
      <span style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                   padding:5px 14px;color:#94a3b8;font-size:11px;font-family:monospace">
        📝 TF-IDF Vectorizer</span>
      <span style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                   padding:5px 14px;color:#94a3b8;font-size:11px;font-family:monospace">
        🐍 scikit-learn</span>
      <span style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                   padding:5px 14px;color:#94a3b8;font-size:11px;font-family:monospace">
        🐳 Docker</span>
      <span style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                   padding:5px 14px;color:#94a3b8;font-size:11px;font-family:monospace">
        🤗 HF Spaces</span>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: Single ───────────────────────
        with gr.TabItem("✍️ Single Analysis"):

            with gr.Row():
                with gr.Column(scale=3):
                    txt_in = gr.Textbox(
                        label="Input Text",
                        placeholder="Type a review, tweet or comment...",
                        lines=4,
                    )
                    with gr.Row():
                        btn_go    = gr.Button("🔍 Analyse", variant="primary", scale=3)
                        btn_clear = gr.Button("✕ Clear",   variant="secondary", scale=1)

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=txt_in,
                        label="💡 Try an example",
                        examples_per_page=4,
                    )

            out_card  = gr.HTML(visible=False)
            out_label = gr.Label(label="Confidence Scores",
                                 num_top_classes=3, visible=False)
            out_meta  = gr.Markdown(visible=False)

            def clear_all():
                return (
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            btn_go.click(predict, txt_in, [out_card, out_label, out_meta])
            txt_in.submit(predict, txt_in, [out_card, out_label, out_meta])
            btn_clear.click(clear_all,
                            outputs=[txt_in, out_card, out_label, out_meta])

        # ── Tab 2: Batch ────────────────────────
        with gr.TabItem("📋 Batch Analysis"):
            gr.Markdown("Enter **one sentence per line** (max 20 lines).")
            b_in  = gr.Textbox(
                label="Texts",
                placeholder="Great product!\nTerrible service.\nIt was okay.",
                lines=8,
            )
            b_btn = gr.Button("🚀 Analyse All", variant="primary")
            b_out = gr.Markdown()
            b_btn.click(batch_predict, b_in, b_out)

            gr.Examples(
                examples=[["\n".join([e[0] for e in EXAMPLES])]],
                inputs=b_in,
                label="📦 Load all examples",
            )

        # ── Tab 3: Metrics ──────────────────────
        with gr.TabItem("📊 Model Metrics"):
            gr.Markdown(f"""
## Training Metrics

| Metric | Score |
|--------|-------|
| ✅ Accuracy  | **{METRICS['accuracy']}%** |
| 🎯 Precision | **{METRICS['precision']}%** |
| 🔍 Recall    | **{METRICS['recall']}%** |
| ⚖️ F1-Score  | **{METRICS['f1']}%** |

## Recall Formula
$$Recall = \\frac{{TP}}{{TP + FN}}$$

**TP** = True Positives &nbsp;|&nbsp; **FN** = False Negatives

## Model Pipeline
```
Input Text
    ↓
TF-IDF Vectorizer  →  converts words to numbers
    ↓
Logistic Regression  →  classifies into 3 classes
    ↓
Output: Positive 😊 / Neutral 😐 / Negative 😠
```

## Dataset
- 60 hand-labelled sentences (20 per class)
- Built-in — no external download needed
- Classes: **Positive · Neutral · Negative**
            """)

        # ── Tab 4: About ────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## About SentimentScope

| Layer | Tool |
|-------|------|
| Model | Logistic Regression |
| Features | TF-IDF (unigrams + bigrams) |
| UI | Gradio |
| Container | Docker |
| Cloud | Hugging Face Spaces |

## Run Locally

```bash
# Python
pip install -r requirements.txt
python app.py

# Docker
docker build -t sentimentscope .
docker run -p 7860:7860 sentimentscope

# Docker Compose
docker compose up --build
```
Open → **http://localhost:7860**
            """)

    gr.HTML("""
    <div style="text-align:center;padding:24px;border-top:1px solid #1e293b;
                margin-top:32px;color:#475569;font-size:.8rem">
      Built with 🐍 scikit-learn · 🎨 Gradio · 🐳 Docker · 🤗 HuggingFace Spaces
    </div>""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
