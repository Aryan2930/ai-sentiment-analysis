---
title: SentimentScope
emoji: 🎭
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sentiment Analysis — Logistic Regression + TF-IDF + Gradio
tags:
  - sentiment-analysis
  - text-classification
  - scikit-learn
  - gradio
  - nlp
---

# 🎭 SentimentScope

Real-time sentiment analysis using **Logistic Regression + TF-IDF** — lightweight, fast, no GPU needed.

## Features
- ✅ Single text analysis with confidence scores
- ✅ Batch processing up to 20 texts at once
- ✅ Model metrics (Accuracy, Precision, Recall, F1)
- ✅ No GPU needed — runs on free HF Spaces CPU
- ✅ Docker ready for local deployment

## Tech Stack
| Layer | Tool |
|-------|------|
| Model | Logistic Regression |
| Features | TF-IDF Vectorizer |
| UI | Gradio |
| Container | Docker |
| Cloud | Hugging Face Spaces |

## Run Locally

### Python
```bash
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -t sentimentscope .
docker run -p 7860:7860 sentimentscope
```

### Docker Compose
```bash
docker compose up --build
```

Open → **http://localhost:7860**

## File Structure
```
sentimentscope/
├── app.py              ← Gradio app + ML model
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Container definition
├── docker-compose.yml  ← Local orchestration
├── .dockerignore       ← Docker build exclusions
├── ui_preview.html     ← Static UI preview
└── README.md           ← This file
```
