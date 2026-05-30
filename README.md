---
title: Fraud Detection App
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Fraud Detection App

A Gradio web application for detecting fraudulent bank transactions using supervised machine learning. Built as part of a thesis project, the app covers the full pipeline from data exploration and feature engineering to model training, scoring, and version management.

> **Dataset note:** The full training dataset (`all_transactions_2024.csv`, ~460k rows) is not included in this repository due to GitHub's 100 MB file size limit. A representative sample of 300,000 randomly selected rows (`data/sample_train_300k.csv`) is provided instead and can be used to train or reproduce the models. The fraud rate in the sample (1.79%) closely matches the full dataset (1.80%). **Users are advised to use `data/sample_train_300k.csv` for training the model.**

**Live demo:** https://huggingface.co/spaces/nolinjoo/fraud-detection

---

## Notebook

`Thesis-2 v15 final.ipynb` is the original research notebook where all analysis, feature engineering, and model experiments were first developed. Once the approach was validated, the notebook code was refactored into the modular `src/` package to make it reusable, testable, and easy to serve through the Gradio app. The notebook is included here as a reference to the underlying research but is not required to run the application.

---

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` in your browser. A pre-trained model is included in `artifacts/` so the Scoring tab works immediately.

---

## Upload Schema

All CSV uploads require these 13 columns (comma or pipe delimited):

```
trans_date, trans_time, amt, category, gender, state,
dob, lat, long, merch_lat, merch_long, city_pop, profile
```

`is_fraud` is optional for scoring and required for training.

---

## Tabs

**Fraud Scoring** — Upload a transaction CSV to get a fraud probability, binary flag, and risk label (Low / Review / High) per row. Results sorted highest risk first with a download link.

**Data Analysis** — Schema validation, quality checks, and a preview of all 23 engineered features. Run this before training to catch issues early.

**Visualizations** — 25 interactive Plotly charts across time, amount, geography, merchant category, and customer profile dimensions.

**Model Training** — Train all six model families on your own labelled data. The winning model is saved automatically and goes live in the Scoring tab immediately.

**Model Management** — Browse all saved model versions by timestamp and activate any previous version with one click.

---

## Files

| File / Folder | Purpose |
|---|---|
| `app.py` | Gradio application — all five tabs |
| `src/` | Python package: features, models, training, scoring, charts |
| `artifacts/` | Pre-trained model and result tables |
| `requirements.txt` | Python dependencies |
| `train.py` | Optional CLI training script (`python train.py --help`) |
| `data/sample_train_300k.csv` | 300k-row sample of the full training dataset (see Dataset note above) |
| `Thesis-2 v15 final.ipynb` | Original research notebook (reference only — see below) |
---

## Notes

Built on synthetic Sparkov-style transaction data. Not intended for production use without validation on real banking data.
