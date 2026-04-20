
================================================================================
  EARNINGS CALL INTELLIGENCE ENGINE
  Predicting Post-Earnings Stock Movement from Transcript NLP Signals
================================================================================

PROBLEM STATEMENT:
    Earnings calls contain rich linguistic signals — executive hedging, sentiment
    shifts, uncertainty markers — that precede market-moving events. This system
    ingests raw earnings call transcripts and predicts whether the stock will move
    UP, DOWN, or NEUTRAL in the 48-hour post-call window.

    Hedge funds and quant desks pay 7-figures for this signal (Bloomberg Terminal,
    Refinitiv Eikon). This pipeline replicates that capability using open-source ML.

MARKET-STANDARD ML ENGINEERING TECHNIQUES USED:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  HuggingFace Transformers  → FinBERT financial sentiment embeddings │
    │  Zero-Shot Classification  → Tone labeling without fine-tuning      │
    │  TF-IDF + Linguistic Feats → Handcrafted NLP feature engineering    │
    │  XGBoost                   → Gradient boosting (quant industry std) │
    │  SHAP                      → Model explainability (enterprise req.)  │
    │  Optuna                    → Bayesian hyperparameter optimization    │
    │  imbalanced-learn (SMOTE)  → Class imbalance correction             │
    │  MLflow                    → Experiment tracking & model registry   │
    │  FastAPI + Pydantic        → Production REST API serving            │
    │  scikit-learn Pipelines    → Reproducible preprocessing             │
    └─────────────────────────────────────────────────────────────────────┘

DATASET:
    Synthetic but statistically realistic earnings call transcripts generated
    with domain-accurate vocabulary. Labels simulate real price movements using
    sentiment-correlated logic. Swap in real transcripts from:
    → https://earningscall.biz  (free tier)
    → SEC EDGAR full-text search (EDGAR-FULL-TEXT API)
    → Motley Fool / Seeking Alpha scrapers

AUTHOR: Arnav Dikshit
DATE:   2026

RUN:
    pip install -r requirements.txt
    python main.py --mode train      # Train & evaluate full pipeline
    python main.py --mode api        # Launch FastAPI inference server
    python main.py --mode demo       # Quick demo on sample transcripts
================================================================================
