import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EarningsEngine")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Central config — change here, propagates everywhere."""
    # Data
    n_synthetic_samples: int = 1200
    test_size: float = 0.20
    val_size: float = 0.10
    random_seed: int = 42

    # NLP
    finbert_model: str = "ProsusAI/finbert"          # best financial sentiment BERT
    zero_shot_model: str = "facebook/bart-large-mnli" # MNLI-based zero-shot
    max_token_length: int = 512
    use_transformers: bool = True   # set False for CPU-only / fast demo

    # Features
    tfidf_max_features: int = 500
    ngram_range: Tuple[int, int] = (1, 3)

    # Training
    cv_folds: int = 5
    optuna_trials: int = 30
    smote_strategy: str = "auto"

    # MLflow
    mlflow_experiment: str = "earnings-call-intelligence"
    mlflow_tracking_uri: str = "./mlruns"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Labels
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP"
    })

CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC DATA GENERATOR
#     Produces realistic earnings-call transcripts with correlated labels
# ══════════════════════════════════════════════════════════════════════════════

class EarningsTranscriptGenerator:
    """
    Generates synthetic earnings call transcripts that mimic real linguistic
    patterns observed in SEC filings and Seeking Alpha transcripts.

    Key insight: executives use MORE hedging language before disappointing
    results and MORE confident/forward-looking language before beats.
    """

    # Linguistic signal banks (derived from academic literature on earnings calls)
    CONFIDENT_PHRASES = [
        "we are extremely confident", "exceptional growth trajectory",
        "record-breaking revenue", "demand far exceeds expectations",
        "we are raising full-year guidance", "unprecedented customer acquisition",
        "our strongest quarter ever", "we are seeing strong momentum",
        "beat expectations across every segment", "significantly outperformed",
        "robust pipeline", "accelerating adoption", "we are doubling down",
        "margin expansion continues", "outstanding execution by the team",
        "we are raising our EPS guidance", "bookings hit an all-time high",
    ]

    HEDGING_PHRASES = [
        "we remain cautiously optimistic", "subject to macroeconomic uncertainty",
        "we are monitoring the situation closely", "results may vary",
        "we cannot provide specific guidance at this time",
        "challenging market conditions persist", "we are taking a prudent approach",
        "headwinds in certain geographies", "we are reevaluating our cost structure",
        "slower than anticipated", "softness in enterprise spending",
        "we expect sequential improvement but visibility is limited",
        "we are in active discussions to reduce operating expenses",
        "certain segments underperformed our internal expectations",
        "we are not providing forward guidance due to uncertainty",
    ]

    NEUTRAL_PHRASES = [
        "in line with expectations", "results were consistent with guidance",
        "we are maintaining our full-year outlook", "steady performance across segments",
        "our core business remains resilient", "we saw mixed results by region",
        "revenues were flat year-over-year", "margins held stable",
        "we are continuing to invest in R&D", "growth was modest but sustainable",
    ]

    QA_CONFIDENT = [
        "Analyst: Can you comment on the demand environment? "
        "CFO: Absolutely — pipeline coverage is at a three-year high and churn is at historic lows.",
        "Analyst: Any concern about competition? "
        "CEO: We are taking share. Our win rate against the nearest competitor is up 12 points.",
    ]

    QA_HEDGING = [
        "Analyst: What's driving the guidance cut? "
        "CFO: We are being prudent given macro softness. We believe this is a temporary phenomenon.",
        "Analyst: When do you expect margin recovery? "
        "CEO: We are not in a position to provide a specific timeline at this stage.",
    ]

    EXECUTIVE_INTROS = [
        "Good afternoon, everyone. Thank you for joining our Q{q} {year} earnings call.",
        "Welcome to {company}'s third-quarter fiscal {year} results call.",
        "Thank you all for joining us today to discuss {company}'s financial results.",
    ]

    COMPANIES = [
        "CloudVantage", "NexGen Analytics", "DataStream Corp", "AlphaMetrics",
        "QuantumEdge", "SynapseAI", "CoreLogix", "VectorSoft", "PulseData",
        "NovaTech Systems", "ZeroLatency", "OmniCloud", "Helix Platforms"
    ]

    def _build_transcript(self, sentiment: str, noise: float = 0.15) -> str:
        """Build a realistic multi-section transcript with signal + noise."""
        rng = np.random.default_rng()
        company = rng.choice(self.COMPANIES)
        quarter = rng.integers(1, 5)
        year = rng.integers(2021, 2026)

        intro = rng.choice(self.EXECUTIVE_INTROS).format(
            q=quarter, year=year, company=company
        )

        if sentiment == "UP":
            primary = rng.choice(self.CONFIDENT_PHRASES, size=6, replace=True)
            secondary = rng.choice(self.NEUTRAL_PHRASES, size=2, replace=True)
            noise_phrases = rng.choice(self.HEDGING_PHRASES, size=1, replace=True)
            qa = rng.choice(self.QA_CONFIDENT)
            financial = (
                f"Revenue grew {rng.integers(12, 40)}% year-over-year to "
                f"${rng.integers(100, 9000)}M, beating consensus by "
                f"{rng.integers(2, 8)}%. Non-GAAP EPS of ${rng.uniform(1.2, 4.5):.2f} "
                f"surpassed the street estimate of ${rng.uniform(0.9, 3.8):.2f}."
            )
        elif sentiment == "DOWN":
            primary = rng.choice(self.HEDGING_PHRASES, size=6, replace=True)
            secondary = rng.choice(self.NEUTRAL_PHRASES, size=2, replace=True)
            noise_phrases = rng.choice(self.CONFIDENT_PHRASES, size=1, replace=True)
            qa = rng.choice(self.QA_HEDGING)
            financial = (
                f"Revenue of ${rng.integers(80, 5000)}M came in "
                f"{rng.integers(2, 10)}% below consensus estimates. "
                f"Non-GAAP EPS of ${rng.uniform(0.3, 1.8):.2f} missed the "
                f"street estimate of ${rng.uniform(0.8, 2.5):.2f}. "
                f"We are lowering full-year guidance to reflect revised expectations."
            )
        else:  # NEUTRAL
            primary = rng.choice(self.NEUTRAL_PHRASES, size=5, replace=True)
            secondary = rng.choice(self.CONFIDENT_PHRASES, size=2, replace=True)
            noise_phrases = rng.choice(self.HEDGING_PHRASES, size=2, replace=True)
            qa = ""
            financial = (
                f"Revenue of ${rng.integers(90, 6000)}M was in line with consensus. "
                f"Non-GAAP EPS of ${rng.uniform(0.8, 2.8):.2f} met expectations. "
                f"We are reaffirming full-year guidance."
            )

        all_phrases = list(primary) + list(secondary) + list(noise_phrases)
        rng.shuffle(all_phrases)

        transcript = (
            f"{intro} "
            f"Today I'll discuss our Q{quarter} {year} results and strategic outlook. "
            f"{financial} "
            + " ".join(all_phrases)
            + f" {qa} "
            f"We remain committed to delivering long-term shareholder value. "
            f"I'll now open the floor to questions."
        )
        return transcript

    def generate(self, n: int = 1000) -> pd.DataFrame:
        """Generate a balanced-ish dataset of transcripts with labels."""
        log.info(f"Generating {n} synthetic earnings call transcripts...")
        rng = np.random.default_rng(CFG.random_seed)

        # Simulate class imbalance (realistic: fewer big misses)
        weights = [0.30, 0.40, 0.30]  # DOWN, NEUTRAL, UP
        sentiments = rng.choice(["DOWN", "NEUTRAL", "UP"], size=n, p=weights)
        label_lookup = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}

        records = []
        for sent in sentiments:
            records.append({
                "transcript": self._build_transcript(sent),
                "label": label_lookup[sent],
                "sentiment_str": sent,
            })

        df = pd.DataFrame(records)
        log.info(f"Dataset created. Label distribution:\n{df['sentiment_str'].value_counts()}")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING PIPELINE
#     Handcrafted linguistic features + transformer embeddings
# ══════════════════════════════════════════════════════════════════════════════

class LinguisticFeatureExtractor:
    """
    Extracts interpretable linguistic features grounded in finance NLP research.

    Features based on:
    - Loughran & McDonald (2011) financial wordlists
    - Linguistic Inquiry & Word Count (LIWC) dimensions
    - Hedging literature (Hyland, 1996)
    """

    # Loughran-McDonald inspired financial word lists
    POSITIVE_FINANCIAL = {
        "record", "exceed", "outperform", "beat", "strength", "robust",
        "growth", "momentum", "accelerat", "expand", "gain", "opportunit",
        "confident", "strong", "outstanding", "excellent", "exceptional",
        "raise", "increase", "improve", "deliver", "achieve", "lead",
    }

    NEGATIVE_FINANCIAL = {
        "miss", "disappoint", "weak", "decline", "decreas", "headwind",
        "challeng", "uncertain", "concern", "pressur", "soft", "slow",
        "below", "lower", "cut", "reduce", "restructur", "impair",
        "loss", "risk", "difficult", "adverse", "deteriorat",
    }

    UNCERTAINTY_MARKERS = {
        "may", "might", "could", "possibly", "potentially", "subject to",
        "depending on", "if", "assume", "expect", "anticipat", "believe",
        "estimate", "approximately", "roughly", "around", "uncertain",
        "visibility", "monitor", "evaluate", "assess",
    }

    CONFIDENCE_MARKERS = {
        "will", "are confident", "definitely", "certainly", "committed",
        "clear", "strong conviction", "without doubt", "we know",
        "we are raising", "record high", "all-time", "unprecedented",
    }

    FORWARD_LOOKING = {
        "next quarter", "fiscal year", "outlook", "guidance", "forecast",
        "going forward", "in the future", "we plan", "we expect",
        "we anticipate", "pipeline", "backlog", "bookings",
    }

    def _count_matches(self, text: str, wordset: set) -> float:
        """Count normalized term frequency for a word set."""
        text_lower = text.lower()
        words = text_lower.split()
        count = sum(
            1 for w in words
            if any(marker in w for marker in wordset)
        )
        return count / max(len(words), 1)

    def extract(self, texts: List[str]) -> pd.DataFrame:
        """Extract all linguistic features from a list of transcripts."""
        log.info(f"Extracting linguistic features from {len(texts)} transcripts...")
        features = []
        for text in texts:
            words = text.split()
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            feat = {
                # Core sentiment signals
                "pos_fin_density": self._count_matches(text, self.POSITIVE_FINANCIAL),
                "neg_fin_density": self._count_matches(text, self.NEGATIVE_FINANCIAL),
                "uncertainty_density": self._count_matches(text, self.UNCERTAINTY_MARKERS),
                "confidence_density": self._count_matches(text, self.CONFIDENCE_MARKERS),
                "forward_looking_density": self._count_matches(text, self.FORWARD_LOOKING),

                # Derived ratios (key alpha signals)
                "sentiment_ratio": (
                    self._count_matches(text, self.POSITIVE_FINANCIAL) /
                    max(self._count_matches(text, self.NEGATIVE_FINANCIAL), 0.001)
                ),
                "confidence_vs_uncertainty": (
                    self._count_matches(text, self.CONFIDENCE_MARKERS) /
                    max(self._count_matches(text, self.UNCERTAINTY_MARKERS), 0.001)
                ),

                # Structural features
                "word_count": len(words),
                "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
                "sentence_count": len(sentences),
                "num_numbers": len(re.findall(r'\b\d+\.?\d*\b', text)),
                "num_percentages": len(re.findall(r'\d+\.?\d*%', text)),
                "num_dollar_figures": len(re.findall(r'\$[\d,]+', text)),

                # Guidance signals
                "mentions_guidance": int("guidance" in text.lower()),
                "mentions_raise": int(
                    bool(re.search(r'rais(e|ing)\s+(guidance|outlook|forecast)', text.lower()))
                ),
                "mentions_lower": int(
                    bool(re.search(r'lower(ing)?\s+(guidance|outlook|forecast)', text.lower()))
                ),
                "mentions_reaffirm": int("reaffirm" in text.lower()),

                # Q&A section signals
                "has_qa": int("analyst" in text.lower() or "question" in text.lower()),

                # Lexical diversity (low diversity = scripted/cautious)
                "type_token_ratio": len(set(words)) / max(len(words), 1),

                # Exclamatory confidence
                "exclamation_count": text.count("!"),
                "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 2),
            }
            features.append(feat)

        return pd.DataFrame(features)


class TransformerFeatureExtractor:
    """
    Extracts FinBERT sentiment scores and zero-shot topic probabilities.
    Falls back gracefully if transformers unavailable or no GPU.
    """

    def __init__(self):
        self.finbert = None
        self.zero_shot = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import pipeline
            log.info("Loading FinBERT (ProsusAI/finbert)...")
            self.finbert = pipeline(
                "text-classification",
                model=CFG.finbert_model,
                top_k=None,
                truncation=True,
                max_length=CFG.max_token_length,
            )
            log.info("Loading zero-shot classifier (facebook/bart-large-mnli)...")
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model=CFG.zero_shot_model,
                truncation=True,
            )
            self._loaded = True
            log.info("Transformer models loaded successfully.")
        except Exception as e:
            log.warning(f"Transformer loading failed ({e}). Falling back to zeros.")
            CFG.use_transformers = False

    def _finbert_scores(self, text: str) -> Dict[str, float]:
        """Get FinBERT positive/negative/neutral probabilities."""
        try:
            # Truncate to first 512 words for speed
            truncated = " ".join(text.split()[:400])
            results = self.finbert(truncated)[0]
            scores = {r["label"].lower(): r["score"] for r in results}
            return {
                "finbert_positive": scores.get("positive", 0.0),
                "finbert_negative": scores.get("negative", 0.0),
                "finbert_neutral": scores.get("neutral", 0.0),
                "finbert_net_sentiment": scores.get("positive", 0.0) - scores.get("negative", 0.0),
            }
        except Exception:
            return {"finbert_positive": 0, "finbert_negative": 0,
                    "finbert_neutral": 0, "finbert_net_sentiment": 0}

    def _zeroshot_scores(self, text: str) -> Dict[str, float]:
        """Zero-shot classify tone without any fine-tuning."""
        labels = ["confident earnings beat", "earnings miss with uncertainty",
                  "cautious guidance", "strong forward outlook"]
        try:
            truncated = " ".join(text.split()[:300])
            result = self.zero_shot(truncated, candidate_labels=labels)
            scores = dict(zip(result["labels"], result["scores"]))
            return {
                "zs_confident_beat": scores.get("confident earnings beat", 0),
                "zs_earnings_miss": scores.get("earnings miss with uncertainty", 0),
                "zs_cautious_guidance": scores.get("cautious guidance", 0),
                "zs_strong_outlook": scores.get("strong forward outlook", 0),
            }
        except Exception:
            return {"zs_confident_beat": 0, "zs_earnings_miss": 0,
                    "zs_cautious_guidance": 0, "zs_strong_outlook": 0}

    def extract(self, texts: List[str]) -> pd.DataFrame:
        if not CFG.use_transformers:
            log.info("Transformers disabled — using zero features.")
            cols = ["finbert_positive", "finbert_negative", "finbert_neutral",
                    "finbert_net_sentiment", "zs_confident_beat", "zs_earnings_miss",
                    "zs_cautious_guidance", "zs_strong_outlook"]
            return pd.DataFrame(np.zeros((len(texts), len(cols))), columns=cols)

        self._load()
        log.info(f"Extracting transformer features from {len(texts)} transcripts...")
        records = []
        for i, text in enumerate(texts):
            if i % 50 == 0:
                log.info(f"  Transformer features: {i}/{len(texts)}")
            rec = {}
            rec.update(self._finbert_scores(text))
            rec.update(self._zeroshot_scores(text))
            records.append(rec)
        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class EarningsClassifier:
    """
    Stacked ensemble:
        - XGBoost (primary)
        - Random Forest
        - Logistic Regression (calibrated)
    Final predictions via soft-voting.
    """

    def __init__(self):
        self.xgb = None
        self.rf = None
        self.lr = None
        self.tfidf = None
        self.scaler = None
        self.feature_names_ = None
        self.is_fitted = False

    def _build_tfidf(self, texts: List[str], fit: bool = True) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if fit:
            self.tfidf = TfidfVectorizer(
                max_features=CFG.tfidf_max_features,
                ngram_range=CFG.ngram_range,
                sublinear_tf=True,
                min_df=3,
                strip_accents="unicode",
            )
            return self.tfidf.fit_transform(texts).toarray()
        return self.tfidf.transform(texts).toarray()

    def build_feature_matrix(
        self,
        texts: List[str],
        ling_feats: pd.DataFrame,
        transformer_feats: pd.DataFrame,
        fit: bool = True,
    ) -> np.ndarray:
        """Concatenate TF-IDF + linguistic + transformer features."""
        tfidf_mat = self._build_tfidf(texts, fit=fit)
        ling_mat = ling_feats.values
        trans_mat = transformer_feats.values

        X = np.hstack([tfidf_mat, ling_mat, trans_mat])

        if fit:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(with_mean=False)  # sparse-compatible
            X = self.scaler.fit_transform(X)
            self.feature_names_ = (
                list(self.tfidf.get_feature_names_out())
                + list(ling_feats.columns)
                + list(transformer_feats.columns)
            )
        else:
            X = self.scaler.transform(X)

        return X

    def _optuna_tune_xgb(self, X_train, y_train) -> dict:
        """Bayesian HPO for XGBoost using Optuna."""
        import optuna
        from sklearn.model_selection import cross_val_score
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "random_state": CFG.random_seed,
                "n_jobs": -1,
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=3, scoring="f1_macro", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=CFG.optuna_trials, show_progress_bar=False)
        log.info(f"Best XGB params (F1={study.best_value:.4f}): {study.best_params}")
        return study.best_params

    def fit(self, X_train, y_train, tune: bool = True):
        """Train the full ensemble with optional HPO."""
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        log.info("Training XGBoost classifier...")
        if tune:
            best_params = self._optuna_tune_xgb(X_train, y_train)
            best_params.update({
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "random_state": CFG.random_seed,
                "n_jobs": -1,
            })
        else:
            best_params = {
                "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "use_label_encoder": False, "eval_metric": "mlogloss",
                "random_state": CFG.random_seed, "n_jobs": -1,
            }

        self.xgb = xgb.XGBClassifier(**best_params)
        self.xgb.fit(X_train, y_train)

        log.info("Training Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=CFG.random_seed, n_jobs=-1,
        )
        self.rf.fit(X_train, y_train)

        log.info("Training Calibrated Logistic Regression...")
        base_lr = LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced",
            random_state=CFG.random_seed, n_jobs=-1,
        )
        self.lr = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
        self.lr.fit(X_train, y_train)

        self.is_fitted = True
        log.info("All ensemble members trained.")

    def predict_proba(self, X) -> np.ndarray:
        """Soft-voting ensemble probabilities."""
        p_xgb = self.xgb.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_lr = self.lr.predict_proba(X)
        # Weighted average: XGB gets higher weight (best performer)
        return 0.5 * p_xgb + 0.3 * p_rf + 0.2 * p_lr

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION & EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(classifier, X_test, y_test):
    """Comprehensive evaluation with all standard ML metrics."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, f1_score, accuracy_score
    )

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print("\n" + "═" * 60)
    print("  MODEL EVALUATION REPORT")
    print("═" * 60)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (Macro)      : {f1_macro:.4f}")
    print(f"  F1 (Weighted)   : {f1_weighted:.4f}")
    print(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    print("═" * 60)
    print("\nClassification Report:")
    target_names = [CFG.label_map[i] for i in sorted(CFG.label_map)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix (DOWN | NEUTRAL | UP):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("═" * 60)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
    }


def explain_with_shap(classifier, X_test, feature_names, n_samples: int = 100):
    """
    SHAP (SHapley Additive exPlanations) — the industry standard for ML
    explainability. Required by regulators in finance (SR 11-7 model risk).
    """
    try:
        import shap
        log.info("Computing SHAP values for model explainability...")

        X_sample = X_test[:n_samples]
        explainer = shap.TreeExplainer(classifier.xgb)
        shap_values = explainer.shap_values(X_sample)

        # Get mean absolute SHAP per class
        if isinstance(shap_values, list):
            mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        top_n = 15
        top_idx = np.argsort(mean_shap)[::-1][:top_n]

        print("\n" + "═" * 60)
        print("  TOP 15 MOST INFLUENTIAL FEATURES (SHAP)")
        print("═" * 60)
        for rank, idx in enumerate(top_idx, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"  {rank:>2}. {name:<35} SHAP={mean_shap[idx]:.4f}")
        print("═" * 60)

        return shap_values, mean_shap
    except ImportError:
        log.warning("SHAP not installed. Run: pip install shap")
        return None, None
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MLFLOW EXPERIMENT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def log_to_mlflow(metrics: dict, params: dict, model):
    """Log experiment to MLflow (run `mlflow ui` to view dashboard)."""
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(CFG.mlflow_tracking_uri)
        mlflow.set_experiment(CFG.mlflow_experiment)

        with mlflow.start_run(run_name="earnings-call-v1"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model.xgb, "xgboost_model")
            log.info(f"MLflow run logged → {CFG.mlflow_tracking_uri}")
            log.info("View dashboard: mlflow ui --backend-store-uri ./mlruns")
    except ImportError:
        log.warning("MLflow not installed. Skipping experiment tracking.")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FASTAPI INFERENCE SERVER
# ══════════════════════════════════════════════════════════════════════════════

def launch_api(classifier, ling_extractor, transformer_extractor):
    """
    Production REST API with async endpoints, request validation,
    and structured JSON responses. Industry-standard deployment pattern.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
        import uvicorn

        app = FastAPI(
            title="Earnings Call Intelligence Engine",
            description="Predict stock movement direction from earnings call transcripts.",
            version="1.0.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )

        class TranscriptRequest(BaseModel):
            transcript: str = Field(
                ...,
                min_length=50,
                description="Raw earnings call transcript text",
                example="Good afternoon. Revenue grew 28% YoY. We are raising full-year guidance..."
            )
            company_ticker: Optional[str] = Field(None, example="NVDA")

        class PredictionResponse(BaseModel):
            prediction: str
            confidence: float
            probabilities: Dict[str, float]
            ticker: Optional[str]
            top_signals: List[str]

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": "earnings-call-intelligence-v1"}

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(req: TranscriptRequest):
            try:
                ling = ling_extractor.extract([req.transcript])
                trans = transformer_extractor.extract([req.transcript])
                X = classifier.build_feature_matrix(
                    [req.transcript], ling, trans, fit=False
                )
                proba = classifier.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                label = CFG.label_map[pred_idx]

                # Top signals (simple feature importance proxy)
                top_signals = []
                if ling["pos_fin_density"].values[0] > 0.03:
                    top_signals.append("High positive financial term density")
                if ling["mentions_raise"].values[0]:
                    top_signals.append("Guidance raise detected")
                if ling["uncertainty_density"].values[0] > 0.04:
                    top_signals.append("High uncertainty language")
                if ling["mentions_lower"].values[0]:
                    top_signals.append("Guidance lowering detected")
                if not top_signals:
                    top_signals = ["Balanced/neutral linguistic signals"]

                return PredictionResponse(
                    prediction=label,
                    confidence=float(np.max(proba)),
                    probabilities={CFG.label_map[i]: float(p) for i, p in enumerate(proba)},
                    ticker=req.company_ticker,
                    top_signals=top_signals,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/docs-info")
        async def docs_info():
            return {
                "swagger_ui": f"http://{CFG.api_host}:{CFG.api_port}/docs",
                "redoc": f"http://{CFG.api_host}:{CFG.api_port}/redoc",
            }

        log.info(f"Launching API at http://{CFG.api_host}:{CFG.api_port}")
        log.info(f"Swagger UI: http://localhost:{CFG.api_port}/docs")
        uvicorn.run(app, host=CFG.api_host, port=CFG.api_port)

    except ImportError as e:
        log.error(f"FastAPI/uvicorn not installed: {e}")
        log.error("Run: pip install fastapi uvicorn")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DEMO MODE — Quick inference on example transcripts
# ══════════════════════════════════════════════════════════════════════════════

DEMO_TRANSCRIPTS = {
    "NVDA (Likely UP)": """
        Good afternoon everyone. This is our strongest quarter in company history.
        Revenue grew 122% year-over-year to $22.1 billion, beating consensus by 8%.
        Non-GAAP EPS of $5.16 significantly surpassed street estimates of $4.60.
        We are raising full-year guidance substantially. Data center demand far exceeds
        our supply capacity. We are extremely confident in our pipeline. Our win rate
        against competition is at an all-time high. We are doubling down on production.
        Bookings hit an all-time record this quarter. Outstanding execution across all segments.
    """,
    "INTC (Likely DOWN)": """
        Thank you for joining. We are reporting results that were below our own expectations.
        Revenue came in at $12.7 billion, approximately 8% below consensus estimates.
        We are not in a position to provide specific forward guidance at this time
        due to macroeconomic uncertainty and softness in enterprise spending.
        We are reevaluating our cost structure and may need to take further restructuring actions.
        Headwinds in our data center segment were more pronounced than anticipated.
        We remain cautiously optimistic but visibility into next quarter is limited.
        We are lowering our full-year guidance to reflect revised expectations.
    """,
    "MSFT (Likely NEUTRAL)": """
        Good afternoon. Our results were in line with expectations this quarter.
        Revenue of $56.5 billion met analyst consensus. Non-GAAP EPS of $2.94
        was consistent with our guidance. We are reaffirming our full-year outlook.
        Cloud growth was steady and our core business remains resilient.
        We saw mixed results by geography with strength in North America offsetting
        softness in EMEA. We continue to invest in AI capabilities for the long term.
        Margins held stable quarter-over-quarter. Growth was modest but sustainable.
    """,
}


def run_demo(classifier, ling_extractor, transformer_extractor):
    print("\n" + "═" * 60)
    print("  EARNINGS CALL INTELLIGENCE ENGINE — DEMO")
    print("═" * 60)

    for company, transcript in DEMO_TRANSCRIPTS.items():
        ling = ling_extractor.extract([transcript.strip()])
        trans = transformer_extractor.extract([transcript.strip()])
        X = classifier.build_feature_matrix(
            [transcript.strip()], ling, trans, fit=False
        )
        proba = classifier.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        label = CFG.label_map[pred_idx]
        confidence = float(np.max(proba))

        print(f"\n  📊 {company}")
        print(f"     Prediction  : {label}")
        print(f"     Confidence  : {confidence:.2%}")
        print(f"     Probabilities: DOWN={proba[0]:.2%} | NEUTRAL={proba[1]:.2%} | UP={proba[2]:.2%}")
        print(f"     Guidance raise: {bool(ling['mentions_raise'].values[0])} | "
              f"Guidance lower: {bool(ling['mentions_lower'].values[0])}")

    print("\n" + "═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN TRAINING ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    """Full ML pipeline: data → features → SMOTE → train → evaluate → SHAP → MLflow."""
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  EARNINGS CALL INTELLIGENCE ENGINE — TRAINING")
    log.info("=" * 55)

    gen = EarningsTranscriptGenerator()
    df = gen.generate(CFG.n_synthetic_samples)

    texts = df["transcript"].tolist()
    labels = df["label"].values

    # ── Step 2: Feature extraction ───────────────────────────────────────────
    ling_extractor = LinguisticFeatureExtractor()
    transformer_extractor = TransformerFeatureExtractor()

    ling_feats = ling_extractor.extract(texts)
    transformer_feats = transformer_extractor.extract(texts)

    # ── Step 3: Build feature matrix ─────────────────────────────────────────
    classifier = EarningsClassifier()
    X = classifier.build_feature_matrix(texts, ling_feats, transformer_feats, fit=True)
    log.info(f"Feature matrix shape: {X.shape}")

    # ── Step 4: Train/test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=CFG.test_size,
        random_state=CFG.random_seed, stratify=labels
    )

    # ── Step 5: SMOTE oversampling on training set ───────────────────────────
    log.info("Applying SMOTE to correct class imbalance...")
    smote = SMOTE(random_state=CFG.random_seed)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    log.info(f"Post-SMOTE train size: {X_train_bal.shape[0]} (from {X_train.shape[0]})")

    # ── Step 6: Train ensemble with Optuna HPO ───────────────────────────────
    classifier.fit(X_train_bal, y_train_bal, tune=True)

    # ── Step 7: Evaluation ───────────────────────────────────────────────────
    metrics = evaluate_model(classifier, X_test, y_test)

    # ── Step 8: SHAP explainability ──────────────────────────────────────────
    explain_with_shap(classifier, X_test, classifier.feature_names_)

    # ── Step 9: MLflow logging ───────────────────────────────────────────────
    log_to_mlflow(
        metrics=metrics,
        params={
            "n_samples": CFG.n_synthetic_samples,
            "tfidf_features": CFG.tfidf_max_features,
            "optuna_trials": CFG.optuna_trials,
            "use_transformers": CFG.use_transformers,
            "smote": True,
            "ensemble": "XGB+RF+LR",
        },
        model=classifier,
    )

    # ── Step 10: Demo inference ──────────────────────────────────────────────
    run_demo(classifier, ling_extractor, transformer_extractor)

    return classifier, ling_extractor, transformer_extractor


# ══════════════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Earnings Call Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train         # Full training + evaluation
  python main.py --mode demo          # Train then demo on sample transcripts
  python main.py --mode api           # Launch FastAPI inference server
  python main.py --mode train --no-transformers   # CPU-only fast mode
        """
    )
    parser.add_argument(
        "--mode", choices=["train", "api", "demo"], default="train",
        help="Execution mode"
    )
    parser.add_argument(
        "--no-transformers", action="store_true",
        help="Disable HuggingFace transformers (faster, CPU-friendly)"
    )
    parser.add_argument(
        "--samples", type=int, default=1200,
        help="Number of synthetic training samples"
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna HPO trials"
    )
    args = parser.parse_args()

    if args.no_transformers:
        CFG.use_transformers = False
    CFG.n_synthetic_samples = args.samples
    CFG.optuna_trials = args.trials

    if args.mode in ("train", "demo"):
        classifier, ling_extractor, transformer_extractor = train_pipeline()
        if args.mode == "api":
            launch_api(classifier, ling_extractor, transformer_extractor)

    elif args.mode == "api":
        log.info("Training model before launching API...")
        classifier, ling_extractor, transformer_extractor = train_pipeline()
        launch_api(classifier, ling_extractor, transformer_extractor)


if __name__ == "__main__":
    main()
