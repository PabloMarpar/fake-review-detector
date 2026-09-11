"""Entrena y evalúa las señales de texto de la Fase 0.

Uso:
    python train.py t3      # estilometría + LightGBM (rápido, dataset completo)
    python train.py t1      # T1 zero-shot (Binoculars) sobre una muestra -- lento en CPU,
                             # pensado para correr en background; reanudable (guarda por lotes).

T2 (DeBERTa-v3 afinado) se añade en un paso posterior -- necesita más tiempo de
CPU todavía y merece su propio script de entrenamiento.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from features_text import binoculars_score, stylometric_features

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / "reviews_baseline.csv"
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
STYLO_CACHE = OUTPUTS / "stylometric_features.csv"
BINOC_CACHE = OUTPUTS / "binoculars_sample_scores.csv"
METRICS_JSON = OUTPUTS / "metrics.json"

STYLO_FEATURE_COLS = [
    "n_words",
    "sentence_length_mean",
    "sentence_length_variance",
    "type_token_ratio",
    "avg_word_length",
    "superlative_density",
    "adj_ratio",
    "adv_ratio",
    "noun_ratio",
    "verb_ratio",
    "exclamation_density",
    "flesch_reading_ease",
]


def _tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def _load_metrics() -> dict:
    if METRICS_JSON.exists():
        return json.loads(METRICS_JSON.read_text())
    return {}


def _save_metrics(update: dict) -> None:
    metrics = _load_metrics()
    metrics.update(update)
    OUTPUTS.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))


def build_stylometric_dataset() -> pd.DataFrame:
    """Calcula T3 sobre TODO el dataset (es rápido, sin red neuronal) y cachea a disco."""
    df = pd.read_csv(DATA_CSV)
    if STYLO_CACHE.exists():
        cached = pd.read_csv(STYLO_CACHE)
        if len(cached) == len(df):
            return pd.concat([df.reset_index(drop=True), cached[STYLO_FEATURE_COLS]], axis=1)

    print(f"Calculando estilometría para {len(df)} reviews...")
    records = []
    for i, text in enumerate(df["text"]):
        records.append(stylometric_features(str(text)))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(df)}")
    feats = pd.DataFrame(records)
    OUTPUTS.mkdir(exist_ok=True)
    feats.to_csv(STYLO_CACHE, index=False)
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def train_t3_baseline() -> None:
    df = build_stylometric_dataset()
    X = df[STYLO_FEATURE_COLS]
    y = df["is_fake"].astype(int)

    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, df["source_dataset"], test_size=0.2, random_state=0, stratify=df[["source_dataset", "is_fake"]]
    )

    model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=0)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "t3_stylometry": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "tpr_at_1pct_fpr": _tpr_at_fpr(y_test, scores, 0.01),
            "tpr_at_5pct_fpr": _tpr_at_fpr(y_test, scores, 0.05),
            "auc": float(pd.Series(scores).corr(pd.Series(y_test.values), method="spearman")),
        }
    }

    # desglose por dataset de origen -- Ott (humano vs. humano-MTurk) es un problema
    # distinto de Salminen (humano vs. GPT-2), no tiene sentido reportarlos solo mezclados
    for source in src_test.unique():
        mask = (src_test == source).values
        metrics["t3_stylometry"][f"tpr_at_5pct_fpr_{source}"] = _tpr_at_fpr(
            y_test[mask], scores[mask], 0.05
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "t3_lightgbm.txt"))
    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


def score_binoculars_sample(n_per_group: int = 300) -> None:
    """T1 sobre una muestra estratificada. Lento en CPU (~2-12s/review) -- se
    guarda incrementalmente para poder reanudar si se interrumpe.
    """
    df = pd.read_csv(DATA_CSV)
    groups = df.groupby(["source_dataset", "is_fake"])
    sample = groups.apply(
        lambda g: g.sample(min(n_per_group, len(g)), random_state=0), include_groups=False
    ).reset_index(level=[0, 1])

    done = {}
    if BINOC_CACHE.exists():
        prev = pd.read_csv(BINOC_CACHE)
        done = dict(zip(prev["text"], prev["binoculars_score"]))
        print(f"Reanudando: {len(done)} ya calculados de una ejecución anterior.")

    OUTPUTS.mkdir(exist_ok=True)
    results = []
    pending = sample[~sample["text"].isin(done.keys())]
    print(f"Calculando T1 (Binoculars) para {len(pending)}/{len(sample)} reviews restantes...")

    for i, (_, row) in enumerate(pending.iterrows()):
        score = binoculars_score(str(row["text"]))
        results.append({"text": row["text"], "is_fake": row["is_fake"],
                         "source_dataset": row["source_dataset"], "binoculars_score": score})
        if (i + 1) % 20 == 0:
            pd.concat([pd.DataFrame(results), pd.read_csv(BINOC_CACHE) if BINOC_CACHE.exists() else pd.DataFrame()]) \
                .to_csv(BINOC_CACHE, index=False)
            print(f"  {i + 1}/{len(pending)} guardado")

    all_results = pd.concat(
        [pd.DataFrame(results), pd.read_csv(BINOC_CACHE) if BINOC_CACHE.exists() else pd.DataFrame()]
    ).drop_duplicates(subset="text")
    all_results.to_csv(BINOC_CACHE, index=False)

    y = all_results["is_fake"].astype(int)
    # Binoculars: score BAJO = más probable IA -> invertimos el signo para que
    # "score alto" = "más probable fake", consistente con el resto del proyecto.
    scores = -all_results["binoculars_score"]
    metrics = {
        "t1_binoculars": {
            "n_samples": len(all_results),
            "tpr_at_1pct_fpr": _tpr_at_fpr(y, scores, 0.01),
            "tpr_at_5pct_fpr": _tpr_at_fpr(y, scores, 0.05),
        }
    }
    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "t3"
    if step == "t3":
        train_t3_baseline()
    elif step == "t1":
        score_binoculars_sample()
    else:
        print(f"Paso desconocido: {step} (usa 't3' o 't1')")
