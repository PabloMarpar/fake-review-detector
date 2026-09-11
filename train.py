"""Entrena y evalúa las señales de texto de la Fase 0.

Uso:
    python train.py t3      # estilometría + LightGBM (rápido, dataset completo)
    python train.py t1      # T1 zero-shot (Binoculars) sobre una muestra -- lento en CPU,
                             # pensado para correr en background; reanudable (guarda por lotes).
    python train.py t2      # DeBERTa-v3-base afinado sobre Ott + corpus propio (GPU recomendada).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from features_text import binoculars_score, get_device, stylometric_features

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / "reviews_baseline.csv"
OWN_CORPUS_DIR = BASE_DIR / "own_corpus"
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
STYLO_CACHE = OUTPUTS / "stylometric_features.csv"
BINOC_CACHE = OUTPUTS / "binoculars_sample_scores.csv"
METRICS_JSON = OUTPUTS / "metrics.json"

T2_MODEL_REPO = "microsoft/deberta-v3-base"
# huggingface_hub se cuelga de forma reproducible en esta red (ver
# own_corpus/_download_model_direct.py y CONTEXTO.md) -- si ya se descargo a
# mano ahi, se carga desde local en vez de disparar otra descarga por red.
_T2_LOCAL_MODEL_DIR = BASE_DIR / "_model_cache" / "deberta-v3-base"
T2_MODEL_NAME = str(_T2_LOCAL_MODEL_DIR) if (_T2_LOCAL_MODEL_DIR / "pytorch_model.bin").exists() else T2_MODEL_REPO
# gpt-4o(-mini) se reserva 100% como generador "nunca visto" -- ni entrena ni
# calibra T2, solo se usa en evaluacion (ver CONTEXTO.md, seccion corpus propio).
T2_HELD_OUT_GENERATOR_FILE = "openai_generated.csv"
T2_TRAIN_GENERATOR_FILES = ["claude_generated.csv", "qwen_generated.csv"]

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
            # Bug encontrado (2026-09-11): `results` no se vaciaba aqui, asi que
            # cada guardado volvia a concatenar TODO lo acumulado en memoria con
            # TODO el fichero ya guardado -> duplicados crecientes (a las ~1100
            # iteraciones el CSV tenia 1820 filas para solo 260 reviews unicas).
            # drop_duplicates() de mas abajo lo disimulaba al final, pero de
            # camino cada guardado reescribia un CSV cada vez mas grande sin
            # necesidad. Fix: vaciar `results` tras cada guardado.
            new_batch = pd.DataFrame(results)
            results = []
            existing = pd.read_csv(BINOC_CACHE) if BINOC_CACHE.exists() else pd.DataFrame()
            pd.concat([existing, new_batch]).drop_duplicates(subset="text").to_csv(BINOC_CACHE, index=False)
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

    # desglose por dataset de origen -- mismo motivo que en T3 (ver
    # train_t3_baseline): Salminen (humano vs. GPT-2) y Ott (humano real vs.
    # humano mintiendo, sin LLM de por medio) son problemas distintos, y
    # mezclarlos en un solo número agregado esconde cuál de los dos está
    # tirando del resultado.
    for source in all_results["source_dataset"].unique():
        mask = (all_results["source_dataset"] == source).values
        metrics["t1_binoculars"][f"tpr_at_5pct_fpr_{source}"] = _tpr_at_fpr(
            y[mask], scores[mask], 0.05
        )
        metrics["t1_binoculars"][f"tpr_at_1pct_fpr_{source}"] = _tpr_at_fpr(
            y[mask], scores[mask], 0.01
        )

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


class _ReviewDataset(Dataset):
    def __init__(self, encodings: dict, labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _build_t2_datasets():
    """Train/val (Ott + generadores de train del corpus propio) y dos sets
    held-out que nunca entran en entrenamiento ni calibracion: el generador
    "nunca visto" (T2_HELD_OUT_GENERATOR_FILE) y Salminen/GPT-2 completo
    (generador viejo, dominio Amazon en vez de hoteles) -- ver README,
    seccion T2, y CONTEXTO.md.
    """
    baseline = pd.read_csv(DATA_CSV)
    ott = baseline[baseline["source_dataset"] == "ott_2013"]
    salminen = baseline[baseline["source_dataset"] == "salminen_2022_gpt2"]

    human = pd.DataFrame({"text": ott["text"], "label": 0})

    ai_frames = []
    for fname in T2_TRAIN_GENERATOR_FILES:
        path = OWN_CORPUS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            ai_frames.append(pd.DataFrame({"text": df["text"], "label": 1}))
    if not ai_frames:
        raise RuntimeError(
            f"Ningun generador de entrenamiento disponible en {OWN_CORPUS_DIR} "
            f"(se esperaba alguno de {T2_TRAIN_GENERATOR_FILES})"
        )
    ai = pd.concat(ai_frames, ignore_index=True)

    data = pd.concat([human, ai], ignore_index=True).dropna(subset=["text"])
    train_df, val_df = train_test_split(
        data, test_size=0.15, random_state=0, stratify=data["label"]
    )

    held_out_path = OWN_CORPUS_DIR / T2_HELD_OUT_GENERATOR_FILE
    held_out_ai = pd.read_csv(held_out_path) if held_out_path.exists() else pd.DataFrame()
    # Humano de referencia para el held-out: Salminen "OR" (humano real,
    # dominio Amazon, distinto al Ott usado en train) -- no reutiliza humano
    # ya visto en entrenamiento.
    held_out_human = salminen[~salminen["is_fake"]]["text"]

    return train_df, val_df, held_out_ai, held_out_human, salminen


def train_t2_deberta(epochs: int = 3, batch_size: int = 16, lr: float = 2e-5) -> None:
    device = get_device()
    train_df, val_df, held_out_ai, held_out_human, salminen = _build_t2_datasets()

    print(f"Train: {len(train_df)} {train_df['label'].value_counts().to_dict()}")
    print(f"Val: {len(val_df)} {val_df['label'].value_counts().to_dict()}")
    print(f"Held-out generador nunca visto ({T2_HELD_OUT_GENERATOR_FILE}): {len(held_out_ai)}")

    tokenizer = AutoTokenizer.from_pretrained(T2_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(T2_MODEL_NAME, num_labels=2).to(device)

    def _encode(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=256, return_tensors="pt")

    train_ds = _ReviewDataset(_encode(train_df["text"]), torch.tensor(train_df["label"].values))
    val_ds = _ReviewDataset(_encode(val_df["text"]), torch.tensor(val_df["label"].values))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        print(f"  epoch {epoch + 1}/{epochs} loss={total_loss / len(train_loader):.4f}")

    model_dir = MODELS_DIR / "t2_deberta"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    def _predict_scores(texts) -> np.ndarray:
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size * 2):
                chunk = list(texts[i:i + batch_size * 2])
                enc = tokenizer(chunk, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
                scores.extend(probs.cpu().tolist())
        model.train()
        return np.array(scores)

    val_scores = _predict_scores(val_df["text"].tolist())
    val_y = val_df["label"].values
    metrics = {
        "t2_deberta": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "tpr_at_1pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.01),
            "tpr_at_5pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.05),
        }
    }

    if len(held_out_ai) and len(held_out_human):
        ho_texts = list(held_out_ai["text"]) + list(held_out_human)
        ho_y = np.array([1] * len(held_out_ai) + [0] * len(held_out_human))
        ho_scores = _predict_scores(ho_texts)
        metrics["t2_deberta"]["n_held_out_generator"] = len(held_out_ai)
        metrics["t2_deberta"]["tpr_at_1pct_fpr_held_out_generator"] = _tpr_at_fpr(ho_y, ho_scores, 0.01)
        metrics["t2_deberta"]["tpr_at_5pct_fpr_held_out_generator"] = _tpr_at_fpr(ho_y, ho_scores, 0.05)

    if len(salminen):
        sal_scores = _predict_scores(salminen["text"].tolist())
        sal_y = salminen["is_fake"].astype(int).values
        metrics["t2_deberta"]["tpr_at_1pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.01)
        metrics["t2_deberta"]["tpr_at_5pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.05)

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


OWN_CORPUS_FILES = {
    "claude_generated.csv": "claude-sonnet-5",
    "qwen_generated.csv": "qwen2.5-1.5b-instruct",
    "openai_generated.csv": "gpt-4o / gpt-4o-mini",
}


def evaluate_own_corpus() -> None:
    """Mide T1 y T3 (ya entrenados contra Ott/GPT-2) frente al corpus propio
    con LLMs modernos de verdad -- hasta ahora nunca se habian evaluado
    contra nada mas reciente que GPT-2 (2019). Reutiliza como referencia
    humana lo ya cacheado para Ott truthful (mismo dataset que T1/T3 ya
    conocen) en vez de recalcular: 300 reviews con score T1 ya calculado
    (`binoculars_sample_scores.csv`) y las features T3 de todo Ott
    (`stylometric_features.csv`, alineado por orden de fila con
    `reviews_baseline.csv`).
    """
    baseline = pd.read_csv(DATA_CSV)
    stylo_all = pd.read_csv(STYLO_CACHE)
    ott_truthful_mask = (baseline["source_dataset"] == "ott_2013") & (~baseline["is_fake"])
    ott_t3_human = stylo_all[ott_truthful_mask.values][STYLO_FEATURE_COLS]

    binoc_cache = pd.read_csv(BINOC_CACHE)
    ott_t1_human = binoc_cache[
        (binoc_cache["source_dataset"] == "ott_2013") & (~binoc_cache["is_fake"])
    ]["binoculars_score"]

    t3_model = lgb.Booster(model_file=str(MODELS_DIR / "t3_lightgbm.txt"))

    metrics = {"t1_own_corpus": {}, "t3_own_corpus": {}}
    for fname, generator_label in OWN_CORPUS_FILES.items():
        path = OWN_CORPUS_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        print(f"Evaluando {fname} ({len(df)} reviews, generador: {generator_label})...")

        # T1
        t1_ai_scores = -pd.Series([binoculars_score(str(t)) for t in df["text"]])
        t1_y = np.array([1] * len(t1_ai_scores) + [0] * len(ott_t1_human))
        t1_scores = pd.concat([t1_ai_scores, -ott_t1_human], ignore_index=True)
        key = fname.replace("_generated.csv", "")
        metrics["t1_own_corpus"][f"n_{key}"] = len(df)
        metrics["t1_own_corpus"][f"tpr_at_1pct_fpr_{key}"] = _tpr_at_fpr(t1_y, t1_scores, 0.01)
        metrics["t1_own_corpus"][f"tpr_at_5pct_fpr_{key}"] = _tpr_at_fpr(t1_y, t1_scores, 0.05)

        # T3
        ai_feats = pd.DataFrame([stylometric_features(str(t)) for t in df["text"]])[STYLO_FEATURE_COLS]
        t3_X = pd.concat([ai_feats, ott_t3_human], ignore_index=True)
        t3_scores = t3_model.predict(t3_X)
        t3_y = np.array([1] * len(ai_feats) + [0] * len(ott_t3_human))
        metrics["t3_own_corpus"][f"n_{key}"] = len(df)
        metrics["t3_own_corpus"][f"tpr_at_1pct_fpr_{key}"] = _tpr_at_fpr(t3_y, t3_scores, 0.01)
        metrics["t3_own_corpus"][f"tpr_at_5pct_fpr_{key}"] = _tpr_at_fpr(t3_y, t3_scores, 0.05)

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "t3"
    if step == "t3":
        train_t3_baseline()
    elif step == "t1":
        score_binoculars_sample()
    elif step == "t2":
        train_t2_deberta()
    elif step == "eval_own_corpus":
        evaluate_own_corpus()
    else:
        print(f"Paso desconocido: {step} (usa 't3', 't1', 't2' o 'eval_own_corpus')")
