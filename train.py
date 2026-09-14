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

import data
from features_text import binoculars_score, get_device, stylometric_features

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / "reviews_baseline.csv"
OWN_CORPUS_DIR = BASE_DIR / "own_corpus"
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
STYLO_CACHE = OUTPUTS / "stylometric_features.csv"
BINOC_CACHE = OUTPUTS / "binoculars_sample_scores.csv"
METRICS_JSON = OUTPUTS / "metrics.json"

def _resolve_t2_model_name(size: str = "base") -> str:
    # huggingface_hub se cuelga de forma reproducible en esta red (ver
    # own_corpus/_download_model_direct.py y CONTEXTO.md) -- si ya se
    # descargo a mano ahi, se carga desde local en vez de disparar otra
    # descarga por red.
    repo = f"microsoft/deberta-v3-{size}"
    local_dir = BASE_DIR / "_model_cache" / f"deberta-v3-{size}"
    return str(local_dir) if (local_dir / "pytorch_model.bin").exists() else repo
# OpenAI y DeepSeek se reservan 100% como generadores "nunca vistos" -- ni
# entrenan ni calibran T2, solo se usan en evaluacion (ver CONTEXTO.md,
# seccion corpus propio). Dos held-out independientes, no uno.
T2_HELD_OUT_GENERATOR_FILES = {"openai_generated.csv": "openai", "deepseek_generated.csv": "deepseek"}
T2_TRAIN_GENERATOR_FILES = ["claude_generated.csv", "qwen_generated.csv", "qwen3_generated.csv"]

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
    """Train/val (Ott + generadores de train del corpus propio) y held-out
    que nunca entran en entrenamiento ni calibracion: los generadores
    "nunca vistos" (T2_HELD_OUT_GENERATOR_FILES, uno o mas) y Salminen/GPT-2
    completo (generador viejo, dominio Amazon en vez de hoteles) -- ver
    README, seccion T2, y CONTEXTO.md.
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

    held_out = {}
    for fname, gname in T2_HELD_OUT_GENERATOR_FILES.items():
        path = OWN_CORPUS_DIR / fname
        if path.exists():
            held_out[gname] = pd.read_csv(path)
    # Humano de referencia para el held-out: Salminen "OR" (humano real,
    # dominio Amazon, distinto al Ott usado en train) -- no reutiliza humano
    # ya visto en entrenamiento.
    held_out_human = salminen[~salminen["is_fake"]]["text"]

    return train_df, val_df, held_out, held_out_human, salminen


def train_t2_deberta(epochs: int = 3, batch_size: int | None = None, lr: float = 2e-5, model_size: str = "base") -> None:
    batch_size = batch_size or (16 if model_size == "base" else 8)
    device = get_device()
    model_name = _resolve_t2_model_name(model_size)
    train_df, val_df, held_out, held_out_human, salminen = _build_t2_datasets()

    print(f"Modelo: {model_name}")
    print(f"Train: {len(train_df)} {train_df['label'].value_counts().to_dict()}")
    print(f"Val: {len(val_df)} {val_df['label'].value_counts().to_dict()}")
    for gname, df in held_out.items():
        print(f"Held-out '{gname}' (nunca visto): {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dtype=float32 explicito: el checkpoint de deberta-v3-base/large esta en
    # fp16, y DeBERTa-v2/v3 es conocido por producir NaN en entrenamiento con
    # fp16 (issue documentado del propio modelo) -- confirmado en esta
    # sesion, loss se iba a NaN en la primera epoca hasta forzar fp32.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, dtype=torch.float32
    ).to(device)

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

    model_dir = MODELS_DIR / ("t2_deberta" if model_size == "base" else f"t2_deberta_{model_size}")
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
    metric_key = "t2_deberta" if model_size == "base" else f"t2_deberta_{model_size}"
    metrics = {
        metric_key: {
            "model": model_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "tpr_at_1pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.01),
            "tpr_at_5pct_fpr_val": _tpr_at_fpr(val_y, val_scores, 0.05),
        }
    }

    for gname, ai_df in held_out.items():
        if not len(held_out_human):
            continue
        ho_texts = list(ai_df["text"]) + list(held_out_human)
        ho_y = np.array([1] * len(ai_df) + [0] * len(held_out_human))
        ho_scores = _predict_scores(ho_texts)
        metrics[metric_key][f"n_held_out_{gname}"] = len(ai_df)
        metrics[metric_key][f"tpr_at_1pct_fpr_{gname}"] = _tpr_at_fpr(ho_y, ho_scores, 0.01)
        metrics[metric_key][f"tpr_at_5pct_fpr_{gname}"] = _tpr_at_fpr(ho_y, ho_scores, 0.05)

    if len(salminen):
        sal_scores = _predict_scores(salminen["text"].tolist())
        sal_y = salminen["is_fake"].astype(int).values
        metrics[metric_key]["tpr_at_1pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.01)
        metrics[metric_key]["tpr_at_5pct_fpr_salminen_2022_gpt2"] = _tpr_at_fpr(sal_y, sal_scores, 0.05)

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


OWN_CORPUS_FILES = {
    "claude_generated.csv": "claude-sonnet-5",
    "qwen_generated.csv": "qwen2.5-1.5b-instruct",
    "openai_generated.csv": "gpt-4o / gpt-4o-mini",
    "openai_gpt56luna_generated.csv": "gpt-5.6-luna",
    "openai_gpt56terra_generated.csv": "gpt-5.6-terra",
    "openai_gpt6astra_generated.csv": "gpt-6-astra",
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

    Ademas del agregado por fichero/generador, desglosa por `prompt_style`
    cuando el CSV trae esa columna con mas de un valor -- imprescindible para
    `gpt-6-astra`, donde 560 de las 610 filas son `hard_evasion` (estilo
    pensado explicitamente para evadir deteccion) y solo 50 son
    naive/adversarial/fewshot como el resto de generadores: mezclarlas en un
    unico bloque escondería si hard_evasion es mas o menos detectable que el
    resto, que es justo el dato que importa. T1 es lento en esta maquina
    (CPU, sin GPU) -- para no duplicar coste, el score de cada review se
    calcula una unica vez por fichero y se reutiliza (via mascara booleana)
    tanto para el agregado del fichero como para cada desglose por estilo.
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

    # Reanudable por fichero: T1 es lento (horas para el corpus completo en
    # esta maquina sin GPU) y ya se ha visto morir a mitad (portatil
    # suspendido de noche) -- si un fichero ya tiene su checkpoint guardado
    # con el mismo n de filas, se salta en vez de recalcular desde cero.
    cached = _load_metrics()
    metrics = {
        "t1_own_corpus": dict(cached.get("t1_own_corpus", {})),
        "t3_own_corpus": dict(cached.get("t3_own_corpus", {})),
    }
    for fname, generator_label in OWN_CORPUS_FILES.items():
        path = OWN_CORPUS_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path).reset_index(drop=True)
        key = fname.replace("_generated.csv", "").replace(".csv", "")

        already_done = (
            metrics["t1_own_corpus"].get(f"n_{key}") == len(df)
            and metrics["t3_own_corpus"].get(f"n_{key}") == len(df)
        )
        if already_done:
            print(f"Saltando {fname} ({len(df)} reviews): ya evaluado en un checkpoint previo.")
            continue

        print(f"Evaluando {fname} ({len(df)} reviews, generador: {generator_label})...")

        print(f"  Calculando T1 (Binoculars) para {len(df)} reviews...")
        t1_ai_scores_full = -pd.Series([binoculars_score(str(t)) for t in df["text"]])
        print(f"  Calculando T3 (estilometria) para {len(df)} reviews...")
        ai_feats_full = pd.DataFrame(
            [stylometric_features(str(t)) for t in df["text"]]
        )[STYLO_FEATURE_COLS]

        def _eval_mask(mask: np.ndarray, sub_key: str) -> None:
            n = int(mask.sum())
            if not n:
                return
            t1_sub = t1_ai_scores_full[mask]
            t1_y = np.array([1] * n + [0] * len(ott_t1_human))
            t1_scores = pd.concat([t1_sub, -ott_t1_human], ignore_index=True)
            metrics["t1_own_corpus"][f"n_{sub_key}"] = n
            metrics["t1_own_corpus"][f"tpr_at_1pct_fpr_{sub_key}"] = _tpr_at_fpr(t1_y, t1_scores, 0.01)
            metrics["t1_own_corpus"][f"tpr_at_5pct_fpr_{sub_key}"] = _tpr_at_fpr(t1_y, t1_scores, 0.05)

            feats_sub = ai_feats_full[mask]
            t3_X = pd.concat([feats_sub, ott_t3_human], ignore_index=True)
            t3_scores = t3_model.predict(t3_X)
            t3_y = np.array([1] * n + [0] * len(ott_t3_human))
            metrics["t3_own_corpus"][f"n_{sub_key}"] = n
            metrics["t3_own_corpus"][f"tpr_at_1pct_fpr_{sub_key}"] = _tpr_at_fpr(t3_y, t3_scores, 0.01)
            metrics["t3_own_corpus"][f"tpr_at_5pct_fpr_{sub_key}"] = _tpr_at_fpr(t3_y, t3_scores, 0.05)

        _eval_mask(np.ones(len(df), dtype=bool), key)

        if "prompt_style" in df.columns and df["prompt_style"].nunique() > 1:
            styles = df["prompt_style"].unique().tolist()
            for style in styles:
                mask = (df["prompt_style"] == style).values
                sub_key = f"{key}__{style}"
                print(f"  -> desglose {sub_key} ({mask.sum()} reviews)...")
                _eval_mask(mask, sub_key)

            # Astra en concreto necesita, ademas del desglose fino por estilo,
            # el contraste explicito "hard_evasion vs. el resto" pedido: es la
            # comparacion que importa para saber si el estilo anti-deteccion
            # es realmente mas dificil que naive/adversarial/fewshot juntos.
            if "hard_evasion" in styles:
                mask = (df["prompt_style"] != "hard_evasion").values
                sub_key = f"{key}__non_hard_evasion"
                print(f"  -> desglose {sub_key} ({mask.sum()} reviews)...")
                _eval_mask(mask, sub_key)

        # Checkpoint tras cada fichero: T1 es lento (CPU, sin GPU) y esto
        # puede tardar horas -- no perder progreso si algo falla a mitad.
        _save_metrics(metrics)

    print(json.dumps(metrics, indent=2))


def _load_t2_model():
    model_dir = MODELS_DIR / "t2_deberta"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), dtype=torch.float32).to(device)
    model.eval()
    return tokenizer, model, device


def _t2_scores(tokenizer, model, device, texts, batch_size: int = 32) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(chunk, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return np.array(scores)


def eval_t2_on_yelpnyc(n_per_class: int = 1500) -> None:
    """Evalúa el T2 ya entrenado (Ott + Claude + Qwen2.5 + Qwen3, ver
    `train_t2_deberta`) sobre el texto real de Yelp-NYC (Fase 1).

    **Expectativa honesta antes de correrlo, no una sorpresa a posteriori**:
    T2 fue entrenado para detectar texto GENERADO POR IA (humano vs. LLM).
    La etiqueta `is_fake` de Yelp-NYC son reviews HUMANAS que Yelp considera
    engañosas (proxy de spam, ver `data.load_yelpnyc_dataset`) — sin ningún
    LLM de por medio. Es el mismo tipo de tarea que Ott (humano real vs.
    humano mintiendo), donde T1 y T3 salieron prácticamente al nivel del
    azar (ver CONTEXTO.md / README, sección "Resultados de Fase 0"). Es
    razonable esperar que T2 tenga poca o ninguna señal aquí tampoco — se
    corre y se documenta el número real que salga, sea alto o bajo, sin
    ajustar el pipeline para forzar el resultado esperado.

    **Muestreado, no las 359.052 reviews completas**: `deberta-v3-base` en
    esta máquina es CPU-only — una pasada de inferencia sobre el dataset
    completo tardaría un orden de horas (referencia real: T1, mucho más
    ligero que T2, tardaba 2-12s/review en CPU en la Fase 0, de ahí el
    muestreo de 300/grupo que ya se usó entonces). Aquí se muestrean
    `n_per_class` reviews por clase (estratificado por `is_fake`), mismo
    patrón que T1 en Fase 0, no las 359k completas.

    Guarda en `outputs/metrics.json` bajo la clave `t2_yelpnyc`.
    """
    df = data.load_yelpnyc_dataset()
    fake = df[df["is_fake"]]
    genuine = df[~df["is_fake"]]
    n_fake = min(n_per_class, len(fake))
    n_genuine = min(n_per_class, len(genuine))
    sample = pd.concat(
        [fake.sample(n_fake, random_state=0), genuine.sample(n_genuine, random_state=0)]
    ).reset_index(drop=True)

    print(f"Cargando T2 desde {MODELS_DIR / 't2_deberta'} ...")
    tokenizer, model, device = _load_t2_model()
    print(f"Evaluando T2 sobre {len(sample)} reviews de Yelp-NYC "
          f"({n_fake} is_fake=True, {n_genuine} is_fake=False)...")
    scores = _t2_scores(tokenizer, model, device, sample["text"].tolist())
    y = sample["is_fake"].astype(int).to_numpy()

    metrics = {
        "t2_yelpnyc": {
            "n_fake": int(n_fake),
            "n_genuine": int(n_genuine),
            "tpr_at_1pct_fpr": _tpr_at_fpr(y, scores, 0.01),
            "tpr_at_5pct_fpr": _tpr_at_fpr(y, scores, 0.05),
        }
    }
    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


def fuse_signals() -> None:
    """Combina T1 (Binoculars) + T3 (estilometria) + T2 (DeBERTa) con una
    regresion logistica simple. Expectativa honesta antes de medir: como T1
    y T3 salieron casi al nivel del azar contra LLMs modernos en
    `eval_own_corpus`, es probable que aporten poco o nada sobre T2 solo --
    pero eso hay que comprobarlo, no asumirlo (por eso se guardan tambien
    las metricas de T2-solo en las mismas poblaciones, para comparar
    manzanas con manzanas).
    """
    from sklearn.linear_model import LogisticRegression

    train_df, val_df, held_out_ai, held_out_human, _salminen = _build_t2_datasets()
    t3_model = lgb.Booster(model_file=str(MODELS_DIR / "t3_lightgbm.txt"))
    tokenizer, t2_model, device = _load_t2_model()

    def _score_all(texts) -> np.ndarray:
        texts = [str(t) for t in texts]
        t1 = np.array([-binoculars_score(t) for t in texts])
        feats = pd.DataFrame([stylometric_features(t) for t in texts])[STYLO_FEATURE_COLS]
        t3 = t3_model.predict(feats)
        t2 = _t2_scores(tokenizer, t2_model, device, texts)
        return np.column_stack([t1, t3, t2])

    print(f"Calculando T1+T3+T2 para train ({len(train_df)})...")
    X_train, y_train = _score_all(train_df["text"]), train_df["label"].values
    print(f"Calculando T1+T3+T2 para val ({len(val_df)})...")
    X_val, y_val = _score_all(val_df["text"]), val_df["label"].values

    fusion = LogisticRegression(max_iter=1000)
    fusion.fit(X_train, y_train)
    val_fused = fusion.predict_proba(X_val)[:, 1]

    metrics = {
        "fusion_t1_t2_t3": {
            "coef_t1_t3_t2": fusion.coef_[0].tolist(),
            "tpr_at_1pct_fpr_val": _tpr_at_fpr(y_val, val_fused, 0.01),
            "tpr_at_5pct_fpr_val": _tpr_at_fpr(y_val, val_fused, 0.05),
            "tpr_at_1pct_fpr_val_t2_only": _tpr_at_fpr(y_val, X_val[:, 2], 0.01),
            "tpr_at_5pct_fpr_val_t2_only": _tpr_at_fpr(y_val, X_val[:, 2], 0.05),
        }
    }

    if len(held_out_ai) and len(held_out_human):
        ho_human_sample = held_out_human.sample(n=min(300, len(held_out_human)), random_state=0)
        print(f"Calculando T1+T3+T2 para held-out ({len(held_out_ai)} IA + {len(ho_human_sample)} humano)...")
        ho_texts = list(held_out_ai["text"]) + list(ho_human_sample)
        ho_y = np.array([1] * len(held_out_ai) + [0] * len(ho_human_sample))
        X_ho = _score_all(ho_texts)
        ho_fused = fusion.predict_proba(X_ho)[:, 1]
        metrics["fusion_t1_t2_t3"]["n_held_out_generator"] = len(held_out_ai)
        metrics["fusion_t1_t2_t3"]["tpr_at_1pct_fpr_held_out_generator"] = _tpr_at_fpr(ho_y, ho_fused, 0.01)
        metrics["fusion_t1_t2_t3"]["tpr_at_5pct_fpr_held_out_generator"] = _tpr_at_fpr(ho_y, ho_fused, 0.05)
        metrics["fusion_t1_t2_t3"]["tpr_at_1pct_fpr_held_out_generator_t2_only"] = _tpr_at_fpr(ho_y, X_ho[:, 2], 0.01)
        metrics["fusion_t1_t2_t3"]["tpr_at_5pct_fpr_held_out_generator_t2_only"] = _tpr_at_fpr(ho_y, X_ho[:, 2], 0.05)

    _save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "t3"
    if step == "t3":
        train_t3_baseline()
    elif step == "t1":
        score_binoculars_sample()
    elif step == "t2":
        train_t2_deberta(model_size=sys.argv[2] if len(sys.argv) > 2 else "base")
    elif step == "eval_own_corpus":
        evaluate_own_corpus()
    elif step == "fusion":
        fuse_signals()
    elif step == "yelpnyc_t2":
        eval_t2_on_yelpnyc()
    else:
        print(f"Paso desconocido: {step} (usa 't3', 't1', 't2', 'eval_own_corpus', 'fusion' o 'yelpnyc_t2')")
