"""Fase 1 -- features de comportamiento (Rayana & Akoglu) + similitud de
texto entre reviews (Nivel C del README), sobre Yelp-NYC.

## Por qué existe este fichero

`features_graph.py` mide cuatro veces (fusión, OddBall, co-bursting,
FRAUDAR/Leiden) que las señales de grafo actuales (`net_rur`/`net_rtr`/
`net_rsr`) son en realidad *target encoding*: el score de cada nodo es la
tasa de fraude de su propia comunidad, calculada con las etiquetas reales
(`leave_one_out_cluster_scores`). Son útiles para medir "¿hay señal en la
estructura de comunidades?" pero **no son features usables en producto**
-- un cliente real no tiene etiquetas de fraude para calcular esa tasa.

Lo que falta, y es lo que implementa este módulo, son las features de
comportamiento clásicas (Rayana & Akoglu, "Collective Opinion Spam
Detection", KDD 2015 -- el mismo paper de SpEagle que el proyecto ya cita
como referencia) que **no usan ninguna etiqueta**: se calculan solo a
partir de rating/fecha/texto/reviewer/negocio, exactamente lo que un
cliente B2B real sí tiene el día 1. Y la similitud de texto entre reviews
del mismo negocio escritas por reviewers DISTINTOS (Nivel C, descrito en
`README.md` pero nunca implementado, pese a que Yelp-NYC sí trae texto).

## Qué NO hace este módulo

No toca `features_graph.py`, `data.py`, `train.py`, `graph_speagle.py` ni
`CONTEXTO.md` (otros agentes trabajando en paralelo sobre esos ficheros).
Reutiliza en modo lectura `features_graph.topk_capture` (evaluación
top-k, ya escrita y usada en todo el proyecto) y
`features_graph.compute_yelpnyc_graph_signal_scores` (para el experimento
(c): behavior+texto+grafo) -- ninguna de las dos se modifica.

## Protocolo de evaluación (obligatorio, para comparar manzanas con
manzanas contra el resto del proyecto)

Copiado literal de `features_graph._fuse_graph_signal_scores`: split
70/30 estratificado, `random_state=SEED=42`, sobre
`data.load_yelpnyc_dataset()`. Métricas guardadas en
`outputs/metrics_behavior.json` (fichero propio, nunca
`outputs/metrics.json`).

## CLI

    python features_behavior.py block1       # features de comportamiento (rápido, ~1 min)
    python features_behavior.py similarity    # TF-IDF + similitud Nivel C (lento, ver timings reales abajo)
    python features_behavior.py graph_signals # net_rur/net_rtr/net_rsr/burst por review, vía features_graph
    python features_behavior.py evaluate      # combina los tres cachés + evalúa (a)/(b)/(c)
    python features_behavior.py all           # los cuatro pasos en orden

Cada paso cachea su resultado en `outputs/behavior_*.csv` para poder
ejecutarlos como llamadas independientes (esta máquina no tiene GPU y
`run_in_background` explícito tiene incidentes documentados de procesos
duplicados -- mejor varias llamadas cortas en primer plano que una sola
larga).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import data
from features_graph import topk_capture

SEED = 42
OUTPUTS_DIR = Path(__file__).parent / "outputs"
METRICS_JSON = OUTPUTS_DIR / "metrics_behavior.json"

BLOCK1_CACHE = OUTPUTS_DIR / "behavior_block1_features.csv"
SIMILARITY_CACHE = OUTPUTS_DIR / "behavior_text_similarity.csv"
GRAPH_SIGNALS_CACHE = OUTPUTS_DIR / "behavior_graph_signals.csv"

REVIEWER_COLS = [
    "reviewer_n_reviews", "reviewer_mnr", "reviewer_pr", "reviewer_nr",
    "reviewer_avg_rd", "reviewer_bst", "reviewer_erd", "reviewer_etg",
    "reviewer_active_life_days",
]
REVIEW_COLS = [
    "rating_deviation", "singleton", "rating_extremity", "temporal_position",
    "text_length_words", "uppercase_ratio", "exclamation_ratio", "lexical_richness",
]
BUSINESS_COLS = ["business_n_reviews", "business_rating_std", "business_singleton_reviewer_ratio"]
SIMILARITY_COLS = [
    "text_max_sim_business_diff_reviewer", "text_n_above_thresh_business_diff_reviewer",
    "text_max_sim_business_week_diff_reviewer", "text_n_above_thresh_business_week_diff_reviewer",
]
GRAPH_COLS = ["net_rur", "net_rtr", "net_rsr", "burst"]


def _save_metrics(update: dict) -> None:
    metrics = {}
    if METRICS_JSON.exists():
        metrics = json.loads(METRICS_JSON.read_text())
    metrics.update(update)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str))


def load_df() -> pd.DataFrame:
    """Mismo dataset y mismo orden de filas en las tres cachés -- las tres
    se construyen a partir de esta llamada, para poder combinarlas después
    por posición (`reset_index(drop=True)`), sin necesidad de guardar un id
    de fila explícito."""
    return data.load_yelpnyc_dataset().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bloque 1 -- features de comportamiento (Rayana & Akoglu), CERO etiquetas
# ---------------------------------------------------------------------------

def compute_business_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas por negocio. `singleton_reviewer_ratio` usa el conteo
    total de reviews por reviewer (no la etiqueta) -- disponible con
    cualquier CSV de cliente real."""
    reviewer_counts = df["reviewer_id"].value_counts()
    singleton_reviewers = set(reviewer_counts[reviewer_counts == 1].index)
    is_singleton = df["reviewer_id"].isin(singleton_reviewers)
    biz = df.groupby("business_id").agg(
        n_reviews=("rating", "size"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
    )
    biz["rating_std"] = biz["rating_std"].fillna(0.0)
    biz["singleton_reviewer_ratio"] = is_singleton.groupby(df["business_id"]).mean()
    return biz


def _rating_entropy(ratings: pd.Series) -> float:
    counts = ratings.value_counts(normalize=True)
    counts = counts[counts > 0]
    return float(-(counts * np.log2(counts)).sum())


_GAP_BINS = [-1, 0, 1, 2, 3, 7, 14, 30, 90, 365, np.inf]


def _gap_entropy(dates_sorted_diff_days: pd.Series) -> float:
    d = dates_sorted_diff_days.dropna()
    if len(d) == 0:
        return 0.0
    binned = pd.cut(d, bins=_GAP_BINS)
    counts = binned.value_counts(normalize=True)
    counts = counts[counts > 0]
    return float(-(counts * np.log2(counts)).sum())


def compute_reviewer_stats(df: pd.DataFrame, business_rating_mean: pd.Series) -> pd.DataFrame:
    """Features por reviewer -- MNR, PR/NR, avgRD, BST, ERD, ETG, n_reviews,
    vida activa. Definiciones siguiendo Rayana & Akoglu (KDD 2015) y la
    literatura clásica de spam de reviews (Mukherjee et al. 2013 para
    MNR/PR/NR/avgRD; BST/ERD/ETG del propio Rayana & Akoglu).

    - **MNR**: máximo de reviews del reviewer en un único día, normalizado
      por el máximo GLOBAL entre todos los reviewers (no por el propio
      `n_reviews` del reviewer -- así un reviewer con 1 review en 1 día no
      empata con uno de 40 reviews en 1 día cada uno).
    - **BST**: burstiness. `L` = ventana de actividad del reviewer en días
      (última review - primera). `BST = max(0, 1 - L/28)` (ventana de 28
      días, el valor que usa el paper original) para reviewers con >=2
      reviews -- 1 si todo concentrado en un día, 0 si la actividad se
      extiende 28+ días. Reviewers con una sola review no tienen ventana
      que medir -- se les asigna 0 (decisión explícita, no un artefacto:
      "no hay ráfaga observable con un solo punto de datos", distinto de
      "ráfaga mínima").
    - **ERD**: entropía de Shannon (base 2) de la distribución de ratings
      *propia* del reviewer -- un reviewer que solo pone 5 estrellas tiene
      ERD=0; uno que reparte por igual entre 1-5 tiene ERD=log2(5)≈2,32.
    - **ETG**: entropía de los huecos temporales (en días) entre reviews
      consecutivas del reviewer, discretizados en bins
      [0,1,2,3,7,14,30,90,365,inf] -- huecos muy regulares (todos en el
      mismo bin) dan ETG bajo, huecos erráticos dan ETG alto.
    """
    work = df[["reviewer_id", "business_id", "rating"]].copy()
    work["biz_mean_rating"] = df["business_id"].map(business_rating_mean)
    work["abs_dev"] = (work["rating"] - work["biz_mean_rating"]).abs()
    work["date"] = pd.to_datetime(df["date"])

    g = work.groupby("reviewer_id")
    n_reviews = g.size().rename("reviewer_n_reviews")
    avg_rd = g["abs_dev"].mean().rename("reviewer_avg_rd")
    pr = g["rating"].apply(lambda s: float((s >= 4).mean())).rename("reviewer_pr")
    nr = g["rating"].apply(lambda s: float((s <= 2).mean())).rename("reviewer_nr")
    erd = g["rating"].apply(_rating_entropy).rename("reviewer_erd")

    per_day = work.groupby(["reviewer_id", "date"]).size()
    max_per_day = per_day.groupby("reviewer_id").max()
    max_per_day = max_per_day.reindex(n_reviews.index).fillna(1)
    mnr = (max_per_day / max_per_day.max()).rename("reviewer_mnr")

    date_min = g["date"].min()
    date_max = g["date"].max()
    active_life_days = (date_max - date_min).dt.days.rename("reviewer_active_life_days")
    bst = (1.0 - (active_life_days / 28.0)).clip(lower=0.0)
    bst = bst.where(n_reviews > 1, 0.0).rename("reviewer_bst")

    def _gaps(dates: pd.Series) -> pd.Series:
        return dates.sort_values().diff().dt.days

    etg = g["date"].apply(lambda s: _gap_entropy(_gaps(s))).rename("reviewer_etg")

    return pd.concat(
        [n_reviews, mnr, pr, nr, avg_rd, bst, erd, etg, active_life_days], axis=1
    )


def compute_review_level_features(df: pd.DataFrame, business_stats: pd.DataFrame,
                                   reviewer_n_reviews: pd.Series) -> pd.DataFrame:
    """Features calculadas directamente sobre la review individual."""
    dates = pd.to_datetime(df["date"])
    biz_mean = df["business_id"].map(business_stats["rating_mean"])
    biz_n = df["business_id"].map(business_stats["n_reviews"])

    out = pd.DataFrame(index=df.index)
    out["rating_deviation"] = (df["rating"] - biz_mean).abs()
    out["singleton"] = (df["reviewer_id"].map(reviewer_n_reviews) == 1).astype(int)
    out["rating_extremity"] = (df["rating"] - 3).abs() / 2.0

    rank = dates.groupby(df["business_id"]).rank(method="first")
    denom = (biz_n - 1).where(biz_n > 1, 1)
    out["temporal_position"] = ((rank - 1) / denom).where(biz_n > 1, 0.5)

    text = df["text"].astype(str)
    words = text.str.split()
    out["text_length_words"] = words.str.len().astype(float)
    n_alpha = text.str.count(r"[A-Za-z]")
    n_upper = text.str.count(r"[A-Z]")
    out["uppercase_ratio"] = (n_upper / n_alpha.replace(0, np.nan)).fillna(0.0)
    n_chars = text.str.len().replace(0, np.nan)
    out["exclamation_ratio"] = (text.str.count("!") / n_chars).fillna(0.0)
    out["lexical_richness"] = words.apply(
        lambda ws: (len(set(w.lower() for w in ws)) / len(ws)) if ws else 0.0
    )
    return out


def build_block1_features(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    business_stats = compute_business_stats(df)
    reviewer_stats = compute_reviewer_stats(df, business_stats["rating_mean"])
    review_feats = compute_review_level_features(df, business_stats, reviewer_stats["reviewer_n_reviews"])
    elapsed = time.time() - t0

    reviewer_feats = df[["reviewer_id"]].merge(
        reviewer_stats[REVIEWER_COLS], left_on="reviewer_id", right_index=True, how="left"
    )[REVIEWER_COLS].reset_index(drop=True)
    business_feats = df[["business_id"]].merge(
        business_stats.rename(columns={
            "n_reviews": "business_n_reviews", "rating_std": "business_rating_std",
            "singleton_reviewer_ratio": "business_singleton_reviewer_ratio",
        })[BUSINESS_COLS],
        left_on="business_id", right_index=True, how="left",
    )[BUSINESS_COLS].reset_index(drop=True)

    X = pd.concat([review_feats.reset_index(drop=True), reviewer_feats, business_feats], axis=1)
    timings = {"block1_seconds": round(elapsed, 2), "n_rows": int(len(df))}
    if verbose:
        print(f"Bloque 1 (features de comportamiento): {timings}")
    return X, timings


# ---------------------------------------------------------------------------
# Bloque 2 -- similitud de texto (Nivel C): near-duplicates entre reviewers
# distintos del mismo negocio
# ---------------------------------------------------------------------------

def build_tfidf_matrix(texts: pd.Series, verbose: bool = True) -> tuple[csr_matrix, dict]:
    """TF-IDF a nivel de carácter (n-gramas 3-5), normalizado L2 (para que
    el producto escalar sea directamente similitud coseno).

    `min_df=3, max_df=0.3`: sondeo previo con la config por defecto
    (`min_df=1`, sin `max_df`) sobre el negocio más grande (7.378 reviews)
    dio una matriz de similitud con **densidad 99,36%** -- casi todos los
    pares comparten AL MENOS un n-grama de 3-5 caracteres, lo esperable en
    inglés corriente ("the ", " and ", terminaciones "-ing", etc.). El TF-IDF
    ya debería bajar el peso de esos n-gramas ubicuos vía IDF, pero
    `max_df=0.3` los elimina del vocabulario directamente (descarta
    n-gramas presentes en más del 30% de las 359.052 reviews) -- reduce
    tiempo/memoria Y concentra la señal en n-gramas realmente distintivos
    (frases/plantillas compartidas, no partículas del idioma). `min_df=3`
    descarta n-gramas que aparecen en <3 reviews (mayoritariamente ruido:
    erratas, nombres propios únicos) sin coste de señal real para detectar
    coordinación (un n-grama que solo aparece 1-2 veces en todo el corpus no
    puede, por definición, ser evidencia de reutilización entre reviewers).
    """
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.3, dtype=np.float32
    )
    X = vec.fit_transform(texts.str.lower())
    X = normalize(X, copy=False)
    elapsed = time.time() - t0
    timings = {
        "tfidf_fit_seconds": round(elapsed, 1),
        "vocab_size": int(X.shape[1]),
        "nnz": int(X.nnz),
    }
    if verbose:
        print(f"TF-IDF char(3-5), min_df=3, max_df=0.3: {timings}")
    return X, timings


def _max_similarity_within_groups(X: csr_matrix, group_key: pd.Series, reviewer_id: pd.Series,
                                   threshold: float, label: str, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Para cada review, similitud coseno máxima con OTRA review del mismo
    grupo escrita por un reviewer DISTINTO -- eso es lo que convierte esto en
    señal de coordinación entre cuentas y no de "el mismo tío copiando y
    pegando su propia review". Calculado por bloques (un negocio o
    negocio+semana a la vez), nunca la matriz 359.052x359.052 completa.
    """
    group_key = pd.Series(group_key).reset_index(drop=True)
    reviewer_arr = pd.Series(reviewer_id).reset_index(drop=True).to_numpy()
    n = len(group_key)
    max_sim = np.zeros(n, dtype=np.float32)
    n_above = np.zeros(n, dtype=np.int32)

    groups = pd.DataFrame({"i": np.arange(n), "k": group_key.to_numpy()}).groupby("k")["i"]
    t0 = time.time()
    n_groups_multi = 0
    max_group_size = 0
    for _, idx_series in groups:
        idx = idx_series.to_numpy()
        if len(idx) < 2:
            continue
        n_groups_multi += 1
        max_group_size = max(max_group_size, len(idx))
        Xg = X[idx]
        S = (Xg @ Xg.T).toarray()
        rid = reviewer_arr[idx]
        same_rev = rid[:, None] == rid[None, :]
        S[same_rev] = -1.0
        row_max = S.max(axis=1)
        has_other_reviewer = row_max >= 0.0
        row_max = np.where(has_other_reviewer, row_max, 0.0)
        max_sim[idx] = row_max
        n_above[idx] = np.where(has_other_reviewer, (S >= threshold).sum(axis=1), 0)
    elapsed = time.time() - t0
    timings = {
        "seconds": round(elapsed, 1),
        "n_groups_multi_review": n_groups_multi,
        "max_group_size": int(max_group_size),
    }
    if verbose:
        print(f"[{label}] similitud por bloques: {timings}")
    return max_sim, n_above, timings


def build_similarity_features(df: pd.DataFrame, threshold: float = 0.5, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    X, tfidf_timings = build_tfidf_matrix(df["text"], verbose=verbose)

    max_sim_biz, n_above_biz, t_biz = _max_similarity_within_groups(
        X, df["business_id"], df["reviewer_id"], threshold=threshold, label="business", verbose=verbose
    )
    dates = pd.to_datetime(df["date"])
    week_key = df["business_id"].astype(str) + "_" + dates.dt.to_period("W").astype(str)
    max_sim_week, n_above_week, t_week = _max_similarity_within_groups(
        X, week_key, df["reviewer_id"], threshold=threshold, label="business+week", verbose=verbose
    )

    feats = pd.DataFrame({
        "text_max_sim_business_diff_reviewer": max_sim_biz,
        "text_n_above_thresh_business_diff_reviewer": n_above_biz,
        "text_max_sim_business_week_diff_reviewer": max_sim_week,
        "text_n_above_thresh_business_week_diff_reviewer": n_above_week,
    })
    timings = {
        "threshold": threshold,
        "tfidf": tfidf_timings,
        "business_level": t_biz,
        "business_week_level": t_week,
        "total_seconds": round(tfidf_timings["tfidf_fit_seconds"] + t_biz["seconds"] + t_week["seconds"], 1),
    }
    if verbose:
        print(f"Bloque 2 (similitud de texto) -- tiempo total: {timings['total_seconds']}s")
    return feats, timings


# ---------------------------------------------------------------------------
# Señales de grafo existentes (solo lectura de features_graph, para el
# experimento (c): behavior + texto + grafo) -- NO se modifica ese fichero.
# ---------------------------------------------------------------------------

def build_graph_signal_features(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    from features_graph import compute_yelpnyc_graph_signal_scores

    t0 = time.time()
    signals = compute_yelpnyc_graph_signal_scores(df, verbose=verbose)
    elapsed = time.time() - t0
    feats = pd.DataFrame({name: signals[name] for name in GRAPH_COLS})
    timings = {"seconds": round(elapsed, 1)}
    if verbose:
        print(f"Señales de grafo (net_rur/net_rtr/net_rsr/burst) recalculadas: {timings}")
    return feats, timings


# ---------------------------------------------------------------------------
# Bloque 3 -- evaluación
# ---------------------------------------------------------------------------

def evaluate_single_feature(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    auc = roc_auc_score(y, x)
    ap = average_precision_score(y, x)
    return {"auc": round(float(auc), 4), "ap": round(float(ap), 4)}


def _fit_lgb(X_train, y_train, feature_names):
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED,
        importance_type="gain", verbosity=-1,
    )
    model.fit(X_train, y_train)
    importances = dict(zip(feature_names, [round(float(v), 2) for v in model.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    return model, importances


def _classifier_block(X: pd.DataFrame, y: np.ndarray, idx_train: np.ndarray, idx_test: np.ndarray,
                       cold_start_mask: np.ndarray, label: str, verbose: bool = True) -> dict:
    feature_names = list(X.columns)
    Xarr = np.nan_to_num(X.to_numpy(dtype=float))
    X_train, X_test = Xarr[idx_train], Xarr[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    model, importances = _fit_lgb(X_train, y_train, feature_names)
    proba_test = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    topk = topk_capture(proba_test, y_test, fracs=(0.01, 0.05, 0.10))

    cold_test_mask = cold_start_mask[idx_test]
    n_cold = int(cold_test_mask.sum())
    cold_result = None
    if n_cold >= 20 and y_test[cold_test_mask].sum() >= 2:
        cold_result = {
            "n": n_cold,
            "base_rate": round(float(y_test[cold_test_mask].mean()), 4),
            "auc": round(float(roc_auc_score(y_test[cold_test_mask], proba_test[cold_test_mask])), 4),
            "ap": round(float(average_precision_score(y_test[cold_test_mask], proba_test[cold_test_mask])), 4),
        }

    result = {
        "features": feature_names,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "base_rate_test": round(float(y_test.mean()), 4),
        "auc": round(float(auc), 4),
        "ap": round(float(ap), 4),
        "topk": topk,
        "feature_importances_gain": importances,
        "cold_start_singleton_reviewers": cold_result,
    }
    if verbose:
        print(f"\n--- Clasificador: {label} ---")
        print(f"n_train={result['n_train']}, n_test={result['n_test']}, base_rate={result['base_rate_test']}")
        print(f"AUC={result['auc']}, AP={result['ap']}")
        print(f"top-1/5/10%: {topk}")
        print(f"Importancias (gain, top 10): {dict(list(importances.items())[:10])}")
        print(f"Cold start (reviewers de una sola review): {cold_result}")
    return result


def run_evaluation(df: pd.DataFrame, block1: pd.DataFrame, similarity: pd.DataFrame,
                    graph: pd.DataFrame | None, verbose: bool = True) -> dict:
    y = df["is_fake"].to_numpy().astype(int)
    cold_start_mask = (block1["reviewer_n_reviews"] == 1).to_numpy()

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y
    )

    X_behavior = pd.concat([block1.reset_index(drop=True), similarity.reset_index(drop=True)], axis=1)

    results = {
        "dataset": "Yelp-NYC",
        "n_total": int(len(df)),
        "base_rate_total": round(float(y.mean()), 4),
        "n_cold_start_singleton_total": int(cold_start_mask.sum()),
        "base_rate_cold_start_total": round(float(y[cold_start_mask].mean()), 4),
    }

    # (a) cada feature sola, sin entrenar nada -- el propio valor crudo
    # como score de ranking, sobre TODO el dataset (no hay ajuste de
    # parámetros que pueda hacer overfitting a un split concreto).
    single = {}
    for col in X_behavior.columns:
        single[col] = evaluate_single_feature(X_behavior[col].to_numpy(), y)
    if graph is not None:
        for col in GRAPH_COLS:
            single[col] = evaluate_single_feature(graph[col].to_numpy(), y)
    results["single_feature_auc_ap"] = dict(sorted(single.items(), key=lambda kv: -kv[1]["ap"]))
    if verbose:
        print("\n--- (a) Cada feature sola (AUC/AP sobre todo el dataset, sin split) ---")
        for name, m in results["single_feature_auc_ap"].items():
            print(f"  {name}: AUC={m['auc']}, AP={m['ap']}")

    # (b) todas las de comportamiento+texto juntas
    results["classifier_behavior_only"] = _classifier_block(
        X_behavior, y, idx_train, idx_test, cold_start_mask, "(b) comportamiento + similitud de texto", verbose
    )

    # (c) + señales de grafo existentes
    if graph is not None:
        X_full = pd.concat([X_behavior, graph.reset_index(drop=True)], axis=1)
        results["classifier_behavior_plus_graph"] = _classifier_block(
            X_full, y, idx_train, idx_test, cold_start_mask, "(c) comportamiento + texto + grafo", verbose
        )
    else:
        results["classifier_behavior_plus_graph"] = None
        if verbose:
            print("\n(c) omitido: no hay caché de señales de grafo -- correr "
                  "`python features_behavior.py graph_signals` primero.")

    _save_metrics({"behavior_features": results})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if target in ("block1", "all"):
        df = load_df()
        X, timings = build_block1_features(df, verbose=True)
        X.to_csv(BLOCK1_CACHE, index=False)
        _save_metrics({"behavior_block1_timings": timings})
        print(f"Guardado {BLOCK1_CACHE}")

    if target in ("similarity", "all"):
        df = load_df()
        feats, timings = build_similarity_features(df, threshold=0.5, verbose=True)
        feats.to_csv(SIMILARITY_CACHE, index=False)
        _save_metrics({"behavior_similarity_timings": timings})
        print(f"Guardado {SIMILARITY_CACHE}")

    if target in ("graph_signals", "all"):
        df = load_df()
        feats, timings = build_graph_signal_features(df, verbose=True)
        feats.to_csv(GRAPH_SIGNALS_CACHE, index=False)
        _save_metrics({"behavior_graph_signals_timings": timings})
        print(f"Guardado {GRAPH_SIGNALS_CACHE}")

    if target in ("evaluate", "all"):
        df = load_df()
        if not BLOCK1_CACHE.exists():
            X_block1, _ = build_block1_features(df, verbose=True)
        else:
            X_block1 = pd.read_csv(BLOCK1_CACHE)
        if not SIMILARITY_CACHE.exists():
            X_sim, _ = build_similarity_features(df, verbose=True)
        else:
            X_sim = pd.read_csv(SIMILARITY_CACHE)
        X_graph = pd.read_csv(GRAPH_SIGNALS_CACHE) if GRAPH_SIGNALS_CACHE.exists() else None
        run_evaluation(df, X_block1, X_sim, X_graph, verbose=True)


if __name__ == "__main__":
    main()
