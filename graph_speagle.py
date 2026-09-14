"""SpEagle (Rayana & Akoglu, KDD 2015, "Collective Opinion Spam Detection
using Behavioral and Content Clues") -- primera implementación real de este
método en el proyecto, tras cuatro sesiones citándolo solo como número de
referencia (`speeagle_benchmark_auc_referencia = 0.78` en
`features_graph.py`, nunca implementado).

## Por qué este módulo, en una frase

Las señales de grafo actuales (`net_rur`/`net_rtr`/`net_rsr` vía
`leave_one_out_cluster_scores` en `features_graph.py`) son *target
encoding*: el score de cada nodo es la tasa de fraude de su comunidad
calculada CON las etiquetas reales (leave-one-out, pero etiquetas reales al
fin y al cabo). En un cliente real sin etiquetas eso no es usable en
producto. SpEagle es atractivo justo porque su prior no necesita ninguna
etiqueta -- se deriva de features de comportamiento, y la propagación de
sospecha (Loopy Belief Propagation) tampoco ve ninguna etiqueta en ningún
momento.

## Qué implementa este módulo

1. **Priors mínimos sin etiquetas** (`compute_minimal_priors`): 5 features de
   comportamiento simples y auto-contenidas (no depende de
   `features_behavior.py`, que otro agente puede estar construyendo en
   paralelo con un conjunto mucho más completo -- esto es deliberadamente el
   mínimo viable, documentado como tal, pensado para integrarse con ese
   módulo más adelante si existe cuando se lea esto):
   - `rating_deviation` (review): |rating - media del negocio|.
   - `temporal_density_z` (review): z-score de ráfaga semanal del negocio en
     la semana de esta review (implementación local, ver nota de scope en
     `_weekly_burst_zscore` sobre por qué no se importa
     `features_graph.detect_bursts_yelpnyc` aunque calcula algo parecido).
   - `user_extreme_ratio` (reviewer): fracción de ratings 1 o 5 del usuario.
   - `user_max_per_day` (reviewer): máximo de reviews que el usuario publicó
     en un mismo día natural, en toda su historia.
   - `product_burst_peak` (negocio): pico (máximo) de `temporal_density_z`
     entre todas las reviews de ese negocio.
2. **MRF bipartito reviewer-review-negocio + Loopy Belief Propagation**
   (`run_lbp`): tres tipos de nodo, dos estados (benigno/fraude) cada uno,
   matriz de compatibilidad `[[1-eps, eps], [eps, 1-eps]]` (homofilia).
   Implementado nativo con NumPy (`np.bincount` para agregar mensajes hacia
   nodos hub de grado alto -- usuarios/negocios), sin NetworkX -- ver
   precedente de que NetworkX revienta memoria a esta escala en
   `features_graph.py` (sección `net_rsr`/`net_homo`, Yelp-NYC). El grafo
   real, si se contraen los nodos de review, es el bipartito
   reviewer<->negocio proyectado (cada review es una "arista etiquetada"
   entre su usuario y su negocio) -- con más de un negocio en común entre
   dos usuarios cualesquiera aparecen ciclos, de ahí que haga falta LBP
   (con bucles) y no BP exacto sobre un árbol.
3. **Evaluación** (`_evaluate_scores`): AUC, AP (PR-AUC), precision/recall/
   lift en top-1/5/10%, base rate -- siempre juntos, nunca un AUC suelto.
   Split 70/30 estratificado `random_state=42`, idéntico al que usa
   `_fuse_graph_signal_scores` en `features_graph.py` -- pero, a diferencia
   de esa función, aquí `idx_train` NO se usa para entrenar nada: SpEagle no
   ve ninguna etiqueta en ningún paso (ni para los priors ni para LBP), el
   split solo sirve para poder comparar el AUC/AP en el mismo held-out que
   las demás señales de grafo del proyecto. Se reporta también el corte
   cold-start (reviewers con una sola review en TODO el dataset), que ningún
   otro resultado del proyecto había mirado todavía -- es el caso donde el
   target encoding de `net_rur` no puede ayudar en absoluto (una comunidad
   de tamaño 1 en `net_rur` recibe la tasa base global, cero información),
   así que es el sitio donde un método real como SpEagle tiene que demostrar
   que aporta algo.

## Lo que NO hace este módulo (a propósito)

- No perfila clusters (eso es `profile_cluster.py`) ni afirma nada sobre
  identidad de las personas -- solo produce un score de sospecha por
  review/usuario/negocio.
- No usa NetworkX, no usa ninguna librería de BP genérica (`pgmpy`,
  `pomegranate`) -- la estructura bipartita concreta de este problema (cada
  review tiene EXACTAMENTE 2 vecinos: un usuario y un negocio) permite
  implementar el mensaje-pasando con operaciones vectorizadas de NumPy/
  `np.bincount`, sin necesidad de un framework de factor graphs genérico.
- No mezcla resultados de train/test: los priors y la propagación corren
  sobre el grafo completo (igual que hace `leave_one_out_cluster_scores` en
  `features_graph.py` con Louvain -- el clustering/la propagación necesitan
  ver toda la estructura), y el split 70/30 se usa solo para decidir sobre
  qué filas se calculan las métricas.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import data

SEED = 42
OUTPUTS_DIR = Path(__file__).parent / "outputs"
METRICS_SPEAGLE_JSON = OUTPUTS_DIR / "metrics_speagle.json"

# Cifras de referencia citadas en la tarea, para comparar sin tener que ir a
# buscarlas -- ver salvedades reales de comparabilidad en
# `features_graph.print_reference_comparison` (no se repiten aquí, mismo
# espíritu: no citar como si fuera una carrera justa sin más).
SPEAGLE_PAPER_AUC_REFERENCIA = 0.78
FUSION_SIN_NET_RUR_HONESTA_REFERENCIA = {
    "auc": 0.6231,
    "ap": 0.1775,
    "base_rate": 0.1027,
    "nota": (
        "Cifras del diagnóstico del maestro (protocolo honesto: net_rur/net_rtr/"
        "net_rsr recalculadas usando SOLO labels de train, no del dataset "
        "completo) que motivó esta tarea -- no recalculadas por este módulo."
    ),
}


def _save_speagle_metrics(update: dict) -> None:
    """Mismo patrón que `_save_graph_metrics`/`_save_metrics` del resto del
    proyecto (lee, `.update()`, reescribe) pero sobre un fichero PROPIO
    (`outputs/metrics_speagle.json`), no `outputs/metrics.json` -- varios
    agentes escribiendo a la vez en el mismo fichero se pisarían.
    """
    metrics = {}
    if METRICS_SPEAGLE_JSON.exists():
        metrics = json.loads(METRICS_SPEAGLE_JSON.read_text())
    metrics.update(update)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_SPEAGLE_JSON.write_text(json.dumps(metrics, indent=2, default=str))


# ---------------------------------------------------------------------------
# Priors sin etiquetas -- mínimos y auto-contenidos, ver docstring del módulo
# ---------------------------------------------------------------------------

def _percentile_prior(raw: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Convierte un array de valores crudos (cualquier escala/distribución)
    en un prior P(fraude) acotado en `[lo, hi]`, vía rango percentil.

    Percentil en vez de z-score/sigmoid con escala fija: robusto a colas
    largas (rating_deviation, z-scores de ráfaga) sin tener que elegir a
    mano un factor de escala por feature. Acotado lejos de 0/1 (`lo=0.05,
    hi=0.95` por defecto) para que ningún prior sature LBP desde la primera
    iteración -- un prior en 0.0 o 1.0 exactos haría que la evidencia previa
    domine sobre cualquier mensaje entrante, sin importar lo que digan los
    vecinos.
    """
    raw = np.asarray(raw, dtype=float)
    ranks = rankdata(raw, method="average")
    pct = (ranks - 1) / max(len(raw) - 1, 1)
    return lo + pct * (hi - lo)


def _weekly_burst_zscore(df: pd.DataFrame) -> np.ndarray:
    """Z-score de ráfaga semanal del negocio, por review (implementación
    LOCAL e independiente, no una llamada a
    `features_graph.detect_bursts_yelpnyc`).

    Calcula esencialmente lo mismo (z-score de cada review frente al ritmo
    semanal propio de su negocio, con ventanas sin actividad rellenadas a 0
    entre la primera y la última semana de vida del negocio) pero
    reimplementado aquí a propósito: `features_graph.py` está siendo editado
    en paralelo por otro agente en esta misma sesión (ver tarea), e importar
    una función de un fichero en edición activa sería frágil (su firma o
    comportamiento podría cambiar bajo los pies de este módulo). Es
    duplicación de ~15 líneas de pandas, no de una pieza de arquitectura --
    se acepta el coste para mantener este fichero autocontenido.
    """
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period("W")
    business = df["business_id"]

    counts = df.groupby([business, period]).size()
    z_lookup = {}
    for business_id, sub in counts.groupby(level=0):
        sub = sub.droplevel(0)
        full_index = pd.period_range(sub.index.min(), sub.index.max(), freq="W")
        full_counts = sub.reindex(full_index, fill_value=0)
        mean = full_counts.mean()
        std = full_counts.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_values = pd.Series(0.0, index=full_counts.index)
        else:
            z_values = (full_counts - mean) / std
        z_lookup.update({(business_id, p): float(v) for p, v in z_values.items()})

    return np.array([z_lookup[(b, p)] for b, p in zip(business, period)])


def compute_minimal_priors(df: pd.DataFrame) -> dict:
    """Calcula los 5 priors sin etiquetas y los índices review->usuario/
    negocio que necesita `run_lbp`.

    Todo se calcula sobre `df` completo (no sobre train/test) -- son
    features de comportamiento derivadas de la estructura del dato, ninguna
    usa `is_fake` en ningún punto (se puede verificar: `is_fake`/`label` no
    aparece en ninguna línea de esta función).
    """
    df = df.reset_index(drop=True)

    # --- review-level ---
    business_mean_rating = df.groupby("business_id")["rating"].transform("mean")
    raw_rating_deviation = (df["rating"] - business_mean_rating).abs().to_numpy()
    raw_temporal_density = _weekly_burst_zscore(df)

    # --- user-level (una entrada por reviewer_id único) ---
    is_extreme = df["rating"].isin([1, 5])
    user_extreme_ratio_series = is_extreme.groupby(df["reviewer_id"]).mean()
    day = pd.to_datetime(df["date"]).dt.date
    user_max_per_day_series = (
        df.groupby([df["reviewer_id"], day]).size().groupby(level=0).max()
    )

    # --- product-level (una entrada por business_id único) ---
    product_burst_peak_series = (
        pd.Series(raw_temporal_density, index=df.index).groupby(df["business_id"]).max()
    )

    user_ids, reviewer_idx = np.unique(df["reviewer_id"].to_numpy(), return_inverse=True)
    product_ids, product_idx = np.unique(df["business_id"].to_numpy(), return_inverse=True)

    raw_user_extreme_ratio = user_extreme_ratio_series.reindex(user_ids).to_numpy()
    raw_user_max_per_day = user_max_per_day_series.reindex(user_ids).to_numpy()
    raw_product_burst_peak = product_burst_peak_series.reindex(product_ids).to_numpy()

    review_prior_fraud = (
        _percentile_prior(raw_rating_deviation) + _percentile_prior(raw_temporal_density)
    ) / 2.0
    user_prior_fraud = (
        _percentile_prior(raw_user_extreme_ratio) + _percentile_prior(raw_user_max_per_day)
    ) / 2.0
    product_prior_fraud = _percentile_prior(raw_product_burst_peak)

    return {
        "review_prior_fraud": review_prior_fraud,
        "user_prior_fraud": user_prior_fraud,
        "product_prior_fraud": product_prior_fraud,
        "reviewer_idx": reviewer_idx.astype(np.int64),
        "product_idx": product_idx.astype(np.int64),
        "user_ids": user_ids,
        "product_ids": product_ids,
        "raw": {
            "rating_deviation": raw_rating_deviation,
            "temporal_density_z": raw_temporal_density,
            "user_extreme_ratio": raw_user_extreme_ratio,
            "user_max_per_day": raw_user_max_per_day,
            "product_burst_peak": raw_product_burst_peak,
        },
    }


# ---------------------------------------------------------------------------
# MRF bipartito + Loopy Belief Propagation
# ---------------------------------------------------------------------------

def _as_log_prior(p_fraud: np.ndarray) -> np.ndarray:
    """`(n,)` probabilidades de fraude -> `(n, 2)` log-priors
    `[log P(benigno), log P(fraude)]`."""
    p = np.clip(p_fraud, 1e-6, 1 - 1e-6)
    return np.log(np.stack([1 - p, p], axis=1))


def _damp(old_log: np.ndarray, new_log: np.ndarray, damping: float) -> np.ndarray:
    """Mezcla `old`/`new` en espacio de PROBABILIDAD (no en log), para que
    el damping sea una media ponderada de verdad y no una media geométrica
    de logs. `damping=0` (por defecto) devuelve `new_log` sin tocar."""
    if damping <= 0:
        return new_log
    old_p = np.exp(old_log - old_log.max(axis=1, keepdims=True))
    old_p /= old_p.sum(axis=1, keepdims=True)
    new_p = np.exp(new_log - new_log.max(axis=1, keepdims=True))
    new_p /= new_p.sum(axis=1, keepdims=True)
    mixed = damping * old_p + (1 - damping) * new_p
    logm = np.log(np.clip(mixed, 1e-300, None))
    return logm - logm.max(axis=1, keepdims=True)


def run_lbp(
    priors: dict,
    epsilon: float = 0.15,
    max_iter: int = 150,
    tol: float = 1e-4,
    damping: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Loopy Belief Propagation sobre el MRF bipartito reviewer-review-
    negocio, actualización SÍNCRONA (Jacobi: todos los mensajes nuevos se
    calculan a partir de los viejos de la iteración anterior, se intercambian
    al final de la iteración -- evita dependencia del orden de recorrido).

    **Por qué `np.bincount` en vez de un bucle por nodo o NetworkX**: cada
    review tiene EXACTAMENTE 2 vecinos (su usuario y su negocio), así que la
    única agregación "de muchos a uno" real del problema es en los nodos hub
    (usuario/negocio, que pueden tener cientos de reviews). Sumar logs de
    mensajes entrantes agrupados por usuario/negocio es exactamente
    `np.bincount(indice, weights=log_mensaje)` -- sin bucle Python, O(n_reviews)
    por iteración, factible a 359k reviews sin ningún problema de escala (a
    diferencia de Louvain/NetworkX sobre `net_rsr`/`net_homo`, documentado
    como inviable en `features_graph.py`).

    Devuelve, entre otras cosas, `converged` (bool) y `n_iterations` (int) --
    **medidos de verdad, no asumidos**: se compara el cambio máximo en la
    creencia de fraude de las reviews entre iteraciones consecutivas contra
    `tol`.

    **Hallazgo real medido en Yelp-NYC completo (359.052 reviews), no
    asumido de la teoría**: con `damping=0` (Jacobi puro) el LBP **NO
    converge -- oscila con periodo 2 exacto**, atascado en
    `delta=0.939597` durante las últimas ~80 iteraciones de 100 (el cambio
    entre iteraciones consecutivas deja de decrecer y se estabiliza en ese
    valor constante, no en cero). Es un modo de fallo conocido de LBP
    síncrono en grafos con muchos ciclos cortos y nodos hub de grado alto
    (aquí: negocios con cientos de reviews) -- el `damping=0.5` por defecto
    de este módulo lo resuelve, convergiendo limpio en 67 iteraciones
    (56,5s). Curiosamente, en esta ejecución concreta el AUC/AP del score
    SIN converger (tomado en la iteración 100 de la oscilación) salió
    IDÉNTICO al de la versión que sí converge -- indicio de que el orden
    relativo de los scores ya era estable aunque los valores numéricos
    oscilaran, pero no es algo en lo que confiar sin `damping`: se reporta
    `converged`/`n_iterations` siempre para que quien use este módulo pueda
    ver si está en ese caso.
    """
    reviewer_idx = priors["reviewer_idx"]
    product_idx = priors["product_idx"]
    n_reviews = len(reviewer_idx)
    n_users = len(priors["user_ids"])
    n_products = len(priors["product_ids"])

    log_prior_R = _as_log_prior(priors["review_prior_fraud"])
    log_prior_U = _as_log_prior(priors["user_prior_fraud"])
    log_prior_P = _as_log_prior(priors["product_prior_fraud"])

    compat = np.array([[1 - epsilon, epsilon], [epsilon, 1 - epsilon]])

    def compat_transform(log_h: np.ndarray) -> np.ndarray:
        """`log_h` (n,2): log-evidencia no normalizada en el nodo origen,
        excluyendo el mensaje del destino. Aplica la matriz de compatibilidad
        (`m(x_j) = sum_{x_i} h(x_i) * compat(x_i, x_j)`) y devuelve el
        mensaje saliente normalizado (log-espacio, log-suma-exp = 0)."""
        h = np.exp(log_h - log_h.max(axis=1, keepdims=True))
        m = h @ compat
        m = np.clip(m, 1e-300, None)
        logm = np.log(m)
        return logm - logm.max(axis=1, keepdims=True)

    # mensajes iniciales = uniformes (sin información), log(1)=0 en ambos estados
    msg_U_to_R = np.zeros((n_reviews, 2))
    msg_P_to_R = np.zeros((n_reviews, 2))
    msg_R_to_U = np.zeros((n_reviews, 2))
    msg_R_to_P = np.zeros((n_reviews, 2))

    prev_belief_fraud = None
    converged = False
    n_iter_run = 0
    delta_history = []
    t0 = time.time()

    for it in range(1, max_iter + 1):
        incoming_at_user = np.empty((n_users, 2))
        incoming_at_user[:, 0] = log_prior_U[:, 0] + np.bincount(
            reviewer_idx, weights=msg_R_to_U[:, 0], minlength=n_users
        )
        incoming_at_user[:, 1] = log_prior_U[:, 1] + np.bincount(
            reviewer_idx, weights=msg_R_to_U[:, 1], minlength=n_users
        )
        pre_h_u = incoming_at_user[reviewer_idx] - msg_R_to_U
        new_msg_U_to_R = compat_transform(pre_h_u)

        incoming_at_product = np.empty((n_products, 2))
        incoming_at_product[:, 0] = log_prior_P[:, 0] + np.bincount(
            product_idx, weights=msg_R_to_P[:, 0], minlength=n_products
        )
        incoming_at_product[:, 1] = log_prior_P[:, 1] + np.bincount(
            product_idx, weights=msg_R_to_P[:, 1], minlength=n_products
        )
        pre_h_p = incoming_at_product[product_idx] - msg_R_to_P
        new_msg_P_to_R = compat_transform(pre_h_p)

        # review -> usuario: usa el prior propio + mensaje ENTRANTE del negocio
        # (el único vecino que no es "usuario", ya que review solo tiene 2)
        new_msg_R_to_U = compat_transform(log_prior_R + msg_P_to_R)
        # review -> negocio: simétrico, usa el mensaje entrante del usuario
        new_msg_R_to_P = compat_transform(log_prior_R + msg_U_to_R)

        new_msg_U_to_R = _damp(msg_U_to_R, new_msg_U_to_R, damping)
        new_msg_P_to_R = _damp(msg_P_to_R, new_msg_P_to_R, damping)
        new_msg_R_to_U = _damp(msg_R_to_U, new_msg_R_to_U, damping)
        new_msg_R_to_P = _damp(msg_R_to_P, new_msg_R_to_P, damping)

        msg_U_to_R, msg_P_to_R = new_msg_U_to_R, new_msg_P_to_R
        msg_R_to_U, msg_R_to_P = new_msg_R_to_U, new_msg_R_to_P

        belief_R_log = log_prior_R + msg_U_to_R + msg_P_to_R
        belief_R = np.exp(belief_R_log - belief_R_log.max(axis=1, keepdims=True))
        belief_R = belief_R / belief_R.sum(axis=1, keepdims=True)
        belief_fraud = belief_R[:, 1]

        n_iter_run = it
        if prev_belief_fraud is not None:
            delta = float(np.max(np.abs(belief_fraud - prev_belief_fraud)))
            delta_history.append(round(delta, 6))
            if delta < tol:
                converged = True
                prev_belief_fraud = belief_fraud
                break
        prev_belief_fraud = belief_fraud

    lbp_time = time.time() - t0

    # creencias finales de usuario/negocio (para el score a nivel reviewer,
    # útil en el diagnóstico cold-start y para un futuro profile_cluster.py)
    belief_U_log = np.empty((n_users, 2))
    belief_U_log[:, 0] = log_prior_U[:, 0] + np.bincount(
        reviewer_idx, weights=msg_R_to_U[:, 0], minlength=n_users
    )
    belief_U_log[:, 1] = log_prior_U[:, 1] + np.bincount(
        reviewer_idx, weights=msg_R_to_U[:, 1], minlength=n_users
    )
    belief_U = np.exp(belief_U_log - belief_U_log.max(axis=1, keepdims=True))
    belief_U = belief_U / belief_U.sum(axis=1, keepdims=True)

    belief_P_log = np.empty((n_products, 2))
    belief_P_log[:, 0] = log_prior_P[:, 0] + np.bincount(
        product_idx, weights=msg_R_to_P[:, 0], minlength=n_products
    )
    belief_P_log[:, 1] = log_prior_P[:, 1] + np.bincount(
        product_idx, weights=msg_R_to_P[:, 1], minlength=n_products
    )
    belief_P = np.exp(belief_P_log - belief_P_log.max(axis=1, keepdims=True))
    belief_P = belief_P / belief_P.sum(axis=1, keepdims=True)

    if verbose:
        status = "CONVERGIÓ" if converged else "NO convergió (máx. iteraciones alcanzado)"
        print(
            f"LBP {status} en {n_iter_run} iteraciones ({lbp_time:.2f}s), "
            f"epsilon={epsilon}, damping={damping}, último delta={delta_history[-1] if delta_history else 'n/a'}"
        )

    return {
        "review_fraud_score": prev_belief_fraud,
        "user_fraud_score": belief_U[:, 1],
        "product_fraud_score": belief_P[:, 1],
        "converged": converged,
        "n_iterations": n_iter_run,
        "delta_history": delta_history,
        "lbp_time_s": round(lbp_time, 2),
        "epsilon": epsilon,
        "damping": damping,
    }


# ---------------------------------------------------------------------------
# Evaluación -- siempre AUC + AP + top-k, nunca una métrica suelta
# ---------------------------------------------------------------------------

def _evaluate_scores(scores: np.ndarray, label: np.ndarray, idx: np.ndarray, name: str = "") -> dict:
    s = np.asarray(scores)[idx]
    y = np.asarray(label)[idx].astype(int)
    base_rate = float(y.mean()) if len(y) else float("nan")

    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "name": name,
            "n": int(len(idx)),
            "base_rate": round(base_rate, 4) if len(y) else None,
            "warning": "n=0 o una sola clase presente -- AUC/AP no definidos",
        }

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    order = np.argsort(-s)
    n = len(y)
    n_fraud = int(y.sum())
    topk = {}
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(round(n * frac)))
        sel = order[:k]
        captured = int(y[sel].sum())
        precision = captured / k
        recall = captured / n_fraud if n_fraud else float("nan")
        lift = precision / base_rate if base_rate else float("nan")
        topk[frac] = {
            "k": k,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "lift_vs_base_rate": round(float(lift), 2),
        }

    return {
        "name": name,
        "n": int(len(idx)),
        "n_fraud": n_fraud,
        "base_rate": round(base_rate, 4),
        "auc": round(float(auc), 4),
        "ap": round(float(ap), 4),
        "topk": topk,
    }


# ---------------------------------------------------------------------------
# Orquestación -- Yelp-NYC (dataset principal de la tarea)
# ---------------------------------------------------------------------------

def run_speagle_yelpnyc(
    epsilon: float = 0.15,
    max_iter: int = 150,
    damping: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Pipeline completo SpEagle sobre Yelp-NYC: carga, priors, LBP,
    evaluación con el protocolo obligatorio de la tarea (split 70/30
    estratificado `SEED=42`, idéntico al de `_fuse_graph_signal_scores` en
    `features_graph.py` -- pero recuérdese: aquí `idx_train` no entrena nada,
    SpEagle no ve ninguna etiqueta).
    """
    t0 = time.time()
    df = data.load_yelpnyc_dataset()
    load_time = time.time() - t0
    if verbose:
        print(f"Yelp-NYC cargado: {len(df)} reviews, {df['reviewer_id'].nunique()} reviewers, "
              f"{df['business_id'].nunique()} negocios ({load_time:.1f}s)")

    t0 = time.time()
    priors = compute_minimal_priors(df)
    priors_time = time.time() - t0
    if verbose:
        print(f"Priors calculados en {priors_time:.1f}s")

    lbp = run_lbp(priors, epsilon=epsilon, max_iter=max_iter, damping=damping, verbose=verbose)

    label = df["is_fake"].to_numpy().astype(int)
    scores = lbp["review_fraud_score"]
    all_idx = np.arange(len(label))

    idx_train, idx_test = train_test_split(
        all_idx, test_size=0.30, random_state=SEED, stratify=label
    )

    review_counts_per_user = np.bincount(priors["reviewer_idx"], minlength=len(priors["user_ids"]))
    is_cold_start_review = review_counts_per_user[priors["reviewer_idx"]] == 1
    idx_test_cold = idx_test[is_cold_start_review[idx_test]]

    result = {
        "dataset": "Yelp-NYC (Rayana & Akoglu)",
        "n_reviews": int(len(df)),
        "n_users": int(len(priors["user_ids"])),
        "n_products": int(len(priors["product_ids"])),
        "base_rate_dataset_completo": round(float(label.mean()), 4),
        "epsilon": epsilon,
        "damping": damping,
        "max_iter": max_iter,
        "lbp_converged": lbp["converged"],
        "lbp_n_iterations": lbp["n_iterations"],
        "lbp_delta_history": lbp["delta_history"],
        "timings_s": {
            "load_data": round(load_time, 2),
            "compute_priors": round(priors_time, 2),
            "lbp": lbp["lbp_time_s"],
        },
        "metrics_test_30pct": _evaluate_scores(scores, label, idx_test, "Yelp-NYC, held-out 30% (protocolo estándar del proyecto)"),
        "metrics_full_dataset": _evaluate_scores(scores, label, all_idx, "Yelp-NYC, dataset completo (referencia, no comparable 1:1 con el held-out)"),
        "metrics_cold_start_test_30pct": _evaluate_scores(
            scores, label, idx_test_cold,
            "Yelp-NYC, held-out 30% -- SOLO reviewers con 1 sola review en TODO el dataset",
        ),
        "n_reviewers_cold_start_total": int((review_counts_per_user == 1).sum()),
        "n_reviews_cold_start_en_test": int(len(idx_test_cold)),
        "referencia_speagle_paper_auc": SPEAGLE_PAPER_AUC_REFERENCIA,
        "referencia_fusion_sin_net_rur_honesta": FUSION_SIN_NET_RUR_HONESTA_REFERENCIA,
    }

    if verbose:
        print(f"\n--- SpEagle Yelp-NYC, held-out 30% (n={result['metrics_test_30pct']['n']}) ---")
        print(f"AUC: {result['metrics_test_30pct']['auc']} | AP: {result['metrics_test_30pct']['ap']} "
              f"| base_rate: {result['metrics_test_30pct']['base_rate']}")
        print(f"top-1%/5%/10%: {result['metrics_test_30pct']['topk']}")
        print(f"\n--- Cold-start (reviewers con 1 sola review, n={result['metrics_cold_start_test_30pct'].get('n')}) ---")
        print(result["metrics_cold_start_test_30pct"])
        print(f"\nReferencia SpEagle paper: {SPEAGLE_PAPER_AUC_REFERENCIA} AUC")
        print(f"Referencia fusión sin net_rur (protocolo honesto): {FUSION_SIN_NET_RUR_HONESTA_REFERENCIA}")

    _save_speagle_metrics({"speagle_yelpnyc": result})
    return result


def run_speagle_bretthollenbeck(
    epsilon: float = 0.15,
    max_iter: int = 150,
    damping: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Mismo pipeline sobre `data.load_bretthollenbeck_dataset()` --
    validación secundaria pedida en la tarea si sobra tiempo.

    Diferencia real frente a Yelp-NYC: `is_fake` es *nullable* (solo 80.281
    de 381.734 filas tienen etiqueta manual, ver docstring de
    `data.load_bretthollenbeck_dataset`) -- los priors y LBP corren sobre las
    381.734 filas completas (más estructura de grafo real = mejor
    propagación), pero la evaluación se restringe a las filas CON etiqueta
    manual (`is_fake.notna()`), intersecado con el split 70/30 y con el
    filtro cold-start.
    """
    t0 = time.time()
    df = data.load_bretthollenbeck_dataset()
    load_time = time.time() - t0
    if verbose:
        print(f"bretthollenbeck cargado: {len(df)} reviews, {df['reviewer_id'].nunique()} reviewers, "
              f"{df['business_id'].nunique()} productos ({load_time:.1f}s)")

    t0 = time.time()
    priors = compute_minimal_priors(df)
    priors_time = time.time() - t0
    if verbose:
        print(f"Priors calculados en {priors_time:.1f}s")

    lbp = run_lbp(priors, epsilon=epsilon, max_iter=max_iter, damping=damping, verbose=verbose)

    labeled_mask = df["is_fake"].notna().to_numpy()
    label_full = df["is_fake"].fillna(False).to_numpy().astype(int)  # solo se USA donde labeled_mask=True
    scores = lbp["review_fraud_score"]
    labeled_idx = np.where(labeled_mask)[0]

    idx_train, idx_test = train_test_split(
        labeled_idx, test_size=0.30, random_state=SEED, stratify=label_full[labeled_idx]
    )

    review_counts_per_user = np.bincount(priors["reviewer_idx"], minlength=len(priors["user_ids"]))
    is_cold_start_review = review_counts_per_user[priors["reviewer_idx"]] == 1
    idx_test_cold = idx_test[is_cold_start_review[idx_test]]

    result = {
        "dataset": "bretthollenbeck (Amazon)",
        "n_reviews_total": int(len(df)),
        "n_reviews_con_etiqueta_manual": int(labeled_mask.sum()),
        "n_users": int(len(priors["user_ids"])),
        "n_products": int(len(priors["product_ids"])),
        "base_rate_etiquetadas": round(float(label_full[labeled_idx].mean()), 4),
        "epsilon": epsilon,
        "damping": damping,
        "max_iter": max_iter,
        "lbp_converged": lbp["converged"],
        "lbp_n_iterations": lbp["n_iterations"],
        "lbp_delta_history": lbp["delta_history"],
        "timings_s": {
            "load_data": round(load_time, 2),
            "compute_priors": round(priors_time, 2),
            "lbp": lbp["lbp_time_s"],
        },
        "metrics_test_30pct": _evaluate_scores(scores, label_full, idx_test, "bretthollenbeck, held-out 30% de las filas etiquetadas"),
        "metrics_cold_start_test_30pct": _evaluate_scores(
            scores, label_full, idx_test_cold,
            "bretthollenbeck, held-out 30% -- SOLO reviewers con 1 sola review en TODO el dataset",
        ),
        "n_reviewers_cold_start_total": int((review_counts_per_user == 1).sum()),
        "n_reviews_cold_start_en_test": int(len(idx_test_cold)),
        "referencia_speagle_paper_auc": SPEAGLE_PAPER_AUC_REFERENCIA,
    }

    if verbose:
        print(f"\n--- SpEagle bretthollenbeck, held-out 30% de etiquetadas (n={result['metrics_test_30pct']['n']}) ---")
        print(f"AUC: {result['metrics_test_30pct']['auc']} | AP: {result['metrics_test_30pct']['ap']} "
              f"| base_rate: {result['metrics_test_30pct']['base_rate']}")
        print(f"top-1%/5%/10%: {result['metrics_test_30pct']['topk']}")
        print(f"\n--- Cold-start (n={result['metrics_cold_start_test_30pct'].get('n')}) ---")
        print(result["metrics_cold_start_test_30pct"])

    _save_speagle_metrics({"speagle_bretthollenbeck": result})
    return result


# ---------------------------------------------------------------------------
# Smoke test -- muestra pequeña, antes de lanzar la escala completa (criterio
# ya establecido del proyecto: medir antes de lanzar a ciegas)
# ---------------------------------------------------------------------------

def run_smoke_test(n_sample: int = 20000, verbose: bool = True) -> dict:
    """Corre el pipeline completo sobre una muestra aleatoria de Yelp-NYC
    (estratificada por `is_fake`) para verificar corrección y medir tiempo
    real antes de lanzar las 359.052 filas completas."""
    df = data.load_yelpnyc_dataset()
    label = df["is_fake"].to_numpy()
    idx_sample, _ = train_test_split(
        np.arange(len(df)), train_size=n_sample, random_state=SEED, stratify=label
    )
    df_sample = df.iloc[idx_sample].reset_index(drop=True)

    t0 = time.time()
    priors = compute_minimal_priors(df_sample)
    lbp = run_lbp(priors, epsilon=0.15, max_iter=100, verbose=verbose)
    total_time = time.time() - t0

    label_sample = df_sample["is_fake"].to_numpy().astype(int)
    metrics = _evaluate_scores(lbp["review_fraud_score"], label_sample, np.arange(len(df_sample)), "smoke test")

    if verbose:
        print(f"\nSmoke test ({n_sample} filas): {total_time:.2f}s total, "
              f"{lbp['n_iterations']} iteraciones LBP, converged={lbp['converged']}")
        print(metrics)

    return {"n_sample": n_sample, "total_time_s": round(total_time, 2), "lbp": lbp, "metrics": metrics}


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if target == "smoke":
        run_smoke_test()
    elif target == "yelpnyc":
        run_speagle_yelpnyc()
    elif target == "bretthollenbeck":
        run_speagle_bretthollenbeck()
    else:
        run_smoke_test()
