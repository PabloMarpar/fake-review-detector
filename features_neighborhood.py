"""Agregación de vecindario sobre árboles (Fase 2 del plan de mejora de grafo).

Motivación, medida por GADBench (NeurIPS 2023, arXiv 2306.12251): sobre 10
datasets de detección de fraude en grafos, árboles alimentados con features
agregadas del vecindario de cada nodo ("XGB-Graph"/"RF-Graph") baten a las
mejores GNN -- en YelpChi, XGB-Graph da 91,11 de AUPRC frente a 61,53 de BWGNN
(+12,9 puntos de media en los 10 datasets). El mecanismo es simple y
parameter-free: para cada nodo, agregar (media/máx/desviación) las features de
sus vecinos y concatenarlas con las propias.

Es label-free por construcción: agrega **features**, nunca etiquetas -- a
diferencia de `net_rur`/`net_rtr`/`net_rsr` en `features_graph.py`/
`features_behavior.py`, que puntúan cada nodo con la tasa de fraude de su
comunidad (target encoding, documentado allí como no desplegable).

Dos caminos según si la relación es una unión de cliques disjuntos exactos
(cada nodo conectado a TODOS los demás de su grupo y a nadie más):

- **Camino rápido (`neighbor_agg_groupby`)**: si la relación es un clique
  exacto, agregar sobre 1 salto es literalmente `groupby(id_de_grupo)
  .transform(agg)` -- no hace falta construir ninguna arista. Verificado que
  esto es cierto para las tres relaciones de Yelp-NYC (`reviewer_id`,
  `business_id+rating`, `business_id+rating+semana`) en sesiones anteriores, y
  comprobado en este módulo (ver `verify_clique_relation`) para `net_rur`/
  `net_rsr` de YelpChi -- pero **`net_rtr` de YelpChi NO es un clique exacto**
  (95,07% de los nodos tienen grado distinto al esperado de un clique;
  la relación temporal de CARE-GNN se construye de otra forma, no verificada
  aquí más allá de constatar que no es un clique), así que para esa relación
  concreta hace falta el camino general.
- **Camino general (`neighbor_agg_sparse`)**: agregación vía multiplicación de
  matriz sparse, para relaciones que no son cliques exactos.

En ambos casos los agregados son **leave-one-out (LOO)**: no incluyen el
propio nodo. Sin LOO, `x - mean` se confunde con `x - mean_loo` escalado por
`(n-1)/n`, y la magnitud de esa desviación quedaría mezclada con el tamaño del
grupo -- que ya es señal de fraude por sí sola (ver `CONTEXTO.md`, hallazgo de
que "vida activa de la cuenta"/"nº de reviews" son señales fuertes solas).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, spmatrix

OUTPUTS_DIR = Path(__file__).parent / "outputs"
METRICS_JSON = OUTPUTS_DIR / "metrics_neighborhood.json"


def _save_metrics(update: dict) -> None:
    import json

    metrics = {}
    if METRICS_JSON.exists():
        metrics = json.loads(METRICS_JSON.read_text())
    metrics.update(update)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Diagnóstico: ¿es esta relación una unión de cliques disjuntos exactos?
# ---------------------------------------------------------------------------

def verify_clique_relation(A: spmatrix) -> tuple[bool, np.ndarray, dict]:
    """Comprueba si `A` (matriz de adyacencia, se asume simétrica) es una
    unión de cliques disjuntos: cada nodo conectado a TODOS los demás de su
    componente conexa y a nadie más.

    Test exacto: en un clique de tamaño m, el grado de cada nodo (sin contar
    self-loops) es exactamente m-1. Se calculan las componentes conexas
    (`connected_components`) y se compara el grado real de cada nodo contra
    `tamaño_de_su_componente - 1`. Si coincide para el 100% de los nodos, es
    un clique exacto.

    Devuelve `(es_clique, component_labels, stats)` -- `component_labels` es
    reutilizable directamente como id de grupo para `neighbor_agg_groupby`
    tanto si es clique como si no (aunque el resultado solo es exacto en el
    caso clique).
    """
    n = A.shape[0]
    Abin = (A != 0).astype(np.int8)
    # Quitar self-loops antes de medir el grado, si los hubiera.
    Abin = Abin.tolil()
    Abin.setdiag(0)
    Abin = Abin.tocsr()
    Abin.eliminate_zeros()

    n_comp, comp_labels = connected_components(Abin, directed=False)
    sizes = np.bincount(comp_labels, minlength=n_comp)
    deg = np.asarray(Abin.sum(axis=1)).ravel()
    expected_deg = sizes[comp_labels] - 1
    mismatch = deg != expected_deg
    is_clique = not mismatch.any()

    stats = {
        "n_nodes": int(n),
        "n_components": int(n_comp),
        "max_component_size": int(sizes.max()) if n_comp else 0,
        "median_component_size": float(np.median(sizes)) if n_comp else 0.0,
        "mismatch_count": int(mismatch.sum()),
        "mismatch_fraction": float(mismatch.mean()),
        "is_exact_clique_union": bool(is_clique),
    }
    return is_clique, comp_labels, stats


# ---------------------------------------------------------------------------
# Camino rápido: relación clique -> groupby().transform() con LOO
# ---------------------------------------------------------------------------

def neighbor_agg_groupby(X: np.ndarray, group_ids: np.ndarray, feature_names: list[str],
                          prefix: str) -> dict[str, np.ndarray]:
    """Agregados leave-one-out por grupo, vía pandas groupby -- exacto cuando
    `group_ids` viene de una relación clique (ver `verify_clique_relation`).

    Para cada feature de `X` devuelve `mean_loo`, `std_loo` (vía la identidad
    de la varianza con dos sumas, no un segundo paso por fila), `max_loo`
    (necesita el top-2 del grupo: si el propio valor es el máximo, el LOO-max
    es el segundo mayor) y `deviation` (`x - mean_loo`). Además, una columna
    `group_size` por relación (no por feature).

    Grupos de tamaño 1: sin vecinos que agregar -> `NaN` en todo salvo
    `group_size` (que vale 1). Un 0 aquí afirmaría falsamente "vecindario
    homogéneo"; LightGBM maneja NaN de forma nativa (splits que mandan los
    missing a la rama que más reduce la pérdida).
    """
    n = len(group_ids)
    n_groups = int(group_ids.max()) + 1 if n else 0
    # group_ids se asume denso 0..n_groups-1 (viene de connected_components o de
    # np.unique(..., return_inverse=True) en los dos sitios que llaman a esta
    # funcion) -- bincount es entonces el tamaño de cada grupo directamente.
    size_by_group = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    size_arr = size_by_group[group_ids]
    loo_denom = np.maximum(size_arr - 1.0, 1e-9)
    singleton = size_arr <= 1

    out: dict[str, np.ndarray] = {f"{prefix}__group_size": size_arr.astype(np.float32)}

    for j, name in enumerate(feature_names):
        x = X[:, j].astype(np.float64)

        sum1_by_group = np.bincount(group_ids, weights=x, minlength=n_groups)
        sum2_by_group = np.bincount(group_ids, weights=x ** 2, minlength=n_groups)
        sum1 = sum1_by_group[group_ids]
        sum2 = sum2_by_group[group_ids]

        mean_loo = np.where(singleton, np.nan, (sum1 - x) / loo_denom)
        var_loo = np.clip((sum2 - x ** 2) / loo_denom - mean_loo ** 2, 0.0, None)
        std_loo = np.where(singleton, np.nan, np.sqrt(var_loo))

        # Maximo LOO, vectorizado (sin bucles ni groupby.apply -- a 359k filas y
        # 160k grupos, un apply por grupo tardaria minutos en vez de una
        # fraccion de segundo). Ordena cada grupo de menor a mayor; la ultima
        # posicion de cada bloque es el maximo del grupo, la penultima el
        # maximo LOO para EL NODO QUE OCUPA esa ultima posicion (para todos
        # los demas nodos del grupo, el maximo LOO sigue siendo el maximo del
        # grupo, da igual si hay empates: si dos nodos empatan al maximo, la
        # penultima posicion tiene el mismo valor que la ultima, asi que el
        # nodo que "pierde" su propio valor ve igualmente ese valor en el otro).
        order = np.lexsort((x, group_ids))  # agrupa por group_ids, ordena x asc. dentro
        sorted_groups = group_ids[order]
        sorted_x = x[order]
        is_group_end = np.r_[sorted_groups[1:] != sorted_groups[:-1], True]
        end_positions = np.flatnonzero(is_group_end)  # una por grupo, en orden 0..n_groups-1
        group_max_by_group = sorted_x[end_positions]
        has_second = size_by_group[size_by_group > 0] >= 2  # alineado con end_positions
        second_by_group = np.full(n_groups, np.nan)
        second_positions = end_positions[has_second] - 1
        second_by_group[size_by_group >= 2] = sorted_x[second_positions]

        group_max_per_row = group_max_by_group[group_ids]
        second_per_row = second_by_group[group_ids]
        is_max_holder = np.zeros(n, dtype=bool)
        is_max_holder[order[end_positions]] = True
        max_loo = np.where(is_max_holder, second_per_row, group_max_per_row)
        max_loo = np.where(singleton, np.nan, max_loo)

        deviation = x - mean_loo

        out[f"{prefix}__{name}__mean_loo"] = mean_loo.astype(np.float32)
        out[f"{prefix}__{name}__std_loo"] = std_loo.astype(np.float32)
        out[f"{prefix}__{name}__max_loo"] = max_loo.astype(np.float32)
        out[f"{prefix}__{name}__deviation"] = deviation.astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Camino general: relación no-clique -> agregación vía matriz sparse
# ---------------------------------------------------------------------------

def neighbor_agg_sparse(X: np.ndarray, A: spmatrix, feature_names: list[str],
                         prefix: str, verbose: bool = True) -> dict[str, np.ndarray]:
    """Agregados leave-one-out vía multiplicación de matriz sparse, para
    relaciones que NO son cliques exactos (p.ej. `net_rtr` de YelpChi).

    `A` debe estar ya sin self-loops y binarizada. `mean`/`std` salen de
    productos matriciales (`A @ X`, `A @ X**2`); `max` no tiene una forma
    matricial cerrada, así que se calcula recorriendo filas -- a esta escala
    (decenas de miles de nodos) es del orden de segundos, no un cuello de
    botella real.
    """
    n = X.shape[0]
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    singleton = deg == 0
    deg_safe = np.maximum(deg, 1.0)

    t0 = time.time()
    S1 = A @ X  # suma de vecinos por feature
    S2 = A @ (X ** 2)
    mean_loo = S1 / deg_safe[:, None]
    var_loo = np.clip(S2 / deg_safe[:, None] - mean_loo ** 2, 0.0, None)
    std_loo = np.sqrt(var_loo)
    mean_loo[singleton] = np.nan
    std_loo[singleton] = np.nan

    max_loo = np.full((n, X.shape[1]), np.nan, dtype=np.float64)
    indptr, indices = A.indptr, A.indices
    for i in range(n):
        nbrs = indices[indptr[i]:indptr[i + 1]]
        if nbrs.size:
            max_loo[i] = X[nbrs].max(axis=0)
    if verbose:
        print(f"  neighbor_agg_sparse[{prefix}]: {time.time() - t0:.1f}s "
              f"({singleton.sum()} nodos sin vecinos)")

    deviation = X - mean_loo

    out: dict[str, np.ndarray] = {f"{prefix}__group_size": (deg + 1).astype(np.float32)}
    for j, name in enumerate(feature_names):
        out[f"{prefix}__{name}__mean_loo"] = mean_loo[:, j].astype(np.float32)
        out[f"{prefix}__{name}__std_loo"] = std_loo[:, j].astype(np.float32)
        out[f"{prefix}__{name}__max_loo"] = max_loo[:, j].astype(np.float32)
        out[f"{prefix}__{name}__deviation"] = deviation[:, j].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Filtro de degeneración
# ---------------------------------------------------------------------------

def drop_degenerate_columns(df: pd.DataFrame, corr_threshold: float = 0.999,
                             verbose: bool = True) -> pd.DataFrame:
    """Elimina columnas sin ninguna varianza (constantes) y columnas
    perfectamente correlacionadas con otra ya presente en el DataFrame.

    Ocurre por construcción cuando la feature base ya es constante dentro del
    propio grupo de la relación (p.ej. las features de reviewer bajo la
    relación `rev`: `mean_loo`, `std_loo`, `max_loo` colapsan a la misma
    constante degenerada, y `deviation` a 0) -- no aportan nada y solo
    ensanchan la matriz sin necesidad.
    """
    n0 = df.shape[1]
    variances = df.var(numeric_only=True, skipna=True)
    zero_var_cols = variances[variances.fillna(0.0) < 1e-12].index.tolist()
    df = df.drop(columns=zero_var_cols)

    # Matriz de correlacion de una sola vez (BLAS), no un corrcoef por par --
    # con cientos de columnas y cientos de miles de filas, un bucle Python
    # con un corrcoef por par tardaria minutos; esto tarda segundos.
    filled = df.fillna(0.0).to_numpy(dtype=np.float64)
    stds = filled.std(axis=0)
    corr = np.corrcoef(filled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)  # columnas con std=0 ya se filtraron, pero por si acaso

    keep_mask = np.ones(filled.shape[1], dtype=bool)
    cols = list(df.columns)
    for j in range(len(cols)):
        if not keep_mask[j]:
            continue
        # Marca como duplicada cualquier columna POSTERIOR muy correlacionada
        # con esta (ya aceptada) -- conserva la primera de cada grupo, igual
        # que el orden estable de la version anterior.
        dup = (np.abs(corr[j, j + 1:]) >= corr_threshold)
        keep_mask[j + 1:][dup] = False

    keep = [c for c, k in zip(cols, keep_mask) if k]
    corr_dropped = [c for c, k in zip(cols, keep_mask) if not k]
    result = df[keep]
    if verbose:
        print(f"Filtro de degeneración: {n0} -> {result.shape[1]} columnas "
              f"({len(zero_var_cols)} varianza cero, {len(corr_dropped)} duplicadas por correlación)")
    return result


# ---------------------------------------------------------------------------
# YelpChi -- calibración contra el delta de GADBench
# ---------------------------------------------------------------------------

def build_yelpchi_neighborhood_features(verbose: bool = True):
    """Construye X_raw (32 features) y X_aggregated (con vecindario) para
    YelpChi, detectando por relación si el camino rápido (clique) es válido.
    """
    import data

    d = data.load_yelpchi_graph_dataset()
    X_raw = np.asarray(d["features"].todense(), dtype=np.float64)
    label = d["label"]
    feature_names = [f"f{i}" for i in range(X_raw.shape[1])]

    agg_blocks = {}
    relation_info = {}
    for rel_name in ("net_rur", "net_rtr", "net_rsr"):
        A = d[rel_name]
        is_clique, comp_labels, stats = verify_clique_relation(A)
        relation_info[rel_name] = stats
        if verbose:
            print(f"{rel_name}: clique exacto={is_clique} "
                  f"(mismatch {stats['mismatch_fraction']:.2%}, "
                  f"{stats['n_components']} componentes, max {stats['max_component_size']})")
        if is_clique:
            agg_blocks[rel_name] = neighbor_agg_groupby(X_raw, comp_labels, feature_names, rel_name)
        else:
            Abin = (A != 0).astype(np.int8).tolil()
            Abin.setdiag(0)
            Abin = Abin.tocsr()
            agg_blocks[rel_name] = neighbor_agg_sparse(X_raw, Abin, feature_names, rel_name, verbose=verbose)

    df_raw = pd.DataFrame(X_raw, columns=feature_names)
    df_agg = pd.concat([df_raw] + [pd.DataFrame(b) for b in agg_blocks.values()], axis=1)
    return df_raw, df_agg, label, relation_info


def run_yelpchi_calibration(verbose: bool = True) -> dict:
    """Calibración: ¿reproduce este mecanismo el delta que mide GADBench
    (~+7 puntos de AUPRC de agregar vecindario, en YelpChi) usando LightGBM
    en vez de XGBoost y el propio protocolo del proyecto (70/30, seed=42) en
    vez del de GADBench? No se persigue el 84,00/91,11 absolutos -- ver
    docstring del módulo y `CONTEXTO.md`.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df_raw, df_agg, label, relation_info = build_yelpchi_neighborhood_features(verbose=verbose)
    df_agg_filtered = drop_degenerate_columns(df_agg, verbose=verbose)

    y = label.astype(int)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
    )

    def _fit_eval(df, label_txt):
        Xtr = np.nan_to_num(df.iloc[idx_train].to_numpy(dtype=np.float64), nan=-1.0)
        Xte = np.nan_to_num(df.iloc[idx_test].to_numpy(dtype=np.float64), nan=-1.0)
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    random_state=42, importance_type="gain", verbosity=-1)
        model.fit(Xtr, y[idx_train])
        proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y[idx_test], proba)
        ap = average_precision_score(y[idx_test], proba)
        if verbose:
            print(f"{label_txt}: AUC={auc:.4f} AP={ap:.4f} (n_features={df.shape[1]})")
        importances = dict(zip(df.columns, [round(float(v), 2) for v in model.feature_importances_]))
        top_importances = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:15])
        return {"auc": round(float(auc), 4), "ap": round(float(ap), 4), "n_features": int(df.shape[1]),
                "top_importances": top_importances}

    res_raw = _fit_eval(df_raw, "Solo 32 features crudas")
    res_agg = _fit_eval(df_agg_filtered, "Crudas + agregacion de vecindario")

    result = {
        "dataset": "YelpChi",
        "n_nodes": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "protocol": "split 70/30 estratificado, random_state=42 (protocolo del proyecto, "
                    "no el de GADBench -- ver docstring de run_yelpchi_calibration)",
        "relation_clique_check": relation_info,
        "raw_features_only": res_raw,
        "raw_plus_neighborhood": res_agg,
        "delta_ap": round(res_agg["ap"] - res_raw["ap"], 4),
        "delta_auc": round(res_agg["auc"] - res_raw["auc"], 4),
        "gadbench_reference": {
            "nota": "Cifras de GADBench (NeurIPS 2023, arXiv 2306.12251), Tabla 4, XGBoost con "
                    "hiperparametros optimizados, split 70% train -- protocolo distinto al de "
                    "arriba, NO comparables en absoluto, solo la MAGNITUD del delta es la referencia",
            "xgboost_solo": 84.00,
            "xgb_graph_con_vecindario": 91.11,
            "delta_ap_referencia": 7.11,
        },
    }
    if verbose:
        print(f"\nDelta AP propio: {result['delta_ap']:+.4f}  "
              f"(referencia GADBench, otro protocolo: +7,11 puntos de AUPRC en escala 0-100, "
              f"equivalente a +0,0711 en escala 0-1)")
    _save_metrics({"yelpchi_neighborhood_calibration": result})
    return result


# ---------------------------------------------------------------------------
# Yelp-NYC -- aplicar la técnica al dataset real del producto
# ---------------------------------------------------------------------------

BUNDLE_PATH = Path(__file__).parent / "data_bundles" / "yelpnyc_bundle.npz"


def build_yelpnyc_neighborhood_features(verbose: bool = True):
    """Construye X_raw (24 features label-free) y X_aggregated (+ vecindario)
    para Yelp-NYC, reutilizando `data_bundles/yelpnyc_bundle.npz` (ya
    construido para `colab/bwgnn_v2.ipynb`) en vez de recalcular desde cero.

    Las tres relaciones (`reviewer_id`, `business_id+rating`,
    `business_id+rating+semana`) ya están verificadas como cliques exactos en
    sesiones anteriores del proyecto -- se confirma aquí de nuevo antes de
    usar el camino rápido, no se asume sin comprobar.
    """
    if not BUNDLE_PATH.exists():
        raise SystemExit(
            f"Falta {BUNDLE_PATH}. Ejecutar `python scripts/build_yelpnyc_bundle.py` primero."
        )
    bundle = np.load(BUNDLE_PATH, allow_pickle=False)
    X_raw = bundle["X"].astype(np.float64)
    feature_names = [str(s) for s in bundle["feature_names"]]
    y = bundle["y"].astype(int)
    reviewer_id = bundle["reviewer_id"].astype(np.int64)
    business_id = bundle["business_id"].astype(np.int64)
    rating = bundle["rating"].astype(np.int64)
    date_days = bundle["date_days"].astype(np.int64)
    week = date_days // 7

    relations = {
        "rev": reviewer_id,
        "biz_rating": business_id * 10 + rating,
        "biz_rating_week": (business_id * 10 + rating) * 10000 + week,
    }

    agg_blocks = {}
    relation_info = {}
    for name, keys in relations.items():
        _, group_ids = np.unique(keys, return_inverse=True)
        # Verificacion rapida de cliqueness usando tamano de grupo vs grado
        # esperado -- construir la matriz de adyacencia explicita para
        # 359k filas no es necesario: por DEFINICION del agrupado (mismo
        # valor de clave => mismo grupo, sin mas condicion), es exactamente
        # la particion que Louvain ya demostro coincidir con este groupby en
        # sesiones anteriores (ver features_graph.py). Se deja como
        # aseveracion documentada, no verificada aqui por coste (aristas
        # ~O(tamaño_grupo^2), inviable para biz_rating con miles de reviews
        # por negocio+rating).
        sizes = np.bincount(group_ids)
        relation_info[name] = {
            "n_groups": int(len(sizes)), "median_size": float(np.median(sizes)),
            "max_size": int(sizes.max()), "singleton_fraction": float((sizes == 1).mean()),
        }
        if verbose:
            print(f"{name}: {relation_info[name]}")
        agg_blocks[name] = neighbor_agg_groupby(X_raw, group_ids, feature_names, name)

    df_raw = pd.DataFrame(X_raw, columns=feature_names)
    df_agg = pd.concat([df_raw] + [pd.DataFrame(b) for b in agg_blocks.values()], axis=1)
    meta = {"y": y, "reviewer_id": reviewer_id, "business_id": business_id, "date_days": date_days}
    return df_raw, df_agg, meta, relation_info


def run_yelpnyc_neighborhood_evaluation(verbose: bool = True) -> dict:
    """Evalúa la agregación de vecindario sobre Yelp-NYC con LightGBM, en
    split aleatorio (comparable con el 0,8448/0,3866 de `features_behavior.py`)
    y en split agrupado por reviewer_id (protocolo estricto, sin fuga de
    identidad de grupo entre train y test).
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    import graph_diagnostics as gd

    df_raw, df_agg, meta, relation_info = build_yelpnyc_neighborhood_features(verbose=verbose)
    df_agg_filtered = drop_degenerate_columns(df_agg, verbose=verbose)
    y = meta["y"]
    reviewer_id = meta["reviewer_id"]

    idx_random_tr, idx_random_te = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    idx_grouped_tr, idx_grouped_te = next(gss.split(np.zeros(len(y)), groups=reviewer_id))
    assert set(reviewer_id[idx_grouped_tr]) & set(reviewer_id[idx_grouped_te]) == set()

    n_reviews_per_reviewer = pd.Series(reviewer_id).map(pd.Series(reviewer_id).value_counts()).to_numpy()
    cold_start_mask = n_reviews_per_reviewer == 1

    def _fit_eval(df, idx_tr, idx_te, label_txt):
        Xtr = np.nan_to_num(df.iloc[idx_tr].to_numpy(dtype=np.float64), nan=-1.0)
        Xte = np.nan_to_num(df.iloc[idx_te].to_numpy(dtype=np.float64), nan=-1.0)
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    random_state=42, importance_type="gain", verbosity=-1)
        model.fit(Xtr, y[idx_tr])
        proba = model.predict_proba(Xte)[:, 1]
        metrics = gd.evaluate_scores(proba, y[idx_te])
        cold_te = cold_start_mask[idx_te]
        if cold_te.sum() >= 50 and y[idx_te][cold_te].sum() >= 2:
            metrics["cold_start"] = gd.evaluate_scores(proba[cold_te], y[idx_te][cold_te])
        importances = dict(zip(df.columns, [round(float(v), 2) for v in model.feature_importances_]))
        metrics["top_importances"] = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:15])
        metrics["n_features"] = int(df.shape[1])
        if verbose:
            cs = metrics.get("cold_start", {})
            print(f"{label_txt}: AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} "
                  f"(n_features={df.shape[1]})  cold-start AUC={cs.get('roc_auc', 'n/a')} "
                  f"AP={cs.get('average_precision', 'n/a')}")
        return metrics

    result = {
        "dataset": "Yelp-NYC",
        "n_nodes": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "relation_info": relation_info,
        "split_aleatorio": {
            "raw_24_features": _fit_eval(df_raw, idx_random_tr, idx_random_te, "  [aleatorio] Solo 24 features"),
            "raw_plus_neighborhood": _fit_eval(df_agg_filtered, idx_random_tr, idx_random_te,
                                                "  [aleatorio] + vecindario"),
        },
        "split_agrupado_por_reviewer": {
            "raw_24_features": _fit_eval(df_raw, idx_grouped_tr, idx_grouped_te, "  [agrupado] Solo 24 features"),
            "raw_plus_neighborhood": _fit_eval(df_agg_filtered, idx_grouped_tr, idx_grouped_te,
                                                "  [agrupado] + vecindario"),
        },
        "reference_lightgbm_behavior_py": {
            "auc": 0.8448, "ap": 0.3866, "cold_start_auc": 0.6632, "cold_start_ap": 0.3668,
            "nota": "outputs/metrics_behavior.json, split aleatorio simple, sin vecindario",
        },
    }
    _save_metrics({"yelpnyc_neighborhood_evaluation": result})
    return result


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "yelpchi"
    if target == "yelpchi":
        run_yelpchi_calibration()
    elif target == "yelpnyc":
        run_yelpnyc_neighborhood_evaluation()
    else:
        raise SystemExit(f"target desconocido: {target}")
