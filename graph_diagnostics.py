"""Métricas y cortes de diagnóstico para las señales de grafo.

Hasta ahora el proyecto evaluaba cada señal de grafo con una sola cifra:
ROC-AUC (ver `evaluate_communities` en `features_graph.py`). Con un 10,3% de
positivos en Yelp-NYC eso da una imagen demasiado optimista y, sobre todo, no
dice *dónde* falla una señal — solo si "va bien" en promedio. Este módulo
añade dos cosas que faltaban:

1. **Métricas sensibles al desbalance** (`evaluate_scores`): Average Precision
   junto al ROC-AUC, y recall a FPR fijo. El AUC pondera igual todo el rango
   de umbrales, incluidos los que ningún producto usaría; el AP y el recall a
   1%/5% de FPR miden la zona en la que de verdad se opera (revisar unos pocos
   casos sospechosos sin ahogar al cliente en falsos positivos).

2. **Cortes de diagnóstico** (`diagnostic_slices`): la misma señal medida por
   subconjuntos de la población. El corte que más importa es `cold_start`
   (reviews de reviewers con una sola review): es donde cualquier señal basada
   en el historial de la cuenta deja de tener nada que mirar, y donde una señal
   de coordinación real tiene que demostrar que sirve. Un AUC global alto
   sostenido solo por reviewers con mucho historial es un resultado
   sobrevalorado para el caso de uso del producto.

Además, `loo_cluster_scores` reimplementa `leave_one_out_cluster_scores` de
`features_graph.py` con una corrección de protocolo: permite estimar la tasa
de fraude de cada comunidad usando SOLO las etiquetas visibles en train. La
versión original usa las etiquetas de todo el dataset, incluido el held-out, y
el split se hace después — medido en Yelp-NYC, eso infla la fusión de las tres
señales de 0,9028 a 0,9233 de AUC (0,5961 a 0,6559 de AP).

Advertencia de lectura, importante para interpretar cualquier número de este
módulo: las señales `net_rur`/`net_rtr`/`net_rsr` son *target encoding* (el
score de un nodo es la tasa de fraude de sus vecinos según las etiquetas), no
features derivadas de la estructura del grafo. Necesitan etiquetas para
existir, así que no son trasladables tal cual a un cliente sin historial
etiquetado. Se miden aquí como referencia interna, no como candidatas a
producto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEED = 42


def recall_at_fpr(scores: np.ndarray, label: np.ndarray, fpr_target: float) -> float:
    """Recall alcanzable sin superar `fpr_target` de falsos positivos.

    Es la misma métrica que el proyecto ya usa para las señales de texto
    (TPR@1%FPR / TPR@5%FPR en `train.py`), traída a las señales de grafo para
    que ambas mitades del producto se puedan comparar en los mismos términos.
    """
    fpr, tpr, _ = roc_curve(label, scores)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate_scores(
    scores: np.ndarray,
    label: np.ndarray,
    fracs: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict:
    """Batería completa de métricas para un vector de scores.

    Devuelve ROC-AUC y Average Precision (la métrica que de verdad discrimina
    con clases desbalanceadas), recall a 1% y 5% de FPR, y precisión/recall/
    lift en los top-k habituales. `lift` es precisión dividida por la tasa base:
    responde "¿cuántas veces mejor que elegir al azar?", que es como se explica
    el valor de la detección a un cliente.
    """
    scores = np.asarray(scores, dtype=float)
    label = np.asarray(label).astype(int)
    n = len(label)
    n_fraud = int(label.sum())
    base_rate = n_fraud / n if n else 0.0

    out = {
        "n": n,
        "n_fraud": n_fraud,
        "base_rate": round(base_rate, 4),
    }
    if n_fraud == 0 or n_fraud == n:
        out["nota"] = "una sola clase presente: AUC/AP no definidos en este corte"
        return out

    out["roc_auc"] = round(float(roc_auc_score(label, scores)), 4)
    out["average_precision"] = round(float(average_precision_score(label, scores)), 4)
    out["ap_lift_vs_base"] = round(out["average_precision"] / base_rate, 2)
    out["recall_at_1pct_fpr"] = round(recall_at_fpr(scores, label, 0.01), 4)
    out["recall_at_5pct_fpr"] = round(recall_at_fpr(scores, label, 0.05), 4)

    order = np.argsort(-scores)
    topk = {}
    for frac in fracs:
        k = max(1, int(round(n * frac)))
        captured = int(label[order[:k]].sum())
        precision = captured / k
        topk[f"top_{frac:.0%}"] = {
            "k": k,
            "precision": round(precision, 4),
            "recall": round(captured / n_fraud, 4),
            "lift": round(precision / base_rate, 2),
        }
    out["topk"] = topk
    return out


def loo_cluster_scores(
    keys: pd.Series | np.ndarray,
    label: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Tasa de fraude de la comunidad de cada nodo, excluyendo su propio label.

    `keys` identifica la comunidad de cada fila (por ejemplo `reviewer_id`, o
    `business_id + rating`); para las relaciones que el proyecto usa, la
    partición por `groupby` es exactamente la misma que devuelve Louvain — ya
    verificado en `features_graph.py`, que documenta la coincidencia exacta del
    número de comunidades en ambos datasets.

    Con `train_mask`, la tasa de cada comunidad se estima usando solo las
    etiquetas de train: un nodo de test recibe la tasa de sus vecinos de train,
    sin que su propia etiqueta ni la de otros nodos de test participen. Sin
    `train_mask` reproduce el comportamiento original (todas las etiquetas
    disponibles), que sirve para análisis retrospectivo pero sobreestima lo que
    se obtendría sobre datos nuevos.
    """
    label = np.asarray(label).astype(float)
    keys = pd.Series(np.asarray(keys)).reset_index(drop=True)

    if train_mask is None:
        grouped = pd.Series(label).groupby(keys)
        total = grouped.transform("sum").to_numpy()
        size = grouped.transform("size").to_numpy()
        base = label.mean()
        return np.where(size > 1, (total - label) / np.maximum(size - 1, 1), base)

    train_mask = np.asarray(train_mask).astype(bool)
    visible_label = np.where(train_mask, label, 0.0)
    visible_count = train_mask.astype(float)
    sum_label = pd.Series(visible_label).groupby(keys).transform("sum").to_numpy()
    sum_count = pd.Series(visible_count).groupby(keys).transform("sum").to_numpy()

    # Un nodo de train se excluye a sí mismo; uno de test nunca estuvo dentro.
    numerator = np.where(train_mask, sum_label - label, sum_label)
    denominator = np.where(train_mask, sum_count - 1.0, sum_count)
    base = label[train_mask].mean() if train_mask.any() else label.mean()
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-9), base)


def diagnostic_slices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras booleanas alineadas con `df` para medir por subpoblación.

    `cold_start` es el corte que más importa: son las reviews cuyo autor no
    tiene ninguna otra review en el dataset, así que ninguna señal basada en el
    historial de la cuenta tiene nada que mirar. Si una señal solo funciona
    fuera de este corte, está midiendo reincidencia de cuentas conocidas, no
    coordinación entre cuentas nuevas — que es el escenario en el que un
    cliente real necesita ayuda.
    """
    reviews_por_reviewer = df.groupby("reviewer_id")["reviewer_id"].transform("size").to_numpy()
    reviews_por_negocio = df.groupby("business_id")["business_id"].transform("size").to_numpy()
    rating = df["rating"].to_numpy()

    return {
        "global": np.ones(len(df), dtype=bool),
        "cold_start": reviews_por_reviewer == 1,
        "reviewer_2_a_5": (reviews_por_reviewer >= 2) & (reviews_por_reviewer <= 5),
        "reviewer_6_o_mas": reviews_por_reviewer >= 6,
        "negocio_poco_resenado": reviews_por_negocio < np.median(reviews_por_negocio),
        "rating_extremo": np.isin(rating, [1, 5]),
    }


def evaluate_by_slice(
    scores: np.ndarray,
    label: np.ndarray,
    slices: dict[str, np.ndarray],
    restrict_to: np.ndarray | None = None,
) -> dict:
    """Aplica `evaluate_scores` dentro de cada corte de `slices`.

    `restrict_to` permite evaluar solo en el held-out manteniendo los cortes
    definidos sobre el dataset completo (el historial de un reviewer se calcula
    con todas sus reviews, no solo con las que cayeron en el held-out).
    """
    scores = np.asarray(scores, dtype=float)
    label = np.asarray(label).astype(int)
    out = {}
    for name, mask in slices.items():
        mask = np.asarray(mask).astype(bool)
        if restrict_to is not None:
            mask = mask & np.asarray(restrict_to).astype(bool)
        if mask.sum() < 50:
            out[name] = {"n": int(mask.sum()), "nota": "corte demasiado pequeño"}
            continue
        out[name] = evaluate_scores(scores[mask], label[mask])
    return out


def compare_protocols(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Mide el efecto real de la fuga de etiquetas y el perfil por cortes.

    Calcula las tres señales de comunidad con ambos protocolos (todas las
    etiquetas / solo las de train) sobre el mismo split 70/30 estratificado que
    usa `_fuse_graph_signal_scores` en `features_graph.py`, para que las cifras
    sean directamente comparables con las ya documentadas en el proyecto.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = df.reset_index(drop=True)
    label = df["is_fake"].to_numpy().astype(int)
    n = len(label)

    idx_train, idx_test = train_test_split(
        np.arange(n), test_size=0.30, random_state=SEED, stratify=label
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[idx_train] = True

    semana = pd.to_datetime(df["date"]).dt.to_period("W").astype(str)
    negocio_rating = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    community_keys = {
        "net_rur": df["reviewer_id"].astype(str),
        "net_rsr": negocio_rating,
        "net_rtr": negocio_rating + "_" + semana,
    }

    resultados = {"con_fuga": {}, "sin_fuga": {}}
    features = {"con_fuga": {}, "sin_fuga": {}}
    for nombre, keys in community_keys.items():
        for protocolo, mask in (("con_fuga", None), ("sin_fuga", train_mask)):
            scores = loo_cluster_scores(keys, label, train_mask=mask)
            features[protocolo][nombre] = scores
            resultados[protocolo][nombre] = evaluate_scores(scores[idx_test], label[idx_test])

    nombres = list(community_keys)
    for protocolo in ("con_fuga", "sin_fuga"):
        X = np.column_stack([features[protocolo][k] for k in nombres])
        scaler = StandardScaler().fit(X[idx_train])
        for etiqueta, columnas in (
            ("fusion", list(range(len(nombres)))),
            ("fusion_sin_net_rur", [i for i, k in enumerate(nombres) if k != "net_rur"]),
        ):
            modelo = LogisticRegression(max_iter=1000, random_state=SEED)
            modelo.fit(scaler.transform(X[idx_train])[:, columnas], label[idx_train])
            pred = modelo.predict_proba(scaler.transform(X[idx_test])[:, columnas])[:, 1]
            resultados[protocolo][etiqueta] = evaluate_scores(pred, label[idx_test])
            if protocolo == "sin_fuga":
                resultados.setdefault("cortes_sin_fuga", {})[etiqueta] = evaluate_by_slice(
                    _scatter(pred, idx_test, n), label, diagnostic_slices(df),
                    restrict_to=~train_mask,
                )

    if verbose:
        _print_comparison(resultados, nombres)
    return resultados


def _scatter(values: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Coloca los scores del held-out en un array del tamaño del dataset."""
    full = np.full(n, np.nan)
    full[idx] = values
    return full


def _print_comparison(resultados: dict, nombres: list[str]) -> None:
    filas = nombres + ["fusion", "fusion_sin_net_rur"]
    print("\n--- Efecto de la fuga de etiquetas (Yelp-NYC, held-out 30%) ---")
    print(f"{'señal':<20} {'AUC fuga':>9} {'AUC limpio':>11} {'AP fuga':>9} {'AP limpio':>10}")
    for fila in filas:
        a, b = resultados["con_fuga"][fila], resultados["sin_fuga"][fila]
        print(f"{fila:<20} {a['roc_auc']:>9.4f} {b['roc_auc']:>11.4f} "
              f"{a['average_precision']:>9.4f} {b['average_precision']:>10.4f}")

    print("\n--- Perfil por cortes, protocolo limpio (AUC / AP / lift del AP) ---")
    for etiqueta, cortes in resultados.get("cortes_sin_fuga", {}).items():
        print(f"\n  {etiqueta}:")
        for corte, m in cortes.items():
            if "roc_auc" not in m:
                print(f"    {corte:<22} {m.get('nota', 'sin datos')}")
                continue
            print(f"    {corte:<22} n={m['n']:>7}  AUC={m['roc_auc']:.4f}  "
                  f"AP={m['average_precision']:.4f}  lift={m['ap_lift_vs_base']:.2f}x  "
                  f"base={m['base_rate']:.4f}")


def run_yelpnyc_diagnostics(verbose: bool = True) -> dict:
    """Informe de diagnóstico completo sobre Yelp-NYC."""
    import json
    from pathlib import Path

    import data

    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    if verbose:
        print(f"Yelp-NYC: {len(df)} reviews, tasa de fraude {df['is_fake'].mean():.4f}")
    resultados = compare_protocols(df, verbose=verbose)

    destino = Path("outputs") / "metrics_diagnostics.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"yelpnyc": resultados}, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"\nGuardado en {destino}")
    return resultados


if __name__ == "__main__":
    run_yelpnyc_diagnostics()
