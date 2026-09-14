"""Perfilado de clusters — Fase 1.

Diseño ya cerrado en `README.md`, sección "Perfilado de clusters" (Pacheco et
al., ICWSM 2021: red bipartita reviewer↔negocio → proyección reviewer-reviewer
→ cluster). Este módulo toma un cluster ya identificado (por ejemplo, una
comunidad de Louvain sobre `net_rur`/`net_rtr`/`net_rsr` de
`features_graph.py`, o la partición directa de `net_rsr` vía
`groupby_cliques_as_communities`) y genera una ficha con evidencia por nivel,
de más a menos fiable, terminando siempre con un bloque fijo de "qué no se
afirma".

**Ámbito real de esta sesión, con Yelp-NYC como único dataset con texto +
grafo disponible**:

- **Nivel A (estructural/temporal)**: implementado y validable aquí — es la
  parte que sí tiene datos reales de grafo y fecha para medir (sincronía de
  co-reviews, participación en ráfagas, densidad frente a un modelo de
  configuración aleatoria, ratio de cuentas de una sola review).
- **Nivel B (cohorte de cuenta)**: implementado y documentado tal como pide
  el README, pero **explícitamente NO validable con Yelp-NYC** — el dataset
  no trae fecha de creación de cuenta (solo fecha de cada review), así que
  no hay forma real de medir "concentración de altas de cuenta frente a la
  tasa de alta de la plataforma" ni "edad de la cuenta en su primera
  review". Se devuelve un bloque explícito diciendo que no hay evidencia,
  no un número inventado ni un proxy que aparente serlo — ver README,
  sección "Riesgos clave": "el Nivel B del perfilado no es validable con los
  datasets académicos disponibles [...] hasta un piloto real".
- **Niveles C (texto) y D (metadatos)**: fuera de alcance de esta sesión —
  Nivel C necesitaría un índice offline de near-duplicates (no construido
  aquí) y Nivel D necesita metadatos que Yelp-NYC no trae (avatar, username,
  email). No se implementan como placeholders vacíos para no aparentar más
  cobertura de la que hay; el README ya los documenta como parte del roadmap
  de Fase 2.

**Línea roja respetada en todo el módulo** (ver README, "Sobre qué se puede
prometer de verdad"): ninguna función de aquí calcula ni infiere
nacionalidad, etnia o identidad de una persona. Todo lo que se mide es
estructural (grafo, tiempo, conteos) — el bloque final de cada ficha lo dice
explícitamente, no como enunciado suelto sino como parte fija y obligatoria
del formato de salida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix


# ---------------------------------------------------------------------------
# Nivel A — estructural / temporal (validable con Yelp-NYC)
# ---------------------------------------------------------------------------

def coreview_synchrony(net_rtr: spmatrix, cluster_idx: np.ndarray) -> dict:
    """Sincronía de co-reviews: densidad de `net_rtr` (mismo negocio+rating+
    ventana temporal) DENTRO del cluster, frente a la densidad global de
    `net_rtr` en todo el dataset (modelo nulo: "¿qué tan probable es que dos
    reviews cualquiera compartan negocio+rating+ventana, si no hubiera nada
    especial en este grupo?").

    Un múltiplo alto (`sync_multiplier` >> 1) es evidencia de que los
    miembros del cluster escriben en el mismo negocio, con el mismo rating,
    en la misma ventana de tiempo mucho más de lo que el azar explicaría —
    justo el patrón de "coordinación", no solo de "cuentas relacionadas".
    """
    n_total = net_rtr.shape[0]
    total_possible_pairs = n_total * (n_total - 1)
    global_density = net_rtr.nnz / total_possible_pairs if total_possible_pairs else 0.0

    k = len(cluster_idx)
    if k < 2:
        return {
            "n_cuentas": k,
            "observado": None,
            "esperado_bajo_modelo_nulo": None,
            "sync_multiplier": None,
            "nota": "cluster de tamaño < 2, no hay pares que medir",
        }

    sub = net_rtr[np.ix_(cluster_idx, cluster_idx)]
    observed_pairs = sub.nnz
    possible_pairs = k * (k - 1)
    observed_density = observed_pairs / possible_pairs
    multiplier = (observed_density / global_density) if global_density > 0 else float("inf")

    return {
        "n_cuentas": k,
        "aristas_net_rtr_dentro": int(observed_pairs / 2),
        "densidad_observada": round(observed_density, 6),
        "densidad_global_net_rtr": round(global_density, 6),
        "sync_multiplier": round(float(multiplier), 2) if np.isfinite(multiplier) else None,
    }


def density_vs_configuration_model(adjacency: spmatrix, cluster_idx: np.ndarray) -> dict:
    """Densidad del cluster frente a un modelo de configuración (respeta el
    grado real de cada nodo en el grafo completo, no asume grafo aleatorio
    uniforme tipo Erdős–Rényi) — la fórmula estándar de la modularidad de
    Newman: `E[aristas dentro del cluster] = (Σ grados del cluster)² /
    (4 · nº total de aristas)`.

    Es la señal que el README marca como "la más importante": separa una
    granja real (densidad muy por encima de lo que sus propios grados ya
    predicen) de un grupo de cuentas activas por casualidad (grados altos
    por sí solos ya explican la densidad, sin necesitar coordinación).

    **Limitación real encontrada al probar esto con Yelp-NYC, documentada
    explícitamente, no escondida**: con un grafo de fondo tan grande
    (359.052 nodos) y clusters pequeños (10-15 cuentas, el tamaño típico de
    un negocio+rating), el número esperado de aristas bajo el modelo de
    configuración es prácticamente cero para cualquier cluster (los grados
    medios son minúsculos frente al tamaño total del grafo) — así que
    CUALQUIER cluster con una sola arista interna produce un
    `ratio_vs_modelo_nulo` astronómico, tenga o no tenga fraude real. Probado
    con un cluster fraudulento real (13 cuentas, 92,3% fraude) y uno benigno
    de tamaño casi idéntico (12 cuentas, 0% fraude): **si el grafo de fondo
    incluye `net_rsr` y el cluster es un grupo negocio+rating, el problema es
    circular** (el cluster ES un clique de `net_rsr`, comparado contra un
    fondo que ya contiene ese mismo clique como pieza — ambos casos dieron
    ratios del mismo orden de magnitud, ~840.000x-960.000x, sin discriminar
    nada). Si se excluye `net_rsr` del fondo (usar solo `net_rur ∪ net_rtr`),
    el benigno pasa a 0 aristas internas (ratio 0) y el fraudulento a 1
    arista interna (ratio ~1.880.000x) — la DIRECCIÓN es correcta (0 vs. algo
    > 0), pero la magnitud del ratio en sí no es una medida fiable de
    "cuánto más denso de lo esperado" a este tamaño de cluster. **Se
    devuelve igualmente el ratio (con este aviso), más los recuentos
    absolutos (`aristas_observadas`/`aristas_esperadas...`) para que quien
    lea la ficha no se quede solo con un múltiplo inflado sin contexto** —
    y se recomienda no usar el grafo que define el propio cluster como
    fondo de comparación (ver `nivel_a_evidence`, que excluye `net_rsr` del
    fondo cuando el cluster proviene de esa misma relación).
    """
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    total_edges = adjacency.sum() / 2.0
    if total_edges == 0:
        return {"n_cuentas": len(cluster_idx), "ratio_vs_modelo_nulo": None, "nota": "grafo sin aristas"}

    sub = adjacency[np.ix_(cluster_idx, cluster_idx)]
    observed_edges = sub.sum() / 2.0
    expected_edges = (degrees[cluster_idx].sum() ** 2) / (4.0 * total_edges)
    ratio = (observed_edges / expected_edges) if expected_edges > 0 else float("inf")

    out = {
        "n_cuentas": len(cluster_idx),
        "aristas_observadas": round(float(observed_edges), 2),
        "aristas_esperadas_modelo_configuracion": round(float(expected_edges), 4),
        "ratio_vs_modelo_nulo": round(float(ratio), 2) if np.isfinite(ratio) else None,
    }
    if observed_edges < 5 and expected_edges < 0.01:
        out["aviso"] = (
            "cluster pequeño frente al grafo completo -- el ratio puede ser "
            "numéricamente inestable (ver docstring), leer junto a "
            "aristas_observadas, no solo el múltiplo"
        )
    return out


def burst_participation(burst_scores: np.ndarray, cluster_idx: np.ndarray, z_threshold: float = 2.0) -> dict:
    """Participación en ráfagas — reutiliza `features_graph.detect_bursts_yelpnyc`
    (Nivel A, real aquí; era imposible en Yelp-Chi por falta de fecha).

    Reporta el z-score medio de las reviews del cluster y qué fracción cae
    por encima de `z_threshold` (una ráfaga real del propio negocio, no solo
    "más reviews de lo normal en general").
    """
    scores = burst_scores[cluster_idx]
    return {
        "n_reviews": len(cluster_idx),
        "z_medio": round(float(np.mean(scores)), 3),
        "z_max": round(float(np.max(scores)), 3),
        "pct_en_rafaga_fuerte": round(float((scores >= z_threshold).mean()), 4),
    }


def single_review_ratio(df: pd.DataFrame, cluster_idx: np.ndarray, reviewer_counts: pd.Series | None = None) -> dict:
    """Ratio de cuentas con una sola review en TODA su vida (no solo dentro
    del cluster) — evidencia de cuentas de usar-y-tirar, uno de los cuatro
    indicios de Nivel A que pide el README.
    """
    if reviewer_counts is None:
        reviewer_counts = df["reviewer_id"].value_counts()
    reviewer_ids = df.loc[cluster_idx, "reviewer_id"]
    is_single = reviewer_ids.map(reviewer_counts) == 1
    return {
        "n_cuentas": len(cluster_idx),
        "n_una_sola_review_en_su_vida": int(is_single.sum()),
        "ratio": round(float(is_single.mean()), 4),
    }


def nivel_a_evidence(
    df: pd.DataFrame,
    cluster_idx: np.ndarray,
    net_rtr: spmatrix,
    density_backdrop: spmatrix,
    burst_scores: np.ndarray,
    reviewer_counts: pd.Series | None = None,
) -> dict:
    """Junta las cuatro señales de Nivel A del README en un solo dict.

    `density_backdrop` es el grafo de fondo para `density_vs_configuration_model`
    — **no debe incluir la misma relación que se usó para definir el
    cluster** (ver docstring de esa función: usarla es circular, un cluster
    de "mismo negocio+rating" comparado contra un fondo que ya contiene ese
    mismo clique como pieza no mide nada). Quien llame a esta función decide
    qué pasar aquí según cómo se identificó el cluster (p. ej. `net_rur ∪
    net_rtr` si el cluster viene de un grupo `net_rsr`, o viceversa).
    """
    return {
        "sincronia_coreviews": coreview_synchrony(net_rtr, cluster_idx),
        "densidad_vs_modelo_configuracion": density_vs_configuration_model(density_backdrop, cluster_idx),
        "participacion_rafagas": burst_participation(burst_scores, cluster_idx),
        "cuentas_una_sola_review": single_review_ratio(df, cluster_idx, reviewer_counts),
    }


# ---------------------------------------------------------------------------
# Nivel B — cohorte de cuenta (NO validable con Yelp-NYC, ver docstring del
# módulo y README, sección "Riesgos clave")
# ---------------------------------------------------------------------------

def nivel_b_evidence(df: pd.DataFrame, cluster_idx: np.ndarray) -> dict:
    """Nivel B tal como lo define el README: concentración de fechas de
    creación de cuenta frente a la tasa de alta real de la plataforma, edad
    de la cuenta en su primera review, ratio de cero interacción recibida.

    **No calculable con Yelp-NYC**: el dataset (`data.load_yelpnyc_dataset`)
    no trae `reviewer_created_at` ni ninguna interacción recibida (votos
    útiles, respuestas) — solo `reviewer_id`, `business_id`, `rating`,
    `date` (de la review, no de alta de cuenta) y `text`. Se devuelve un
    bloque explícito de "sin evidencia", no un número aproximado ni un
    proxy que aparente ser esto — la fecha de la PRIMERA review de un
    reviewer en este dataset NO es su fecha de alta real (pudo haberse
    registrado antes y no haber escrito nada), así que usarla como sustituto
    silencioso sería inventar un dato, no estimarlo con incertidumbre.

    Coherente con el propio README (sección "Riesgos clave"): "el Nivel B
    del perfilado no es validable con los datasets académicos disponibles —
    se documenta como no validado hasta un piloto real."
    """
    return {
        "disponible": False,
        "motivo": (
            "Yelp-NYC no trae fecha de creación de cuenta ni interacción "
            "recibida (votos/respuestas) — solo fecha de cada review. No "
            "validable hasta un piloto real con un cliente que aporte esos "
            "campos (ver README, 'Riesgos clave')."
        ),
        "n_cuentas": len(np.unique(df.loc[cluster_idx, "reviewer_id"])),
    }


# ---------------------------------------------------------------------------
# Ficha de cluster — formato fijado en README.md
# ---------------------------------------------------------------------------

def _label_es(value: float, thresholds: tuple[float, float] = (2.0, 5.0)) -> str:
    """Traduce un ratio/multiplicador numérico a "débil"/"moderada"/"fuerte",
    lenguaje estimativo tipo ICD-203 que pide el README (probabilidad y
    confianza, nunca mezcladas en la misma frase)."""
    if value is None or not np.isfinite(value):
        return "sin evidencia"
    low, high = thresholds
    if value < low:
        return "débil"
    if value < high:
        return "moderada"
    return "fuerte"


def build_cluster_card(
    cluster_id,
    df: pd.DataFrame,
    cluster_idx: np.ndarray,
    net_rtr: spmatrix,
    density_backdrop: spmatrix,
    burst_scores: np.ndarray,
    reviewer_counts: pd.Series | None = None,
) -> str:
    """Genera la ficha de texto de un cluster, mismo formato que el ejemplo
    fijado en README.md ("CLUSTER #7 · ..."). Nivel B se incluye siempre
    marcado como "sin evidencia" para Yelp-NYC (ver `nivel_b_evidence`), no
    se omite en silencio.

    `density_backdrop` — ver aviso en `density_vs_configuration_model` y en
    `nivel_a_evidence`: no debe ser el mismo grafo que se usó para definir
    `cluster_idx` (circular).
    """
    sub = df.loc[cluster_idx]
    n_cuentas = sub["reviewer_id"].nunique()
    n_reviews = len(cluster_idx)
    n_negocios = sub["business_id"].nunique()

    nivel_a = nivel_a_evidence(df, cluster_idx, net_rtr, density_backdrop, burst_scores, reviewer_counts)
    nivel_b = nivel_b_evidence(df, cluster_idx)

    sync = nivel_a["sincronia_coreviews"]["sync_multiplier"]
    dens_info = nivel_a["densidad_vs_modelo_configuracion"]
    dens = dens_info["ratio_vs_modelo_nulo"]
    burst = nivel_a["participacion_rafagas"]["z_medio"]
    single_ratio = nivel_a["cuentas_una_sola_review"]["ratio"]

    dens_line = (
        f"DENSIDAD (vs. modelo de configuración)   {_label_es(dens)}   "
        f"ratio observado/esperado = {dens}x ({dens_info['aristas_observadas']} aristas reales)"
        if dens is not None else
        "DENSIDAD (vs. modelo de configuración)   sin evidencia"
    )
    if dens_info.get("aviso"):
        dens_line += f"  [aviso: {dens_info['aviso']}]"

    lines = [
        f"CLUSTER #{cluster_id} · {n_cuentas} cuentas · {n_reviews} reviews · {n_negocios} negocio(s)",
        "─" * 60,
        dens_line,
        f"SINCRONÍA DE CO-REVIEWS                  {_label_es(sync)}   "
        f"{sync}x la densidad global de net_rtr" if sync is not None else
        f"SINCRONÍA DE CO-REVIEWS                  sin evidencia",
        f"PARTICIPACIÓN EN RÁFAGAS                 {_label_es(burst, (1.0, 2.0))}   "
        f"z medio = {burst} (frente a la tasa base propia del negocio)",
        f"CUENTAS DE UNA SOLA REVIEW               {_label_es(single_ratio, (0.3, 0.7))}   "
        f"{nivel_a['cuentas_una_sola_review']['n_una_sola_review_en_su_vida']}/{n_cuentas} "
        f"({single_ratio * 100:.1f}%)",
        f"COHORTE (Nivel B)                        sin evidencia   {nivel_b['motivo']}",
        "",
        "No se afirma nacionalidad, origen geográfico, identidad de personas ni quién se beneficia.",
        "Esto es una pista para revisión humana, no un veredicto.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import numpy as _np

    import data
    import features_graph as fg

    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    graphs = fg.build_yelpnyc_graphs(df, rtr_window="W")
    burst_scores = fg.detect_bursts_yelpnyc(df, window="W")
    reviewer_counts = df["reviewer_id"].value_counts()

    # Cluster real de demostración: negocio 826, rating 1 -- encontrado
    # buscando, entre los grupos negocio+rating de tamaño 10-30, el de mayor
    # tasa de fraude real (92,3%, 13 reviews, ver CONTEXTO.md para el detalle
    # de cómo se localizó). No es un cluster de juguete inventado para que
    # la ficha salga bonita -- es el grupo negocio+rating con más
    # concentración de `is_fake=True` de todo el dataset en ese rango de
    # tamaño.
    cluster_idx = df[(df["business_id"] == 826) & (df["rating"] == 1)].index.to_numpy()
    print(f"Tasa de fraude real de este cluster: {df.loc[cluster_idx, 'is_fake'].mean():.4f}")
    print()

    # backdrop de densidad SIN net_rsr: el cluster de demo es justo un grupo
    # negocio+rating (una pieza de net_rsr) -- usarlo también como fondo de
    # comparación sería circular (ver docstring de density_vs_configuration_model).
    density_backdrop = ((graphs["net_rur"] + graphs["net_rtr"]) > 0).astype(_np.float32).tocsr()

    print(build_cluster_card(
        "826-rating1", df, cluster_idx, graphs["net_rtr"], density_backdrop, burst_scores, reviewer_counts
    ))
