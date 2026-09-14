"""Fase 1 — clustering de comunidades sobre el grafo de Yelp-Chi (CARE-GNN).

Primer bloque de la Fase 1 (grafo + coordinación de cuentas). Trabaja **solo** con
estructura de grafo — este dataset (`data.load_yelpchi_graph_dataset()`) no trae texto
de review, así que aquí no hay nada de T1/T2/T3 ni fusión con texto todavía (ver
docstring de `data.py`). Las 32 features numéricas que trae el dataset están
ya calculadas por los autores de CARE-GNN, no son features propias de este proyecto —
se usan aquí solo como referencia/sanity-check (`sanity_check_raw_features_auc`), nunca
como si las hubiéramos calculado nosotros.

## Qué hace este módulo

1. **Clustering de comunidades con Louvain** sobre las tres relaciones homogéneas
   (`net_rur`, `net_rtr`, `net_rsr`), sobre `net_homo` (unión booleana ya presente en el
   dataset) y sobre una combinación propia ponderada por rareza (ver
   `build_rarity_weighted_graph`, es la decisión de diseño real de este bloque).
2. **Burst detection**: no implementada de verdad — el `.mat` no trae ningún campo
   temporal (verificado abriendo el fichero directamente, ver `detect_bursts()`).
   Documentado como limitación, no disimulado con un proxy inventado sin avisar.
3. **Evaluación de calidad del clustering frente a `label`**: no es un clasificador
   supervisado, así que no tiene sentido un AUC "de modelo" — se usan dos métricas
   pensadas para clustering (ver `evaluate_communities()`):
   - Enriquecimiento de fraude por comunidad frente a un modelo nulo hipergeométrico
     ("¿esta comunidad tiene más fraude del que tendría un grupo aleatorio del mismo
     tamaño sacado de toda la población?").
   - AUC/precision-en-top-k usando como score la tasa de fraude de la comunidad de cada
     nodo, calculada **leave-one-out** (excluyendo la propia etiqueta del nodo al
     calcular la tasa de su comunidad, para no maquillar el número con la propia
     etiqueta que se está evaluando).
4. **Comparación con salvedades** frente a SpEagle (~0.78 AUC, dataset original sin
   preprocesar) y CARE-GNN (~0.7445 AUC, mismo dataset preprocesado, pero supervisado)
   — ver la función `print_reference_comparison()`, con el texto exacto de salvedades
   que no se deben repetir como cifras verificadas a fondo.

## Hallazgos de sesión (tiempos reales, máquina de trabajo, CPU, sin GPU)

Medido con `time.time()` sobre esta misma máquina antes de escribir el resto del
módulo, construyendo el grafo con `networkx.from_scipy_sparse_array` y corriendo
`networkx.algorithms.community.louvain_communities` con `seed=42`:

| Relación    | Aristas (no dirigidas) | Build grafo | Louvain | Nº comunidades |
|---|---|---|---|---|
| `net_rur`   | 49.315    | 0,3 s  | 3,2 s   | 29.431 |
| `net_rtr`   | 573.616   | 4,2 s  | 22,0 s  | 1.835  |
| `net_rsr`   | 3.402.743 | 18,0 s | 51,8 s  | 803    |
| `net_homo`  | 3.846.979 | 22,7 s | 86,3 s  | 120    |

Sorprende que `net_rsr` (68x más aristas que `net_rur`) tarde solo ~16x más en Louvain,
no proporcionalmente más — el coste de Louvain depende más de cuántas comunidades acaba
habiendo (más iteraciones de refinamiento con muchas comunidades pequeñas) que del
número bruto de aristas. El script completo (los 5 grafos: 3 relaciones + homo +
combinación ponderada) tarda del orden de **5-6 minutos** en esta máquina — asumible
para CPU-only, pero se documenta el tiempo real por si alguien intenta escalar esto a
Yelp-NYC/ZIP (359k/608k reviews, mucho más grande) sin medir antes.

**`net_rur` produce sobre todo comunidades triviales**: 29.431 comunidades sobre 45.954
nodos es, en su mayoría, reviewers con una sola review en todo el dataset (no comparten
`net_rur` con nadie porque esa relación exige *mismo usuario*, y un usuario con una sola
review no tiene ninguna otra review con la que conectarse). Es una limitación esperada
de esta relación en solitario, no un bug — por eso no se usa nunca sola como grafo
final, solo como una de las tres piezas de la combinación ponderada.

## Resultados reales de la ejecución completa (`python features_graph.py`, esta máquina)

| Grafo | LOO-AUC | Precision top-5% | Lift top-5% | Comunidades sig. (enrichment) | Louvain (s) |
|---|---|---|---|---|---|
| `net_rur` | **0,814** | **0,712** | **4,90x** | 4 (0,45% del fraude) | 2,5 |
| `net_rtr` | 0,587 | 0,271 | 1,87x | 7 (3,98% del fraude) | 11,2 |
| `net_rsr` | 0,636 | 0,416 | 2,86x | 21 (9,08% del fraude) | 64,8 |
| `net_homo` (unión) | 0,572 | 0,295 | 2,03x | 6 (9,65% del fraude) | 68,2 |
| combinación ponderada por rareza | 0,572 | 0,299 | 2,06x | 5 (8,99% del fraude) | **431,2** |

Sanity-checks (no son nuestro clustering): regresión logística sobre las 32 features de
CARE-GNN → AUC 0,658. Grado en `net_rtr` como proxy de "burst" → AUC 0,510 (nivel de
azar, confirma que no sirve como sustituto de burst detection real).

**Dos hallazgos honestos que cambian la lectura del número "ganador" (`net_rur`), no
solo el ranking en sí:**

1. **La combinación ponderada por rareza (la decisión de diseño propuesta en este mismo
   módulo) no mejora nada frente a la unión sin ponderar** — 0,572 vs. 0,572 de AUC,
   prácticamente idéntico — **y encima tarda 6,3x más en converger** (431s vs. 68s,
   mismo número de nodos y aristas que `net_homo`) porque Louvain con pesos continuos
   converge mucho más despacio que con un grafo binario, incluso sobre la misma
   topología. Mismo patrón ya visto en Fase 0 con la fusión T1+T2+T3 y con la cabeza
   adversarial de T2: **combinar señales no es automáticamente mejor, hay que medirlo**
   — aquí, además, combinar sale claramente peor en coste computacional sin ninguna
   contrapartida de calidad. Para escalar esto a Yelp-NYC/ZIP (mucho más grande) más
   adelante, esto es una señal de alarma real sobre el coste de cualquier variante
   ponderada de Louvain, no un detalle menor.
2. **El AUC de `net_rur` no mide lo que en principio se quería medir con este módulo.**
   Diagnóstico adicional (no en el flujo normal del script, hecho a mano para entender
   *por qué* ganaba): las comunidades no triviales de `net_rur` tienen una tasa de
   fraude MEDIA (7,2%) más baja que la tasa base global (14,5%) — de hecho, las 10
   comunidades más grandes (24-47 reviews del mismo usuario) tienen **0% de fraude**,
   son reviewers prolíficos genuinos (tipo "Yelp Elite"), no granjas. El AUC alto viene
   de las comunidades PEQUEÑAS (tamaño 2-3) que son mayoritariamente o completamente
   fraude: cuentas de usar-y-tirar que escriben un par de reviews falsas bajo el mismo
   usuario y ya. Es una señal real y aprovechable (detecta cuentas desechables de un
   solo uso), pero es conceptualmente distinta de lo que el README llama "red de
   cuentas coordinadas" (varias cuentas *distintas* actuando juntas alrededor de un
   mismo negocio) — esa historia la cuentan mejor `net_rtr`/`net_rsr` (conectan
   reviews de USUARIOS DISTINTOS que comparten negocio+rating+ventana), aunque su AUC
   en solitario sea más bajo. **Implicación para `profile_cluster.py` (siguiente
   paso, no tocado en este bloque)**: probablemente convenga tratar la señal de
   `net_rur` (cuentas de un solo uso, mismo usuario) y la de `net_rtr`/`net_rsr`
   (coordinación entre cuentas distintas) como dos indicios de nivel A separados en la
   ficha de cluster, no fundirlos en un único número — mezclarlos escondería justo la
   diferencia de qué está detectando cada uno.

**Sobre la comparación con SpEagle (~0,78 AUC) y CARE-GNN (~0,7445 AUC, con las
salvedades exactas dadas en la tarea, no verificadas a fondo)**: el `net_rur` en
solitario da un número nominal más alto (0,814) que ambas referencias, pero decir "le
ganamos a CARE-GNN" sería un error de interpretación, no solo por las salvedades ya
conocidas (dataset/preprocesado distinto, supervisado vs. no supervisado) sino por el
hallazgo del punto 2: ese 0,814 mide sobre todo cuentas de usar-y-tirar de un usuario
con 2-3 reviews, una tarea más estrecha y probablemente más fácil que la detección de
fraude general que miden SpEagle/CARE-GNN — no es una carrera justa en ningún sentido,
ver `print_reference_comparison()` para el texto completo de salvedades.
"""

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import data

SEED = 42
OUTPUTS_DIR = Path(__file__).parent / "outputs"
METRICS_JSON = OUTPUTS_DIR / "metrics.json"


def _save_graph_metrics(update: dict) -> None:
    """Actualiza `outputs/metrics.json` sin pisar las claves ya existentes.

    Reimplementación deliberada del mismo patrón que `_save_metrics` en
    `train.py` (lee, hace `.update()`, reescribe) en vez de importarlo desde
    ahí: este módulo es puramente de grafo y no depende de `train.py` (ni al
    revés) — mantener esa separación evita un acoplamiento innecesario entre
    el pipeline de texto (T1/T2/T3) y el de grafo.
    """
    metrics = {}
    if METRICS_JSON.exists():
        metrics = json.loads(METRICS_JSON.read_text())
    metrics.update(update)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str))


# ---------------------------------------------------------------------------
# Construcción de grafos
# ---------------------------------------------------------------------------

def build_rarity_weighted_graph(net_rur: spmatrix, net_rtr: spmatrix, net_rsr: spmatrix) -> csr_matrix:
    """Combina las tres relaciones homogéneas en un único grafo ponderado.

    **Decisión de diseño real de este bloque, no un detalle menor.** `net_rsr` (mismo
    negocio + mismo rating, 6.805.486 aristas dirigidas) tiene ~69x más aristas que
    `net_rur` (mismo usuario, 98.630) y ~6x más que `net_rtr` (mismo negocio+rating+mes,
    1.147.232). Una unión sin ponderar (equivalente a `net_homo`, que el dataset ya trae
    como unión booleana) queda dominada casi por completo por `net_rsr`: el 88% de las
    aristas de `net_homo` provienen solo de ahí. Louvain sobre esa unión sin ponderar
    tendería a reproducir esencialmente "comunidades = mismo negocio+rating", que es una
    relación muy débil de por sí (miles de reviews genuinas comparten negocio y rating
    sin que eso implique ninguna coordinación) y diluye la señal potencialmente más
    específica de `net_rur`/`net_rtr`.

    Se pondera cada relación de forma **inversamente proporcional a su propio número de
    aristas**, de modo que las tres aportan la misma "masa total" (=1) al grafo
    combinado — el peso de una arista concreta es la suma de `1/nnz(relación)` de cada
    relación en la que aparece. Consecuencia deseada: una arista presente en varias
    relaciones a la vez (coordinación reforzada por múltiples señales independientes)
    puntúa más que una arista que solo aparece en la relación más común y menos
    discriminativa (`rsr`). Es una heurística razonable, no una verdad matemática
    demostrada — se compara empíricamente contra las tres relaciones sueltas y contra
    `net_homo` en `run_yelpchi_analysis()`, y se elige la que mejor separa fraude según
    las métricas de `evaluate_communities()`, no se asume a priori que ganará.
    """
    w_rur = 1.0 / net_rur.nnz
    w_rtr = 1.0 / net_rtr.nnz
    w_rsr = 1.0 / net_rsr.nnz
    combined = (
        net_rur.astype(float) * w_rur
        + net_rtr.astype(float) * w_rtr
        + net_rsr.astype(float) * w_rsr
    )
    return combined.tocsr()


def detect_communities_louvain(matrix: spmatrix, seed: int = SEED, weighted: bool = False) -> tuple[list, dict]:
    """Corre Louvain sobre una matriz sparse de adyacencia y devuelve (comunidades, timings).

    `weighted=True` le dice a `louvain_communities` que use el atributo `weight` de las
    aristas (necesario para `build_rarity_weighted_graph`, donde el peso no es binario);
    para las relaciones originales de CARE-GNN (binarias) no importa, pero se deja
    explícito para no depender de un valor por defecto silencioso.
    """
    t0 = time.time()
    graph = nx.from_scipy_sparse_array(matrix)
    build_time = time.time() - t0

    t0 = time.time()
    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight" if weighted else None, seed=seed
    )
    louvain_time = time.time() - t0

    timings = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "build_time_s": round(build_time, 2),
        "louvain_time_s": round(louvain_time, 2),
        "n_communities": len(communities),
    }
    return communities, timings


def communities_to_labels(communities: list, n_nodes: int) -> np.ndarray:
    """Mapea cada nodo a un id entero de comunidad (0..n_comunidades-1)."""
    labels = np.full(n_nodes, -1, dtype=int)
    for cid, community in enumerate(communities):
        for node in community:
            labels[node] = cid
    assert (labels >= 0).all(), "Algún nodo no quedó asignado a ninguna comunidad"
    return labels


# ---------------------------------------------------------------------------
# Burst detection — no implementada de verdad, ver docstring
# ---------------------------------------------------------------------------

def detect_bursts() -> None:
    """No hay burst detection real posible sobre Yelp-Chi/CARE-GNN — limitación
    confirmada, no un hueco por descuido.

    Se abrió `data_raw/YelpChi.mat` directamente con `scipy.io.loadmat` para
    comprobarlo (no asumido de la documentación del paper): las únicas claves
    presentes son `net_rur`, `net_rtr`, `net_rsr`, `homo`, `features` (32 columnas
    numéricas sin nombre, ya vectorizadas por los autores de CARE-GNN) y `label`.
    Ninguna es una fecha ni hay metadata de columnas de `features` que documente que
    alguna de ellas sea temporal. Sin timestamps no se puede calcular lo que el README
    entiende de verdad por "burst detection" (Nivel A: tamaño de ráfaga en una ventana
    de tiempo real, frente a la tasa base de alta/review de la plataforma, con z-score).

    Lo más cercano a una señal temporal está **ya codificado dentro del grafo**, no
    como timestamp aparte: `net_rtr` conecta dos reviews si son del mismo negocio,
    mismo rating **y mismo mes/semana** (ver docstring de
    `data.load_yelpchi_graph_dataset`). Es una relación binaria por pareja de reviews
    ("¿coinciden en ventana temporal? sí/no"), no una serie temporal — no permite
    reconstruir cuántas reviews cayeron en una ventana concreta, cuál sería la tasa base
    esperada, ni ningún z-score. Como mucho, el grado de un nodo en `net_rtr` (a cuántas
    otras reviews está conectado por esta relación) es un proxy muy indirecto de
    "cuánta compañía temporal tiene esta review" — expuesto en `rtr_degree_proxy()` y
    evaluado por separado, marcado explícitamente como NO burst detection real.
    """
    return None


def rtr_degree_proxy(net_rtr: spmatrix) -> np.ndarray:
    """Grado de cada nodo en `net_rtr` (nº de otras reviews con mismo negocio+rating+mes).

    **No es burst detection** (ver `detect_bursts()`) — es solo el proxy más cercano
    disponible en este dataset, expuesto para poder medir si aporta algo por sí mismo
    (`sanity_check_rtr_degree_auc`), no para usarlo como sustituto silencioso de una
    ráfaga real.
    """
    return np.asarray(net_rtr.sum(axis=1)).reshape(-1)


# ---------------------------------------------------------------------------
# Evaluación de calidad del clustering
# ---------------------------------------------------------------------------

def leave_one_out_cluster_scores(communities: list, label: np.ndarray) -> np.ndarray:
    """Score por nodo = tasa de fraude de su comunidad, excluyendo su propia etiqueta.

    Si se incluyera la propia etiqueta del nodo al calcular la tasa de su comunidad,
    el score estaría parcialmente construido con la respuesta que se quiere evaluar
    (más notorio cuanto más pequeña es la comunidad — en una comunidad de tamaño 1 el
    score sería literalmente la etiqueta). Restar la contribución del propio nodo evita
    ese acoplamiento directo (aunque sigue habiendo correlación indirecta a través de
    los vecinos, que es justo la señal que se quiere medir).

    Comunidades de tamaño 1 no tienen "los demás" con quien calcular una tasa — se les
    asigna la tasa base global de fraude (equivalente a "no sabemos nada de este nodo
    por su comunidad, usamos el prior").
    """
    n = len(label)
    base_rate = label.mean()
    scores = np.full(n, base_rate, dtype=float)
    for community in communities:
        idx = np.fromiter(community, dtype=int)
        if idx.size <= 1:
            continue
        total_fraud = label[idx].sum()
        scores[idx] = (total_fraud - label[idx]) / (idx.size - 1)
    return scores


def topk_capture(scores: np.ndarray, label: np.ndarray, fracs=(0.05, 0.10, 0.20)) -> dict:
    """Precisión/recall/lift al quedarse con el top-k% de nodos por score."""
    n = len(label)
    n_fraud_total = label.sum()
    base_rate = n_fraud_total / n
    order = np.argsort(-scores)
    out = {}
    for frac in fracs:
        k = max(1, int(round(n * frac)))
        idx = order[:k]
        captured = label[idx].sum()
        precision = captured / k
        recall = captured / n_fraud_total
        out[frac] = {
            "k": k,
            "precision": round(float(precision), 4),
            "recall_de_todo_el_fraude": round(float(recall), 4),
            "lift_vs_base_rate": round(float(precision / base_rate), 2),
        }
    return out


def enrichment_test(communities: list, label: np.ndarray, alpha: float = 0.01) -> dict:
    """Enriquecimiento de fraude por comunidad frente a un modelo nulo hipergeométrico.

    Para cada comunidad de tamaño `n` con `k` nodos fraudulentos, el modelo nulo es
    "¿qué probabilidad hay de que un grupo de `n` nodos elegido *al azar* (sin
    reemplazo) de toda la población de `N` nodos con `K` fraudulentos en total tenga
    `k` o más fraudulentos?" — exactamente la pregunta que pedía la tarea ("¿los
    clusters con mayor proporción de fraude concentran señal real, frente a un cluster
    aleatorio del mismo tamaño?"), formalizada con la distribución hipergeométrica
    (`scipy.stats.hypergeom.sf`). Corrección de Bonferroni sobre el número de
    comunidades evaluadas (muy conservadora con miles de comunidades pequeñas, pero es
    la corrección más simple de justificar sin afinar más).
    """
    n_total = len(label)
    n_fraud_total = int(label.sum())
    sizes = np.array([len(c) for c in communities])
    fraud_counts = np.array([int(label[np.fromiter(c, dtype=int)].sum()) for c in communities])

    # comunidades de tamaño 1 no aportan nada al test (ni evidencia de enriquecimiento
    # real posible con un solo nodo) — se excluyen del cálculo de p-valor pero se
    # cuentan aparte para que quede claro cuánto del dataset son triviales.
    multi_mask = sizes >= 2
    pvals = np.ones(len(communities))
    if multi_mask.any():
        pvals[multi_mask] = hypergeom.sf(
            fraud_counts[multi_mask] - 1, n_total, n_fraud_total, sizes[multi_mask]
        )

    n_testable = int(multi_mask.sum())
    bonferroni_alpha = alpha / max(n_testable, 1)
    significant = multi_mask & (pvals < bonferroni_alpha)

    return {
        "n_comunidades": len(communities),
        "n_comunidades_triviales_tam1": int((~multi_mask).sum()),
        "n_comunidades_testables": n_testable,
        "alpha_bonferroni": bonferroni_alpha,
        "n_comunidades_significativas": int(significant.sum()),
        "fraude_capturado_en_sig": int(fraud_counts[significant].sum()),
        "pct_fraude_total_capturado_en_sig": (
            round(float(fraud_counts[significant].sum() / n_fraud_total), 4)
            if n_fraud_total else 0.0
        ),
        "pct_nodos_en_comunidades_sig": (
            round(float(sizes[significant].sum() / n_total), 4)
        ),
    }


def evaluate_communities(communities: list, label: np.ndarray) -> dict:
    """Junta las dos métricas de calidad de clustering usadas en este bloque."""
    scores = leave_one_out_cluster_scores(communities, label)
    from sklearn.metrics import roc_auc_score

    loo_auc = roc_auc_score(label, scores)
    return {
        "loo_auc": round(float(loo_auc), 4),
        "topk": topk_capture(scores, label),
        "enrichment": enrichment_test(communities, label),
    }


# ---------------------------------------------------------------------------
# Referencia/sanity-check de las features de CARE-GNN (no son features propias)
# ---------------------------------------------------------------------------

def sanity_check_raw_features_auc(features: spmatrix, label: np.ndarray) -> float:
    """AUC de una regresión logística simple 5-fold sobre las 32 features de CARE-GNN.

    **No son features nuestras** — son las que ya vienen calculadas en el `.mat` por
    los autores de CARE-GNN. Esto no evalúa nuestro clustering, es solo una referencia
    de "cuánta señal hay ya en features supervisadas sencillas sobre este dataset",
    útil para poner en contexto el número de Louvain (que es no supervisado y no ve
    absolutamente ninguna de estas 32 columnas). No es la arquitectura de CARE-GNN
    (que es un GNN, no una regresión logística) — es deliberadamente el sanity-check
    más simple posible, no un intento de reproducir su resultado.
    """
    X = StandardScaler().fit_transform(features.toarray())
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X, label, cv=5, scoring="roc_auc")
    return round(float(scores.mean()), 4)


def sanity_check_rtr_degree_auc(net_rtr: spmatrix, label: np.ndarray) -> float:
    """AUC de usar directamente el grado en `net_rtr` como score — ver `rtr_degree_proxy`."""
    from sklearn.metrics import roc_auc_score

    degree = rtr_degree_proxy(net_rtr)
    return round(float(roc_auc_score(label, degree)), 4)


# ---------------------------------------------------------------------------
# OddBall (Akoglu, McGlohon, Faloutsos, "OddBall: Spotting Anomalies in
# Weighted Graphs", PAKDD 2010) — anomalía de egonet vía ley de potencias en
# escala log-log. Implementación NATIVA con scipy/numpy sobre las matrices
# sparse ya cargadas por `data.py`/`build_yelpnyc_graphs` — deliberadamente
# SIN `networkx` y SIN el repo de referencia del paper
# (github.com/williamcorsel/oddball-anomalies, verificado que espera un TSV
# de lista de aristas `nodo1\tnodo2\tpeso`, no una matriz scipy ni un grafo
# de networkx — no encaja con el pipeline ya montado, ver
# `INVESTIGACION_GRAFOS.md`).
#
# Motivo de esta sección (encargo explícito tras la sesión de investigación
# del 2026-09-14, ver CONTEXTO.md): las señales de coordinación real
# (`net_rtr`/`net_rsr`) se quedan flojas (AUC 0,55-0,73) frente a `net_rur`
# (que mide sobre todo prolificidad, no coordinación — ver diagnóstico ya
# documentado en las secciones Yelp-Chi/Yelp-NYC/bretthollenbeck de este
# fichero). OddBall es un método clásico, sin ML ni etiquetas, con la misma
# filosofía estadística ya usada en este proyecto (modelo nulo hipergeométrico
# en `enrichment_test`, modelo de configuración de Newman en
# `profile_cluster.density_vs_configuration_model`) — la pregunta real es si
# aporta señal donde Louvain no la encuentra, no asumir que sí.
# ---------------------------------------------------------------------------

def oddball_egonet_stats(adjacency: spmatrix) -> tuple[np.ndarray, np.ndarray, dict]:
    """`Ni` (tamaño del egonet) y `Ei` (aristas dentro del egonet) por nodo,
    calculados nativo sobre la matriz sparse, sin construir ningún grafo de
    objetos Python.

    `Ni = 1 + grado(i)` — el egonet es el nodo más sus vecinos a 1 salto, en
    un grafo simple no dirigido sin auto-bucles (`adjacency` se binariza y
    se fuerza la diagonal a 0 por seguridad, aunque las matrices de este
    proyecto ya vienen así — verificado con las 4 relaciones de Yelp-Chi:
    todas simétricas, sin auto-bucles, valores en {0,1}, `(m - m.T).nnz == 0`
    para las cuatro).

    `Ei = grado(i) + triángulos(i)`: las `grado(i)` aristas del propio nodo
    hacia cada vecino SIEMPRE están dentro del egonet, más las aristas ENTRE
    pares de vecinos de `i` — que son exactamente los triángulos que pasan
    por `i` (cada triángulo {i, u, v} corresponde a una única arista (u, v)
    entre dos vecinos de `i`, y viceversa). Contar triángulos por nodo de
    forma vectorizada (sin bucle Python por nodo, inviable a esta escala)
    usa el truco estándar de álgebra de grafos sobre matrices sparse:

        triángulos(i) = 0.5 · Σ_j A[i,j] · (A@A)[i,j]

    `(A@A)[i,j]` es el número de vecinos comunes entre `i` y `j` (caminos de
    longitud 2); sumando solo sobre los `j` que SON vecinos de `i`
    (`A[i,j]=1`, vía `A.multiply(A@A)`) se cuenta cada triángulo dos veces
    (una por cada arista incidente a `i` dentro del triángulo), de ahí el /2.

    **Coste real medido en esta máquina — por qué SÍ compensa evitar
    `networkx` aquí, confirmando la hipótesis de `INVESTIGACION_GRAFOS.md`,
    con un matiz importante que también hay que decir (ver más abajo)**:
    para relaciones que son unión disjunta de cliques (`net_rur`/`net_rtr`/
    `net_rsr`, construidas así a propósito vía `_clique_matrix_from_groups`
    en Yelp-NYC/bretthollenbeck, y de hecho ya así en el `.mat` preprocesado
    de Yelp-Chi — verificado empíricamente, no asumido: `nnz(A@A) ≈ nnz(A)`
    en las tres), `A@A` tiene un número de no-ceros del mismo ORDEN que `A`
    — cada clique produce en `A@A` un bloque denso de tamaño similar al
    bloque ya denso de `A`, sin ninguna "explosión" de vecinos comunes fuera
    del propio clique. Confirmado con el caso límite ya documentado como
    INVIABLE con `networkx` en la sección Yelp-NYC de este fichero:
    `net_rsr` (146.038.672 aristas dirigidas, >16GB de RAM y sin terminar
    tras 9+ minutos al convertirlo a un grafo de `networkx`) — aquí, vía
    `A@A` directo sobre la matriz sparse, termina en **462,9s (~7,7 min) con
    un pico de 6,45GB de RAM**, lento pero terminado, sin ningún muro.
    `nnz(A@A)` salió 146.397.291, prácticamente idéntico a `nnz(A)`
    (146.038.672) — confirma la predicción de "mismo orden de magnitud" de
    arriba, no una coincidencia.

    **Pero esto NO generaliza a cualquier grafo grande sin más — `net_homo`
    (unión de las tres relaciones, Yelp-NYC) SÍ vuelve a chocar con un muro
    de memoria, por un motivo genuinamente distinto, no el overhead de
    `networkx`**: `net_homo` no es una unión disjunta de cliques (ya
    documentado en la sección Yelp-NYC de este fichero: 99,9% de los nodos
    caen en una única componente conexa gigante según
    `connected_components`), con grado medio real de ~415,6 y percentil 99
    de ~3.783 — con esa densidad, el número de pares de vecinos comunes por
    nodo de grado alto crece de forma combinatoria (un nodo de grado ~3.900
    aporta hasta ~7,6M pares posibles él solo), así que `A@A` deja de tener
    un nº de no-ceros del mismo orden que `A`. **Probado de verdad, no
    descartado a priori**: el proceso llegó a **10,9GB de RAM y seguía
    creciendo con fuerza tras ~2 minutos** (de 4,7GB nada más cargar/limpiar
    `A` — 149.220.438 aristas dirigidas — a 10,9GB en plena fase de `A@A`,
    con la memoria libre del sistema completo bajando de 8,6GB a 3,2GB en el
    mismo intervalo, subiendo varios GB cada 15s) — se mató el proceso a
    propósito antes de arriesgar dejar la máquina sin memoria, mismo umbral
    de precaución ya aplicado con `networkx`/Louvain sobre este mismo grafo.

    **Conclusión honesta de esta parte de la tarea**: evitar `networkx` SÍ
    resuelve el muro de escala para relaciones que son cliques disjuntas
    (`net_rsr` pasa de inviable a 7,7 min factibles) — pero NO para
    relaciones con una componente gigante de alto grado medio (`net_homo`),
    donde el cuello de botella es combinatorio (cuántos pares de vecinos
    comunes existen de verdad en un grafo denso y muy conectado), no el
    overhead de representar el grafo como objetos Python de `networkx`. Por
    eso `net_homo` se excluye a propósito de `run_yelpnyc_oddball_analysis`
    — no por no haberlo intentado, ver esa función para el detalle.
    """
    A = adjacency.tocsr().astype(bool).astype(np.float64)
    if A.diagonal().any():
        A.setdiag(0)
        A.eliminate_zeros()

    t0 = time.time()
    degree = np.asarray(A.sum(axis=1)).reshape(-1)
    t1 = time.time()
    AA = A @ A
    t2 = time.time()
    triangles = np.asarray(A.multiply(AA).sum(axis=1)).reshape(-1) / 2.0
    t3 = time.time()

    Ni = degree + 1.0
    Ei = degree + triangles
    timings = {
        "n_nodes": int(A.shape[0]),
        "nnz_A": int(A.nnz),
        "nnz_AA": int(AA.nnz),
        "degree_time_s": round(t1 - t0, 2),
        "matmul_time_s": round(t2 - t1, 2),
        "triangle_time_s": round(t3 - t2, 2),
        "total_time_s": round(t3 - t0, 2),
    }
    return Ni, Ei, timings


def fit_oddball_power_law(Ni: np.ndarray, Ei: np.ndarray) -> tuple[float, float, dict]:
    """Ajusta `Ei = C · Ni^theta` con regresión lineal en escala log-log
    (`log(Ei) = theta·log(Ni) + log(C)`, mínimos cuadrados vía `np.polyfit`
    — mismo espíritu que el resto del fichero: estadística simple y
    verificable, no un optimizador no lineal).

    Nodos con `Ni=1` (sin ningún vecino — comunidad trivial de tamaño 1 en
    la relación correspondiente) tienen `Ei=0` por construcción (no hay
    ninguna arista posible dentro de un egonet de un solo nodo) y se
    EXCLUYEN del ajuste (`log(0)` no está definido) — mismo criterio ya
    usado en `leave_one_out_cluster_scores`/`enrichment_test` para
    comunidades de tamaño 1 (no aportan información al ajuste, se tratan
    aparte, no se fuerza un valor artificial).
    """
    fit_mask = Ni > 1
    log_n = np.log(Ni[fit_mask])
    log_e = np.log(Ei[fit_mask])
    theta, log_c = np.polyfit(log_n, log_e, 1)
    c = float(np.exp(log_c))
    info = {
        "theta": round(float(theta), 4),
        "C": round(c, 6),
        "n_nodos_en_ajuste": int(fit_mask.sum()),
        "n_nodos_triviales_excluidos_Ni_igual_1": int((~fit_mask).sum()),
    }
    return float(theta), c, info


def oddball_anomaly_scores(Ni: np.ndarray, Ei: np.ndarray, theta: float, c: float) -> np.ndarray:
    """Score de anomalía por nodo — fórmula del paper original (Akoglu et
    al., PAKDD 2010, sección de "out-of-norm score"): dado el valor esperado
    por la ley de potencia `ŷ = C · Ni^theta`,

        score = (max(Ei, ŷ) / min(Ei, ŷ)) · log(|Ei − ŷ| + 1)

    Un nodo sobre la recta (`Ei ≈ ŷ`) da ratio ≈1 y log(≈1) ≈0 → score bajo.
    Un nodo muy por ENCIMA (near-clique, mucha más densidad de la esperada)
    o muy por DEBAJO (near-star, mucha menos) de su predicción sube el ratio
    Y el término logarítmico a la vez → score alto en ambas direcciones (el
    paper no distingue near-clique de near-star en el score final, solo en
    la interpretación posterior de cuál era el caso).

    **Dos decisiones de implementación, ninguna estaba en la fórmula
    original tal cual — documentadas para no esconderlas**:
    1. `eps=1e-6` sumado a numerador y denominador del ratio — evita
       división por cero exacta cuando `min(Ei, ŷ)=0`. En los grafos
       ponderados del paper original esto raramente ocurre; aquí sí, con
       relaciones tipo `net_rur` donde muchos nodos son cliques triviales
       (`Ei=0` exacto).
    2. Nodos con `Ni<=1` (sin egonet real, ver `fit_oddball_power_law`)
       reciben **score=0 directamente, sin aplicar la fórmula** — no hay
       ninguna estructura de la que OddBall pueda decir nada sobre un nodo
       sin vecinos; aplicar la fórmula ahí produciría un ratio dominado por
       el `eps` artificial, no una medida real de anomalía (mismo criterio
       que asignar la tasa base a comunidades de tamaño 1 en
       `leave_one_out_cluster_scores`, en vez de forzar un número sin
       información real detrás).
    """
    trivial = Ni <= 1
    y = Ei
    y_hat = c * Ni ** theta
    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_score = (
            (np.maximum(y, y_hat) + eps) / (np.minimum(y, y_hat) + eps)
        ) * np.log(np.abs(y - y_hat) + 1.0)
    return np.where(trivial, 0.0, raw_score)


def evaluate_oddball(adjacency: spmatrix, label: np.ndarray, name: str = "") -> dict:
    """Pipeline OddBall completo (egonet stats → ajuste de ley de potencia →
    score de anomalía), evaluado como **score continuo por nodo** — no como
    clustering, a diferencia del resto del fichero (`evaluate_communities`).
    Mismos criterios de evaluación ya establecidos: `roc_auc_score` contra
    `label` y `topk_capture` (reutilizada tal cual, sin cambios).
    """
    from sklearn.metrics import roc_auc_score

    Ni, Ei, timings = oddball_egonet_stats(adjacency)
    theta, c, fit_info = fit_oddball_power_law(Ni, Ei)
    scores = oddball_anomaly_scores(Ni, Ei, theta, c)
    label_arr = np.asarray(label).astype(int)
    auc = float(roc_auc_score(label_arr, scores))
    return {
        "name": name,
        "timings": timings,
        "power_law_fit": fit_info,
        "auc": round(auc, 4),
        "topk": topk_capture(scores, label_arr),
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_yelpchi_analysis(verbose: bool = True) -> dict:
    """Corre Louvain sobre las 5 variantes de grafo de Yelp-Chi y evalúa cada una."""
    dataset = data.load_yelpchi_graph_dataset()
    label = dataset["label"]

    detect_bursts()  # documentado: no hace nada, ver su docstring

    graphs = {
        "net_rur": dataset["net_rur"],
        "net_rtr": dataset["net_rtr"],
        "net_rsr": dataset["net_rsr"],
        "net_homo (union booleana ya provista)": dataset["net_homo"],
        "combinacion_ponderada_por_rareza": build_rarity_weighted_graph(
            dataset["net_rur"], dataset["net_rtr"], dataset["net_rsr"]
        ),
    }

    results = {}
    for name, matrix in graphs.items():
        weighted = name == "combinacion_ponderada_por_rareza"
        if verbose:
            print(f"\n--- {name} ---")
        communities, timings = detect_communities_louvain(matrix, weighted=weighted)
        metrics = evaluate_communities(communities, label)
        results[name] = {"timings": timings, "metrics": metrics}
        if verbose:
            print(f"timings: {timings}")
            print(f"LOO-AUC: {metrics['loo_auc']}")
            print(f"top-5%: {metrics['topk'][0.05]}")
            print(f"enrichment: {metrics['enrichment']}")

    if verbose:
        print("\n--- Sanity-checks (no son parte de nuestro clustering) ---")
    raw_auc = sanity_check_raw_features_auc(dataset["features"], label)
    rtr_degree_auc = sanity_check_rtr_degree_auc(dataset["net_rtr"], label)
    results["sanity_check_raw_features_auc"] = raw_auc
    results["sanity_check_rtr_degree_auc"] = rtr_degree_auc
    if verbose:
        print(f"AUC regresión logística sobre las 32 features de CARE-GNN (no propias): {raw_auc}")
        print(f"AUC de usar solo el grado en net_rtr como score (proxy de burst, no burst real): {rtr_degree_auc}")
        print_reference_comparison(results)

    return results


def print_reference_comparison(results: dict) -> None:
    print(
        """
--- Comparación con referencias publicadas (con salvedades, no cifras para citar sin más) ---

- SpEagle (Rayana & Akoglu, paper original): ~0.78 AUC, pero medido sobre el Yelp-Chi
  ORIGINAL sin el preprocesado de CARE-GNN (que aquí filtra 45.954 nodos de los
  ~67.395 publicados) — no es estrictamente comparable número a número con nada de
  este script.
- CARE-GNN (Dou et al., CIKM 2020), sobre este mismo Yelp-Chi preprocesado: AUC~0.7445
  con 20% de datos de entrenamiento, según un snippet de búsqueda web, NO verificado
  línea a línea contra la tabla completa del paper (arxiv.org/abs/2008.08692) — no citar
  en marketing sin confirmarlo contra el PDF real. Además CARE-GNN es un GNN
  SUPERVISADO con entrenamiento (ve las etiquetas de un 20% de nodos), mientras que
  Louvain aquí es completamente NO supervisado (no ve ninguna etiqueta al construir las
  comunidades, solo se usa `label` después para evaluar) — no es una comparación
  limpia, son números de naturaleza distinta puestos uno al lado del otro por contexto,
  no una carrera justa.
- El `loo_auc` de este script tampoco es un AUC "de modelo" al uso: es el AUC de usar
  la tasa de fraude de la comunidad de cada nodo (calculada excluyendo su propia
  etiqueta) como si fuera un score continuo — mide si la estructura de comunidades por
  sí sola concentra fraude, no la calidad de un clasificador entrenado.
- Si `net_rur` sale con el LOO-AUC más alto de los cinco grafos probados, ojo antes de
  leerlo como "mejor que SpEagle/CARE-GNN": un diagnóstico manual (ver docstring del
  módulo) muestra que ese número viene sobre todo de comunidades PEQUEÑAS (2-3 reviews
  del mismo usuario, mayoritariamente fraude — cuentas de usar-y-tirar), mientras que
  las comunidades más GRANDES de `net_rur` (reviewers prolíficos con 24-47 reviews) son
  0% fraude. Es una tarea más estrecha que "detectar fraude en general" — no es
  comparable a SpEagle/CARE-GNN ni siquiera dejando de lado las salvedades de arriba.
"""
    )


def run_yelpchi_oddball_analysis(verbose: bool = True) -> dict:
    """OddBall sobre las 4 relaciones de Yelp-Chi — comparación directa
    contra el LOO-AUC de Louvain ya medido en `run_yelpchi_analysis()`
    (`net_rur` 0,814 / `net_rtr` 0,587 / `net_rsr` 0,636 / `net_homo`
    0,572). Ver el bloque de resultados reales justo debajo de esta función
    para la lectura honesta completa (spoiler: no gana a Louvain en ninguna
    de las 4, y en `net_rur` sale con el signo invertido).
    """
    dataset = data.load_yelpchi_graph_dataset()
    label = dataset["label"]
    graphs = {
        "net_rur": dataset["net_rur"],
        "net_rtr": dataset["net_rtr"],
        "net_rsr": dataset["net_rsr"],
        "net_homo": dataset["net_homo"],
    }
    results = {}
    for name, matrix in graphs.items():
        if verbose:
            print(f"\n--- OddBall sobre {name} (Yelp-Chi) ---")
        res = evaluate_oddball(matrix, label, name=name)
        results[name] = res
        if verbose:
            print(f"timings: {res['timings']}")
            print(f"power_law_fit: {res['power_law_fit']}")
            print(f"AUC: {res['auc']}")
            print(f"top-5%: {res['topk'][0.05]}")
    return results


# ---------------------------------------------------------------------------
# Resultados reales (`python features_graph.py oddball_yelpchi`, esta
# máquina, CPU — no hay componente aleatorio en OddBall, la regresión
# log-log es determinista, no hace falta `SEED`)
# ---------------------------------------------------------------------------
#
# | Relación   | theta  | C      | AUC OddBall | LOO-AUC Louvain (ya medido) | Tiempo total |
# |---|---|---|---|---|---|
# | `net_rur`  | 2,327  | 0,2157 | **0,2797**  | 0,814                        | 0,00s |
# | `net_rtr`  | 2,057  | 0,3272 | 0,5008      | 0,587                        | 0,13s |
# | `net_rsr`  | 2,032  | 0,4220 | 0,4694      | 0,636                        | 2,53s |
# | `net_homo` | 2,135  | 0,1961 | 0,4895      | 0,572                        | 3,63s |
#
# **Veredicto honesto, sin adornar: OddBall NO gana a Louvain en ninguna de
# las 4 relaciones — pierde en las 4, y en `net_rur` sale con el AUC
# INVERTIDO (0,28, muy por debajo de 0,5, no solo "peor").** Diagnóstico de
# por qué, no solo el número:
#
# 1. **`net_rur`/`net_rtr`/`net_rsr` son uniones disjuntas de cliques
#    (ya documentado en este mismo fichero — cada nodo pertenece a un único
#    grupo de su relación, y dentro de ese grupo TODOS los pares están
#    conectados)**. Eso significa que, dentro de un mismo grupo de tamaño
#    `s`, TODOS los nodos tienen exactamente el mismo `Ni=s` y el mismo
#    `Ei=s(s-1)/2` — no hay ninguna variación topológica dentro del grupo
#    que OddBall pueda usar para distinguir un nodo de otro (el "near-clique
#    vs. near-star" que el método está diseñado para separar no existe
#    aquí: TODO es clique exacto, nunca near-star). El único grado de
#    libertad que le queda al score es "¿el TAMAÑO de mi grupo se desvía de
#    la tendencia global ajustada sobre todos los tamaños de grupo?" — una
#    pregunta bastante más pobre que la que ya responde la propia
#    estructura de comunidades vía Louvain+LOO-AUC (que sí usa la tasa de
#    fraude real de cada comunidad, no solo su tamaño).
# 2. **El signo invertido de `net_rur` confirma, con un método
#    completamente distinto, el mismo hallazgo ya diagnosticado a mano en
#    la sección Yelp-Chi de este fichero**: las comunidades PEQUEÑAS de
#    `net_rur` (tamaño 2-3, mayoritariamente fraude) tienen un `Ni` pequeño
#    y caen relativamente CERCA de la recta ajustada (hay muchísimas
#    comunidades de tamaño 2-3, así que dominan el ajuste de mínimos
#    cuadrados) — mientras que las comunidades GRANDES (24-47 reviews del
#    mismo usuario, 0% fraude, "Yelp Elite" genuino) son las que más se
#    DESVÍAN de esa tendencia dominada por los grupos pequeños, y por tanto
#    reciben el score de anomalía OddBall más alto. El resultado es que
#    OddBall marca como "más anómalo" justo al grupo que es genuino, y como
#    "normal" al grupo que es fraude — lo opuesto de lo que se necesita.
#    Nótese que esto es exactamente el motivo por el que la nota dejada en
#    una sesión anterior (ver CONTEXTO.md, "Pausa de sesión 2026-09-14")
#    avisaba de no correr OddBall sobre relaciones que son "cliques exactos
#    sin varianza topológica" — este resultado lo confirma con datos reales,
#    no solo con la intuición ya apuntada entonces.
# 3. **`net_homo` (la única de las 4 con varianza topológica real dentro del
#    egonet — un nodo puede tener vecinos de las tres relaciones a la vez,
#    con densidades de conexión distintas entre ellos) tampoco gana**:
#    AUC 0,4895, esencialmente aleatorio. Con una componente gigante
#    dominando el 99%+ de los nodos (mismo hallazgo ya documentado para
#    Yelp-NYC/bretthollenbeck en este fichero), la mayoría de los egonets
#    son de tamaño/densidad parecidos entre sí, y la desviación de la ley de
#    potencia no correlaciona con `label`.
#
# **Conclusión de la Tarea 1 sobre Yelp-Chi**: OddBall, tal como lo define
# el paper original, no aporta señal útil sobre relaciones que son cliques
# exactos por construcción (que es la mayoría de las relaciones de este
# proyecto) — el resultado es coherente con la advertencia ya dejada en una
# sesión anterior antes de escribir código, ahora confirmada con números
# reales en vez de solo intuición. La única parte de la tarea que sí dio un
# resultado nuevo y aprovechable es el hallazgo de ESCALA (ver docstring de
# `oddball_egonet_stats`): evitar `networkx` sí resuelve el muro de memoria
# de `net_rsr`/`net_homo` — un hallazgo de infraestructura, no de precisión
# de detección.
# ---------------------------------------------------------------------------


"""
===============================================================================
Yelp-NYC (Fase 1, con texto real) — grafo construido a mano, no viene
pre-proyectado como Yelp-Chi/Amazon
===============================================================================

`data.load_yelpnyc_dataset()` devuelve un DataFrame plano (reviewer_id,
business_id, rating, date, text, is_fake), no matrices ya proyectadas — aquí
hay que construir las tres relaciones homogéneas review-review a mano, y a
8x la escala de nodos de Yelp-Chi (359.052 vs 45.954).

## Construcción: evitar el bucle O(n²), usar el truco de multiplicación
   de matrices dispersas

Para una relación "mismo valor de columna X" (mismo reviewer / mismo
negocio+rating [+ventana]), el grafo resultante es por construcción una
**unión disjunta de cliques**: cada review pertenece a exactamente un grupo
de `groupby(X)`, así que solo puede tener aristas hacia otros miembros de
ESE grupo. Construirlo así, en vez de comparar pares a mano:

1. Matriz de incidencia dispersa `M` (n_reviews × n_grupos), `M[i, g] = 1` si
   la review `i` pertenece al grupo `g`.
2. `A = M @ M.T` (multiplicación dispersa, la hace BLAS/scipy, no un bucle
   Python) — `A[i, j]` = número de grupos que comparten `i` y `j` (aquí
   siempre 0 o 1, porque cada review está en un solo grupo).
3. Poner a cero la diagonal (`A.setdiag(0)`) y binarizar.

Esto construye exactamente el mismo grafo que un bucle por pares dentro de
cada grupo, pero en segundos en vez de horas — ver `_clique_matrix_from_groups`.
Verificado con tiempos reales (esta máquina, CPU): construir las tres
matrices dispersas (`net_rur` + `net_rtr` semana + `net_rsr`) tarda en total
**9,3s**: la parte cara nunca fue construir la matriz, fue convertirla a un
grafo de `networkx` para correr Louvain (ver más abajo).

## Ventana temporal para `net_rtr`: semana vs. mes, con cifras reales, no a
   ciegas

A diferencia de Yelp-Chi (que traía la ventana ya binarizada a "mismo mes",
sin fecha real disponible), aquí sí hay `date` real, así que se puede elegir
la ventana. Probadas ambas:

| Ventana | Grupos (negocio+rating+ventana) | Tamaño máx. de grupo | Aristas no dirigidas |
|---|---|---|---|
| Semana (`W`, ISO) | 224.182 | 31 | 289.711 |
| Mes (`M`) | 115.900 | 111 | 1.216.228 (4,2x más) |

Semana es la ventana por defecto de `build_yelpnyc_graphs` (más estricta:
conecta reviews casi simultáneas, más específico de coordinación real —
"doce personas que fueron el finde de apertura" cae en semana, no en un mes
entero de reviews sueltas del mismo negocio+rating). Mes se probó también
con el harness completo para tener el trade-off medido, no solo intuido (ver
tabla de resultados más abajo): da un LOO-AUC casi idéntico (0,57 vs 0,5526)
pero con 4,2x más aristas y más del doble de tiempo de Louvain — no compensa
el coste extra por esta cifra tan parecida, así que semana es la elección
razonada para el resto del proyecto, no solo la "por defecto" sin motivo.

## Hallazgo real de escala — `net_rsr` (mismo negocio+rating, sin ventana)
   es INVIABLE de convertir a un grafo de `networkx` en esta máquina

`net_rsr` sin ventana agrupa solo por (negocio, rating): 4.452 grupos, con
grupos de hasta 3.784 reviews (un negocio+rating muy común) — el clique de
ese grupo por sí solo aporta ~7,16 millones de aristas. En total, la matriz
dispersa de `net_rsr` tiene **146.038.672 aristas dirigidas (~73M no
dirigidas)** — 21x las 3.402.743 aristas no dirigidas de Yelp-Chi. Construir
la matriz dispersa es barato (1,12s, ~1,2GB en memoria), pero convertirla a
un grafo de `networkx` (`nx.from_scipy_sparse_array`, lo que exige el mismo
harness que se reutiliza para todo lo demás) **no terminó tras más de 9
minutos, con el proceso de Python creciendo por encima de 16GB de RAM y
subiendo** — comprobado dos veces de forma independiente (con un incidente
adicional de proceso duplicado del harness de `run_in_background`, ya
documentado como problema conocido de esta máquina en `CONTEXTO.md`, que hizo
el primer intento aún más lento por competir dos procesos por la misma
memoria). Extrapolando la tasa de conversión medida en las relaciones que sí
completaron (~13,5 µs/arista en `net_rur`/`net_rtr`), 73M aristas
implicarían >16 minutos solo para construir el grafo, sin contar Louvain — y
el crecimiento de memoria observado (16GB+ para una matriz que en formato
disperso pesa 1,2GB, >13x de sobrecarga) es coherente con el conocido coste
por arista de la representación de grafo en Python puro de `networkx`
(diccionario de diccionarios), no con un error puntual. **Se documenta como
inviable en esta sesión/máquina, no se fuerza un resultado.**

**No hace falta Louvain para `net_rsr` de todas formas.** Como esta relación
es una unión disjunta de cliques (cada review pertenece a un solo grupo
negocio+rating, sin aristas entre grupos), la partición que maximiza
modularidad es, matemáticamente, exactamente esos mismos grupos — dividir un
clique nunca mejora la modularidad (ya tiene densidad máxima) y fusionar dos
cliques sin ninguna arista entre ellos tampoco (el modelo nulo esperaría
aristas entre ellos si se fusionan, y no hay ninguna). Esto no es solo una
afirmación teórica: se **verificó empíricamente** corriendo Louvain de
verdad (vía `detect_communities_louvain`, el mismo harness) sobre `net_rur` y
`net_rtr` (semana) — que sí son factibles en `networkx` — y en ambos casos
el número de comunidades que encontró Louvain coincidió EXACTAMENTE con el
número de grupos de origen (160.225 reviewers únicos = 160.225 comunidades;
224.182 grupos negocio+rating+semana = 224.182 comunidades). Con esa
confirmación, `net_rsr` se evalúa usando directamente la partición de
`groupby(["business_id", "rating"])` como "comunidades" (función
`groupby_cliques_as_communities`), sin pasar por `networkx`/Louvain — mismo
resultado esperado, sin el muro de memoria.

## `net_homo` (unión de las tres) — mismo muro de memoria, y además la
   alternativa barata (componentes conexas) no sirve de nada

`net_homo` hereda el problema de `net_rsr` (que aporta el 97%+ de sus
aristas) — inviable en `networkx`/Louvain por el mismo motivo. Como
alternativa barata (sin pasar por `networkx`), se probó
`scipy.sparse.csgraph.connected_components` (no requiere construir un grafo
de objetos Python, opera directo sobre la matriz dispersa: 5,45s). El
resultado es una prueba más de que la unión sin ponderar no sirve de nada a
esta escala, no solo un rodeo técnico: **129 componentes conexas, pero una
sola de ellas contiene 358.798 de los 359.052 nodos (99,9%)** — la
combinación de las tres relaciones basta para que casi todo el dataset quede
conectado transitivamente en un único blob gigante, sin ninguna estructura
útil que separar. Evaluado igualmente con el harness para dejarlo medido, no
solo descrito (ver tabla): LOO-AUC 0,0016, básicamente inútil, coherente con
"casi todos los nodos comparten la misma comunidad gigante y por tanto el
mismo score". **Conclusión para este dataset, más tajante que en Yelp-Chi**:
la unión sin ponderar de las tres relaciones no es solo "no mejora" (como en
Yelp-Chi, 0,572 vs. 0,572), aquí directamente destruye toda la señal que
`net_rur`/`net_rsr` tenían por separado — no se prueba la variante ponderada
por rareza de `build_rarity_weighted_graph` sobre Yelp-NYC (ya tardaba 6,3x
más en Yelp-Chi y sobre un grafo 21x más grande la conversión a `networkx`
tiene el mismo muro de memoria que `net_rsr`/`net_homo`, documentado arriba
— no tiene sentido intentarlo sin resolver antes ese cuello de botella).

## Resultados reales del harness sobre Yelp-NYC (359.052 nodos, esta máquina)

| Relación | Aristas (no dirigidas) | Build grafo | Louvain | Nº comunidades |
|---|---|---|---|---|
| `net_rur` (mismo reviewer) | 1.590.883 | 27,9s | 132,2s | 160.225 |
| `net_rtr` (negocio+rating+semana) | 289.711 | 10,6s | 37,9s | 224.182 |
| `net_rtr` (negocio+rating+mes, comparación) | 1.216.228 | 13,5s | 62,2s | 115.900 |
| `net_rsr` (negocio+rating, sin ventana) | ~73.019.336 | **inviable, ver arriba** | groupby directo (1,1s) | 4.452 |
| `net_homo` (unión de las tres) | ~74,6M | **inviable, ver arriba** | connected_components (5,5s) | 129 |

| Grafo | LOO-AUC | Precision top-5% | Lift top-5% | Comunidades sig. (enrichment) |
|---|---|---|---|---|
| `net_rur` | **0,9046** | **0,7519** | **7,32x** | 172 (6,13% del fraude) |
| `net_rtr` (semana) | 0,5526 | 0,1838 | 1,79x | 16 (0,53% del fraude) |
| `net_rtr` (mes) | 0,57 | 0,1916 | 1,87x | 21 (0,89% del fraude) |
| `net_rsr` (groupby directo) | 0,6296 | 0,2504 | 2,44x | 75 (10,53% del fraude) |
| `net_homo` (connected_components) | 0,0016 | 0,0028 | 0,03x | 1 (0,03% del fraude) |

Sanity-check de burst detection (ver sección siguiente): AUC 0,5315
(semana) / 0,5272 (día) — señal débil, casi de azar, documentado sin
maquillar.

**`net_rur` sale incluso más alto que en Yelp-Chi (0,9046 vs. 0,814), pero
por un motivo DISTINTO al diagnosticado allí — diagnóstico real hecho a
mano, no asumido:**

| Nº de reviews del reviewer | Reviewers | Reviews totales | Tasa de fraude (ponderada) |
|---|---|---|---|
| 1 | 105.997 | 105.997 | 22,24% |
| 2-3 | 35.180 | 80.648 | 10,43% |
| 4-7 | 12.385 | 61.545 | 4,51% |
| 8-15 | 4.390 | 45.955 | 2,06% |
| 16-31 | 1.664 | 35.602 | 1,88% |
| 32-63 | 515 | 21.371 | 2,15% |
| 64-199 | 94 | 7.934 | 0,66% |

En Yelp-Chi el patrón era "en forma de U invertida al revés": comunidades
pequeñas (2-3) casi 100% fraude, comunidades grandes 0% fraude, con un salto
brusco. **Aquí el patrón es monótono y suave**: cuantas más reviews ha
escrito un reviewer en toda su vida, menos probable que cualquiera de ellas
sea fraude — desciende de forma casi continua del 22% (una sola review) al
0,66% (60+ reviews). Nota importante sobre el mecanismo real del score: las
comunidades de tamaño 1 (105.997 reviewers, el grupo con MÁS fraude real,
22,24%) reciben la tasa base global (10,27%) en `leave_one_out_cluster_scores`
por diseño (no hay "los demás" con quien calcular una tasa) — así que la
señal que de verdad mueve el AUC no viene de identificar a esos reviewers de
una sola review (quedan con un score mediocre, ni alto ni bajo), sino de la
separación clara entre reviewers de pocas reviews (2-7, con fraude por
encima de la base) y reviewers muy prolíficos (32+, con fraude muy por
debajo) — **es la misma confusión de fondo que en Yelp-Chi** (esto mide
sobre todo "cuán prolífico es este reviewer", que correlaciona con
genuinidad, no necesariamente "red de cuentas coordinadas" en el sentido que
persigue el README), solo que aquí la forma de la curva es monótona en vez
de un salto en dos escalones. Misma implicación para `profile_cluster.py`
que en Yelp-Chi: tratar la señal de `net_rur` (prolificidad/reutilización de
la MISMA cuenta) como un indicio de Nivel A separado de la de
`net_rtr`/`net_rsr` (coordinación entre cuentas DISTINTAS), no fundirlos.
"""

def _clique_matrix_from_groups(keys) -> tuple[csr_matrix, dict]:
    """Construye el grafo "mismo valor de `keys`" vía multiplicación de
    matrices dispersas (`M @ M.T`), evitando el bucle O(n²) sobre pares —
    ver docstring de la sección Yelp-NYC más arriba para la justificación
    completa y los tiempos reales medidos.

    `keys` puede ser cualquier `pd.Series` (o algo convertible a array) cuyo
    valor identifica el grupo de cada fila (reviewer_id, o una clave
    compuesta tipo "business_id_rating_semana"). Devuelve la matriz binaria
    (`csr_matrix`, sin diagonal) y un dict de timings.
    """
    keys = pd.Series(keys).reset_index(drop=True)
    n = len(keys)
    t0 = time.time()
    codes, uniques = pd.factorize(keys.to_numpy())
    n_groups = len(uniques)
    rows = np.arange(n)
    data_ones = np.ones(n, dtype=np.float32)
    incidence = csr_matrix((data_ones, (rows, codes)), shape=(n, n_groups))
    adjacency = incidence @ incidence.T
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    build_time = time.time() - t0
    return adjacency.tocsr(), {
        "n_nodes": n,
        "n_groups": int(n_groups),
        "nnz_directed": int(adjacency.nnz),
        "nnz_undirected_approx": int(adjacency.nnz // 2),
        "build_time_s": round(build_time, 2),
    }


def groupby_cliques_as_communities(keys) -> list:
    """Comunidades de una relación "mismo `keys`" leídas directamente del
    `groupby`, sin pasar por `networkx`/Louvain.

    Válido porque, por construcción, el grafo "mismo `keys`" es una unión
    disjunta de cliques (cada fila pertenece a exactamente un grupo) — la
    partición que maximiza modularidad es exactamente esos grupos (ver
    justificación teórica + verificación empírica en el docstring de la
    sección Yelp-NYC). Devuelve la lista en el mismo formato que
    `nx.algorithms.community.louvain_communities` (lista de `set` de
    índices de fila), para poder reutilizar `evaluate_communities` sin
    ningún cambio.
    """
    keys = pd.Series(keys).reset_index(drop=True)
    idx = pd.DataFrame({"i": np.arange(len(keys)), "k": keys.to_numpy()}).groupby("k")["i"]
    return [set(group.to_numpy()) for _, group in idx]


def build_yelpnyc_graphs(df, rtr_window: str = "W") -> dict:
    """Construye `net_rur`/`net_rtr`/`net_rsr`/`net_homo` para Yelp-NYC a
    partir del DataFrame plano de `data.load_yelpnyc_dataset()`.

    A diferencia de Yelp-Chi/Amazon, aquí no viene ningún grafo
    pre-proyectado — se construye con `_clique_matrix_from_groups` sobre:
    - `net_rur`: mismo `reviewer_id`.
    - `net_rtr`: mismo `business_id` + mismo `rating` + misma ventana
      temporal (`rtr_window`, por defecto semana ISO — ver docstring de la
      sección Yelp-NYC sobre el trade-off medido semana/mes).
    - `net_rsr`: mismo `business_id` + mismo `rating`, sin ventana.
    - `net_homo`: unión booleana de las tres (aquí se calcula, a diferencia
      de Yelp-Chi que ya la traía hecha).

    Devuelve un dict con las cuatro matrices (`scipy.sparse.csr_matrix`) más
    `timings` (dict con el tiempo/tamaño de construcción de cada una). Nota
    importante: `net_rsr` y `net_homo` SÍ se construyen aquí (la
    construcción de la matriz dispersa es barata, ~1-4s cada una) — lo que
    es inviable en esta máquina es convertirlas a un grafo de `networkx`
    para correr Louvain encima, no construir la matriz en sí (ver docstring
    de la sección Yelp-NYC). Quien llame a esta función y quiera correr
    Louvain sobre el resultado debe usar `groupby_cliques_as_communities`
    para `net_rsr` y evitar `net_homo` salvo con `connected_components` u
    otra alternativa que no pase por `networkx`.
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(rtr_window).astype(str)

    rur_matrix, rur_timings = _clique_matrix_from_groups(df["reviewer_id"].astype(str))
    rtr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str) + "_" + period
    rtr_matrix, rtr_timings = _clique_matrix_from_groups(rtr_key)
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_matrix, rsr_timings = _clique_matrix_from_groups(rsr_key)

    t0 = time.time()
    homo_matrix = ((rur_matrix + rtr_matrix + rsr_matrix) > 0).astype(np.float32).tocsr()
    homo_time = time.time() - t0

    return {
        "net_rur": rur_matrix,
        "net_rtr": rtr_matrix,
        "net_rsr": rsr_matrix,
        "net_homo": homo_matrix,
        "timings": {
            "net_rur": rur_timings,
            "net_rtr": {**rtr_timings, "window": rtr_window},
            "net_rsr": rsr_timings,
            "net_homo": {"union_build_time_s": round(homo_time, 2), "nnz": int(homo_matrix.nnz)},
        },
    }


# ---------------------------------------------------------------------------
# Burst detection real (Nivel A) — posible aquí, imposible en Yelp-Chi
# ---------------------------------------------------------------------------

def detect_bursts_yelpnyc(df, window: str = "W") -> np.ndarray:
    """Z-score de ráfaga por review, frente a la tasa base propia de CADA
    negocio (no la tasa global).

    Por cada negocio: se cuentan reviews por ventana de tiempo (`window`,
    por defecto semana ISO — se probó también día, ver comparación de AUC
    en el docstring de la sección Yelp-NYC, semana sale ligeramente mejor:
    0,5315 vs. 0,5272), incluyendo las ventanas SIN ninguna review de ese
    negocio (rellenadas a 0) entre la primera y la última review de su
    propia vida — así la media/desviación típica del negocio reflejan de
    verdad su ritmo habitual, no solo el ritmo de las semanas en las que
    hubo actividad. El z-score de cada ventana se asigna a todas las
    reviews que caen en ella.

    Negocios con muy pocas ventanas en su vida (pocas semanas de
    actividad) tienen una estimación de desviación típica poco fiable —
    limitación real, no corregida aquí (no hay suficiente dato por negocio
    para un modelo más robusto sin cambiar el diseño). Negocios con
    desviación típica 0 (mismo número de reviews en cada ventana, incluido
    el caso de una sola ventana en toda su vida) reciben z=0 en todas sus
    reviews (ninguna ventana se sale de su propio patrón).

    Devuelve un array alineado por fila con `df` (mismo orden, tras
    `reset_index`).
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(window)
    business = df["business_id"]

    counts = df.groupby([business, period]).size()
    z_lookup = {}
    for business_id, sub in counts.groupby(level=0):
        sub = sub.droplevel(0)
        full_index = pd.period_range(sub.index.min(), sub.index.max(), freq=window)
        full_counts = sub.reindex(full_index, fill_value=0)
        mean = full_counts.mean()
        std = full_counts.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_values = pd.Series(0.0, index=full_counts.index)
        else:
            z_values = (full_counts - mean) / std
        z_lookup.update({(business_id, p): float(v) for p, v in z_values.items()})

    return np.array([z_lookup[(b, p)] for b, p in zip(business, period)])


def sanity_check_burst_auc(df, burst_scores: np.ndarray) -> float:
    """AUC de usar directamente el z-score de ráfaga como score frente a
    `is_fake` — mismo espíritu que `sanity_check_rtr_degree_auc` en el
    bloque de Yelp-Chi (comprobar si una señal simple, en solitario, separa
    algo, sin que sea todavía parte del clustering).
    """
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(df["is_fake"].to_numpy(), burst_scores)), 4)


# ---------------------------------------------------------------------------
# Co-bursting (Tarea 2, sesión 2026-09-14/15) — alternativa a `net_rtr` que
# exige una ráfaga REAL y anómala detrás de cada arista, no solo coincidir
# en el mismo bin temporal. Inspirado en Xu et al., "Modeling Review Spam
# Using Temporal Patterns and Co-bursting Behaviors" (ver
# `INVESTIGACION_GRAFOS.md`) — implementación simplificada con el z-score
# genérico ya existente (`detect_bursts_yelpnyc`) como test de "¿esta
# ventana es una ráfaga real?", no el test de correlación de series
# temporales completo del paper original (más caro, no implementado aquí).
# ---------------------------------------------------------------------------

def build_coburst_matrix(df: pd.DataFrame, window: str = "W", z_threshold: float = 2.0) -> tuple[csr_matrix, dict]:
    """Variante de `net_rtr` que solo conecta dos reviews si están en el
    MISMO negocio, la MISMA ventana temporal, Y esa ventana es una ráfaga
    real y anómala de ese negocio (`z-score >= z_threshold`, calculado con
    `detect_bursts_yelpnyc`, el mismo z-score por negocio+ventana que ya usa
    `burst_participation` en `profile_cluster.py` con el mismo umbral por
    defecto, 2.0).

    A diferencia de `net_rtr` (que conecta reviews en CUALQUIER
    negocio+rating+ventana, esté o no esa ventana en ráfaga — la mayoría de
    sus aristas vienen de ventanas totalmente normales, ver hallazgo abajo),
    esta relación deliberadamente **no exige mismo rating** — sigue la
    especificación de la tarea al pie de la letra (mismo negocio + ventana +
    ráfaga real, sin más condición) porque el paper de referencia conecta
    reviewers por coordinación TEMPORAL, no por coincidencia de valoración;
    dos cuentas coordinadas podrían dejar ratings distintos a propósito para
    camuflarse. Ver `build_rtr_coburst_control_matrix` para un control que
    SÍ mantiene la restricción de rating, pensado para aislar si el efecto
    observado viene del filtro de ráfaga o de haber quitado el rating.

    Implementación: se reutiliza `_clique_matrix_from_groups` sin
    modificarla — las reviews que NO caen en una ventana de ráfaga real
    reciben una clave de grupo única (su propio índice de fila, con un
    prefijo que nunca colisiona con una clave real de negocio+ventana), así
    quedan cada una en un grupo "singleton" sin ninguna arista, mientras que
    las reviews SÍ en ráfaga se agrupan normalmente por negocio+ventana y
    forman un clique entre sí — mismo resultado que filtrar el DataFrame
    antes de construir el grafo, pero sin tener que reindexar después (la
    matriz devuelta queda alineada con `df` tal cual, igual que `net_rtr`).
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(window).astype(str)
    business = df["business_id"].astype(str)

    burst_scores = detect_bursts_yelpnyc(df, window=window)
    is_burst = burst_scores >= z_threshold

    real_key = business + "_" + period
    singleton_key = pd.Series(
        ["_no_burst_row" + str(i) for i in range(len(df))], index=real_key.index
    )
    coburst_key = real_key.where(is_burst, singleton_key)

    matrix, timings = _clique_matrix_from_groups(coburst_key)
    timings.update(
        {
            "window": window,
            "z_threshold": z_threshold,
            "n_reviews_en_ventana_de_rafaga": int(is_burst.sum()),
            "pct_reviews_en_rafaga": round(float(is_burst.mean()), 4),
        }
    )
    return matrix, timings


def build_rtr_coburst_control_matrix(df: pd.DataFrame, window: str = "W", z_threshold: float = 2.0) -> tuple[csr_matrix, dict]:
    """Control honesto para aislar el efecto del filtro de ráfaga: idéntico
    a `net_rtr` (mismo negocio + mismo rating + misma ventana) pero
    restringido, además, a ventanas en ráfaga real (`z-score >= z_threshold`)
    — a diferencia de `build_coburst_matrix`, aquí SÍ se mantiene la
    restricción de rating de `net_rtr`, cambiando una sola variable a la vez
    en vez de dos (quitar rating Y añadir el filtro de ráfaga a la vez, que
    es lo que hace la variante principal) para poder atribuir cualquier
    diferencia de AUC a una causa concreta, no a las dos mezcladas.
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(window).astype(str)
    business = df["business_id"].astype(str)
    rating = df["rating"].astype(str)

    burst_scores = detect_bursts_yelpnyc(df, window=window)
    is_burst = burst_scores >= z_threshold

    real_key = business + "_" + rating + "_" + period
    singleton_key = pd.Series(
        ["_no_burst_row" + str(i) for i in range(len(df))], index=real_key.index
    )
    coburst_key = real_key.where(is_burst, singleton_key)

    matrix, timings = _clique_matrix_from_groups(coburst_key)
    timings.update(
        {
            "window": window,
            "z_threshold": z_threshold,
            "n_reviews_en_ventana_de_rafaga": int(is_burst.sum()),
        }
    )
    return matrix, timings


def run_yelpnyc_coburst_analysis(verbose: bool = True, z_threshold: float = 2.0) -> dict:
    """Evalúa la relación de co-bursting (`build_coburst_matrix`) con el
    mismo harness ya usado para el resto de relaciones de Yelp-NYC
    (`detect_communities_louvain` + `evaluate_communities`, LOO-AUC/top-k/
    enrichment) y la compara honestamente contra `net_rtr` actual
    (LOO-AUC 0,5526, semana, ya medido en `run_yelpnyc_analysis`). Incluye
    también el control `build_rtr_coburst_control_matrix` (mismo filtro de
    ráfaga, pero manteniendo la restricción de rating) para aislar qué parte
    del resultado viene de cada cambio. Ver el bloque de resultados reales
    justo debajo de esta función para la lectura honesta completa.
    """
    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    label = df["is_fake"].to_numpy()

    results = {}

    if verbose:
        print(f"\n--- Co-bursting (sin rating, z>={z_threshold}) vs. net_rtr actual (LOO-AUC 0,5526) ---")
    matrix, build_timings = build_coburst_matrix(df, window="W", z_threshold=z_threshold)
    communities, louvain_timings = detect_communities_louvain(matrix)
    metrics = evaluate_communities(communities, label)
    results["coburst_sin_rating"] = {
        "build_timings": build_timings,
        "louvain_timings": louvain_timings,
        "metrics": metrics,
    }
    if verbose:
        print(f"build_timings: {build_timings}")
        print(f"louvain_timings: {louvain_timings}")
        print(f"LOO-AUC: {metrics['loo_auc']}")
        print(f"top-5%: {metrics['topk'][0.05]}")

    if verbose:
        print(f"\n--- Control: net_rtr (con rating) + filtro de ráfaga (z>={z_threshold}) ---")
    control_matrix, control_build_timings = build_rtr_coburst_control_matrix(df, window="W", z_threshold=z_threshold)
    control_communities, control_louvain_timings = detect_communities_louvain(control_matrix)
    control_metrics = evaluate_communities(control_communities, label)
    results["coburst_con_rating_control"] = {
        "build_timings": control_build_timings,
        "louvain_timings": control_louvain_timings,
        "metrics": control_metrics,
    }
    if verbose:
        print(f"build_timings: {control_build_timings}")
        print(f"louvain_timings: {control_louvain_timings}")
        print(f"LOO-AUC: {control_metrics['loo_auc']}")
        print(f"top-5%: {control_metrics['topk'][0.05]}")

    return results


# ---------------------------------------------------------------------------
# Resultados reales (`python features_graph.py coburst_yelpnyc`, esta
# máquina, CPU, Louvain con `seed=SEED=42`) — z_threshold=2.0 (mismo umbral
# por defecto que `burst_participation` en `profile_cluster.py`)
# ---------------------------------------------------------------------------
#
# 53.580 de 359.052 reviews (14,92%) caen en una ventana negocio+semana que
# es una ráfaga real (z>=2.0) — el resto (85,08%) queda en grupos
# "singleton" sin ninguna arista de co-bursting.
#
# | Relación | LOO-AUC | Precision top-5% | Lift top-5% | Comunidades sig. (enrichment) | Aristas (no dirig.) | Louvain (s) |
# |---|---|---|---|---|---|---|
# | `net_rtr` actual (negocio+rating+semana, SIN filtro de ráfaga — ya medido) | **0,5526** | 0,1838 | 1,79x | 16 (0,53% del fraude) | 289.711 | 37,9 |
# | **Co-bursting SIN rating** (negocio+semana, SOLO ventanas en ráfaga real) | 0,5229 | 0,1727 | 1,68x | 15 (0,59% del fraude) | 218.626 | 9,3 |
# | Control: co-bursting CON rating (negocio+rating+semana, SOLO ráfaga real) | 0,52 | 0,1697 | 1,65x | 16 (0,53% del fraude) | 76.538 | 8,3 |
#
# **Sensibilidad al umbral** (mismo grafo "sin rating", variando
# `z_threshold`, para no quedarse con un único punto): z>=1.5 (90.629
# reviews en ráfaga, 25,24% del dataset) → LOO-AUC 0,5317; z>=2.0 (como
# arriba) → 0,5229; z>=3.0 (15.805 reviews, 4,40% del dataset) → 0,5115.
# Patrón monótono: cuanto más estricto el umbral (menos aristas), peor el
# AUC — ninguno de los tres puntos se acerca a 0,5526.
#
# **Veredicto honesto: restringir las aristas a ventanas que son ráfagas
# REALES (no solo "mismo bin temporal") NO mejora la señal de coordinación
# frente al `net_rtr` actual — la empeora ligeramente, de forma consistente
# en ambas variantes probadas (con y sin rating) y en los tres umbrales de
# z-score probados.** No es un resultado límite/ambiguo: las tres cifras de
# co-bursting (0,5229 / 0,52 / y el rango 0,5115-0,5317 de la sensibilidad)
# caen por debajo de 0,5526 de forma consistente, nunca por encima.
#
# Diagnóstico de por qué, no solo el número: `net_rtr` actual tiene 289.711
# aristas — MUCHAS más que las 218.626 (sin rating) o 76.538 (con rating) de
# las variantes de co-bursting. Restringir a ventanas en ráfaga real quita
# aristas (correctamente, es justo lo que pide la tarea) pero el precio es
# perder cobertura: el 85% de reviews que quedan sin ninguna ráfaga fuerte
# detrás incluye, aparentemente, suficiente coordinación real de bajo perfil
# (ráfagas moderadas, z entre 0 y 2) como para que su ausencia pese más que
# el ruido que se elimina. Dicho de otro modo: el "ruido" que `net_rtr`
# conecta en ventanas normales no es solo ruido — una parte no trivial es
# señal real que el filtro de ráfaga descarta junto con el ruido genuino.
# Esto es coherente con el propio AUC del z-score de ráfaga en solitario, ya
# documentado en este fichero como débil (0,5315) — si la propia intensidad
# de la ráfaga apenas correlaciona con `is_fake`, exigirla como condición
# para las aristas no iba a mejorar mucho la relación resultante, y aquí se
# confirma que además la empeora un poco (menos aristas → menos señal
# capturada, sin compensación suficiente en precisión).
#
# **Conclusión de la Tarea 2**: la hipótesis de `INVESTIGACION_GRAFOS.md`
# (que restringir a ráfagas reales mejoraría sobre la coincidencia de bin
# temporal a secas) no se confirma con datos en Yelp-NYC — se documenta el
# resultado negativo tal cual, con el mismo criterio de honestidad que el
# resto del fichero (ver la combinación ponderada por rareza en Yelp-Chi,
# descartada con el mismo espíritu). El z-score genérico de ráfaga sigue
# siendo, con o sin este experimento, una señal débil en este dataset.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fusión con banda de abstención (grafo + texto T2, cuando T2 esté disponible)
# ---------------------------------------------------------------------------

def compute_abstention_thresholds(scores: np.ndarray, low_pct: float = 10.0, high_pct: float = 90.0) -> tuple[float, float]:
    """Umbrales de la banda de abstención, calibrados como percentiles del
    propio score observado (no umbrales fijos a ciegas) — el S2 del README
    promete justo esto ("banda de inconcluyente visible").
    """
    return float(np.percentile(scores, low_pct)), float(np.percentile(scores, high_pct))


def apply_abstention_band(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """`scores <= low` -> "genuina", `scores >= high` -> "sospechosa", resto
    -> "inconcluyente" (la banda central, "no lo sé" en vez de forzar)."""
    out = np.full(len(scores), "inconcluyente", dtype=object)
    out[scores <= low] = "genuina"
    out[scores >= high] = "sospechosa"
    return out


def evaluate_abstention_band(scores: np.ndarray, label: np.ndarray, low_pct: float = 10.0, high_pct: float = 90.0) -> dict:
    """Mide con datos reales si la banda separa de verdad: la tasa de fraude
    real dentro de "inconcluyente" debería quedar cerca de la tasa base
    (si la banda captura bien la incertidumbre), y alejarse de la tasa base
    en ambas direcciones fuera de la banda ("genuina" mucho más baja,
    "sospechosa" mucho más alta) — si no pasa eso, la banda no está
    cumpliendo su función y hay que decirlo, no maquillarlo.
    """
    low, high = compute_abstention_thresholds(scores, low_pct, high_pct)
    bands = apply_abstention_band(scores, low, high)
    out = {"low_threshold": round(low, 4), "high_threshold": round(high, 4), "base_rate": round(float(label.mean()), 4)}
    for name in ["genuina", "inconcluyente", "sospechosa"]:
        mask = bands == name
        out[name] = {
            "n": int(mask.sum()),
            "pct_del_total": round(float(mask.mean()), 4),
            "fraud_rate_real": round(float(label[mask].mean()), 4) if mask.sum() else None,
        }
    return out


def fuse_scores(component_scores: dict, weights: dict | None = None) -> np.ndarray:
    """Combinación lineal simple de scores ya normalizados a [0, 1]
    (min-max sobre la propia muestra) — pensada para admitir cualquier
    número de señales (aquí, cuando T2 esté disponible en esta máquina:
    `{"community": ..., "t2": ...}`). Sin `weights`, promedio simple.

    Deliberadamente simple (no una regresión logística entrenada): el
    proyecto ya se ha encontrado dos veces que combinar señales sin medir
    no es automáticamente mejor (fusión T1+T2+T3 en Fase 0, combinación
    ponderada por rareza en Yelp-Chi) — cualquier fusión aquí debe
    evaluarse con `evaluate_communities`/AUC antes de asumir que mejora
    sobre la señal individual más fuerte, no asumirlo por diseñarla.
    """
    names = list(component_scores.keys())
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    normalized = []
    for name in names:
        arr = np.asarray(component_scores[name], dtype=float)
        lo, hi = arr.min(), arr.max()
        norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        normalized.append(norm * weights[name])
    return np.sum(normalized, axis=0)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_yelpnyc_analysis(verbose: bool = True) -> dict:
    """Corre el harness de evaluación sobre las relaciones de Yelp-NYC +
    burst detection + banda de abstención sobre la mejor señal disponible.

    **T2 (texto) no se evalúa aquí** — vive en `train.py`
    (`eval_t2_on_yelpnyc`), fuera del ámbito de este módulo (que es
    puramente de grafo). Ver `CONTEXTO.md` para el estado real de esa pieza
    en esta sesión (el modelo entrenado no está presente en esta máquina).
    """
    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    label = df["is_fake"].to_numpy()

    graphs = build_yelpnyc_graphs(df, rtr_window="W")
    results = {"build_timings": graphs["timings"]}

    communities_by_relation = {}
    for name in ["net_rur", "net_rtr"]:
        if verbose:
            print(f"\n--- {name} (Louvain real vía networkx) ---")
        communities, timings = detect_communities_louvain(graphs[name])
        communities_by_relation[name] = communities
        metrics = evaluate_communities(communities, label)
        results[name] = {"timings": timings, "metrics": metrics}
        if verbose:
            print(f"timings: {timings}")
            print(f"LOO-AUC: {metrics['loo_auc']}")

    if verbose:
        print("\n--- net_rsr (partición groupby directa, sin networkx/Louvain — ver docstring) ---")
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_communities = groupby_cliques_as_communities(rsr_key)
    results["net_rsr"] = {
        "n_comunidades": len(rsr_communities),
        "metrics": evaluate_communities(rsr_communities, label),
    }
    if verbose:
        print(f"LOO-AUC: {results['net_rsr']['metrics']['loo_auc']}")

    if verbose:
        print("\n--- net_homo (connected_components, NO Louvain — inviable, ver docstring) ---")
    from scipy.sparse.csgraph import connected_components

    n_comp, cc_labels = connected_components(graphs["net_homo"], directed=False)
    homo_communities = [set(np.where(cc_labels == c)[0]) for c in range(n_comp)]
    results["net_homo"] = {
        "n_componentes": int(n_comp),
        "tam_mayor_componente": int(pd.Series(cc_labels).value_counts().iloc[0]),
        "metrics": evaluate_communities(homo_communities, label),
    }
    if verbose:
        print(f"n_componentes: {n_comp}, LOO-AUC: {results['net_homo']['metrics']['loo_auc']}")

    if verbose:
        print("\n--- Burst detection (Nivel A, real aquí — imposible en Yelp-Chi) ---")
    burst_scores = detect_bursts_yelpnyc(df, window="W")
    burst_auc = sanity_check_burst_auc(df, burst_scores)
    results["burst_detection"] = {"window": "W", "auc": burst_auc}
    if verbose:
        print(f"AUC del z-score de ráfaga (semana) frente a is_fake: {burst_auc}")

    if verbose:
        print("\n--- Banda de abstención sobre net_rur (mejor señal individual medida) ---")
    rur_scores = leave_one_out_cluster_scores(communities_by_relation["net_rur"], label)
    band = evaluate_abstention_band(rur_scores, label)
    results["abstention_band_net_rur"] = band
    if verbose:
        print(band)

    return results


# ---------------------------------------------------------------------------
# OddBall sobre Yelp-NYC (Tarea 1, segundo dataset) — mismas funciones
# genéricas de la sección Yelp-Chi (`oddball_egonet_stats`/
# `fit_oddball_power_law`/`oddball_anomaly_scores`/`evaluate_oddball`), sin
# ningún cambio: son agnósticas al dataset, solo reciben una matriz sparse y
# un array de label.
# ---------------------------------------------------------------------------

def run_yelpnyc_oddball_analysis(verbose: bool = True) -> dict:
    """OddBall sobre `net_rur`/`net_rtr`/`net_rsr` de Yelp-NYC — comparación
    directa contra el LOO-AUC de Louvain ya medido en `run_yelpnyc_analysis`
    (`net_rur` 0,9046 / `net_rtr` 0,5526 / `net_rsr` 0,6296).

    **`net_homo` se excluye A PROPÓSITO, no por omisión** — ver el docstring
    completo de `oddball_egonet_stats` para el detalle real medido: el
    intento directo de `A@A` sobre `net_homo` (149.220.438 aristas
    dirigidas, componente gigante con grado medio ~415,6) llegó a 10,9GB de
    RAM y seguía creciendo con fuerza tras ~2 minutos (partiendo de 4,7GB
    nada más cargar la matriz) — se mató el proceso a propósito antes de
    arriesgar dejar la máquina sin memoria, mismo criterio de seguridad ya
    aplicado con `networkx`/Louvain sobre esta misma relación en
    `run_yelpnyc_analysis`. A diferencia de `net_rsr` (donde SÍ compensó
    evitar `networkx`, ver más abajo), aquí el muro persiste con o sin
    `networkx` — el cuello de botella es combinatorio (grafo denso, muchos
    pares de vecinos comunes), no el overhead de representar el grafo como
    objetos Python.

    **`net_rsr` SÍ es viable aquí, a diferencia de con Louvain/`networkx`
    (donde era inviable — ver docstring de la sección Yelp-NYC más arriba,
    >16GB de RAM y sin terminar tras 9+ minutos)**: vía `A@A` directo sobre
    la matriz sparse tarda del orden de varios minutos (ver timings reales
    en el bloque de resultados justo debajo de esta función) — lento, pero
    termina, con un pico de RAM muy por debajo del límite de esta máquina.
    Es la confirmación real, con un dataset 8x más grande que Yelp-Chi, de
    que evitar `networkx` sí resuelve el muro de escala para relaciones que
    son uniones disjuntas de cliques.
    """
    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    label = df["is_fake"].to_numpy().astype(int)
    graphs = build_yelpnyc_graphs(df, rtr_window="W")

    results = {}
    for name in ["net_rur", "net_rtr", "net_rsr"]:
        if verbose:
            print(f"\n--- OddBall sobre {name} (Yelp-NYC) ---")
        res = evaluate_oddball(graphs[name], label, name=name)
        results[name] = res
        if verbose:
            print(f"timings: {res['timings']}")
            print(f"power_law_fit: {res['power_law_fit']}")
            print(f"AUC: {res['auc']}")
            print(f"top-5%: {res['topk'][0.05]}")

    results["net_homo"] = {
        "estado": "NO ejecutado a propósito -- muro de memoria confirmado (>10,9GB y creciendo en ~2 min, proceso matado antes de completar), ver docstring de oddball_egonet_stats y de esta función",
    }
    if verbose:
        print("\n--- net_homo: NO ejecutado a propósito, ver docstring (muro de memoria confirmado, no networkx esta vez) ---")

    return results


# ---------------------------------------------------------------------------
# Resultados reales (`python features_graph.py oddball_yelpnyc`, esta
# máquina, CPU)
# ---------------------------------------------------------------------------
#
# | Relación   | theta  | C      | AUC OddBall | LOO-AUC Louvain (ya medido) | Tiempo total OddBall |
# |---|---|---|---|---|---|
# | `net_rur`  | 2,198  | 0,2581 | **0,2581**  | 0,9046                       | 1,43s |
# | `net_rtr`  | 2,387  | 0,2021 | 0,5023      | 0,5526                       | 0,04s |
# | `net_rsr`  | 2,015  | 0,4550 | 0,4824      | 0,6296                       | **286,18s (~4,8 min)** |
# | `net_homo` | — | — | **no ejecutado (muro de memoria, ver arriba)** | 0,0016 | — |
#
# **Mismo veredicto que en Yelp-Chi, ahora confirmado en un segundo dataset
# 8x más grande: OddBall pierde frente a Louvain en las 3 relaciones
# evaluables, y en `net_rur` sale otra vez con el AUC INVERTIDO** (0,2581,
# el mismo patrón que en Yelp-Chi 0,2797 — coherente porque `net_rur` es
# unión disjunta de cliques en ambos datasets, mismo mecanismo diagnosticado
# arriba: las comunidades pequeñas de `net_rur`, que aquí SON las de mayor
# tasa de fraude —22,24% con 1 review, ver tabla ya documentada en la
# sección Yelp-NYC—, quedan cerca de la tendencia global dominada por los
# grupos pequeños, mientras que los reviewers muy prolíficos, que aquí son
# genuinos, se desvían más y reciben el score de anomalía más alto).
#
# **Hallazgo de escala, la parte de verdad nueva y aprovechable de esta
# tarea, confirmado ahora con un dataset real 8x mayor que Yelp-Chi**:
# `net_rsr` (146.038.672 aristas dirigidas) — INVIABLE de convertir a
# `networkx` en esta misma máquina (>16GB de RAM, sin terminar tras 9+
# minutos, ya documentado en la sección Yelp-NYC de este fichero) — vía
# `A@A` directo sobre la matriz sparse **termina en 286,18s (~4,8 min) sin
# ningún problema de memoria** (`nnz(A@A)`=146.397.291, prácticamente
# idéntico a `nnz(A)`, confirmando de nuevo que es una unión disjunta de
# cliques). Es la confirmación real de que evitar `networkx` SÍ resuelve
# el muro de escala que bloqueaba a Louvain en esta relación concreta — con
# el matiz honesto ya documentado en `oddball_egonet_stats`: el mismo truco
# NO funciona en `net_homo` (muro de memoria distinto, de origen
# combinatorio, no de `networkx`).
#
# **Conclusión de la Tarea 1 sobre los dos datasets pedidos**: OddBall, tal
# como lo define el paper original, no mejora ninguna de las señales de
# grafo ya medidas en este proyecto — el resultado es negativo de forma
# consistente en Yelp-Chi y Yelp-NYC, con el mismo mecanismo de fondo en
# ambos (relaciones construidas como cliques exactos, sin la varianza
# topológica que el método necesita para distinguir near-clique de
# near-star). El valor real que deja esta tarea no es una mejora de
# precisión, es la confirmación de escala: evitar `networkx` permite
# calcular egonet stats sobre grafos de decenas de millones de aristas que
# antes eran inviables, siempre que la relación sea una unión de cliques
# disjuntas — no cuando el grafo tiene una componente gigante densa
# (`net_homo`), donde el problema es combinatorio y no de representación.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fusión REAL de las señales de grafo (Yelp-NYC) — no solo señales sueltas
# ---------------------------------------------------------------------------
#
# Motivo de esta sección (pedido explícito del usuario, 2026-09-14, ver
# CONTEXTO.md "Decisión de priorización — sin acceso al ordenador de casa"):
# hasta `run_yelpnyc_analysis()` cada señal de grafo se mide POR SEPARADO
# contra `is_fake` (LOO-AUC de `net_rur`/`net_rtr`/`net_rsr` + AUC del burst
# z-score). La única que sale fuerte (`net_rur`, 0,9046 aquí / 0,9975 en
# bretthollenbeck) ya está documentada arriba como sospechosa de medir
# "prolificidad de cuenta" en vez de coordinación real — cambia de signo
# entre datasets. Las señales más específicas de coordinación (`net_rtr`,
# `net_rsr`) son mucho más flojas en solitario (0,55-0,63). En la literatura
# (SpEagle, ~0,78 AUC, ya citado como benchmark en `print_reference_
# comparison`) estos métodos funcionan fusionando varias señales flojas-
# moderadas en un único clasificador, no juzgando cada una por separado —
# ese es el experimento que faltaba antes de poder decir si el grafo "sirve"
# de verdad para este proyecto. La banda de abstención que ya existía
# (`evaluate_abstention_band`) solo se había aplicado a `net_rur` SOLA (ver
# arriba) — nunca a un score fusionado de verdad.
#
# Mismo patrón que `fuse_signals()` en `train.py` (que fusiona T1+T2+T3 de
# texto con una regresión logística simple, entrena en un split y evalúa en
# un held-out aparte, reportando coeficientes) — aquí se aplica exactamente
# esa misma idea a las señales de grafo en vez de a las de texto.

def compute_yelpnyc_graph_signal_scores(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Calcula las 4 señales de grafo (score por fila, alineado con `df`)
    que se fusionan en `fuse_graph_signals_yelpnyc`: `net_rur`/`net_rtr` vía
    Louvain real + `leave_one_out_cluster_scores`, `net_rsr` vía la
    partición `groupby` directa (`groupby_cliques_as_communities` — sin
    Louvain, ya justificado en el docstring de la sección Yelp-NYC de este
    fichero: es matemáticamente la misma partición y evita el muro de
    memoria de convertir ~73M aristas a un grafo de `networkx`), y el
    z-score de ráfaga semanal (`detect_bursts_yelpnyc`, que ya es un score
    por fila, no pasa por comunidades).

    Reutiliza literalmente el mismo cálculo que ya hace `run_yelpnyc_
    analysis()` para cada señal por separado — esta función solo lo expone
    en forma de 4 arrays alineados en vez de solo imprimir el LOO-AUC de
    cada uno, para poder pasarlos a un clasificador de fusión.

    Deliberadamente NO incluye `net_homo`: ya documentado en este mismo
    fichero como sin señal útil en absoluto (LOO-AUC 0,0016 — la componente
    gigante de `connected_components` se come el 99,9% del dataset y diluye
    cualquier señal, ver docstring de la sección Yelp-NYC). Meterlo en la
    fusión no aportaría nada y sí podría introducir ruido.
    """
    label = df["is_fake"].to_numpy()
    graphs = build_yelpnyc_graphs(df, rtr_window="W")

    if verbose:
        print("Calculando comunidades net_rur (Louvain real, puede tardar ~2 min)...")
    rur_communities, rur_timings = detect_communities_louvain(graphs["net_rur"])
    rur_scores = leave_one_out_cluster_scores(rur_communities, label)

    if verbose:
        print("Calculando comunidades net_rtr (Louvain real)...")
    rtr_communities, rtr_timings = detect_communities_louvain(graphs["net_rtr"])
    rtr_scores = leave_one_out_cluster_scores(rtr_communities, label)

    if verbose:
        print("Calculando comunidades net_rsr (partición groupby directa, sin Louvain)...")
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_communities = groupby_cliques_as_communities(rsr_key)
    rsr_scores = leave_one_out_cluster_scores(rsr_communities, label)

    if verbose:
        print("Calculando burst detection (z-score semanal por negocio)...")
    burst_scores = detect_bursts_yelpnyc(df, window="W")

    return {
        "net_rur": rur_scores,
        "net_rtr": rtr_scores,
        "net_rsr": rsr_scores,
        "burst": burst_scores,
        "label": label,
        "timings": {"net_rur": rur_timings, "net_rtr": rtr_timings},
    }


def _fuse_graph_signal_scores(signals: dict, metrics_key: str, label_dataset: str, verbose: bool = True) -> dict:
    """Núcleo compartido de la fusión: split 70/30 estratificado
    (`random_state=SEED` fijo), regresión logística sobre las 4 señales
    estandarizadas, AUC de cada señal individual SOLA en el mismo held-out,
    coeficientes aprendidos, banda de abstención sobre el score fusionado, y
    el control honesto "fusión sin `net_rur`" (ver más abajo por qué).
    Reutilizado tal cual por `fuse_graph_signals_yelpnyc` y
    `fuse_graph_signals_bretthollenbeck` — las diferencias reales entre
    datasets (cómo se calculan las señales de entrada, si `net_rur` cambia
    de signo, si el label es parcial) ya están resueltas antes de llegar
    aquí, en `compute_yelpnyc_graph_signal_scores`/
    `compute_bretthollenbeck_graph_signal_scores`.

    Se estandarizan las 4 features (`StandardScaler`, ajustado SOLO en
    train) antes de la regresión — a diferencia de `fuse_signals()` en
    `train.py` (que no estandariza T1/T3/T2 porque ya son scores en escalas
    parecidas), aquí las 4 señales tienen escalas muy distintas por
    construcción (`net_rur`/`net_rtr`/`net_rsr` son tasas de fraude en
    [0, 1], `burst` es un z-score sin cota, típicamente en [-3, +10]) — sin
    estandarizar, los coeficientes no serían comparables entre sí y "qué
    señal pesa más" no se podría leer directamente del coeficiente, que es
    justo lo que pide esta tarea.

    **Segundo experimento incluido a propósito, no solo la fusión completa**:
    `net_rur` ya está documentada en este fichero como sospechosa de medir
    "prolificidad de cuenta" (correlaciona con genuinidad en Yelp-NYC, con
    fraude reincidente en bretthollenbeck — cambia de SIGNO entre datasets,
    ver docstrings de cada sección) más que "red de cuentas coordinadas" en
    el sentido que persigue el README. Reportar solo el AUC de la fusión
    completa sin este control sería el mismo error de lectura ya evitado en
    Yelp-Chi (`print_reference_comparison`): un número nominal alto que en
    realidad mide algo más estrecho de lo que sugiere. Se fusionan también,
    por separado, SOLO las tres señales más específicas de coordinación
    entre cuentas DISTINTAS (`net_rtr`, `net_rsr`, burst), sin `net_rur`,
    para poder decir cuánto vale de verdad "coordinación" sin la ayuda de
    "prolificidad/reincidencia de la misma cuenta".

    Guarda el resultado en `outputs/metrics.json` bajo `metrics_key`, sin
    tocar ninguna clave ya existente (ver `_save_graph_metrics`).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    feature_names = ["net_rur", "net_rtr", "net_rsr", "burst"]
    label = signals["label"]
    X = np.column_stack([signals[name] for name in feature_names])
    y = np.asarray(label).astype(int)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fusion = LogisticRegression(max_iter=1000, random_state=SEED)
    fusion.fit(X_train_scaled, y_train)
    fused_scores_test = fusion.predict_proba(X_test_scaled)[:, 1]

    fusion_auc = roc_auc_score(y_test, fused_scores_test)
    individual_auc_held_out = {
        name: round(float(roc_auc_score(y_test, X_test[:, i])), 4)
        for i, name in enumerate(feature_names)
    }
    coefficients_standardized = dict(zip(feature_names, fusion.coef_[0].tolist()))
    band = evaluate_abstention_band(fused_scores_test, y_test)

    no_rur_names = ["net_rtr", "net_rsr", "burst"]
    no_rur_idx = [feature_names.index(n) for n in no_rur_names]
    fusion_no_rur = LogisticRegression(max_iter=1000, random_state=SEED)
    fusion_no_rur.fit(X_train_scaled[:, no_rur_idx], y_train)
    fused_scores_test_no_rur = fusion_no_rur.predict_proba(X_test_scaled[:, no_rur_idx])[:, 1]
    fusion_auc_no_rur = roc_auc_score(y_test, fused_scores_test_no_rur)
    coefficients_no_rur = dict(zip(no_rur_names, fusion_no_rur.coef_[0].tolist()))

    results = {
        "dataset": label_dataset,
        "n_train": int(len(idx_train)),
        "n_held_out": int(len(idx_test)),
        "base_rate_held_out": round(float(y_test.mean()), 4),
        "fusion_auc_held_out": round(float(fusion_auc), 4),
        "individual_auc_held_out": individual_auc_held_out,
        "best_individual_auc_held_out": max(individual_auc_held_out.values()),
        "fusion_coefficients_standardized": coefficients_standardized,
        "abstention_band_fused_held_out": band,
        "speeagle_benchmark_auc_referencia": 0.78,
        "fusion_sin_net_rur": {
            "nota": (
                "net_rur excluida a propósito -- señal confundida con "
                "prolificidad/reincidencia de cuenta, no coordinación entre "
                "cuentas distintas (ver docstring). Este número mide mejor "
                "'coordinación real' que el AUC de la fusión completa de arriba."
            ),
            "fusion_auc_held_out": round(float(fusion_auc_no_rur), 4),
            "fusion_coefficients_standardized": coefficients_no_rur,
        },
    }

    if verbose:
        print(f"\n--- Fusión de señales de grafo ({label_dataset}), held-out 30% ---")
        print(f"n_train={results['n_train']}, n_held_out={results['n_held_out']}, "
              f"base_rate_held_out={results['base_rate_held_out']}")
        print(f"AUC fusión (held-out): {results['fusion_auc_held_out']}")
        print(f"AUC individuales (held-out, misma partición): {individual_auc_held_out}")
        print(f"Mejor señal individual sola (held-out): {results['best_individual_auc_held_out']}")
        print(f"Coeficientes (features estandarizadas): {coefficients_standardized}")
        print(f"Banda de abstención sobre score fusionado: {band}")
        print(f"Benchmark publicado SpEagle: ~0.78 AUC")
        print(f"\n--- Control honesto: fusión SIN net_rur (solo net_rtr+net_rsr+burst) ---")
        print(f"AUC fusión sin net_rur (held-out): {results['fusion_sin_net_rur']['fusion_auc_held_out']}")
        print(f"Coeficientes sin net_rur: {coefficients_no_rur}")

    _save_graph_metrics({metrics_key: results})
    return results


def fuse_graph_signals_yelpnyc(verbose: bool = True) -> dict:
    """Fusiona `net_rur` + `net_rtr` + `net_rsr` + burst (Yelp-NYC) en una
    única regresión logística, entrenada en un split y evaluada en un
    held-out aparte — la pregunta real de esta sección: ¿el grafo "sirve" si
    se fusionan sus señales, en vez de juzgar cada una sola como hasta
    ahora? Ver `_fuse_graph_signal_scores` para el mecanismo completo
    (compartido con la validación en bretthollenbeck) y los resultados
    reales medidos, documentados en el bloque de comentarios justo debajo de
    esta función.

    Guarda el resultado en `outputs/metrics.json` bajo la clave
    `graph_fusion_yelpnyc`.
    """
    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    signals = compute_yelpnyc_graph_signal_scores(df, verbose=verbose)
    return _fuse_graph_signal_scores(signals, "graph_fusion_yelpnyc", "Yelp-NYC", verbose=verbose)


# ---------------------------------------------------------------------------
# Resultados reales (`python features_graph.py graph_fusion_yelpnyc`, esta
# máquina, split 70/30 estratificado, random_state=SEED=42)
# ---------------------------------------------------------------------------
#
# | Señal (held-out, n=107.716, base rate 10,27%) | AUC sola |
# |---|---|
# | `net_rur` | 0,9045 |
# | `net_rsr` | 0,6282 |
# | `net_rtr` | 0,5540 |
# | `burst` | 0,5325 |
# | **Fusión (las 4)** | **0,9232** |
# | **Fusión SIN `net_rur`** (`net_rtr`+`net_rsr`+`burst`) | **0,6310** |
#
# Coeficientes (features estandarizadas, comparables en magnitud entre sí):
# fusión completa `net_rur=3,606 / net_rsr=0,323 / net_rtr=0,137 /
# burst=0,054`; fusión sin `net_rur` `net_rsr=0,360 / net_rtr=0,131 /
# burst=0,090`.
#
# **Hallazgo honesto, en dos lecturas que NO se pueden mezclar en una sola
# cifra de marketing:**
#
# 1. **La fusión de las 4 señales (0,9232) SÍ mejora sobre la mejor señal
#    suelta (`net_rur`, 0,9045)** — una mejora real, aunque modesta (+0,0187
#    de AUC), y el mecanismo de fusión en sí (regresión logística simple,
#    mismo patrón que `fuse_signals()` en `train.py`) funciona: combinar
#    señales flojas-moderadas con la más fuerte sí añade algo, no resta.
#    0,9232 supera con holgura el benchmark publicado de SpEagle (~0,78).
# 2. **Pero ese 0,9232 está dominado casi por completo por `net_rur`**
#    (coeficiente 3,6, 11x el de la siguiente señal más fuerte) — la señal
#    ya documentada en este mismo fichero como sospechosa de medir
#    "prolificidad de cuenta" (correlaciona con GENUINIDAD en Yelp-NYC, ver
#    tabla de arriba) más que "red de cuentas coordinadas" en el sentido
#    que persigue el README. Decir "el grafo fusionado da 0,92, muy por
#    encima de SpEagle" sin esta salvedad sería el mismo error de lectura ya
#    identificado y evitado en Yelp-Chi (`print_reference_comparison`): un
#    número nominal alto que en realidad mide algo más estrecho/distinto de
#    lo que sugiere.
#
# **La cifra que responde de verdad a "¿la coordinación entre cuentas
# DISTINTAS, sin apoyarse en la prolificidad de una misma cuenta, aporta
# algo si se fusiona?" es la fusión sin `net_rur`: 0,631** — apenas por
# encima de la mejor de esas tres señales sola (`net_rsr`, 0,6282, +0,003 de
# AUC) y **muy por debajo del benchmark de SpEagle (~0,78)**. Conclusión
# honesta, sin maquillar: fusionar `net_rtr`+`net_rsr`+`burst` (las señales
# que sí capturan coordinación entre cuentas distintas, no solo reutilización
# de la misma cuenta) no basta por sí solo para acercarse a SpEagle en este
# dataset — la fuerza real de "el grafo fusionado le gana a SpEagle" descansa
# casi enteramente en `net_rur`, y esa señal mide algo más estrecho de lo que
# el README entiende por "coordinación".
#
# **La banda de abstención sobre el score fusionado SÍ funciona mejor que el
# intento anterior con `net_rur` sola** (ver `abstention_band_net_rur` en
# `run_yelpnyc_analysis`, que fallaba porque el 29,5% de nodos con una sola
# review empataban justo en la tasa base y los percentiles 10/90 caían sobre
# ese empate masivo, dejando la banda "inconcluyente" casi vacía —0,34%—).
# Aquí, con el score fusionado y calibrado en el held-out: **"genuina" 10%
# del dataset con 0,05% de fraude real** (prácticamente limpio), **"sospechosa"
# 10% con 55,1% de fraude real** (5,4x la tasa base), y **"inconcluyente" el
# 80% restante con 5,95% de fraude** (por debajo de la tasa base global de
# 10,27%, no muy cerca de ella como sería ideal, pero sin el fallo de banda
# casi vacía del intento anterior). Mejora real y medible, no solo teórica.
#
# **Conclusión general para "¿el grafo sirve?", sin adornar**: sirve, pero
# con una salvedad importante que no se puede omitir — la mayor parte de esa
# utilidad (en el sentido de superar el benchmark publicado) viene de una
# señal que mide sobre todo "cuentas de uso único/reutilización de la misma
# cuenta", no de la coordinación entre cuentas distintas que es la promesa
# central del producto (README, "red de cuentas coordinadas"). Es una señal
# real y aprovechable en producto (detecta cuentas desechables), pero
# tratarla como si fuera "hemos resuelto la detección de granjas de reviews"
# sería una lectura que los propios números no sostienen.
#
# ---------------------------------------------------------------------------
# Segundo punto de validación (opcional, `python features_graph.py
# graph_fusion_bretthollenbeck`) — misma metodología sobre bretthollenbeck/
# Amazon (24.085 filas held-out, subset etiquetado, base rate 27,58%)
# ---------------------------------------------------------------------------
#
# | Señal | AUC sola | Fusión completa | Fusión sin `net_rur` |
# |---|---|---|---|
# | `net_rur` | 0,9978 | — | — |
# | `net_rsr` | 0,7355 | — | — |
# | `net_rtr` | 0,6757 | — | — |
# | `burst` | 0,5387 | — | — |
# | **Todas (4)** | — | **0,9992** | — |
# | `net_rtr`+`net_rsr`+`burst` | — | — | **0,7611** |
#
# Coeficientes fusión completa: `net_rur=7,96 / net_rsr=0,69 / net_rtr=0,68 /
# burst=0,17` — `net_rur` domina aún MÁS que en Yelp-NYC (11,5x la siguiente
# señal, frente a 11x en Yelp-NYC), coherente con el patrón ya documentado
# (aquí `net_rur` mide reincidencia de cuentas profesionales de fraude, con
# un patrón incluso más extremo — 100% de fraude a partir de 32 reviews del
# mismo `reviewer_id`).
#
# **Mismo patrón cualitativo que Yelp-NYC (confirma la metodología, no es un
# hallazgo aislado de un solo dataset) pero con un matiz cuantitativo real
# que hay que decir tal cual**: aquí la fusión SIN `net_rur` (0,7611) queda
# mucho más CERCA del benchmark de SpEagle (~0,78) que en Yelp-NYC (0,631) —
# a solo 0,02 de diferencia, frente a 0,15 en Yelp-NYC. Lectura honesta, ni
# triunfalista ni descartando el resultado: no es evidencia de que la
# metodología de fusión sea "mejor" en Amazon, es más probable que sea del
# propio dataset (igual que ya se documentó para las señales sueltas en la
# sección bretthollenbeck de este fichero) — al estar centrado a propósito
# en productos con campaña de fraude YA conocida, la coordinación entre
# cuentas distintas (`net_rtr`/`net_rsr`) tiene menos ruido de fraude
# genuino diluyéndola que en una muestra más general de negocios como
# Yelp-NYC. **No se puede generalizar de este único punto de datos que
# "la coordinación real sin net_rur se acerca a SpEagle"** — son dos
# datasets con LOO-AUCs individuales ya muy distintos por motivos de
# dominio ya conocidos, y esto es la misma limitación heredada, no una
# nueva.
"""
===============================================================================
bretthollenbeck/fake-reviews-data (Amazon, Fase 1) — cuarto dataset de
grafo: el primero con fecha real de evento (`campaign_start_date`) para
burst detection, y el primero con label solo PARCIAL (21% de las filas)
===============================================================================

`data.load_bretthollenbeck_dataset()` da un DataFrame plano (`reviewer_id`,
`business_id` [de `asin`], `rating`, `date`, `text`, `is_fake` + columnas
extra), sin grafo pre-proyectado — igual que Yelp-NYC, no como Yelp-Chi/
Amazon. Se reutiliza tal cual el mismo patrón de construcción de esa sección
(`_clique_matrix_from_groups`/`groupby_cliques_as_communities`, el mismo
truco de incidencia dispersa `M @ M.T`, sin bucles por pares) — el trabajo
real de este bloque no es reinventar la construcción, es la evaluación con
label parcial y el burst detection con fecha real, que sí son nuevos.

## Decisión de diseño real, sin precedente exacto en este proyecto: label
   PARCIAL (21% de las filas), construcción de grafo con el 100%

A diferencia de Yelp-Chi/Amazon (Dou et al., label en el 100% de nodos) y de
Yelp-NYC (proxy de spam de Rayana & Akoglu en el 100% de filas), aquí el
label real (`is_fake`, manual/curado, ver `data.py`) solo existe en 80.281
de 381.734 filas (21%) — `pandas.NA` en el resto, nunca tratado como
`False`. Decisión tomada y aplicada en todo este bloque:

- **El grafo se construye con las 381.734 filas, no solo con las
  etiquetadas.** Una review sin etiqueta manual sigue siendo comportamiento
  real (mismo reviewer, mismo negocio+rating+ventana) — no hay ningún
  motivo para excluirla de la ESTRUCTURA del grafo, solo de la evaluación.
  Excluirla rompería aristas reales (dos reviews etiquetadas que comparten
  negocio+rating con una tercera sin etiquetar dejarían de estar conectadas
  a través de la misma comunidad) y encogería artificialmente el tamaño de
  cada comunidad.
- **La evaluación (LOO-AUC, top-k, enrichment) se calcula SOLO sobre el
  subconjunto con `is_fake` no nulo.** Se implementa con una función nueva,
  `project_communities_to_labeled_subset`, que reindexa las comunidades ya
  calculadas sobre el grafo COMPLETO al subconjunto etiquetado (cada
  comunidad se queda solo con sus miembros etiquetados, remapeados a un
  índice `0..n_etiquetadas-1`) — así se reutilizan `evaluate_communities`/
  `leave_one_out_cluster_scores`/`topk_capture` tal cual, sin tocarlas ni
  duplicar lógica de evaluación. Comunidades sin ningún miembro etiquetado
  se excluyen de la evaluación (no aportan ninguna fila al array reducido),
  pero sí aportaron estructura real al grafo y a Louvain.

## `net_rsr` SÍ fue viable con Louvain vía `networkx` esta vez — primera
   verificación DIRECTA (no extrapolada) del teorema ya usado en Yelp-NYC

En Yelp-NYC, `net_rsr` (mismo negocio+rating, sin ventana) tenía ~73M
aristas no dirigidas y era inviable de convertir a `networkx` (>16GB RAM,
no terminó tras 9+ minutos) — la partición óptima se tomó del `groupby`
directo por argumento teórico (unión disjunta de cliques) verificado
EMPÍRICAMENTE, pero solo de forma indirecta: corriendo Louvain de verdad
sobre `net_rur`/`net_rtr` (mucho más pequeñas) y comprobando que coincidía
con sus propios grupos de origen, extrapolando esa confianza a `net_rsr`
sin poder correrlo ahí directamente.

Aquí `net_rsr` es mucho más pequeño (este dataset concentra 381.734 reviews
en solo 3.389 productos, pero con muchos menos negocio+rating únicos que
Yelp-NYC): **14.839 grupos, 47.310.962 aristas dirigidas (~23,66M no
dirigidas)** — 3,1x menos que el muro de Yelp-NYC. Medido con una sonda
dedicada antes de decidir nada (mismo criterio que ya exige este fichero:
"no fuerces algo caro, compruébalo primero"): construir el grafo de
`networkx` tardó **81,56s** (RSS pico 11,5GB) y Louvain **229,38s**
(~3,8 min) más, RSS pico 9,7GB — **viable en esta máquina (31,5GB de RAM),
a diferencia de la máquina donde se probó Yelp-NYC**. El resultado confirma
el teorema de forma aún más sólida que en Yelp-NYC, porque aquí SÍ se pudo
verificar directamente sobre la propia relación en vez de por extrapolación:
Louvain encontró **exactamente 14.839 comunidades**, el mismo número que
`groupby(["business_id", "rating"])`. El pipeline de este bloque usa de
todas formas `groupby_cliques_as_communities` para `net_rsr` (1,1s frente a
~5,2 minutos, mismo resultado exacto ya verificado) — por eficiencia, no
porque Louvain fuera inviable esta vez.

## `net_rur`: la señal más fuerte medida hasta ahora en todo el proyecto, y
   en la dirección OPUESTA a Yelp-NYC — diagnóstico real, no solo el número

| Nº de reviews del reviewer | Reviews etiquetadas | Tasa de fraude (etiquetada) |
|---|---|---|
| 1 | 12.140 | 4,56% |
| 2-3 | 46.536 | 14,93% |
| 4-7 | 14.102 | 55,79% |
| 8-15 | 4.754 | 85,65% |
| 16-31 | 1.295 | 95,98% |
| 32-63 | 659 | 100% |
| 64-199 | 795 | 100% |

**Patrón monótono CRECIENTE — justo lo opuesto de Yelp-NYC** (donde 1
review tenía la tasa de fraude más alta, 22,24%, y decaía con más reviews
hasta 0,66% en 64-199) **y distinto también del salto en dos escalones de
Yelp-Chi**. Aquí, cuantas más reviews ha escrito el mismo `reviewer_id` en
todo el dataset, más probable —con certeza casi total a partir de 32— que
sea fraude. Lectura de dominio, no solo estadística: este dataset está
centrado en productos con campaña de fraude YA conocida (no es una muestra
aleatoria de Amazon, ver `data.py`), así que un reviewer que aparece muchas
veces dentro de él es, con mucha probabilidad, una cuenta profesional
reincidente contratada para varias campañas — todo lo contrario de un
"Yelp Elite" genuino. **Implicación honesta para `profile_cluster.py`
(fuera de alcance de esta tarea, ver cierre)**: la lectura de `net_rur`
como indicio de Nivel A depende del dataset/dominio, no es universal — aquí
apunta directo a "cuenta profesional de fraude", en Yelp-NYC apuntaba a
"prolificidad genuina". No se puede fijar una regla única tipo "más
reviews = más sospechoso" sin decir de qué plataforma se está hablando.

Con ese patrón tan limpio, el LOO-AUC sale extremo: **0,9975**, con
precisión del 100% en el top-5/10/20% (algo que no se había visto en
ningún grafo de este proyecto hasta ahora) — ver tabla completa más abajo.

## Resultados reales del harness (381.734 nodos totales, 80.281 con label
   real, esta máquina)

| Relación | Aristas (no dirig.) | Construcción | Louvain/partición | Nº comunidades |
|---|---|---|---|---|
| `net_rur` (mismo reviewer) | 152.344 | 1,77s | 24,1s (Louvain) | 334.342 |
| `net_rtr` (negocio+rating+semana) | 464.030 | 1,43s | 14,29s (Louvain) | 243.472 |
| `net_rsr` (negocio+rating, sin ventana) | 23.655.481 | 81,56s | 229,38s (Louvain, viable aquí) | 14.839 |
| `net_rsr` (partición groupby, usada en el pipeline) | — | 1,1s | — (idéntica a Louvain, verificado) | 14.839 |
| `net_homo` (unión de las tres) | ~23,8M | 4,49s | 1,08s (`connected_components`) | 5.991 componentes |

| Grafo | LOO-AUC (solo subset etiquetado) | Precision top-5% | Lift top-5% | Comunidades sig. (enrichment) |
|---|---|---|---|---|
| `net_rur` | **0,9975** | **1,0** | **3,63x** | 172 (17,54% del fraude, en 4,84% de los nodos) |
| `net_rtr` (semana) | 0,6777 | 0,5416 | 1,96x | 13 (1,45% del fraude) |
| `net_rsr` (groupby) | 0,7336 | 0,5304 | 1,92x | 51 (14,67% del fraude) |
| `net_homo` (`connected_components`) | 0,0133 | 0,001 | 0,0x | 1 (ver aviso abajo) |

**`net_homo` repite el mismo muro que en Yelp-NYC, con un matiz distinto que
también hay que decir tal cual**: la componente gigante se come el 87,4%
del dataset (333.669 de 381.734 nodos) — algo menos extremo que el 99,9% de
Yelp-NYC, pero con el mismo efecto práctico: LOO-AUC 0,0133 (peor que el
azar en la dirección útil, la componente gigante diluye la señal en vez de
concentrarla). El test de enriquecimiento reporta, de forma engañosa si se
lee sin este aviso, "1 comunidad significativa, 99,95% del fraude
capturado, en el 99,04% de los nodos" — no es una señal útil, es
literalmente casi todo el dataset colapsado en un único grupo por la unión
sin ponderar; se documenta el número tal cual sale, con esta advertencia al
lado, no se omite. **No se prueba la variante ponderada por rareza aquí**
por el mismo motivo que en Yelp-NYC: ya no compensó en Yelp-Chi y aquí
tiene el mismo volumen de aristas que `net_rsr`, sin ninguna razón para
esperar que compense esta vez.

## Comparación honesta con Yelp-NYC — ¿aporta algo distinto, o confirma el
   mismo patrón?

| Señal | Yelp-NYC (LOO-AUC) | bretthollenbeck/Amazon (LOO-AUC) |
|---|---|---|
| `net_rur` | 0,9046 | **0,9975** |
| `net_rtr` (semana) | 0,5526 | **0,6777** |
| `net_rsr` | 0,6296 | 0,7336 |
| `net_homo` | 0,0016 | 0,0133 (mismo patrón, magnitud algo distinta) |

Las cuatro señales salen en la misma DIRECCIÓN que en Yelp-NYC (`net_rur`
gana con claridad, `net_homo` es inútil, `net_rtr`/`net_rsr` quedan en un
punto intermedio) pero **más altas en las tres relaciones útiles**, no solo
`net_rur`. Lectura honesta, no triunfalista: esto no es necesariamente
"nuestra metodología es mejor en Amazon que en Yelp" — es más probable que
sea del propio dataset, que está construido *a propósito* alrededor de
productos con campaña de fraude YA CONOCIDA (3.389 productos, no una
muestra aleatoria de Amazon, ver `data.py`), así que la señal de
coordinación está más concentrada y menos diluida por ruido genuino que en
Yelp-NYC (923 negocios normales, con o sin fraude). **La respuesta honesta
a "¿aporta algo que Yelp-NYC no daba?" es dominio distinto (Amazon frente a
restaurantes) y confirmación del mismo patrón estructural — no una mejora
de metodología que vaya a repetirse igual en cualquier plataforma nueva.**
Sí aporta algo que Yelp-NYC no podía dar en absoluto: la fecha real de
campaña para burst detection (ver siguiente sección) — esa es la
contribución distintiva de este dataset, no el LOO-AUC más alto de
`net_rur`.

## Burst detection con fecha real de campaña — primera vez en el proyecto
   con un evento conocido, no un umbral genérico

Yelp-Chi/Amazon (Dou et al.) no tienen fecha en absoluto; Yelp-NYC solo
permite un z-score genérico de ráfaga (`detect_bursts_yelpnyc`) sin nada
externo con que contrastarlo, y ese z-score salió con AUC casi de azar
(0,5315) frente a la etiqueta proxy — no hay forma de saber, con Yelp-NYC,
si el burst detectado ocurre de verdad "cuando algo raro pasó" o si es
ruido. Aquí, por primera vez, hay una fecha real de investigación externa
(`campaign_start_date`, no un proxy) contra la que contrastar.

**1.449 productos tienen `campaign_start_date` conocida** (de 3.389 totales;
143.427 filas de esos productos — comprobado sin ambigüedad: cada producto
tiene como máximo una fecha de campaña, nunca varias distintas).

**Primera pregunta — ¿el volumen SÍ se dispara alrededor de la fecha real?
Sí, con matices reales, no un sí rotundo.** Tasa de fraude por bin de días
desde el inicio de campaña (subconjunto etiquetado, esos mismos 1.449
productos):

| Días desde `campaign_start_date` | Tasa de fraude | Reviews etiquetadas |
|---|---|---|
| < -365 | 28,9% | 218 |
| -365 a -90 | 21,3% | 5.828 |
| -90 a -30 | 24,1% | 6.254 |
| -30 a -7 | 38,7% | 4.191 |
| -7 a 0 | 59,3% | 2.149 |
| 0 a 7 | **63,6%** | 3.139 |
| 7 a 30 | 51,2% | 10.890 |
| 30 a 90 | 32,5% | 15.735 |
| 90 a 365 | 26,2% | 10.314 |

Pico claro justo en la semana de inicio de campaña (63,6%, frente a una
tasa base de ~21-28% lejos de la fecha) que decae de forma gradual en
ambas direcciones — el patrón temporal es real, no artefacto de un solo
bin.

**Segunda pregunta — ¿el burst detectado por z-score genérico (sin usar la
fecha, el mismo mecanismo de `detect_bursts_yelpnyc`, reutilizado tal cual
sobre este subconjunto porque esa función ya es completamente genérica pese
a su nombre) coincide temporalmente con la fecha real?** Medido por
producto: distancia entre la semana de mayor z-score de TODA la vida del
producto y su propia `campaña_start_date`. **El pico del burst genérico cae
dentro de ±30 días de la fecha real en el 44,9% de los productos, y dentro
de ±90 días en el 73,4%** (mediana de distancia absoluta: 38 días) — el
burst SÍ tiende a ocurrir "alrededor" de la fecha conocida en la mayoría de
los casos, con dispersión real, no una coincidencia exacta.

**Tercera pregunta, la que de verdad pedía la tarea — ¿ese burst
correlaciona con `is_fake` a nivel de review? Aquí el hallazgo es más
matizado, y hay que decirlo tal cual**: el z-score genérico de ráfaga
(`burst_z`, magnitud de la anomalía de esa semana) da **AUC 0,5520
(semana) / 0,5422 (día)** frente a `is_fake` — señal débil, del mismo orden
que el 0,5315 ya visto en Yelp-NYC (o el 0,510 del proxy descartado en
Yelp-Chi). **En cambio, usar directamente la proximidad temporal a la fecha
de campaña conocida (`campaign_proximity_score`, `-abs(días desde
campaign_start_date)`) da AUC 0,6504** — sensiblemente más fuerte que el
burst genérico, y top-5%/10%/20% con precisión 0,60-0,62 y lift ~1,7x.
**Conclusión honesta**: saber CUÁNDO ocurrió el evento real (la fecha de
campaña) y medir distancia a esa fecha conocida es una señal notablemente
mejor que intentar DETECTAR el burst por volumen anómalo sin esa fecha —
el burst por volumen se diluye porque en la misma ventana temporal también
hay compras/reviews genuinas (lanzamientos de producto, promociones
legítimas, estacionalidad), mientras que la distancia a un evento conocido
no depende de que el volumen se dispare de forma estadísticamente
detectable. Es la primera vez que el proyecto puede separar estas dos
preguntas ("¿hubo un burst?" vs. "¿coincide con el evento real?") en lugar
de asumir que son lo mismo — y la respuesta es que no lo son del todo: hay
más información en la fecha conocida que en el burst por sí solo.

## Qué queda fuera de esta tarea, a propósito

No se ha tocado `data.py`, `profile_cluster.py` ni `train.py` — fuera de
alcance explícito de esta sesión. Ideas para cuando se retome (dejadas
anotadas, no implementadas):

- `profile_cluster.py` podría incorporar `campaign_proximity_score` como
  una señal de Nivel A nueva ("cercanía a un evento de campaña conocido"),
  pero solo tendría sentido para clientes que efectivamente sepan la fecha
  de una campaña sospechada — no es un campo que un cliente típico vaya a
  tener de entrada (a diferencia de `created_at`, que sí es casi universal,
  ver la sesión de onboarding en `CONTEXTO.md`).
- El contraste `net_rur` opuesto entre Yelp-NYC y este dataset (prolificidad
  genuina vs. prolificidad de fraude) es un argumento más para no fundir
  nunca esa señal en un número único sin contexto de plataforma/dominio —
  ya se avisaba de esto en la sección Yelp-NYC, aquí se confirma que la
  dirección puede invertirse por completo según el dataset.
"""

import numpy as np
import pandas as pd


def project_communities_to_labeled_subset(communities: list, labeled_mask: np.ndarray) -> list:
    """Reindexa una lista de comunidades (calculadas sobre el grafo
    COMPLETO, con todas las filas, etiquetadas o no) al subconjunto de
    nodos con `is_fake` conocida — para poder reutilizar
    `evaluate_communities`/`leave_one_out_cluster_scores`/`topk_capture`
    SIN modificarlas, evaluando solo donde hay verdad de terreno real.

    Cada comunidad se queda solo con los nodos que sí tienen label
    (`labeled_mask[i] == True`), remapeados a su índice dentro del
    subconjunto (`0..n_etiquetadas-1`, mismo orden que `label[labeled_mask]`
    en quien llame a esta función). Los nodos sin label siguen habiendo
    aportado estructura real al grafo y a Louvain (más contexto, aristas
    reales), pero no aparecen en ninguna métrica de evaluación porque no
    hay ninguna etiqueta con la que compararlos — no se rellenan con
    `False` ni se ignoran silenciosamente, simplemente no cuentan.

    Comunidades sin NINGÚN nodo etiquetado se excluyen por completo del
    resultado (no aportarían ninguna fila al array de label reducido).
    """
    old_to_new = -np.ones(len(labeled_mask), dtype=int)
    old_to_new[np.where(labeled_mask)[0]] = np.arange(labeled_mask.sum())
    projected = []
    for community in communities:
        idx = np.fromiter(community, dtype=int)
        idx_labeled = idx[labeled_mask[idx]]
        if idx_labeled.size == 0:
            continue
        new_idx = old_to_new[idx_labeled]
        projected.append(set(new_idx.tolist()))
    return projected


def build_bretthollenbeck_graphs(df: pd.DataFrame, rtr_window: str = "W") -> dict:
    """Construye `net_rur`/`net_rtr`/`net_rsr`/`net_homo` para el dataset de
    Amazon de bretthollenbeck a partir del DataFrame plano de
    `data.load_bretthollenbeck_dataset()` — mismo patrón exacto que
    `build_yelpnyc_graphs` (reutiliza `_clique_matrix_from_groups`, el
    truco de incidencia dispersa `M @ M.T`), sobre las **381.734 filas
    completas**, no solo las etiquetadas (ver docstring de la sección de
    arriba sobre por qué).

    - `net_rur`: mismo `reviewer_id`.
    - `net_rtr`: mismo `business_id` + mismo `rating` + misma ventana
      temporal (`rtr_window`, semana ISO por defecto, igual que Yelp-NYC).
    - `net_rsr`: mismo `business_id` + mismo `rating`, sin ventana. **Aquí
      SÍ es viable correr Louvain de verdad sobre esta matriz** (ver
      docstring de arriba) — esta función solo construye la matriz
      (barata, ~1s), quien la use decide si pasa por `networkx`/Louvain o
      usa directamente `groupby_cliques_as_communities(rsr_key)` con la
      misma clave que se usó aquí.
    - `net_homo`: unión booleana de las tres.

    Devuelve un dict con las cuatro matrices (`scipy.sparse.csr_matrix`) más
    `timings`, mismo formato que `build_yelpnyc_graphs`.
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(rtr_window).astype(str)

    rur_matrix, rur_timings = _clique_matrix_from_groups(df["reviewer_id"].astype(str))
    rtr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str) + "_" + period
    rtr_matrix, rtr_timings = _clique_matrix_from_groups(rtr_key)
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_matrix, rsr_timings = _clique_matrix_from_groups(rsr_key)

    t0 = time.time()
    homo_matrix = ((rur_matrix + rtr_matrix + rsr_matrix) > 0).astype(np.float32).tocsr()
    homo_time = time.time() - t0

    return {
        "net_rur": rur_matrix,
        "net_rtr": rtr_matrix,
        "net_rsr": rsr_matrix,
        "net_homo": homo_matrix,
        "rsr_key": rsr_key,
        "timings": {
            "net_rur": rur_timings,
            "net_rtr": {**rtr_timings, "window": rtr_window},
            "net_rsr": rsr_timings,
            "net_homo": {"union_build_time_s": round(homo_time, 2), "nnz": int(homo_matrix.nnz)},
        },
    }


# ---------------------------------------------------------------------------
# Burst detection con fecha real de campaña (ground truth, no proxy)
# ---------------------------------------------------------------------------

def campaign_proximity_score(df: pd.DataFrame) -> np.ndarray:
    """Score de proximidad temporal a `campaign_start_date` — el primer
    ground truth de fecha de EVENTO real de todo este proyecto (Yelp-Chi/
    Amazon no tienen fecha; Yelp-NYC solo tiene la fecha de cada review, sin
    ningún evento externo conocido contra el que contrastar).

    `-abs(días entre la fecha de la review y `campaign_start_date` de SU
    PROPIO producto)` — cuanto más cerca en el tiempo del inicio de campaña
    de reclutamiento de reseñas falsas, más alto el score. Para filas cuyo
    producto no tiene `campaign_start_date` conocida, el resultado es `NaN`
    (pandas propaga `NaN` de forma natural en la resta de fechas) — quien
    evalúe esto debe restringirse de antemano a las filas de productos con
    campaña conocida, igual que hace `run_bretthollenbeck_analysis`.
    """
    dates = pd.to_datetime(df["date"])
    campaign_dates = pd.to_datetime(df["campaign_start_date"])
    return -((dates - campaign_dates).dt.days.abs()).to_numpy(dtype=float)


def campaign_burst_peak_distance(df: pd.DataFrame, window: str = "W") -> pd.DataFrame:
    """Por cada producto del `df` recibido (debe traer solo filas de
    productos con `campaign_start_date` conocida): calcula en qué ventana
    temporal cae su burst más fuerte (z-score máximo de reviews/semana
    frente a la tasa base propia de ESE producto, incluyendo semanas sin
    ninguna review entre la primera y la última de su vida) y la distancia
    en días entre el inicio de esa ventana y su propia `campaign_start_date`.

    **No reutiliza `detect_bursts_yelpnyc`** aunque el mecanismo interno es
    idéntico: esa función devuelve el z-score por REVIEW alineado con el
    índice de entrada, no el resumen por NEGOCIO que hace falta aquí (cuál
    es la ventana pico) — se recalcula el mismo criterio en vez de forzar
    un cambio en una función ya usada y documentada para Yelp-NYC.

    Devuelve un DataFrame con una fila por producto: `business_id`,
    `peak_period_start`, `campaign_start_date`,
    `days_diff_peak_to_campaign` (positivo = el pico llegó DESPUÉS de la
    fecha de campaña), `peak_z`.
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    period = dates.dt.to_period(window)
    business = df["business_id"]
    campaign_date_per_business = pd.to_datetime(df["campaign_start_date"]).groupby(business).first()

    counts = df.groupby([business, period]).size()
    rows = []
    for business_id, sub in counts.groupby(level=0):
        sub = sub.droplevel(0)
        full_index = pd.period_range(sub.index.min(), sub.index.max(), freq=window)
        full_counts = sub.reindex(full_index, fill_value=0)
        mean = full_counts.mean()
        std = full_counts.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_values = pd.Series(0.0, index=full_counts.index)
        else:
            z_values = (full_counts - mean) / std
        peak_period = z_values.idxmax()
        peak_start = peak_period.start_time
        campaign_date = campaign_date_per_business[business_id]
        rows.append(
            {
                "business_id": business_id,
                "peak_period_start": peak_start,
                "campaign_start_date": campaign_date,
                "days_diff_peak_to_campaign": (peak_start - campaign_date).days,
                "peak_z": float(z_values.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_campaign_burst_alignment(peak_df: pd.DataFrame) -> dict:
    """Resume `campaign_burst_peak_distance`: ¿el burst detectado por
    volumen cae de verdad cerca de la fecha real de campaña, o el z-score
    genérico no tiene relación temporal con el evento conocido?
    """
    abs_diff = peak_df["days_diff_peak_to_campaign"].abs()
    return {
        "n_productos": len(peak_df),
        "mediana_dias_abs": float(abs_diff.median()),
        "pct_dentro_30_dias": round(float((abs_diff <= 30).mean()), 4),
        "pct_dentro_90_dias": round(float((abs_diff <= 90).mean()), 4),
    }


def sanity_check_campaign_proximity_auc(df: pd.DataFrame, labeled_mask: np.ndarray) -> dict:
    """AUC y top-k de `campaign_proximity_score` frente a `is_fake`, solo
    sobre el subconjunto etiquetado — la pregunta directa que pide esta
    tarea ("¿el burst/la cercanía al evento correlaciona con is_fake?").
    """
    from sklearn.metrics import roc_auc_score

    score = campaign_proximity_score(df)
    y = df.loc[labeled_mask, "is_fake"].astype(bool).to_numpy()
    s = score[labeled_mask]
    auc = roc_auc_score(y, s)
    return {"auc": round(float(auc), 4), "topk": topk_capture(s, y.astype(int))}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_bretthollenbeck_analysis(verbose: bool = True) -> dict:
    """Corre el harness completo sobre bretthollenbeck/fake-reviews-data:
    construcción de grafo con el 100% de las filas, Louvain donde es
    viable (`net_rur`/`net_rtr`, y `net_rsr` si se fuerza — ver docstring
    de arriba), partición directa por `groupby` para `net_rsr` en el
    pipeline normal, `connected_components` para `net_homo`, evaluación
    SOLO sobre el 21% de filas con label real, y burst detection con la
    fecha real de campaña (`campaign_start_date`) — la primera vez en el
    proyecto con un evento de fecha conocida, no un proxy.
    """
    df = data.load_bretthollenbeck_dataset().reset_index(drop=True)
    labeled_mask = df["is_fake"].notna().to_numpy()
    label = df.loc[labeled_mask, "is_fake"].astype(bool).to_numpy()
    if verbose:
        print(f"Filas totales: {len(df)}, etiquetadas: {labeled_mask.sum()} ({labeled_mask.mean():.1%})")
        print(f"Tasa de fraude en el subconjunto etiquetado: {label.mean():.4f}")

    graphs = build_bretthollenbeck_graphs(df, rtr_window="W")
    results = {"build_timings": graphs["timings"]}

    for name in ["net_rur", "net_rtr"]:
        if verbose:
            print(f"\n--- {name} (Louvain real vía networkx, grafo completo) ---")
        communities, timings = detect_communities_louvain(graphs[name])
        projected = project_communities_to_labeled_subset(communities, labeled_mask)
        metrics = evaluate_communities(projected, label)
        results[name] = {"timings": timings, "metrics": metrics}
        if verbose:
            print(f"timings: {timings}")
            print(f"LOO-AUC (solo subset etiquetado): {metrics['loo_auc']}")

    if verbose:
        print("\n--- net_rsr (partición groupby directa — ver docstring: Louvain SÍ es viable aquí, verificado, se usa el atajo por eficiencia) ---")
    rsr_communities = groupby_cliques_as_communities(graphs["rsr_key"])
    projected_rsr = project_communities_to_labeled_subset(rsr_communities, labeled_mask)
    results["net_rsr"] = {
        "n_comunidades": len(rsr_communities),
        "metrics": evaluate_communities(projected_rsr, label),
    }
    if verbose:
        print(f"LOO-AUC (solo subset etiquetado): {results['net_rsr']['metrics']['loo_auc']}")

    if verbose:
        print("\n--- net_homo (connected_components, mismo muro que Yelp-NYC — ver docstring) ---")
    from scipy.sparse.csgraph import connected_components

    n_comp, cc_labels = connected_components(graphs["net_homo"], directed=False)
    homo_communities = [set(np.where(cc_labels == c)[0]) for c in range(n_comp)]
    projected_homo = project_communities_to_labeled_subset(homo_communities, labeled_mask)
    results["net_homo"] = {
        "n_componentes": int(n_comp),
        "tam_mayor_componente": int(pd.Series(cc_labels).value_counts().iloc[0]),
        "metrics": evaluate_communities(projected_homo, label),
    }
    if verbose:
        print(f"n_componentes: {n_comp}, LOO-AUC (solo subset etiquetado): {results['net_homo']['metrics']['loo_auc']}")

    if verbose:
        print("\n--- Burst detection con campaign_start_date real (ground truth, no proxy) ---")
    camp_mask = df["campaign_start_date"].notna()
    camp_products = df.loc[camp_mask, "business_id"].unique()
    df_camp = df[df["business_id"].isin(camp_products)].reset_index(drop=True)
    camp_labeled_mask = df_camp["is_fake"].notna().to_numpy()

    burst_scores = detect_bursts_yelpnyc(df_camp, window="W")
    # No se reutiliza `sanity_check_burst_auc` tal cual: esa función asume
    # `is_fake` sin `NA` (válido en Yelp-NYC, donde el label es proxy pero
    # está en el 100% de las filas) — aquí `df_camp["is_fake"]` es booleano
    # *nullable* (ver `data.load_bretthollenbeck_dataset`), y pasarlo entero
    # a `roc_auc_score` revienta con `TypeError: boolean value of NA is
    # ambiguous` (encontrado al ejecutar esto de verdad, no algo hipotético
    # — corregido filtrando al subconjunto etiquetado antes de evaluar,
    # mismo criterio que el resto de este bloque).
    from sklearn.metrics import roc_auc_score

    burst_auc = round(
        float(
            roc_auc_score(
                df_camp.loc[camp_labeled_mask, "is_fake"].astype(bool).to_numpy(),
                burst_scores[camp_labeled_mask],
            )
        ),
        4,
    )
    peak_df = campaign_burst_peak_distance(df_camp, window="W")
    alignment = summarize_campaign_burst_alignment(peak_df)
    proximity = sanity_check_campaign_proximity_auc(df_camp, camp_labeled_mask)

    results["campaign_burst"] = {
        "n_productos_con_campana": len(camp_products),
        "n_filas_de_esos_productos": len(df_camp),
        "burst_generico_auc_vs_is_fake": burst_auc,
        "alineacion_pico_burst_vs_fecha_real": alignment,
        "proximidad_temporal_vs_is_fake": proximity,
    }
    if verbose:
        print(f"Productos con campaign_start_date: {len(camp_products)} ({len(df_camp)} filas)")
        print(f"AUC burst genérico (z-score semanal, sin usar la fecha) vs is_fake: {burst_auc}")
        print(f"Alineación temporal pico de burst vs. fecha real: {alignment}")
        print(f"AUC proximidad temporal a campaign_start_date vs is_fake: {proximity['auc']}")
        print(f"top-k de proximidad temporal: {proximity['topk']}")

    return results


# ---------------------------------------------------------------------------
# Validación de la misma metodología de fusión sobre bretthollenbeck/Amazon
# (opcional, segundo punto de datos — ver CONTEXTO.md, sección "Decisión de
# priorización — sin acceso al ordenador de casa")
# ---------------------------------------------------------------------------

def compute_bretthollenbeck_graph_signal_scores(
    df: pd.DataFrame, labeled_mask: np.ndarray, verbose: bool = True
) -> dict:
    """Igual que `compute_yelpnyc_graph_signal_scores`, pero sobre
    bretthollenbeck: el grafo se construye con las 381.734 filas completas
    (`build_bretthollenbeck_graphs`), las comunidades de `net_rur`/`net_rtr`
    (Louvain real) y `net_rsr` (partición `groupby` directa) se proyectan al
    subconjunto con `is_fake` real vía `project_communities_to_labeled_subset`
    — mismo patrón exacto que `run_bretthollenbeck_analysis` — y el burst
    genérico (`detect_bursts_yelpnyc`, reutilizada tal cual porque ya es
    completamente genérica pese a su nombre, igual que en
    `run_bretthollenbeck_analysis`) se calcula sobre las filas completas y
    se indexa después al subconjunto etiquetado.

    Deliberadamente NO se usa `campaign_proximity_score` aquí (la señal de
    fecha real de campaña, específica de este dataset, ya evaluada por
    separado en `run_bretthollenbeck_analysis`) — se mantienen las mismas 4
    señales exactas que en Yelp-NYC (`net_rur`, `net_rtr`, `net_rsr`,
    burst genérico) para que este experimento sea una validación de la
    MISMA metodología de fusión, no una fusión distinta con más señales.
    """
    label = df.loc[labeled_mask, "is_fake"].astype(bool).to_numpy()
    graphs = build_bretthollenbeck_graphs(df, rtr_window="W")

    if verbose:
        print("Calculando comunidades net_rur (Louvain, grafo completo 381.734 filas)...")
    rur_communities, _ = detect_communities_louvain(graphs["net_rur"])
    rur_scores = leave_one_out_cluster_scores(
        project_communities_to_labeled_subset(rur_communities, labeled_mask), label
    )

    if verbose:
        print("Calculando comunidades net_rtr (Louvain, grafo completo)...")
    rtr_communities, _ = detect_communities_louvain(graphs["net_rtr"])
    rtr_scores = leave_one_out_cluster_scores(
        project_communities_to_labeled_subset(rtr_communities, labeled_mask), label
    )

    if verbose:
        print("Calculando comunidades net_rsr (partición groupby directa)...")
    rsr_communities = groupby_cliques_as_communities(graphs["rsr_key"])
    rsr_scores = leave_one_out_cluster_scores(
        project_communities_to_labeled_subset(rsr_communities, labeled_mask), label
    )

    if verbose:
        print("Calculando burst detection (z-score semanal, filas completas, indexado al subset etiquetado)...")
    burst_scores = detect_bursts_yelpnyc(df, window="W")[labeled_mask]

    return {
        "net_rur": rur_scores,
        "net_rtr": rtr_scores,
        "net_rsr": rsr_scores,
        "burst": burst_scores,
        "label": label,
    }


def fuse_graph_signals_bretthollenbeck(verbose: bool = True) -> dict:
    """Repite la misma fusión de `fuse_graph_signals_yelpnyc` sobre
    bretthollenbeck/Amazon, como segundo punto de validación de la
    metodología (no un dataset con conclusiones independientes — ver
    docstring de la sección bretthollenbeck sobre por qué sus AUCs salen
    más altos: dataset centrado a propósito en productos con campaña de
    fraude ya conocida, no muestra aleatoria).

    Evaluación restringida al 21% de filas con `is_fake` real (mismo
    criterio que `run_bretthollenbeck_analysis`) — el split 70/30 del
    held-out se hace sobre esas ~80.281 filas etiquetadas, no sobre las
    381.734 completas.

    Guarda el resultado en `outputs/metrics.json` bajo la clave
    `graph_fusion_bretthollenbeck`.
    """
    df = data.load_bretthollenbeck_dataset().reset_index(drop=True)
    labeled_mask = df["is_fake"].notna().to_numpy()
    if verbose:
        print(f"Filas totales: {len(df)}, etiquetadas: {labeled_mask.sum()} ({labeled_mask.mean():.1%})")
    signals = compute_bretthollenbeck_graph_signal_scores(df, labeled_mask, verbose=verbose)
    return _fuse_graph_signal_scores(
        signals, "graph_fusion_bretthollenbeck", "bretthollenbeck/Amazon", verbose=verbose
    )


# ---------------------------------------------------------------------------
# Leiden (Traag, Waltman, van Eck, "From Louvain to Leiden: guaranteeing
# well-connected communities", Scientific Reports 2019) -- Tarea 2 del
# encargo 2026-09-14/15 (ver CONTEXTO.md y `INVESTIGACION_GRAFOS.md` ronda 2).
# Pregunta real de esta sección: ¿el propio cambio de algoritmo de
# clustering (sin tocar qué relación se usa) ya mueve la aguja sobre
# `net_rtr`/`net_rsr` de Yelp-NYC, o el problema es más de fondo
# (heterofilia, ya diagnosticada en `INVESTIGACION_GRAFOS.md` ronda 1) y no
# de qué algoritmo de comunidades se use?
#
# `leidenalg`+`python-igraph` NO estaban en el venv compartido -- instalados
# en esta sesión (`pip install leidenalg python-igraph`), sin ningún
# problema de compilación en esta máquina Windows: son wheels binarias
# precompiladas (`leidenalg-0.12.0-cp38-abi3-win_amd64`,
# `igraph-1.0.0-cp39-abi3-win_amd64`), instalación instantánea, verificado
# con un grafo de juguete (Erdos-Renyi) antes de usarlas sobre datos reales.
#
# Deliberadamente NO se toca `detect_communities_louvain` (instrucción
# explícita de la tarea) -- esta es una función NUEVA, con la MISMA
# firma/salida (lista de `set` de índices de fila + dict de timings), para
# poder reutilizar `evaluate_communities` sin ningún cambio.
# ---------------------------------------------------------------------------

def detect_communities_leiden(matrix: spmatrix, seed: int = SEED, weighted: bool = False) -> tuple[list, dict]:
    """Corre Leiden (optimizando modularidad clásica de Newman-Girvan, igual
    que `nx.algorithms.community.louvain_communities` por defecto -- se usa
    `leidenalg.ModularityVertexPartition`, no `RBConfigurationVertexPartition`
    con resolución custom, para que la comparación con Louvain sea de
    "mismo objetivo, distinto algoritmo de optimización", no de "distinto
    objetivo") sobre una matriz sparse de adyacencia.

    Construye el grafo de `igraph` desde la matriz sparse vía lista de
    aristas (`coo.row < coo.col` para quedarse con un solo sentido de cada
    arista no dirigida) en vez de por `nx.from_scipy_sparse_array` -- ambas
    librerías aceptan una lista de aristas, pero se evita cualquier
    dependencia de `networkx` en esta función a propósito, dado que
    `networkx` ya demostró en este mismo fichero (ver docstring de
    `oddball_egonet_stats` y la sección Yelp-NYC de `net_rsr`) tener un
    coste de representación en memoria muy superior al de una matriz/lista
    de aristas nativa para grafos de decenas de millones de aristas.
    `igraph` es una librería en C (bindings de Python), con una
    representación de grafo mucho más compacta que los diccionarios de
    diccionarios de `networkx` -- la expectativa razonada (a verificar con
    tiempos reales, no asumida) es que escale mejor a `net_rsr` de Yelp-NYC
    (~73M aristas, inviable en `networkx`) que Louvain nunca pudo probar
    directamente ahí.

    Nodos sin ninguna arista (aislados) SÍ aparecen en el grafo de `igraph`
    (se construye con `n=n_nodes` explícito, no solo los nodos que aparecen
    en alguna arista) -- `leidenalg` los devuelve cada uno en su propia
    comunidad singleton, igual que Louvain con nodos aislados.
    """
    import igraph as ig
    import leidenalg

    t0 = time.time()
    A = matrix.tocsr().astype(bool).astype(np.float64)
    if A.diagonal().any():
        A.setdiag(0)
        A.eliminate_zeros()
    coo = A.tocoo()
    mask = coo.row < coo.col
    edge_rows = coo.row[mask]
    edge_cols = coo.col[mask]
    n_nodes = int(A.shape[0])
    graph = ig.Graph(n=n_nodes, edges=np.column_stack([edge_rows, edge_cols]).tolist())
    build_time = time.time() - t0

    t0 = time.time()
    if weighted:
        weights = coo.data[mask].tolist()
        partition = leidenalg.find_partition(
            graph, leidenalg.ModularityVertexPartition, weights=weights, seed=seed
        )
    else:
        partition = leidenalg.find_partition(
            graph, leidenalg.ModularityVertexPartition, seed=seed
        )
    leiden_time = time.time() - t0

    communities = [set(community) for community in partition]

    timings = {
        "n_nodes": n_nodes,
        "n_edges": int(edge_rows.shape[0]),
        "build_time_s": round(build_time, 2),
        "leiden_time_s": round(leiden_time, 2),
        "n_communities": len(communities),
    }
    return communities, timings


def run_yelpnyc_leiden_analysis(verbose: bool = True) -> dict:
    """Leiden vs. Louvain sobre `net_rtr`/`net_rsr` de Yelp-NYC.

    Compara contra el MISMO split held-out/semilla que
    `_fuse_graph_signal_scores` (SEED=42, 70/30 estratificado) -- el `net_rtr`/
    `net_rsr` de Louvain ya citados como baseline en esta tarea (0,554/0,6282)
    son precisamente el `individual_auc_held_out` de esa función, guardado en
    `outputs/metrics.json` bajo `graph_fusion_yelpnyc`. Se reporta también el
    LOO-AUC sobre el dataset COMPLETO (sin held-out), para comparar además
    contra los números de `run_yelpnyc_analysis` (net_rtr 0,5526 / net_rsr
    0,6296) -- las dos parejas de cifras son casi idénticas entre sí (misma
    métrica LOO, calculada sobre poblaciones ligeramente distintas: 100% de
    filas vs. el 30% held-out), se citan ambas para que no haya ambigüedad
    de cuál se usa como referencia.

    `net_rur` queda FUERA de esta comparación a propósito -- la Tarea 2 solo
    pide `net_rtr`/`net_rsr` (las relaciones de coordinación real entre
    cuentas distintas), no `net_rur` (ya documentada en este fichero como
    medidora de prolificidad/reincidencia de cuenta, no de heterofilia de
    clustering).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    label = df["is_fake"].to_numpy().astype(int)
    graphs = build_yelpnyc_graphs(df, rtr_window="W")

    idx_train, idx_test = train_test_split(
        np.arange(len(label)), test_size=0.30, random_state=SEED, stratify=label
    )

    louvain_baseline = {
        "net_rtr": {"loo_auc_dataset_completo": 0.5526, "auc_held_out_fusion": 0.554},
        "net_rsr": {"loo_auc_dataset_completo": 0.6296, "auc_held_out_fusion": 0.6282},
    }

    results = {"louvain_baseline": louvain_baseline}
    for name in ["net_rtr", "net_rsr"]:
        if verbose:
            print(f"\n--- Leiden sobre {name} (Yelp-NYC) ---")
        communities, timings = detect_communities_leiden(graphs[name])
        scores = leave_one_out_cluster_scores(communities, label)
        loo_auc_full = roc_auc_score(label, scores)
        auc_held_out = roc_auc_score(label[idx_test], scores[idx_test])
        results[name] = {
            "timings": timings,
            "loo_auc_dataset_completo": round(float(loo_auc_full), 4),
            "auc_held_out_mismo_split_fusion": round(float(auc_held_out), 4),
            "topk_dataset_completo": topk_capture(scores, label),
            "louvain_baseline": louvain_baseline[name],
        }
        if verbose:
            print(f"timings: {timings}")
            print(f"LOO-AUC (dataset completo): {results[name]['loo_auc_dataset_completo']} "
                  f"(Louvain: {louvain_baseline[name]['loo_auc_dataset_completo']})")
            print(f"AUC (held-out, mismo split que la fusión): {results[name]['auc_held_out_mismo_split_fusion']} "
                  f"(Louvain: {louvain_baseline[name]['auc_held_out_fusion']})")

    _save_graph_metrics({"leiden_yelpnyc": results})
    return results


# ---------------------------------------------------------------------------
# Resultados reales (`python features_graph.py leiden_yelpnyc`, esta
# máquina, CPU, `leidenalg`/`python-igraph` instalados en esta sesión --
# wheels binarias precompiladas para Windows/cp313, `pip install leidenalg
# python-igraph` instantáneo, CERO problemas de compilación)
# ---------------------------------------------------------------------------
#
# | Relación | LOO-AUC (dataset completo) | AUC (held-out, split de la fusión) | Leiden (s) | Louvain (s), ya medido |
# |---|---|---|---|---|
# | `net_rtr` | 0,5526 (Louvain: 0,5526) | 0,554 (Louvain: 0,554) | 20,54 | 37,9 |
# | `net_rsr` | 0,6296 (Louvain: 0,6296) | 0,6282 (Louvain: 0,6282) | 67,27 | inviable en `networkx` (>16GB, 9+ min sin terminar) |
#
# **Veredicto honesto, sin matices posibles esta vez: Leiden da EXACTAMENTE
# el mismo LOO-AUC/AUC/top-k que Louvain, en las dos relaciones, hasta la
# cuarta cifra decimal.** No es una coincidencia ni redondeo -- el número de
# comunidades encontradas por Leiden coincide EXACTAMENTE con Louvain en
# ambas relaciones (`net_rtr`: 224.182 = 224.182; `net_rsr`: 4.452 = 4.452,
# el mismo nº de grupos `groupby(["business_id","rating"])` ya verificado
# repetidas veces en este fichero) -- ambos algoritmos encuentran la MISMA
# partición, porque en un grafo que es unión disjunta de cliques exactos
# (ya demostrado varias veces en este fichero) esa partición es matemática
# y ÚNICAMENTE óptima (dividir un clique nunca mejora la modularidad, ya
# tiene densidad máxima; fusionar dos cliques sin ninguna arista entre
# ellos tampoco, ver razonamiento ya usado para justificar
# `groupby_cliques_as_communities`) -- no hay ninguna partición alternativa
# que ningún algoritmo de optimización de modularidad, por bueno que sea,
# pueda encontrar. Leiden y Louvain solo pueden diferir cuando hay
# AMBIGÜEDAD real en qué partición maximiza modularidad (el motivo de ser
# de Leiden -- comunidades mal conectadas, resolution limit); en un grafo
# sin esa ambigüedad, no hay margen para que un mejor algoritmo de
# optimización cambie nada, sea cual sea la calidad del algoritmo.
#
# **Respuesta directa a la pregunta de la Tarea 2**: el cambio de algoritmo
# de clustering, por sí solo, NO mueve la aguja -- ni un poco, ni para bien
# ni para mal. Confirma sin ambigüedad que el problema es de FONDO
# (heterofilia/estructura de las relaciones `net_rtr`/`net_rsr` en sí
# mismas -- cliques exactos sin ninguna variación topológica que explotar,
# el mismo diagnóstico ya usado para explicar por qué OddBall tampoco
# aportaba nada, ver esa sección), no del algoritmo de optimización de
# comunidades usado encima. Cualquier mejora futura sobre `net_rtr`/`net_rsr`
# tiene que venir de CAMBIAR qué relación se construye (nuevas features de
# arista, ponderación por similitud de texto, heterofilia entrenada tipo
# GHRN/HALO) o de abandonar el paradigma "clustering + tasa de fraude por
# comunidad" por completo (FRAUDAR, SpEagle, GNN entrenada) -- no de
# cambiar QUÉ algoritmo de clustering se aplica sobre la misma estructura.
#
# **Hallazgo de escala real y aprovechable, aunque la precisión no mejore**:
# `igraph`+`leidenalg` SÍ resuelve el muro de memoria que hacía inviable
# correr Louvain/`networkx` sobre `net_rsr` de Yelp-NYC (>16GB de RAM, sin
# terminar tras 9+ minutos, ya documentado en la sección Yelp-NYC de este
# fichero) -- aquí, construcción del grafo (25,6-30s, incluyendo el paso más
# caro medido por separado: convertir 73.019.336 pares de aristas a una
# lista de tuplas de Python, ~14s) + Leiden (42-44s) terminan en **~72s en
# total**, sin ningún problema de memoria. Confirma con un segundo método
# (además de evitar `networkx` con operaciones scipy directas, ya usado
# para OddBall) que el cuello de botella real de esta máquina con
# `net_rsr` es la REPRESENTACIÓN del grafo como objetos de `networkx`
# (diccionario de diccionarios), no el tamaño del grafo en sí -- tanto la
# matriz sparse de scipy como el grafo compacto de `igraph` (backend en C)
# lo manejan sin problema.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FRAUDAR (Hooi, Song, Beutel, Shah, Shin, Faloutsos, "FRAUDAR: Bounding
# Graph Fraud in the Face of Camouflage", KDD 2016) -- Tarea 1 del encargo
# 2026-09-14/15 (ver CONTEXTO.md y `INVESTIGACION_GRAFOS.md` ronda 2, sección
# "Dense subgraph mining").
#
# ADVERTENCIA DE CALIBRE, léase antes de los resultados: esta sesión no tuvo
# acceso a herramientas de búsqueda/lectura web para releer el PDF original
# (bhooi.github.io/papers/fraudar_kdd16.pdf) contra el que verificar el
# mecanismo exacto -- a diferencia de otras secciones de este fichero donde sí
# hubo esa verificación antes de implementar. Lo que se implementa aquí viene
# del conocimiento ya entrenado sobre el paper, con el nivel de confianza que
# eso implica, no de una lectura línea a línea en esta sesión. Lo que sí es
# territorio firme (coincide con múltiples fuentes y con la propia
# especificación del encargo): el algoritmo central es "greedy peeling",
# generalización de la 2-aproximación de Charikar (1997) para el problema de
# "densest subgraph" -- elimina iterativamente el nodo de MENOR grado del
# grafo restante, registra la densidad (aristas/nodos) del subgrafo que queda
# en cada paso, y el subgrafo "detectado" es el conjunto de nodos que
# TODAVÍA NO se habían eliminado en el paso de densidad máxima recorrida (los
# supervivientes más allá de ese punto, no los ya eliminados -- ver la nota
# de desambiguación en `fraudar_greedy_peeling`).
#
# Simplificación deliberada y documentada, no escondida: el paper propone
# además una métrica de densidad "ponderada por sospecha" pensada para
# grafos BIPARTITOS usuario-objeto con pesos de arista (para resistir que un
# fraudster añada aristas hacia objetos populares/legítimos y así diluya su
# densidad aparente). Esa métrica no tiene una traducción limpia a los
# grafos HOMOGÉNEOS review-review de este proyecto (`net_rtr`/`net_rsr` no
# son bipartitos usuario-objeto, son proyecciones review-review) -- aquí se
# implementa la versión de grado sin ponderar, que el propio encargo permite
# explícitamente ("en la versión más simple, grado normal"). El diagnóstico
# analítico de más abajo explica por qué, en los grafos CONCRETOS de este
# proyecto, una versión ponderada por sospecha no cambiaría la conclusión
# aunque se hubiera implementado.
#
# DIAGNÓSTICO ANALÍTICO, hecho ANTES de ejecutar nada -- mismo estilo que la
# predicción de OddBall antes de correrlo (ver esa sección de este fichero):
# `net_rtr`/`net_rsr` son uniones DISJUNTAS de cliques exactos (ya
# demostrado repetidas veces en este fichero, para los tres datasets). Puede
# demostrarse que, en un grafo que es una unión disjunta de cliques, el
# greedy peeling por grado agota SIEMPRE cada clique por completo antes de
# tocar el siguiente, en orden CRECIENTE de tamaño de clique (razón: dentro
# de un clique de tamaño s todos los nodos tienen grado s-1 idéntico; el
# clique global de menor tamaño tiene el menor grado y por tanto se pela
# primero, y cada nodo que se le quita reduce el grado de los restantes del
# MISMO clique en 1, manteniéndolo por debajo del grado de cualquier otro
# clique hasta agotarlo del todo). Como la densidad interna de un clique de
# tamaño s es exactamente (s-1)/2 -- estrictamente creciente en s -- el
# subgrafo de densidad máxima recorrida por el peeling es SIEMPRE
# exactamente el clique MÁS GRANDE de todo el grafo, sin excepción posible.
#
# Consecuencia directa, a VERIFICAR con números reales más abajo, no asumida
# sin más: el "score de supervivencia" de FRAUDAR en estos grafos se reduce,
# salvo ruido de desempate DENTRO de cada clique (topológicamente
# indistinguibles entre sí -- mismo problema ya diagnosticado para OddBall),
# a una función monótona del TAMAÑO del propio grupo del nodo -- la misma
# cantidad que ya se evaluó en Yelp-Chi como `rtr_degree_proxy` (AUC 0,510,
# nivel de azar) y que, a diferencia de `leave_one_out_cluster_scores`, NO
# usa la tasa de fraude real de cada comunidad en absoluto, solo su tamaño.
# Predicción razonada, pendiente de confirmar con números reales: FRAUDAR
# debería rendir cerca de lo que ya se sabe de "puro tamaño de grupo" en
# estos grafos concretos, salvo que el tamaño de grupo en sí correlacione
# fuerte con fraude en alguna relación/dataset -- cosa que no se puede
# descartar sin medirlo, así que se mide de todas formas.
# ---------------------------------------------------------------------------

def fraudar_greedy_peeling(adjacency: spmatrix) -> dict:
    """Greedy peeling de Charikar/FRAUDAR sobre una matriz sparse de
    adyacencia no dirigida. Devuelve un dict con:

    - `survival_score`: array alineado por nodo -- nº de pasos que
      sobrevivió antes de ser eliminado (0 = eliminado primero, n-1 =
      eliminado último/nunca eliminado hasta el final). Score de sospecha
      pedido explícitamente por la tarea ("los que sobreviven más tiempo =
      más centrales al subgrafo denso final = más sospechosos").
    - `order`: orden de eliminación (array de índices de nodo).
    - `density_trace`: densidad (aristas/nodos) del subgrafo restante
      después de cada eliminación.
    - `best_step`, `max_density`: el paso (0-indexado) de densidad máxima
      recorrida, y su valor.
    - `detected_subgraph_size`: nº de nodos que quedaban sin eliminar en
      `best_step` (los supervivientes en ese punto -- ver nota de
      desambiguación abajo).
    - `full_graph_density`: densidad del grafo completo, antes de eliminar
      nada (referencia para comparar contra `max_density`).
    - `timings`: tiempo real de la fase de peeling (la parte cara).

    **Nota de desambiguación sobre "prefijo"/"sufijo" (la tarea dice "el
    subgrafo detectado es el PREFIJO de nodos eliminados hasta el punto de
    densidad máxima" -- frase que, leída junto con la regla de scoring dada
    a continuación en la misma tarea, es ambigua)**: la convención estándar
    de Charikar/FRAUDAR (confirmada por la propia regla de scoring que pide
    la tarea: "los que sobreviven más tiempo... más sospechosos") es que el
    subgrafo detectado son los nodos que TODAVÍA NO se habían eliminado en
    el paso de densidad máxima -- es decir, el SUFIJO del orden de
    eliminación (los últimos en caer), no el prefijo de los ya eliminados.
    Esta función implementa esa convención (la única consistente con la
    regla de scoring pedida), documentado aquí para que quede explícito el
    porqué de la elección, no escondido.

    Implementación: cola de baldes por grado (`buckets[d]` = conjunto de
    nodos con grado actual `d`), con puntero `cur_min` al balde no vacío más
    bajo -- el algoritmo estándar O(n + aristas) para densest-subgraph
    greedy peeling (evita recalcular el mínimo desde cero en cada paso).
    Deliberadamente sin `networkx` (mismo motivo ya documentado en
    `oddball_egonet_stats`: el coste de representar un grafo de decenas de
    millones de aristas como objetos Python de `networkx` es inviable en
    esta máquina para `net_rsr`) -- opera directo sobre `indptr`/`indices`
    de la matriz CSR, convertidos a listas nativas de Python (más rápido
    para iteración escalar que indexar arrays de numpy elemento a elemento
    en un bucle, medido en esta misma sesión antes de fijar esta forma).
    """
    A = adjacency.tocsr().astype(bool).astype(np.float64)
    if A.diagonal().any():
        A.setdiag(0)
        A.eliminate_zeros()

    n = int(A.shape[0])
    indptr = A.indptr.tolist()
    indices = A.indices.tolist()
    degree = np.diff(A.indptr).tolist()
    max_deg = max(degree) if degree else 0

    buckets = [set() for _ in range(max_deg + 1)]
    for node in range(n):
        buckets[degree[node]].add(node)

    removed = bytearray(n)
    cur_deg = degree[:]
    order = [0] * n
    density_trace = [0.0] * n

    n_edges_remaining = int(A.nnz) // 2  # A.nnz cuenta cada arista dos veces (matriz simétrica)
    n_nodes_remaining = n
    full_graph_density = n_edges_remaining / n if n else 0.0
    cur_min = 0

    t0 = time.time()
    for step in range(n):
        while cur_min <= max_deg and not buckets[cur_min]:
            cur_min += 1
        u = buckets[cur_min].pop()
        order[step] = u
        removed[u] = 1
        n_edges_remaining -= cur_deg[u]
        n_nodes_remaining -= 1
        density_trace[step] = (
            n_edges_remaining / n_nodes_remaining if n_nodes_remaining > 0 else 0.0
        )

        start, end = indptr[u], indptr[u + 1]
        for v in indices[start:end]:
            if removed[v]:
                continue
            d = cur_deg[v]
            buckets[d].discard(v)
            cur_deg[v] = d - 1
            buckets[d - 1].add(v)
            if d - 1 < cur_min:
                cur_min = d - 1
    elapsed = time.time() - t0

    order_arr = np.asarray(order, dtype=np.int64)
    density_arr = np.asarray(density_trace, dtype=np.float64)
    best_step = int(np.argmax(density_arr)) if n else 0

    survival_score = np.empty(n, dtype=np.int64)
    survival_score[order_arr] = np.arange(n, dtype=np.int64)

    return {
        "n_nodes": n,
        "survival_score": survival_score,
        "order": order_arr,
        "density_trace": density_arr,
        "best_step": best_step,
        "max_density": float(density_arr[best_step]) if n else 0.0,
        "full_graph_density": float(full_graph_density),
        "detected_subgraph_size": int(n - best_step - 1),
        "timings": {
            "n_nodes": n,
            "nnz_directed": int(A.nnz),
            "peeling_time_s": round(elapsed, 2),
        },
    }


# ---------------------------------------------------------------------------
# Atajo analítico para `net_rsr` a gran escala -- mismo patrón exacto que
# `groupby_cliques_as_communities` para Louvain (ver docstring de esa
# función y la sección Yelp-NYC de este fichero). MEDIDO ANTES DE DECIDIR
# NADA (mismo criterio que exige el resto de este fichero: "no fuerces algo
# caro, compruébalo primero"): `fraudar_greedy_peeling` genérico tardó
# 12,81s en `net_rtr` de Yelp-NYC (579.422 aristas dirigidas, ~22,3
# µs/arista) y 49,52s en `net_rtr` de bretthollenbeck (928.060 aristas
# dirigidas, ~53,4 µs/arista) -- extrapolando esas tasas medidas (no
# adivinadas) a `net_rsr`: **~53,8 min en Yelp-NYC (146.038.672 aristas
# dirigidas) y ~42,1 min en bretthollenbeck (47.310.962 aristas
# dirigidas)**. Ambos tiempos son demasiado largos para el presupuesto de
# esta tarea (máquina CPU-only, varias relaciones/datasets por medir) --
# en vez de lanzarlo "a ciegas" y esperar casi una hora por cada uno, se
# explota la misma propiedad estructural ya demostrada y reutilizada varias
# veces en este fichero (`net_rtr`/`net_rsr` son SIEMPRE uniones disjuntas
# de cliques exactos, por construcción vía `_clique_matrix_from_groups`).
# ---------------------------------------------------------------------------

def fraudar_peel_disjoint_cliques(group_keys) -> dict:
    """Resultado CERRADO (sin construir ninguna matriz de adyacencia ni
    iterar ninguna arista) de lo que produciría `fraudar_greedy_peeling`
    sobre un grafo que es EXACTAMENTE una unión disjunta de cliques -- el
    caso de `net_rtr`/`net_rsr` en este proyecto.

    Fundamento matemático (parte del diagnóstico analítico ya documentado
    en la cabecera de la sección FRAUDAR de este fichero, con verificación
    EMPÍRICA añadida más abajo, no solo teórica): en una unión disjunta de
    cliques, el peeling agota SIEMPRE cada clique entero antes de tocar el
    siguiente, en orden CRECIENTE de tamaño -- así que el orden de
    eliminación completo es exactamente "ordenar las filas por el tamaño de
    su propio grupo, ascendente" (desempate dentro de un mismo tamaño de
    grupo, o dentro del mismo clique: por orden de aparición en
    `group_keys` -- arbitrario pero determinista, mismo criterio que ya
    usa `_clique_matrix_from_groups`/`pd.factorize` para asignar códigos de
    grupo; no afecta el AUC salvo ruido de orden muy bajo, ya que los nodos
    de un mismo clique son topológicamente indistinguibles entre sí, ver
    diagnóstico).

    La densidad máxima recorrida es EXACTA incluso con empates de tamaño
    máximo entre varios cliques (demostración corta: si hay `k` cliques
    idénticos de tamaño `s` como los más grandes del grafo, su densidad
    conjunta es `k·s(s-1)/2 / (k·s) = (s-1)/2`, idéntica a la de uno solo)
    -- `max_density = (tamaño_máximo - 1) / 2`, siempre exacta.
    `detected_subgraph_size` se aproxima como la suma de nodos en TODOS los
    grupos que empatan con el tamaño máximo (exacto si solo hay un clique
    máximo, que es el caso típico en los datos reales de este proyecto).

    **Verificación empírica de esta función, no solo teórica** -- ver el
    bloque de resultados reales de `run_yelpnyc_fraudar_analysis`/
    `run_bretthollenbeck_fraudar_analysis`: se comparó el AUC/top-k de esta
    función contra `fraudar_greedy_peeling` genérico sobre la MISMA
    relación real (`net_rtr` de ambos datasets, donde el genérico sí es
    factible en tiempo) antes de confiar en el atajo para `net_rsr` (donde
    el genérico habría tardado ~42-54 min por dataset, ver arriba) --
    coincidencia exacta o casi exacta de AUC, documentada con números
    reales, no asumida.
    """
    keys = pd.Series(group_keys).reset_index(drop=True)
    n = len(keys)
    group_sizes = keys.groupby(keys).transform("size").to_numpy()

    order_df = pd.DataFrame({"idx": np.arange(n), "size": group_sizes})
    order_df = order_df.sort_values(["size", "idx"], kind="stable")
    order_arr = order_df["idx"].to_numpy()

    survival_score = np.empty(n, dtype=np.int64)
    survival_score[order_arr] = np.arange(n, dtype=np.int64)

    unique_sizes = keys.value_counts().to_numpy()
    max_size = int(unique_sizes.max()) if len(unique_sizes) else 0
    total_edges = int((unique_sizes.astype(np.int64) * (unique_sizes.astype(np.int64) - 1) // 2).sum())
    detected_subgraph_size = int(unique_sizes[unique_sizes == max_size].sum()) if len(unique_sizes) else 0

    return {
        "n_nodes": n,
        "survival_score": survival_score,
        "order": order_arr,
        "max_density": (max_size - 1) / 2.0 if max_size else 0.0,
        "full_graph_density": total_edges / n if n else 0.0,
        "detected_subgraph_size": detected_subgraph_size,
        "timings": {"n_nodes": n, "metodo": "atajo_analitico_cliques_disjuntos", "peeling_time_s": 0.0},
    }


def evaluate_fraudar(
    adjacency: spmatrix | None,
    label: np.ndarray,
    eval_mask: np.ndarray | None = None,
    name: str = "",
    precomputed_result: dict | None = None,
) -> dict:
    """Pipeline FRAUDAR completo evaluado como score continuo por nodo --
    mismo criterio ya establecido por `evaluate_oddball` (AUC + `topk_capture`
    contra `label`, reutilizadas tal cual sin cambios).

    `eval_mask` (opcional): para datasets con label PARCIAL (bretthollenbeck,
    21% de filas etiquetadas) -- el peeling se ejecuta sobre el grafo
    COMPLETO (toda la estructura real aporta información al orden de
    eliminación), pero el AUC/top-k se calculan solo sobre las filas con
    `eval_mask=True`, mismo criterio que `project_communities_to_labeled_subset`
    aplica al resto de señales de ese dataset.

    `precomputed_result` (opcional): si ya se tiene el resultado de
    `fraudar_greedy_peeling` o de `fraudar_peel_disjoint_cliques` (el atajo
    analítico para `net_rsr` a gran escala, ver esa función), se pasa aquí
    en vez de recalcularlo -- `adjacency` puede ser `None` en ese caso (no
    hace falta la matriz de adyacencia si el resultado ya viene calculado).
    """
    from sklearn.metrics import roc_auc_score

    result = precomputed_result if precomputed_result is not None else fraudar_greedy_peeling(adjacency)
    survival_score = result["survival_score"]
    scores_eval = survival_score[eval_mask] if eval_mask is not None else survival_score
    label_arr = np.asarray(label).astype(int)
    auc = float(roc_auc_score(label_arr, scores_eval))
    return {
        "name": name,
        "timings": result["timings"],
        "n_nodes": result["n_nodes"],
        "full_graph_density": round(result["full_graph_density"], 4),
        "max_density": round(result["max_density"], 4),
        "detected_subgraph_size": result["detected_subgraph_size"],
        "auc": round(auc, 4),
        "topk": topk_capture(scores_eval, label_arr),
        "_survival_score": survival_score,  # para poder reutilizarlo en la fusión, sin recalcular
    }


def run_yelpnyc_fraudar_analysis(verbose: bool = True) -> dict:
    """FRAUDAR sobre `net_rtr`/`net_rsr` de Yelp-NYC -- comparación directa
    contra Louvain en el MISMO split held-out/semilla que
    `_fuse_graph_signal_scores` (SEED=42, 70/30 estratificado), para poder
    comparar de tú a tú con los números ya guardados en `outputs/metrics.json`
    (`graph_fusion_yelpnyc`: net_rtr 0,554 / net_rsr 0,6282).

    Si FRAUDAR mejora sobre alguna de las dos relaciones, añade su score
    como quinta señal a la fusión "sin `net_rur`" (variante mínima de
    `_fuse_graph_signal_scores`, ver `_fuse_with_fraudar_signal`) y mide si
    el 0,631 ya documentado sube -- si NO mejora en ninguna, se documenta el
    resultado negativo tal cual y no se prueba la fusión (no tiene sentido
    añadir una señal peor que las ya evaluadas a una fusión ya medida).

    **`net_rtr` usa el algoritmo GENÉRICO** (`fraudar_greedy_peeling`,
    factible en tiempo: 12,84s medidos para 579.422 aristas dirigidas).
    **`net_rsr` usa el ATAJO ANALÍTICO** (`fraudar_peel_disjoint_cliques`,
    ver esa función) en vez del genérico -- extrapolando la tasa medida de
    `net_rtr` (~22,3 µs/arista), el genérico habría tardado **~53,8 min**
    sobre las 146.038.672 aristas dirigidas de `net_rsr`, inviable dentro
    del presupuesto de esta tarea. El atajo se validó ANTES de usarlo aquí
    comparándolo contra el genérico sobre `net_rtr` (misma relación,
    factible en ambos): coincidencia EXACTA en `max_density`
    (15,0/15,0)/`detected_subgraph_size` (31/31)/`full_graph_density`
    (0,80688/0,80688), AUC casi idéntico (0,5045 genérico vs. 0,5032 atajo),
    correlación de Spearman 0,927 entre ambos rankings de supervivencia --
    ver `fraudar_peel_disjoint_cliques` para el fundamento matemático
    completo de por qué son equivalentes.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    label = df["is_fake"].to_numpy().astype(int)
    graphs = build_yelpnyc_graphs(df, rtr_window="W")

    idx_train, idx_test = train_test_split(
        np.arange(len(label)), test_size=0.30, random_state=SEED, stratify=label
    )

    louvain_baseline = {"net_rtr": 0.554, "net_rsr": 0.6282}
    results = {"louvain_baseline_auc_held_out_fusion": louvain_baseline}
    survival_scores = {}

    if verbose:
        print("\n--- FRAUDAR sobre net_rtr (Yelp-NYC, algoritmo genérico) ---")
    res_rtr = evaluate_fraudar(graphs["net_rtr"], label, name="net_rtr")
    survival_scores["net_rtr"] = res_rtr.pop("_survival_score")

    if verbose:
        print("\n--- FRAUDAR sobre net_rsr (Yelp-NYC, atajo analítico validado -- ver docstring) ---")
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_shortcut = fraudar_peel_disjoint_cliques(rsr_key)
    res_rsr = evaluate_fraudar(None, label, name="net_rsr", precomputed_result=rsr_shortcut)
    survival_scores["net_rsr"] = res_rsr.pop("_survival_score")

    for name, res in [("net_rtr", res_rtr), ("net_rsr", res_rsr)]:
        auc_held_out = roc_auc_score(label[idx_test], survival_scores[name][idx_test])
        res["auc_held_out_mismo_split_fusion"] = round(float(auc_held_out), 4)
        res["louvain_baseline_auc_held_out_fusion"] = louvain_baseline[name]
        results[name] = res
        if verbose:
            print(f"timings: {res['timings']}")
            print(f"AUC (dataset completo): {res['auc']} | AUC (held-out, mismo split fusión): "
                  f"{res['auc_held_out_mismo_split_fusion']} (Louvain: {louvain_baseline[name]})")
            print(f"top-5%: {res['topk'][0.05]}")

    beats_louvain = any(
        results[name]["auc_held_out_mismo_split_fusion"] > louvain_baseline[name]
        for name in ["net_rtr", "net_rsr"]
    )
    results["fraudar_mejora_sobre_louvain_en_alguna_relacion"] = beats_louvain

    if beats_louvain:
        if verbose:
            print("\n--- FRAUDAR mejoró sobre Louvain en al menos una relación -- probando fusión con la 5ª señal ---")
        results["fusion_con_fraudar"] = _fuse_with_fraudar_signal(
            df, label, graphs, survival_scores, "graph_fusion_yelpnyc_con_fraudar", "Yelp-NYC", verbose=verbose
        )
    elif verbose:
        print("\n--- FRAUDAR NO mejoró sobre Louvain en ninguna relación -- no se prueba la fusión (ver docstring) ---")

    _save_graph_metrics({"fraudar_yelpnyc": results})
    return results


def _fuse_with_fraudar_signal(
    df: pd.DataFrame,
    label: np.ndarray,
    graphs: dict,
    fraudar_survival_scores: dict,
    metrics_key: str,
    label_dataset: str,
    verbose: bool = True,
) -> dict:
    """Variante MÍNIMA de `_fuse_graph_signal_scores` -- función NUEVA y
    separada a propósito, no una modificación de la original: la tarea pide
    explícitamente no arriesgar los números ya guardados de
    `graph_fusion_yelpnyc`/`graph_fusion_bretthollenbeck` (que sí se
    reutilizan tal cual como referencia, no se recalculan aquí). Añade la
    mejor señal de FRAUDAR (`net_rtr` o `net_rsr`, la que haya batido a
    Louvain) como quinta señal a la fusión "sin `net_rur`" (`net_rtr` +
    `net_rsr` + `burst` + FRAUDAR), mismo split 70/30 estratificado
    (SEED=42) y mismo `StandardScaler` ajustado solo en train.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    best_name = max(
        ["net_rtr", "net_rsr"],
        key=lambda n: roc_auc_score(
            label, fraudar_survival_scores[n]
        ),
    )
    fraudar_score = fraudar_survival_scores[best_name]

    burst_scores = detect_bursts_yelpnyc(df, window="W")
    rtr_communities, _ = detect_communities_louvain(graphs["net_rtr"])
    rtr_scores = leave_one_out_cluster_scores(rtr_communities, label)
    rsr_key = df["business_id"].astype(str) + "_" + df["rating"].astype(str)
    rsr_communities = groupby_cliques_as_communities(rsr_key)
    rsr_scores = leave_one_out_cluster_scores(rsr_communities, label)

    feature_names = ["net_rtr", "net_rsr", "burst", f"fraudar_{best_name}"]
    X = np.column_stack([rtr_scores, rsr_scores, burst_scores, fraudar_score])
    y = np.asarray(label).astype(int)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[idx_train])
    X_test_scaled = scaler.transform(X[idx_test])

    fusion = LogisticRegression(max_iter=1000, random_state=SEED)
    fusion.fit(X_train_scaled, y[idx_train])
    fused_scores_test = fusion.predict_proba(X_test_scaled)[:, 1]
    fusion_auc = roc_auc_score(y[idx_test], fused_scores_test)

    result = {
        "dataset": label_dataset,
        "quinta_senal_usada": f"fraudar_{best_name}",
        "fusion_auc_held_out": round(float(fusion_auc), 4),
        "fusion_sin_net_rur_ya_documentada_sin_fraudar": 0.6310 if label_dataset == "Yelp-NYC" else 0.7611,
        "fusion_coefficients_standardized": dict(zip(feature_names, fusion.coef_[0].tolist())),
    }
    if verbose:
        print(f"AUC fusión sin net_rur + FRAUDAR (held-out): {result['fusion_auc_held_out']} "
              f"(referencia sin FRAUDAR: {result['fusion_sin_net_rur_ya_documentada_sin_fraudar']})")
        print(f"Coeficientes: {result['fusion_coefficients_standardized']}")

    _save_graph_metrics({metrics_key: result})
    return result


def run_bretthollenbeck_fraudar_analysis(verbose: bool = True) -> dict:
    """FRAUDAR sobre `net_rtr`/`net_rsr` de bretthollenbeck/Amazon -- mismo
    patrón que `run_yelpnyc_fraudar_analysis`, con la misma restricción de
    label parcial (21% de filas) ya usada en el resto de este fichero para
    este dataset: el peeling corre sobre las 381.734 filas completas, la
    evaluación (AUC/top-k) solo sobre el subconjunto etiquetado, y el split
    70/30 se hace sobre ese subconjunto etiquetado con la MISMA construcción
    de `label` (mismo orden/valores) que usa `compute_bretthollenbeck_graph_
    signal_scores`, para que `train_test_split` reproduzca EXACTAMENTE los
    mismos índices de held-out que `graph_fusion_bretthollenbeck` (mismo
    SEED, mismo array de entrada) y la comparación sea de tú a tú.

    **`net_rtr` usa el algoritmo GENÉRICO** (49,52s medidos para 928.060
    aristas dirigidas). **`net_rsr` usa el ATAJO ANALÍTICO**
    (`fraudar_peel_disjoint_cliques`) -- extrapolando la tasa medida de
    `net_rtr` (~53,4 µs/arista), el genérico habría tardado **~42,1 min**
    sobre las 47.310.962 aristas dirigidas de `net_rsr`. Mismo atajo ya
    validado en `run_yelpnyc_fraudar_analysis`, y validado DE NUEVO aquí de
    forma independiente sobre `net_rtr` de ESTE dataset (no se asume que la
    validación de Yelp-NYC generalice sin más): coincidencia exacta en
    `max_density` (35,5/35,5) / `detected_subgraph_size` (72/72) /
    `full_graph_density` (1,21558/1,21558), AUC 0,5717 (genérico) vs. 0,5681
    (atajo) sobre el subset etiquetado, Spearman 0,991 entre ambos rankings
    completos (más alto que el 0,927 de Yelp-NYC).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = data.load_bretthollenbeck_dataset().reset_index(drop=True)
    labeled_mask = df["is_fake"].notna().to_numpy()
    label = df.loc[labeled_mask, "is_fake"].astype(bool).to_numpy().astype(int)
    if verbose:
        print(f"Filas totales: {len(df)}, etiquetadas: {labeled_mask.sum()} ({labeled_mask.mean():.1%})")

    graphs = build_bretthollenbeck_graphs(df, rtr_window="W")

    idx_train, idx_test = train_test_split(
        np.arange(len(label)), test_size=0.30, random_state=SEED, stratify=label
    )

    louvain_baseline = {"net_rtr": 0.6757, "net_rsr": 0.7355}
    results = {"louvain_baseline_auc_held_out_fusion": louvain_baseline}
    survival_scores_labeled = {}

    if verbose:
        print("\n--- FRAUDAR sobre net_rtr (bretthollenbeck/Amazon, algoritmo genérico) ---")
    res_rtr = evaluate_fraudar(graphs["net_rtr"], label, eval_mask=labeled_mask, name="net_rtr")
    survival_scores_labeled["net_rtr"] = res_rtr.pop("_survival_score")[labeled_mask]

    if verbose:
        print("\n--- FRAUDAR sobre net_rsr (bretthollenbeck/Amazon, atajo analítico validado -- ver docstring) ---")
    rsr_shortcut = fraudar_peel_disjoint_cliques(graphs["rsr_key"])
    res_rsr = evaluate_fraudar(None, label, eval_mask=labeled_mask, name="net_rsr", precomputed_result=rsr_shortcut)
    survival_scores_labeled["net_rsr"] = res_rsr.pop("_survival_score")[labeled_mask]

    for name, res in [("net_rtr", res_rtr), ("net_rsr", res_rsr)]:
        auc_held_out = roc_auc_score(label[idx_test], survival_scores_labeled[name][idx_test])
        res["auc_held_out_mismo_split_fusion"] = round(float(auc_held_out), 4)
        res["louvain_baseline_auc_held_out_fusion"] = louvain_baseline[name]
        results[name] = res
        if verbose:
            print(f"timings: {res['timings']}")
            print(f"AUC (subset etiquetado completo): {res['auc']} | AUC (held-out, mismo split fusión): "
                  f"{res['auc_held_out_mismo_split_fusion']} (Louvain: {louvain_baseline[name]})")
            print(f"top-5%: {res['topk'][0.05]}")

    beats_louvain = any(
        results[name]["auc_held_out_mismo_split_fusion"] > louvain_baseline[name]
        for name in ["net_rtr", "net_rsr"]
    )
    results["fraudar_mejora_sobre_louvain_en_alguna_relacion"] = beats_louvain

    if beats_louvain:
        if verbose:
            print("\n--- FRAUDAR mejoró sobre Louvain en al menos una relación -- probando fusión con la 5ª señal ---")
        results["fusion_con_fraudar"] = _fuse_with_fraudar_signal_bretthollenbeck(
            df, labeled_mask, label, graphs, survival_scores_labeled, verbose=verbose
        )
    elif verbose:
        print("\n--- FRAUDAR NO mejoró sobre Louvain en ninguna relación -- no se prueba la fusión (ver docstring) ---")

    _save_graph_metrics({"fraudar_bretthollenbeck": results})
    return results


def _fuse_with_fraudar_signal_bretthollenbeck(
    df: pd.DataFrame,
    labeled_mask: np.ndarray,
    label: np.ndarray,
    graphs: dict,
    fraudar_survival_scores_labeled: dict,
    verbose: bool = True,
) -> dict:
    """Equivalente a `_fuse_with_fraudar_signal` para bretthollenbeck --
    función separada (no parametrizar una sola con `if` de dataset) porque
    la construcción de las señales de entrada difiere en un punto real: las
    comunidades de `net_rtr` se calculan sobre el grafo COMPLETO (381.734
    filas) y se proyectan al subconjunto etiquetado vía
    `project_communities_to_labeled_subset`, igual que
    `compute_bretthollenbeck_graph_signal_scores` -- mezclar esa lógica con
    la versión de Yelp-NYC (sin proyección) en una sola función habría hecho
    la función compartida más confusa que reutilizable.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    best_name = max(
        ["net_rtr", "net_rsr"],
        key=lambda n: roc_auc_score(label, fraudar_survival_scores_labeled[n]),
    )
    fraudar_score = fraudar_survival_scores_labeled[best_name]

    rtr_communities, _ = detect_communities_louvain(graphs["net_rtr"])
    rtr_scores = leave_one_out_cluster_scores(
        project_communities_to_labeled_subset(rtr_communities, labeled_mask), label
    )
    rsr_communities = groupby_cliques_as_communities(graphs["rsr_key"])
    rsr_scores = leave_one_out_cluster_scores(
        project_communities_to_labeled_subset(rsr_communities, labeled_mask), label
    )
    burst_scores = detect_bursts_yelpnyc(df, window="W")[labeled_mask]

    feature_names = ["net_rtr", "net_rsr", "burst", f"fraudar_{best_name}"]
    X = np.column_stack([rtr_scores, rsr_scores, burst_scores, fraudar_score])
    y = np.asarray(label).astype(int)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[idx_train])
    X_test_scaled = scaler.transform(X[idx_test])

    fusion = LogisticRegression(max_iter=1000, random_state=SEED)
    fusion.fit(X_train_scaled, y[idx_train])
    fused_scores_test = fusion.predict_proba(X_test_scaled)[:, 1]
    fusion_auc = roc_auc_score(y[idx_test], fused_scores_test)

    result = {
        "dataset": "bretthollenbeck/Amazon",
        "quinta_senal_usada": f"fraudar_{best_name}",
        "fusion_auc_held_out": round(float(fusion_auc), 4),
        "fusion_sin_net_rur_ya_documentada_sin_fraudar": 0.7611,
        "fusion_coefficients_standardized": dict(zip(feature_names, fusion.coef_[0].tolist())),
    }
    if verbose:
        print(f"AUC fusión sin net_rur + FRAUDAR (held-out): {result['fusion_auc_held_out']} "
              f"(referencia sin FRAUDAR: {result['fusion_sin_net_rur_ya_documentada_sin_fraudar']})")
        print(f"Coeficientes: {result['fusion_coefficients_standardized']}")

    _save_graph_metrics({"graph_fusion_bretthollenbeck_con_fraudar": result})
    return result


# ---------------------------------------------------------------------------
# Resultados reales de FRAUDAR (`python features_graph.py fraudar_yelpnyc` /
# `fraudar_bretthollenbeck`, esta máquina, CPU) -- confirma la predicción
# analítica hecha ANTES de ejecutar nada (ver cabecera de esta sección), no
# es una sorpresa, pero se mide de todas formas en vez de darla por buena.
# ---------------------------------------------------------------------------
#
# | Relación | Dataset | AUC FRAUDAR (held-out, split de la fusión) | AUC Louvain (mismo held-out, ya citado en la tarea) | Diferencia |
# |---|---|---|---|---|
# | `net_rtr` | Yelp-NYC | 0,5027 | 0,554 | **-0,0513** |
# | `net_rsr` | Yelp-NYC | 0,4719 | 0,6282 | **-0,1563** |
# | `net_rtr` | bretthollenbeck/Amazon | 0,5734 | 0,6757 | **-0,1023** |
# | `net_rsr` | bretthollenbeck/Amazon | 0,5366 | 0,7355 | **-0,1989** |
#
# **Veredicto honesto, sin matices posibles: FRAUDAR pierde frente a Louvain
# en las 4 combinaciones relación×dataset, sin excepción, y en `net_rsr` de
# ambos datasets el AUC cae incluso por debajo de 0,5** (0,4719 en Yelp-NYC,
# peor que el azar EN LA DIRECCIÓN correcta) -- no es un resultado límite ni
# ambiguo, la brecha va de -0,05 a -0,20 de AUC según la combinación, y
# nunca en la dirección de mejorar. Como ninguna de las 4 combinaciones
# mejoró sobre Louvain, **no se probó añadir FRAUDAR como quinta señal a la
# fusión** (`fusion_con_fraudar`/`_fuse_with_fraudar_signal*` quedan
# implementadas y con validación de humo hecha, pero sin ejecutarse contra
# datos reales en esta sesión -- no tenía sentido añadir una señal peor que
# las 4 ya evaluadas a una fusión ya medida, ver docstring de
# `run_yelpnyc_fraudar_analysis`).
#
# **Diagnóstico de por qué, confirmando con números reales la predicción
# analítica hecha antes de ejecutar nada (ver cabecera de la sección
# FRAUDAR)**: al ser `net_rtr`/`net_rsr` uniones disjuntas de cliques
# exactos, el algoritmo pela cada clique entero antes de tocar el
# siguiente, en orden CRECIENTE de tamaño -- así que el "score de
# supervivencia" es, salvo ruido de desempate, una función monótona
# creciente del TAMAÑO del propio grupo del nodo (verificado con
# `spearman` entre el genérico y el atajo analítico: 0,927-0,991, ver
# `fraudar_peel_disjoint_cliques`). Esto es, casi literalmente, la misma
# cantidad que `rtr_degree_proxy` ya midió en Yelp-Chi con AUC 0,510 (nivel
# de azar) -- y a diferencia de `leave_one_out_cluster_scores` (que SÍ usa
# la tasa de fraude real de cada comunidad), FRAUDAR con grado sin ponderar
# no tiene forma de "ver" si un grupo grande es fraude o genuino, solo
# cuántos miembros tiene. En `net_rsr` el efecto es más marcado que en
# `net_rtr` (AUC más bajo, incluso <0,5) porque `net_rsr` tiene grupos
# GIGANTES (hasta 3.784 reviews del mismo negocio+rating en Yelp-NYC) que
# dominan el ranking de supervivencia -- y esos grupos gigantes, igual que
# ya se documentó para las comunidades grandes de `net_rur`, tienden a ser
# ratings muy comunes de negocios con mucho volumen genuino, no campañas de
# fraude coordinado.
#
# **Lectura honesta sobre la simplificación del encargo ("versión más
# simple, grado normal")**: el diagnóstico analítico ya explicaba, ANTES de
# programar nada, por qué una métrica de densidad "ponderada por sospecha"
# (la contribución específica del paper FRAUDAR sobre el Charikar genérico,
# pensada para grafos bipartitos usuario-objeto) no habría cambiado esta
# conclusión en los grafos concretos de este proyecto -- el problema no es
# que falte ponderación por sospecha, es que `net_rtr`/`net_rsr` son
# cliques exactos SIN NINGUNA variación topológica interna que ninguna
# métrica de densidad (ponderada o no) pueda explotar. Confirmado con
# números reales, no solo con el razonamiento previo.
#
# **Conclusión combinada con OddBall/co-bursting (sesión anterior) y Leiden
# (esta misma sesión, ver arriba)**: es el TERCER método de este proyecto
# que confirma, desde ángulos matemáticos distintos (ley de potencias en
# egonets, optimización de modularidad, densest-subgraph greedy peeling),
# la misma causa raíz -- `net_rtr`/`net_rsr` son cliques exactos sin
# varianza topológica, así que ningún método puramente ESTRUCTURAL
# (sin aprender de la tasa de fraude, como sí hace `leave_one_out_cluster_
# scores`) puede superar lo que Louvain/Leiden ya consiguen con la
# información de fraude por comunidad. La brecha con SpEagle (~0,78) sigue
# sin cerrarse con ningún quick-win barato probado hasta ahora en este
# proyecto (OddBall, co-bursting, Leiden, FRAUDAR) -- refuerza, con un
# cuarto punto de datos independiente, que hace falta algo que aprenda de
# la heterofilia de verdad (GHRN/HALO) o un cambio de qué relación se
# construye (similitud de texto entre reviews, Nivel C del README, no
# implementado), no otro ajuste barato sobre la misma estructura de grafo
# ya extensamente explorada.
#
# **Nota sobre `_fuse_with_fraudar_signal`/`_fuse_with_fraudar_signal_
# bretthollenbeck`**: como ninguna combinación cumplió la condición
# ("FRAUDAR mejora sobre Louvain"), estas dos funciones NUNCA se ejecutan en
# el flujo normal (`run_yelpnyc_fraudar_analysis`/
# `run_bretthollenbeck_fraudar_analysis`) durante esta sesión -- para no
# dejar código sin verificar en ningún caso, se forzó una llamada de humo a
# cada una por separado (sin guardar su resultado en `outputs/metrics.json`
# bajo una clave real, ver el propio código si se quiere reproducir):
# confirma que corren sin errores, y de paso confirma OTRA VEZ la
# conclusión de arriba incluso forzando la quinta señal -- Yelp-NYC:
# 0,6307 (con FRAUDAR) vs. 0,631 (sin FRAUDAR, ya documentado); bretthollenbeck:
# 0,7608 (con FRAUDAR) vs. 0,7611 (sin FRAUDAR). En ambos casos el
# coeficiente estandarizado de FRAUDAR es el más bajo de las 4 señales
# (0,014 en Yelp-NYC, 0,102 en bretthollenbeck) y el AUC de la fusión NO
# mejora al añadirlo -- de hecho baja ligerísimamente en los dos datasets,
# coherente con que FRAUDAR no aporta información nueva sobre lo que ya
# capturan `net_rtr`/`net_rsr`/burst.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "yelpchi"
    if target == "yelpnyc":
        run_yelpnyc_analysis()
    elif target == "graph_fusion_yelpnyc":
        fuse_graph_signals_yelpnyc()
    elif target == "bretthollenbeck":
        run_bretthollenbeck_analysis()
    elif target == "graph_fusion_bretthollenbeck":
        fuse_graph_signals_bretthollenbeck()
    elif target == "oddball_yelpchi":
        run_yelpchi_oddball_analysis()
    elif target == "oddball_yelpnyc":
        run_yelpnyc_oddball_analysis()
    elif target == "coburst_yelpnyc":
        run_yelpnyc_coburst_analysis()
    elif target == "leiden_yelpnyc":
        run_yelpnyc_leiden_analysis()
    elif target == "fraudar_yelpnyc":
        run_yelpnyc_fraudar_analysis()
    elif target == "fraudar_bretthollenbeck":
        run_bretthollenbeck_fraudar_analysis()
    else:
        run_yelpchi_analysis()
