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

import time

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


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "yelpchi"
    if target == "yelpnyc":
        run_yelpnyc_analysis()
    else:
        run_yelpchi_analysis()
