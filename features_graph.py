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


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "yelpchi"
    if target == "yelpnyc":
        run_yelpnyc_analysis()
    elif target == "bretthollenbeck":
        run_bretthollenbeck_analysis()
    else:
        run_yelpchi_analysis()
