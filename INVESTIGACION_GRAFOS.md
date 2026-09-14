# Investigación: cómo mejorar la parte de grafo (2026-09-14)

> Documento de investigación puntual, no de bitácora — es una sesión de búsqueda web dirigida
> por el usuario ("estamos un poco pochos con los grafos, busca papers/ideas/datasets"), hecha
> por `agente-maestro` sin tocar código. Recoge papers, técnicas y datasets encontrados,
> priorizados por coste/impacto para este proyecto en concreto — no es una lista genérica de
> "lo que existe en el campo". Nada de esto está implementado todavía; es material de partida
> para decidir en qué invertir tiempo, no una tarea completada. Contexto de por qué surge esta
> sesión: ver `CONTEXTO.md`, sección "Decisión de priorización — sin acceso al ordenador de
> casa (2026-09-14, tarde)" — el usuario tenía dudas fundadas sobre si el grafo "sirve", justo
> antes de esta investigación.

## Estado real, actualizado tras probar cuatro de las ideas de este documento (2026-09-14, noche)

**OddBall, co-bursting, FRAUDAR y Leiden ya se probaron con código real — las cuatro salieron
negativas.** Detalle completo, números y diagnóstico en `CONTEXTO.md` (secciones "OddBall y
co-bursting — resultado real" y "FRAUDAR y Leiden — resultado real"). Resumen de una línea por
si se retoma este documento sin releer CONTEXTO.md: las cuatro confirman, desde ángulos
matemáticos distintos, que `net_rtr`/`net_rsr` son uniones de cliques exactos sin varianza
topológica interna — ningún método "barato" (sin entrenar nada) puede sacarle más señal a la
estructura tal cual está. La fusión de grafo (sin `net_rur`) sigue en 0,631 (Yelp-NYC) /
0,7611 (bretthollenbeck), por debajo del benchmark de SpEagle (~0,78). Las secciones de abajo
(rondas 1-4) se dejan tal cual se escribieron, como registro de lo que se pensó en el momento —
no se han tachado retroactivamente, pero a partir de aquí OddBall/co-bursting/FRAUDAR/Leiden
deben leerse como **ya probadas y descartadas**, no como candidatas pendientes.

## Diagnóstico: por qué el grafo va flojo, no es casualidad

Los propios resultados ya medidos en `features_graph.py` lo estaban diciendo sin que lo
llamáramos por su nombre: `net_rur` "gana" en AUC (0,814 en Yelp-Chi, 0,90-0,99 en
Yelp-NYC/bretthollenbeck) pero mide otra cosa (cuentas de usar-y-tirar / prolificidad, según el
propio diagnóstico ya documentado), mientras que `net_rtr`/`net_rsr` — la señal que de verdad
correspondería a "cuentas distintas coordinándose" — se queda floja (AUC 0,55-0,73).

Hay un motivo publicado para esto, no es mala suerte del dataset: **Louvain asume homofilia**
(nodos similares se conectan entre sí), pero los grafos de fraude son estructuralmente
**heterofílicos por diseño** — un estafador se camufla conectándose con reviews/cuentas
legítimas a propósito, para no formar un cluster obvio. Es el mismo problema que motivó el
propio CARE-GNN citado en el README (Dou et al. 2020, "camuflaje"), y desde entonces hay una
línea de investigación específica sobre esto:

- **[Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum
  (GHRN, WWW 2023)](https://dl.acm.org/doi/10.1145/3543507.3583268)** — trata la heterofilia
  como alta frecuencia en el espectro del grafo y poda aristas inter-clase. Código real y
  funcional: **[github.com/blacksingular/GHRN](https://github.com/blacksingular/GHRN)**.
- **[A Label-Free Heterophily-Guided Approach for Unsupervised Graph Fraud Detection
  (HUGE/HALO, 2025)](https://arxiv.org/abs/2502.13308)** — lo más relevante para este proyecto
  en concreto: una métrica de heterofilia (**HALO**) calculable **sin ninguna etiqueta**,
  pensada justo para el escenario de producción que se persigue aquí (sin ground truth de
  cliente real). Detalle técnico verificado: combina HALO con una arquitectura MLP-GNN
  (pérdida de ranking + pérdida de alineación asimétrica) — sí necesita entrenar una red, no es
  solo una fórmula cerrada, así que es más esfuerzo que las ideas de la sección siguiente.

## Ideas concretas, ordenadas por coste/impacto

### 1. OddBall — quick win, encaja con el estilo ya establecido del proyecto

[OddBall (Akoglu et al.)](https://github.com/williamcorsel/oddball-anomalies) es un método
clásico pero vigente: sin ML, sin etiquetas, sin GPU. Mira el "egonet" (nodo + vecinos a 1
salto) de cada nodo y detecta desviaciones de una ley de potencias esperada (aristas vs. nodos
del egonet) — los near-cliques y near-stars anómalos se salen de esa recta en escala log-log.
Es estadística pura, la misma filosofía que ya usa el proyecto (modelo nulo hipergeométrico,
z-scores, modelo de configuración de Newman). Además ataca directamente la limitación ya
documentada en `density_vs_configuration_model` (el ratio se vuelve numéricamente inestable
con clusters pequeños frente a un fondo enorme, ver aviso ya presente en esa función) — OddBall
compara contra una tendencia local estimada sobre *muchos* egonets, no un único valor esperado
global, así que en teoría debería ser más robusto justo en el caso que falla ahora.

Nota de implementación real (verificada, no solo el abstract): el repo de referencia espera un
fichero TSV de lista de aristas (`nodo1\tnodo2\tpeso`), no una matriz scipy ni un grafo de
networkx directamente — hay que convertir el formato antes, esfuerzo moderado, no trivial.

### 2. Redes de "co-bursting" en vez de `net_rtr` — para la burst detection débil (AUC ~0,53-0,55)

[Modeling Review Spam Using Temporal Patterns and Co-bursting Behaviors (Xu et al.),
usado sobre DianPing](https://arxiv.org/pdf/1611.06625): en vez de conectar reviews por
coincidencia exacta (mismo negocio+rating+ventana, que es lo que hace `net_rtr` ahora),
construye una red donde dos reviewers se conectan si su patrón de ráfagas está
**estadísticamente correlado en el tiempo** (no solo "cayeron en la misma semana"), usando un
test de significancia en vez de una igualdad exacta de bin temporal. Implementable con
pandas/scipy, sin deep learning — mismo nivel de esfuerzo que lo que ya existe en
`detect_bursts_yelpnyc`. Dado que la burst detection actual está casi a nivel de azar en los
dos datasets donde se ha medido (Yelp-NYC 0,53, bretthollenbeck 0,55 con el z-score genérico),
esto es candidato directo a reemplazar o complementar `net_rtr`/`detect_bursts_yelpnyc`.

### 3. Fusión texto+grafo inteligente — para cuando se retome la fusión T2+grafo

[Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph
Neural Network (FraudSquad, sept. 2026)](https://arxiv.org/abs/2510.01801) es casi un calco del
problema de este proyecto: generan reviews falsas con Llama3/Qwen2/DeepSeek sobre productos
reales de Amazon (su corpus propio es conceptualmente igual al de `own_corpus/`) y construyen
el mismo tipo de grafo (mismo usuario / mismo rating / misma ventana temporal) que aquí. Lo
importante: en vez de promediar el score de texto y el de grafo (que es lo que hace
`fuse_scores` en `features_graph.py` ahora mismo), usan una **puerta aprendida** que decide,
por instancia, cuánto pesar cada señal — arquitectura verificada:
`Hi = Xi + PReLU(Xi·β1 + Zi·β2)·β3` (embedding de texto + "embedding de riesgo" combinados de
forma aditiva con no linealidad, no concatenación). En sus ablations, la fusión ingenua
(concatenar/promediar) rinde claramente peor, y features de grafo manuales añaden poco (Δ
0,1-0,8%) frente a dejar que la topología del grafo haga el trabajo.

Dado que este proyecto ya se ha encontrado dos veces que "fusionar sin medir sale peor" (T1+T2+T3
en Fase 0, la combinación ponderada por rareza en Yelp-Chi — ver README, "Resultados de Fase 0" y
`features_graph.py`), este paper da una alternativa concreta ya validada en un problema casi
idéntico, en vez de reinventar la rueda cuando llegue el momento de fusionar T2 con la señal de
grafo (bloqueado ahora mismo a falta de `outputs/models/t2_deberta/`, ver CONTEXTO.md).

Nota honesta: el código/dataset del paper vive en un repo anónimo de revisión
(`anonymous.4open.science/r/FraudSquad-5389/`), no confirmado como reproducible ni descargado —
solo se ha leído el paper, no probado el código.

### 4. Multiplex fusion en vez de suma ponderada — para cuando se reintente combinar net_rur/rtr/rsr

`build_rarity_weighted_graph` (ponderar por rareza y sumar en una sola matriz) no mejoró nada
frente a la unión sin ponderar y tardó 6,3x más (ver docstring de ese módulo) — vale la pena
saber que hay un motivo publicado, no solo mala suerte: sumar relaciones en un único grafo
pierde la semántica de cada una antes de que el clustering vea nada. La línea de **multiplex
graph neural networks** (p. ej. [RestMGFN, Expert Systems with Applications
2024](https://www.sciencedirect.com/science/article/abs/pii/S0957417424024655)) mantiene cada
relación como una capa separada y aprende cómo combinarlas en vez de aplastarlas en una sola
matriz antes de clusterizar — mecánicamente distinto a lo ya probado y descartado, no la misma
idea con otro nombre.

## Datasets nuevos, no cargados hoy por `data.py`

- **[GADBench (NeurIPS 2023 Datasets & Benchmarks)](https://arxiv.org/abs/2306.12251)** —
  benchmark curado con datasets de fraude en grafo reales (no sintéticos). Su versión de
  **Amazon** (11.944 nodos, ~4,4M aristas, 9,5% de fraude, "review correlation") es distinta
  del McAuley Amazon-2023 ya usado (ese no trae etiqueta) — este sí, con features y grafo ya
  construidos: un cuarto/quinto dataset grafo+etiqueta junto a Yelp-Chi/NYC/bretthollenbeck. Su
  hallazgo publicado de que "tree ensembles simples con agregación de vecindario superan a GNNs
  sofisticados" también encaja con la conclusión ya repetida en este proyecto de que lo simple
  gana si no se mide lo contrario.
- **DianPing** (reseñas de restaurantes de Shanghái, ~2,76M reviews / ~633k usuarios,
  2011-2014) — plataforma y cultura distintas a Yelp/Amazon, con información temporal
  explotable para co-bursting (ver idea 2 arriba). Útil por el mismo motivo que se usaron 3
  generadores de IA distintos en Fase 0: generalizar entre plataformas, no solo entre
  generadores. No confirmado si trae fecha de creación de cuenta (relevante para Nivel B) —
  solo se verificó que trae fecha de review, usuario, rating y texto.
- **Xiaohongshu**, usado en [Detecting Fake Reviewer Groups in Dynamic Networks (marzo
  2026)](https://arxiv.org/html/2603.08332) — plataforma social-comercio china, banco de
  pruebas adicional si interesa el caso "reviews con componente social" más allá de e-commerce
  puro. Paper reciente (2026): modelo DS-DGA-GCN sobre grafos producto-review-reviewer, 89,8%
  accuracy en Amazon / 88,3% en Xiaohongshu — cifra de accuracy agregada, no TPR@FPR, leer con
  la misma cautela que el proyecto ya aplica a cifras agregadas de otros papers.
- El corpus de **FraudSquad** (reviews LLM sobre productos reales de Amazon + grafo, ver idea 3)
  — si su repo se hace público tras revisión, es directamente reutilizable como benchmark de
  fusión texto+grafo con generadores de IA modernos, no solo GPT-2.

Ningún dataset de esta lista confirma traer `reviewer_created_at` — el Nivel B del perfilado
sigue sin un dataset académico que lo valide (mismo límite honesto ya documentado en el README).

## Ideas más especulativas, para vigilar sin apostar tiempo todavía

- **[CAMERA — camuflaje semántico en grafos con atributos de texto
  (2026)](https://arxiv.org/pdf/2605.20032)**: fraude no supervisado en grafos con texto —
  relevante porque las reviews de este proyecto tienen texto Y grafo a la vez. Solo leído el
  abstract, no el método completo.
- **Sheaf-hypergraph consensus residuals** ([Detecting AI-Generated E-Commerce Reviews with
  Sheaf-Hypergraph Consensus Residuals](https://doi.org/10.3390/app16147071)): modela una
  review como nodo de un hipergrafo con varias relaciones a la vez (producto, rating, ventana,
  vecindario semántico) y mide cuánto se desvía del "consenso local" en cada una.
  Matemáticamente pesado (teoría de haces/sheaf), pero la idea de fondo — "¿esta review encaja
  con sus vecinos en CADA relación, o solo en algunas?" — es una generalización de lo que el
  proyecto ya hace a mano comparando `net_rur` vs. `net_rtr`/`net_rsr` por separado en vez de
  fundirlos. Prioridad baja: complejidad alta, no evaluado si es implementable sin infraestructura
  pesada.
- **LLM como razonador sobre el grafo** ([Can LLMs Find Fraudsters?,
  2025](https://arxiv.org/pdf/2507.11997), [LLM-Powered Text-Attributed GAD, nov.
  2025](https://arxiv.org/abs/2511.17584)): dado que el proyecto ya usa Claude/GPT-4o/Qwen
  intensivamente (`own_corpus/`), hay una línea de investigación activa sobre usar un LLM para
  razonar directamente sobre subgrafos y explicar por qué son anómalos — encajaría más de forma
  natural con `build_cluster_card` (ya genera texto explicativo a partir de evidencia
  estructurada) que como técnica de detección en sí. Idea de producto/explicabilidad más que de
  precisión.

## Ronda 2, misma sesión (2026-09-14, tarde-noche) — mining de subgrafos densos y alternativas a Louvain

Segunda pasada de investigación, mientras corría en paralelo la tarea delegada a
`agente-codigo` (OddBall + co-bursting, ver CONTEXTO.md). Esta ronda se centra en dos ángulos
que la primera no cubrió: (a) los límites conocidos de Louvain en sí mismo (no solo el problema
de heterofilia ya documentado arriba), y (b) una familia clásica de métodos — "dense subgraph
mining" — pensada desde el origen para grupos pequeños y densos camuflados en un grafo enorme,
que es literalmente la definición de una granja de bots.

### Límites de Louvain, más allá de la heterofilia

- **Resolution limit** (problema distinto y adicional al de heterofilia ya documentado arriba):
  la optimización de modularidad tiende a fusionar comunidades pequeñas dentro de otras más
  grandes cuando el grafo total es grande — cuanto mayor el grafo de fondo, más pequeñas y
  reales quedan "escondidas". Es exactamente el escenario de este proyecto (clusters de 10-15
  cuentas dentro de un grafo de 359k nodos en Yelp-NYC).
- **Comunidades mal conectadas o incluso desconectadas**: problema DISTINTO al resolution limit,
  demostrado empíricamente en [From Louvain to Leiden: guaranteeing well-connected communities
  (Traag et al., Scientific Reports 2019)](https://www.nature.com/articles/s41598-019-41695-z)
  — hasta un 25% de las comunidades que produce Louvain están mal conectadas, hasta un 16%
  completamente desconectadas (dos sub-grupos sin ninguna arista entre ellos, metidos en la
  misma "comunidad" por un artefacto del algoritmo, no por ninguna relación real). **Leiden**
  añade una fase de refinamiento que lo evita por construcción, y en la práctica es más rápido
  y de mejor calidad que Louvain. Es un candidato de coste muy bajo: mismo tipo de entrada
  (grafo/matriz de adyacencia), API parecida (`leidenalg` + `python-igraph`, no viene en
  `networkx` de serie) — se podría sustituir literalmente la llamada a
  `nx.algorithms.community.louvain_communities` en `detect_communities_louvain` y volver a
  correr el mismo harness de evaluación (`evaluate_communities`) ya existente, sin rediseñar
  nada más. No verificado si `leidenalg` está ya en el venv compartido — a comprobar antes de
  añadir la dependencia (regla ya fijada en `agente-codigo.md`).
- **[OSLOM — Order Statistics Local Optimization Method](https://arxiv.org/pdf/1012.2363)**:
  encaja filosóficamente mejor que Leiden con lo que ya hace el proyecto — en vez de optimizar
  modularidad y decidir después si el resultado es significativo (que es lo que hacen
  `enrichment_test`/`leave_one_out_cluster_scores` ahora, como un paso separado tras Louvain),
  OSLOM busca directamente comunidades que sean **estadísticamente improbables frente a un
  modelo nulo de configuración** (el mismo tipo de modelo nulo que ya usa
  `density_vs_configuration_model`) — la significancia está integrada en el propio algoritmo de
  búsqueda, no añadida a posteriori. Permite comunidades solapadas (una cuenta puede pertenecer
  a más de una red de coordinación a la vez, algo que Louvain no permite por diseño) y no sufre
  el resolution limit de la misma forma que la modularidad. Mayor esfuerzo de integración que
  Leiden (no hay binding directo a Python tan maduro, típicamente se usa el binario original en
  C++) — candidato de fase 2, no de "quick win".

### Dense subgraph mining — familia de métodos diseñada para esto exactamente

Esta es la familia de papers más directamente relevante encontrada en toda la sesión: en vez de
"clustering general + comprobar después si hay fraude" (el patrón actual del proyecto), estos
métodos buscan **directamente** el subgrafo más denso posible de forma robusta al camuflaje —
es decir, atacan justo el problema de heterofilia (idea 3 de la ronda 1) pero desde una familia
algorítmica completamente distinta (optimización combinatoria, no GNN ni espectral):

- **[FRAUDAR (Hooi et al., KDD 2016)](https://bhooi.github.io/papers/fraudar_kdd16.pdf)** — el
  paper fundacional de esta línea. Explícitamente diseñado para resistir camuflaje (un
  estafador añadiendo reviews/follows "normales" a propósito para diluir su densidad) y da
  **garantías teóricas** (cotas demostrables de cuánto puede ocultarse un fraude, no solo una
  medida empírica). Probado en producción real: un grafo de Twitter de 1.470 millones de
  aristas, detectó un subgrafo de más de 4.000 cuentas confirmadas como compradoras de
  followers. Sin deep learning, escalable.
- **[HoloScope (Liu et al.)](https://arxiv.org/pdf/1705.02505)** — extiende la idea de FRAUDAR
  fusionando **topología + ráfagas temporales en un único score**, en vez de tratarlas como dos
  señales separadas (que es justo lo que hace este proyecto ahora: `net_rtr` por un lado,
  `detect_bursts_yelpnyc` por otro, sin combinarlas). Verificado con más detalle vía WebFetch:
  no supervisado (sin etiquetas), basado en optimización de matrices (sin red neuronal),
  **evaluado explícitamente sobre Amazon y YelpChi** — los mismos datasets ya usados en este
  proyecto. Aviso de calibre: la cifra "AUC 0,85-0,95 en YelpChi/Amazon" viene de un resumen
  del propio paper hecho por la herramienta de fetch, no confirmada línea a línea contra la
  tabla original — no citar sin verificar contra el PDF, mismo criterio que ya aplica el
  proyecto a cifras de terceros (ver `print_reference_comparison()`). Si se confirma aunque sea
  parcialmente, es el candidato más directo de toda la sesión para atacar a la vez las dos
  señales flojas ya medidas (burst AUC ~0,53, `net_rtr`/`net_rsr` AUC 0,55-0,73).
- **[MRFS — Mining Rating Fraud Subgraph in Bipartite Graph (IEEE TCSS
  2024)](https://ieeexplore.ieee.org/iel7/6570650/10557215/10012333.pdf)** — más reciente,
  trabaja directamente sobre el grafo bipartito reviewer↔negocio (sin proyectar primero a
  reviewer-reviewer, que es el paso que hace este proyecto con `net_rur`/`net_rtr`/`net_rsr`) —
  coincide con el propio diseño ya citado en el README (Pacheco et al., ICWSM 2021). También
  pensado explícitamente contra camuflaje/cuentas secuestradas, escalabilidad lineal.
- **[CopyCatch (Facebook, Beutel et al.)](https://ai.meta.com/research/publications/copycatch-stopping-group-attacks-by-spotting-lockstep-behavior-in-social-networks/)**
  — detecta "lockstep" (varias cuentas actuando en fila, casi al mismo tiempo, sobre el mismo
  objetivo) partiendo de semillas conocidas de spammers (semi-supervisado, no sirve sin al
  menos algunos ejemplos etiquetados — aquí sí se podría usar Yelp-Chi/NYC como semillas).
  Limitación documentada por los propios autores y confirmada en literatura posterior: granjas
  más sofisticadas que evitan el lockstep exacto (variando ligeramente el timing) se le escapan
  — relevante porque el propio `hard_evasion` del corpus de `gpt-6-astra` sigue esa misma
  lógica de evitar patrones demasiado uniformes.

### Lectura priorizada actualizada (incorporando la ronda 2)

Con esta segunda ronda, el orden de prioridad cambia ligeramente frente a la ronda 1: antes de
OddBall/co-bursting (ronda 1, ya delegado a `agente-codigo`, ver resultado en CONTEXTO.md),
el **cambio más barato de todos es probar Leiden en vez de Louvain** — mismo harness de
evaluación ya existente, un cambio de una función, sin nueva dependencia conceptual (solo la
librería). Si Leiden por sí solo ya cambia los números de `net_rtr`/`net_rsr` de forma notable,
sería la señal más barata de que parte del problema era el propio algoritmo de clustering, no
solo la heterofilia. **HoloScope** es la apuesta más ambiciosa pero mejor dirigida de toda la
sesión (ronda 1 + ronda 2 juntas): ataca exactamente las dos señales más flojas ya medidas
(burst + coordinación estructural) a la vez, sin deep learning, y ya evaluado sobre datasets
que este proyecto ya tiene descargados.

## Ronda 3, misma sesión (2026-09-14, noche) — el propio SpEagle, foundation models y cómo validar sin ground truth

Tercera pasada, todavía esperando el resultado de la tarea delegada a `agente-codigo`. Esta
ronda cubre tres cosas que no habían salido: cómo funciona de verdad SpEagle (la referencia que
ya se cita en `features_graph.py` sin haberla implementado nunca), una vía real para desplegar
en un cliente nuevo sin recalibrar desde cero, y — quizás lo más importante de las tres — una
metodología concreta para el problema ya documentado en el README de que "no existe ningún
dataset con verdad de terreno de que este cluster sea una granja de bots".

### SpEagle no es solo una cifra de referencia — es implementable, y resolvería la fusión de paso

`print_reference_comparison()` cita SpEagle (~0,78 AUC) como referencia con salvedades, pero el
proyecto nunca ha implementado el método en sí, solo la cifra publicada. Mecanismo real,
confirmado por varias fuentes independientes (Akoglu et al., el paper original de FraudEagle
del que deriva SpEagle): modela reviewer-review-negocio como un grafo bipartito con un **Markov
Random Field**, asigna a cada nodo una probabilidad previa ("prior") a partir de un vector de
features (texto + metadatos), y propaga la sospecha entre nodos conectados mediante **Loopy
Belief Propagation (LBP)** — un algoritmo de inferencia aproximada, escalable linealmente, nada
de deep learning. Variantes relacionadas encontradas: **[ColluEagle](https://arxiv.org/pdf/1911.01690)**
(MRF por pares sobre reviewers que co-revisan) y **[FairJudge](https://arxiv.org/pdf/1703.10545)**
(fiabilidad de usuario en plataformas de rating, mismo tipo de propagación).

**Por qué esto es más interesante de lo que parece a primera vista para este proyecto en
concreto**: LBP combina de forma nativa una señal de texto (el "prior" de cada review, que
podría ser literalmente el score de T2 ya calculado) con la señal de grafo (la propagación entre
vecinos) **dentro de un único marco probabilístico coherente** — no es "calcular T2 por un lado,
calcular grafo por otro, y promediar los dos números al final" (que es el patrón ya descartado
dos veces en este proyecto: T1+T2+T3 en Fase 0, la combinación ponderada por rareza en Fase 1).
Implementar una versión propia de FraudEagle/SpEagle serviría dos propósitos a la vez: (1) una
comparación honesta con número propio en vez de una cifra citada con salvedades, y (2) un
mecanismo de fusión texto+grafo con fundamento distinto a `fuse_scores` (promedio lineal) y a la
puerta aprendida de FraudSquat (ronda 1) — un tercer enfoque, más clásico y más barato
computacionalmente, que probar antes de comprometerse a entrenar nada.

### Graph foundation models — para el problema real de "cliente nuevo, cero calibración"

**[AnomalyGFM (KDD 2025)](https://arxiv.org/html/2502.09254v2)**: un modelo pre-entrenado sobre
múltiples datasets de GAD (incluye redes sociales, finanzas, **co-review networks** —
explícitamente el mismo dominio de este proyecto) que generaliza a dominios completamente
nuevos **sin reentrenar ni afinar** — alinea "residuos de representación de nodo" contra
prototipos genéricos independientes del dominio concreto. Relevancia directa: el mayor riesgo
de producto ya documentado en el README es que Nivel B/C/D del perfilado "no son validables con
los datasets académicos disponibles... hasta un piloto real" — cada cliente nuevo llega con un
grafo distinto y cero histórico de fraude confirmado. Un foundation model de este tipo, si
cumple lo que promete, sería la única vía de tener una señal de grafo razonable el día 1 de un
cliente nuevo, sin esperar a acumular meses de datos propios para calibrar Louvain/modelos
nulos desde cero. **Nivel de confianza bajo todavía** — solo leído el abstract/resumen, no el
método completo ni verificado si el código es público o reproducible; es una apuesta a más
largo plazo, no un quick-win como Leiden.

### Inyección de anomalías sintéticas — la respuesta práctica al "no hay ground truth"

Esto no es un paper único sino una **metodología ya estándar en el campo** para exactamente el
problema que el README marca como límite honesto ("no existe ningún dataset con verdad de
terreno de que este cluster sea una granja de bots... se evalúa con precisión-en-top-k bajo
revisión manual, nunca con un AUC inventado"). La práctica habitual cuando no hay fraude
confirmado: tomar un grafo real (el de un cliente, sin ninguna etiqueta) e **inyectar
anomalías sintéticas estructurales** — seleccionar un grupo de nodos al azar y conectarlos
completamente entre sí (un clique artificial, simulando una coordinación perfecta) — para tener
al menos un suelo mínimo de "esto SÍ debería detectarse" contra el que medir precisión-en-top-k,
sin necesitar ninguna etiqueta real de fraude. **Aplicación directa y de bajo coste a este
proyecto**: antes de un piloto real, se podría tomar el grafo de un negocio cualquiera (o
sintético) y añadir clusters inyectados de distinta densidad/tamaño, para responder "¿el
pipeline actual (Louvain + `evaluate_communities` + `profile_cluster`) los encuentra en el
top-k, o se pierden entre el ruido genuino?" — una validación intermedia entre "solo datasets
académicos con label real" y "esperar a un piloto real con fraude confirmado", que hoy no
existe en el proyecto. Coste de implementación bajo: son unas pocas líneas sobre las matrices
sparse ya existentes, no una librería nueva.

### Embeddings bipartitos sin proyectar — mención breve, prioridad baja

**[BiNE](http://staff.ustc.edu.cn/~hexn/papers/sigir18-bipartiteNE.pdf)** y **metapath2vec**
aprenden embeddings directamente sobre el grafo bipartito reviewer↔negocio, sin pasar primero
por la proyección a reviewer-reviewer que hace este proyecto (`net_rur`/`net_rtr`/`net_rsr`) —
coincide con la idea de MRFS (ronda 2) de trabajar sobre el bipartito original. En las
comparativas encontradas, BiNE y metapath2vec rinden de forma casi idéntica entre sí (diferencia
<1 punto porcentual) y ambos métodos reconocen sufrir limitaciones parecidas a las de la
proyección clásica — no hay evidencia clara aquí de que esto resuelva el problema de heterofilia
ya diagnosticado en la ronda 1, así que queda como opción de prioridad baja, no como alternativa
clara a lo ya identificado.

## Ronda 4, misma sesión (2026-09-14, noche) — opciones caras, para cuando haya GPU otra vez

A petición explícita del usuario: todo lo de las rondas 1-3 está elegido a propósito para
funcionar en la máquina de trabajo (CPU-only). Esta ronda es la contraria — arquitecturas GNN
completas que sí necesitan entrenar de verdad, pensadas para cuando se retome el acceso al
ordenador de casa (RTX 5060 Ti, 16GB, ver CONTEXTO.md). Ninguna de estas se ha probado ni
verificado con código propio — son candidatas para cuando haya presupuesto de cómputo, no
recomendaciones cerradas.

### Nivel 1 — primer experimento serio con GNN entrenado (esfuerzo medio, ya evaluadas sobre YelpChi/Amazon)

Esta familia ataca el mismo problema de heterofilia ya diagnosticado en la ronda 1, pero
entrenando una red de verdad en vez de una fórmula cerrada — el paso lógico si Leiden/OddBall
(rondas 1-2) confirman que el problema es real pero no basta con arreglarlo sin aprendizaje:

- **[BWGNN — Beta Wavelet GNN](https://arxiv.org/pdf/2312.06441)**: parte de un hallazgo propio
  interesante — los nodos anómalos producen un "desplazamiento a la derecha" del espectro del
  grafo (la energía se concentra en frecuencias altas, no bajas). Diseña un filtro banda-pasante
  a partir de wavelets Beta para capturarlo. Evaluado explícitamente sobre YelpChi/Amazon —
  datasets que este proyecto ya tiene descargados, comparación directa posible sin adaptar nada
  del lado de datos.
- **[GAGA — Group Aggregation enhanced Transformer](https://arxiv.org/pdf/2302.10407)**: mete la
  información de la etiqueta directamente en la agregación de vecinos (con cuidado de evitar
  fuga de la propia etiqueta del nodo, "label leakage"). Cifra citada por varias fuentes
  independientes: hasta un 24% de mejora sobre otros detectores GNN en entornos de baja
  homofilia — no verificada línea a línea contra el paper original, pero repetida de forma
  consistente en varias fuentes secundarias.
- **PC-GNN**: pensado específicamente para el desequilibrio de clases (la clase fraude es
  minoritaria) — relevante porque Yelp-NYC/bretthollenbeck tienen justo ese problema (~10%/~21%
  de fraude, no 50/50). Benchmark ya citado en ronda 1: AUC 0,892 en YelpChi.
- **GTAN**: atención temporal + propagación de riesgo — combina grafo y tiempo en una sola red,
  relevante porque ya hay timestamps reales en Yelp-NYC/bretthollenbeck (`date`,
  `campaign_start_date`) sin explotar con una arquitectura que aprenda de ellos en vez de un
  z-score hecho a mano.

### Nivel 2 — grafos temporales dinámicos, para la burst detection débil de verdad

Todo el burst detection actual (`detect_bursts_yelpnyc`, z-score por ventana fija semana/día) es
una foto fija con "coarsening" temporal — agrupa en bins y pierde granularidad dentro de cada
bin. La familia de **Temporal Graph Networks** trata cada review como un evento en tiempo
continuo y mantiene un estado de memoria por nodo que se actualiza con cada evento, sin bins:

- **[TGN — Temporal Graph Networks](https://arxiv.org/abs/2404.00060)**: módulos de memoria por
  nodo que capturan dependencias de largo plazo sin necesidad de discretizar el tiempo en
  ventanas.
- **DySAT**: autoatención sobre grafos dinámicos (una foto por ventana, pero con atención entre
  fotos en vez de un z-score simple).
- **EvolveGCN**: una RNN genera los propios pesos de la GCN en cada paso de tiempo, para que la
  red "evolucione" con el grafo en vez de reentrenarse desde cero en cada ventana.

Encontrado un dato concreto (no verificado a fondo, de una fuente secundaria): modelos tipo
T-GCN/EvolveGCN mejoraron 12-18% sobre GNNs estáticas en evaluaciones offline, pero sufren el
mismo "coarsening temporal" que ya tiene `detect_bursts_yelpnyc` — los métodos de tiempo
continuo (TGN, DySAT) son el paso siguiente si eso no basta. Esfuerzo de implementación notable
(requiere reformular el pipeline de reviews como flujo de eventos, no como matrices estáticas) —
candidato de "gran inversión", no de una tarde.

### Nivel 3 — frontera de investigación, mayor riesgo/incertidumbre

- **[DiffGAD — Diffusion-based Unsupervised Graph Anomaly Detector (ICLR
  2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5ddc53810fcd805697905f40ffe9102d-Abstract-Conference.html)**:
  no supervisado (encaja con el escenario de producción sin ground truth), pero explícitamente
  **caro** — los propios modelos de difusión consumen mucha memoria de GPU y tienen tiempos de
  muestreo largos, según fuentes generales sobre difusión (no específico de este paper). Sería
  la opción más cara de todo el documento, y la de mayor incertidumbre de si compensa el coste.
- **HUGE/HALO con entrenamiento completo** (ronda 1) y **GHRN a escala completa** (ronda 1) — ya
  mencionados como técnicas de heterofilia, pero su versión con entrenamiento real de la parte
  MLP-GNN es la que de verdad necesita GPU, no solo la métrica HALO en sí (que sí es barata).
- **AnomalyGFM** (ronda 3) — el propio ajuste/uso del foundation model, si se confirma que el
  código es reproducible, es candidato de GPU también, aunque la promesa es precisamente no
  tener que reentrenar por cliente.

### Cómo elegir cuando llegue el momento

Mi sugerencia, no una decisión tomada: si solo hay tiempo para una cosa con GPU, **BWGNN o GAGA
sobre Yelp-Chi primero** (esfuerzo medio, ya evaluados sobre datos que ya tenéis, comparación
directa con los números ya medidos de Louvain) antes que las opciones de Nivel 2/3 — confirmar
primero si una GNN entrenada de verdad supera a Louvain+heterofilia-barata por un margen que
justifique la inversión, antes de ir a por grafos temporales dinámicos o difusión, que son
apuestas mayores con más incertidumbre.

## Recursos para no perder de vista (no un paper suelto, listas activas)

- **[safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers)**
  — lista curada y activa, específica de fraude en grafo (incluye sección de fraude en reviews).
- **[mala-lab/Awesome-Deep-Graph-Anomaly-Detection](https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)**
  — repo oficial de la survey TKDE 2025, con métodos no supervisados/self-supervised y detección
  a nivel de comunidad/subgrafo (no solo nodo), que es la unidad de interés real de este
  proyecto (clusters, no reviews sueltas).

## Lectura priorizada del maestro (opinión, no medida todavía)

**OddBall** (barato, sin ML, ataca la limitación ya documentada de
`density_vs_configuration_model`) y las **redes de co-bursting** (arreglan la burst detection
floja que ya se midió dos veces por debajo de 0,55 AUC) son los siguientes pasos con mejor
relación esfuerzo/resultado — se podrían probar sobre Yelp-NYC o bretthollenbeck sin tocar nada
del resto del pipeline. El enfoque de **heterofilia** (GHRN/HALO) es el que explica de verdad
por qué el grafo va flojo y merece una lectura seria, pero es más esfuerzo (entrenar una red,
no solo una fórmula). **FraudSquad** es la referencia a tener a mano cuando se retome la fusión
T2+grafo, porque ya resolvió con datos casi idénticos el mismo error que este proyecto ya ha
cometido dos veces (fusión ingenua peor que la señal sola).

Nada de esto está verificado con código propio todavía — es investigación de partida, no
resultados medidos. Antes de citar cualquier cifra de estos papers como comparación en
marketing o README, aplicar el mismo criterio que ya usa `print_reference_comparison()` en
`features_graph.py`: son números de naturaleza distinta, no una carrera justa sin más contexto.
