# Investigación de grafos — Ronda 5 (2026-09-14)

> Continuación de `INVESTIGACION_GRAFOS.md` (rondas 1-4). **No repite nada de allí**: heterofilia/
> GHRN/HALO, OddBall, co-bursting, Leiden, FRAUDAR, HoloScope, SpEagle, AnomalyGFM, inyección de
> anomalías sintéticas, BWGNN/GAGA/PC-GNN/GTAN y grafos temporales (TGN/DySAT/EvolveGCN) se dan por
> cubiertos. Cuando esta ronda toca algo ya mencionado antes (HALO, AnomalyGFM, inyección
> sintética) es **para corregirlo o profundizarlo con material nuevo**, y se dice explícitamente.
>
> Sesión de búsqueda web dirigida, sin tocar código. Nada de este documento está implementado ni
> medido con código propio.
>
> **Regla aplicada en todo el documento**: cada hallazgo dice si se ha verificado leyendo la fuente
> (`✅ verificado`), si viene solo de un resumen de búsqueda sin abrir el paper (`⚠️ sin verificar
> a fondo`), o si es razonamiento mío no publicado (`🔵 hipótesis propia`). No hay ningún link ni
> cifra inventados; los que no he podido abrir están marcados como tales.

## El diagnóstico que motiva esta ronda (medido por el maestro, 2026-09-14)

Las tres señales de grafo del proyecto (`net_rur`/`net_rtr`/`net_rsr`) **no son features de grafo:
son target encoding**. El score de cada nodo es la tasa de fraude de su comunidad calculada con las
etiquetas reales (vía `leave_one_out_cluster_scores`). Eso explica a la vez el AUC alto y la
inutilidad en producto: un cliente nuevo no tiene etiquetas.

Cifras honestas medidas en Yelp-NYC (split 70/30, base rate 10,27%):

| | AUC | AP |
|---|---|---|
| Fusión completa (protocolo sin fuga) | 0,9028 | 0,5961 |
| **Fusión sin `net_rur`** | **0,6231** | **0,1775** |

Ese 0,6231 / 0,1775 es el número a batir. Y el AP de 0,1775 sobre una base rate de 0,1027 es un
lift de solo 1,73x — mucho más elocuente que el AUC. Esta ronda se organiza alrededor de esas dos
preguntas: **(A) cómo evaluar sin engañarnos** y **(B) qué se puede detectar sin etiquetas**.

---

## Sección A — Evaluación: lo que la literatura ya denuncia

Esta es la sección con más valor inmediato, porque no requiere implementar ningún método nuevo:
son cambios de protocolo que se aplican sobre lo ya medido.

### A.1. El problema del proyecto tiene nombre en la literatura: fuga de etiquetas en un método "no supervisado"

**[Towards Automated Self-Supervised Learning for Truly Unsupervised Graph Anomaly Detection
(Li, Wang, van Leeuwen; DMKD, aceptado junio 2025)](https://arxiv.org/abs/2501.14694)** — versión de
revista: [doi.org/10.1007/s10618-025-01115-5](https://link.springer.com/article/10.1007/s10618-025-01115-5).
`✅ verificado` (leído el abstract completo).

Denuncia exactamente nuestro pecado, aunque en una variante más sutil: la mayoría de métodos SSL de
detección de anomalías en grafo se presentan como no supervisados pero **usan las etiquetas para
elegir hiperparámetros**, lo que los autores llaman literalmente *"label information leakage and
leads to severe overestimation of a method's performance"*. Evalúan 10 algoritmos SSL recientes y
proponen una estrategia de evaluación **interna** (sin etiquetas, con análisis teórico) para elegir
hiperparámetros.

**Por qué nos importa a nosotros**: nuestro caso es peor que el que denuncian (no usamos las
etiquetas para elegir un hiperparámetro, las usamos para *construir la feature*), pero el marco
conceptual y el criterio de "evaluación interna" es directamente reutilizable: si algún día
calibramos umbrales de `evaluate_communities`/abstención mirando el AUC contra la etiqueta, estamos
cometiendo la versión que denuncia este paper.

Coste: cero implementación, es un criterio. Sin GPU.

### A.2. Corrección importante a la ronda 3: la inyección de anomalías sintéticas TIENE fuga conocida

La ronda 3 recomendaba inyectar cliques sintéticos como "respuesta práctica al no hay ground
truth". **Hay que matizarlo fuerte**:

**[Unsupervised Graph Outlier Detection: Problem Revisit, New Insight, and Superior Method
(Huang, Wang, Zhang, Lin; arXiv 2210.12941, oct. 2022)](https://arxiv.org/abs/2210.12941)**
`✅ verificado`.

Cita textual del abstract: *"the most widely-used outlier injection approach has a serious data
leakage issue. By only utilizing such data leakage, a simple approach can achieve state-of-the-art
performance in detecting outliers."* El mecanismo es obvio una vez dicho: si inyectas un clique,
los nodos inyectados quedan con grado por encima de la media, así que **el grado solo ya los
detecta**; y si inyectas anomalías de atributo, la norma L2 del vector de atributos los detecta.
Proponen VGOD y recomiendan evaluar sobre datasets con anomalías reales, no inyectadas. El mismo
argumento lo repiten trabajos posteriores (p. ej. [Three Revisits to Node-Level Graph Anomaly
Detection, arXiv 2403.04010](https://arxiv.org/abs/2403.04010), `⚠️ sin verificar a fondo`).

**Consecuencia concreta para nosotros**: la idea de la ronda 3 sigue siendo útil como *test de
suelo mínimo* ("¿el pipeline encuentra al menos lo obvio?"), pero **no** como validación de que un
método es bueno, y sobre todo hay que reportar siempre, al lado, el resultado de la línea base
trivial (grado del nodo). Si el método propuesto no supera claramente al grado sobre datos
inyectados, no ha demostrado nada. Esto es barato de añadir y evita publicar una cifra inflada.

### A.3. Splits aleatorios sobre grafos: la trampa está documentada y es grande

**[When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural Networks for
Bitcoin Fraud Detection under Temporal Distribution Shift (Saket Maganti, arXiv 2604.19514, abril
2026)](https://arxiv.org/abs/2604.19514)** `✅ verificado`.

Es el paper más directamente incómodo (en el buen sentido) de toda la ronda. Bajo un protocolo
**estrictamente inductivo** (el modelo nunca ve aristas del periodo de test durante el
entrenamiento) sobre el dataset Elliptic:

- Random Forest sobre features crudas: **F1 = 0,821**
- GraphSAGE: **F1 = 0,689 ± 0,017**
- Un experimento controlado atribuye **39,5 puntos de F1** a la mera exposición, en entrenamiento,
  a la adyacencia del periodo de test.
- Y el remate: *"randomly wired graphs outperform the real transaction graph"* — un grafo cableado
  al azar bate al grafo real bajo cambio de distribución temporal.

El código estaba anunciado como "to be released soon" en el momento de la consulta, no confirmado
disponible.

**Encaje con nosotros**: nuestro split es 70/30 estratificado aleatorio sobre Yelp-NYC. En un grafo
donde las comunidades son grupos `(business_id, rating)`, **un split aleatorio pone miembros del
mismo grupo a ambos lados de la línea** — que es precisamente el canal por el que la tasa de fraude
del grupo (calculada con train) predice el label en test. No es un detalle: es el mecanismo entero
de la fuga. Un split temporal (entrenar con reviews anteriores a una fecha, evaluar con
posteriores) o un split por grupo (`GroupKFold` sobre `business_id`) daría el número honesto.

Complemento en la misma dirección: **[Leakage Safe Graph Features for Interpretable Fraud Detection
in Temporal Transaction Networks (Khaleghpour & McKinney, arXiv 2603.06632, feb.
2026)](https://arxiv.org/abs/2603.06632)** `✅ verificado` — define features de grafo "leakage safe"
como variantes causales que usan *"only edges observed up to each timestep"*, y evalúa con splits
temporales estrictos sobre Elliptic reportando ROC-AUC ≈0,85, **AP ≈0,54, precision@k, curvas de
calibración y Brier score**. Ese conjunto de métricas es justo el que pide el encargo. Aviso
honesto: el abstract **no** da la comparación numérica explícita entre features con fuga y sin
fuga, así que no se puede citar un "delta" concreto de este paper.

### A.4. Qué métricas usa de verdad la literatura seria

- **GADBench (NeurIPS 2023 D&B, [arXiv 2306.12251](https://arxiv.org/abs/2306.12251))** usa
  **AUROC + AUPRC + Rec@K** como terna estándar. Dato concreto muy útil: XGB-Graph supera a BWGNN
  en **+2,0% AUROC, +12,9% AUPRC, +9,8% Rec@K**, y los propios autores atribuyen la diferencia de
  magnitud al desequilibrio de clases. `⚠️ cifras tomadas de un extracto citado del paper, no
  comprobadas línea a línea contra la tabla original` — pero el patrón cualitativo (AUROC comprime
  las diferencias, AUPRC/Rec@K las amplifican) es exactamente lo que estamos viendo nosotros:
  0,9028 → 0,6231 en AUC parece una caída; 0,5961 → 0,1775 en AP muestra que es un desplome.
- **[GAD in the Wild: Benchmarking Graph Anomaly Detection under Realistic Deployment Challenges
  (Zhou et al., arXiv 2605.07133, mayo 2026)](https://arxiv.org/abs/2605.07133)** `✅ verificado` —
  evalúa tres condiciones de despliegue real: grafos de millones de nodos, escasez extrema de
  anomalías y atributos faltantes. Hallazgos: la mayoría de métodos basados en GNN **no escalan** a
  grafos de millones de nodos por memoria, y con ratios de anomalía realistas (p. ej. 0,1%) el
  rendimiento se desploma, *"sometimes yielding zero recall"*. Cinco grafos, dos de escala
  industrial (>3,7M nodos). Código en un repo anónimo de revisión
  (`anonymous.4open.science/r/Benchmark_GAD-E7A3`), no confirmado reproducible.
- El proyecto ya hace lo correcto en el lado de texto (TPR@1%FPR / TPR@5%FPR). **La incoherencia
  actual es reportar solo ROC-AUC en el lado de grafo.** No hace falta inventar nada: aplicar el
  mismo criterio en los dos lados.

### A.5. Target encoding sin fuga: cómo se hace bien, y por qué leave-one-out no basta

**[Interpretable versus Learned Encoders for High-Cardinality Fraud Detection (Han, Liu, Zheng,
Zhang, Wu; arXiv 2607.00477, julio 2026)](https://arxiv.org/abs/2607.00477)** `✅ verificado`.

Sobre IEEE-CIS Fraud Detection (590.540 registros, 3,5% positivos), el target encoding correcto se
hace con **cross-fitting out-of-fold (5 folds internos) + suavizado bayesiano**, fórmula explícita
del paper: `r̂ᵥ = (nᵥ·rᵥ + m·r̄)/(nᵥ + m)` con `m=30`, más un **test de permutación** para
confirmar invarianza a la fuga. Resultados: AUC-ROC 0,9612 (entity embeddings) / 0,9602 (CatBoost)
/ 0,9548 (agrupación en tiers interpretables); AUC-PR 0,8216 (CatBoost) vs 0,7928 (embeddings).
Lo interesante para nosotros es el hallazgo colateral: una versión **auditable** (binning de tasas
suavizadas en tiers ordinales, legible por un humano) pierde muy poco frente al encoder aprendido.

Y el punto que nos toca directamente: **leave-one-out target encoding — que es literalmente lo que
hace `leave_one_out_cluster_scores` — es un mitigante conocido pero insuficiente**. La
documentación de CatBoost lo explica con un ejemplo canónico: con LOO y target binario, todos los
objetos de clase 0 acaban codificados en un valor y todos los de clase 1 en otro, permitiendo a un
árbol partir exactamente en medio y acertar el 100% en train. Por eso CatBoost usa *ordered target
statistics* (permutación aleatoria + estadístico calculado solo con las filas anteriores), no LOO.
`⚠️ esto viene de documentación y tutoriales de CatBoost recogidos en búsqueda, no de un paper
arbitrado`; los tutoriales oficiales están en
[github.com/catboost/catboost](https://github.com/catboost/catboost/blob/master/catboost/tutorials/categorical_features/categorical_features_parameters.ipynb).

### A.6. El sesgo también existe en el lado de texto, y está cuantificado

**[Confounds and Overestimations in Fake Review Detection: Experimentally Controlling for
Product-Ownership and Data-Origin (Soldner, Kleinberg, Johnson; arXiv 2110.15130, 2021, rev.
2022)](https://arxiv.org/abs/2110.15130)** `✅ verificado`.

Accuracy de detección de reviews falsas según qué confusores se controlen:

| Condición | Accuracy |
|---|---|
| Solo veracidad (ambos confusores controlados) | 60,26–69,87% |
| + confusor de propiedad del producto | 66,19–74,17% |
| + confusor de origen de datos (mezclar fuentes) | 84,44–86,94% |
| Ambos confusores sin controlar | 87,78–88,12% |

**18-27 puntos porcentuales de inflación** por no controlar los confusores. Encaja de forma casi
literal con lo que este proyecto ya descubrió por su cuenta en Fase 0 (T1/T3 detectaban "GPT-2", no
"IA"): el "confusor de origen de datos" es exactamente el mecanismo por el que un clasificador
entrenado sobre Ott+Salminen aprende a distinguir *datasets*, no *veracidad*. Vale la pena citarlo
en el README como respaldo publicado de una decisión que ya se tomó a ciegas.

---

## Sección B — Detección sin etiquetas (o con muy pocas)

Aquí está la necesidad real del producto. Separo por si hace falta GPU, que es lo que determina qué
se puede hacer hoy en el portátil.

### B.1. SIN GPU — FreeGAD: detección de anomalías en grafo sin entrenar nada

**[FreeGAD: A Training-Free yet Effective Approach for Graph Anomaly Detection (Zhao, Liu, Li,
Chen, Zheng, Pan; CIKM 2025, arXiv 2508.10594)](https://arxiv.org/abs/2508.10594)** — código real:
**[github.com/yunf-zhao/FreeGAD](https://github.com/yunf-zhao/FreeGAD)**. `✅ verificado` (leída la
versión HTML completa, no solo el abstract).

Es la mejor noticia de esta ronda para el escenario "portátil sin GPU". Tesis del paper: *"the
training phase of deep GAD methods, commonly perceived as crucial, may actually contribute less to
anomaly detection performance than expected"*. Mecanismo: un **encoder de propagación con puerta de
afinidad** (sin pesos entrenados), selección de **nodos ancla** como referencias pseudo-normales y
pseudo-anómalas, y score por desviación estadística respecto a esos anclas.

Números sobre datasets que ya tenemos descargados:

| Dataset | AUROC | AUPRC |
|---|---|---|
| Amazon | 88,57% | 75,06% |
| YelpChi | 78,55% | **15,80%** |
| Reddit | 57,21% | — |

**Lectura honesta, y es importante**: el AUROC de 78,55% en YelpChi suena a "por fin alguien llega
al nivel de SpEagle", pero **el AUPRC de 15,80% dice la verdad** — con una base rate de ~14,5% en
YelpChi, eso es un lift ridículo. Es exactamente la lección de la sección A.4 aplicada a un paper
real: el mismo método parece excelente en Amazon (AUPRC 75%) y casi inútil en YelpChi. **Este es
probablemente el dato más útil de toda la ronda**: pone un techo realista a lo que cabe esperar de
cualquier método no supervisado sobre nuestro tipo de datos, y sugiere que nuestro 0,1775 de AP no
está tan lejos del estado del arte no supervisado como parecía.

Requisitos: **necesita atributos de nodo** (matriz `X`). Yelp-Chi los tiene (32 features); Yelp-NYC
tal como lo tenemos, no — habría que construirlos. Tiempos de test reportados: 0,0141–0,3142 s.
Hiperparámetros a fijar: α, β (0-1), L (capas de propagación, típicamente ≥8), K (anclas, 10-100).
GPU **no requerida explícitamente**. Coste de implementación: **bajo-medio** (código público,
pero hay que adaptar el formato de datos y construir atributos para Yelp-NYC).

⚠️ Y ojo con el hiperparámetro: elegir α/β/L/K mirando el AUC contra la etiqueta sería cometer
exactamente lo que denuncia A.1. Hay que fijarlos a los valores por defecto del repo, o elegirlos
con un criterio interno.

### B.2. SIN GPU — Redes estadísticamente validadas (SVN): la que mejor encaja con la cultura del proyecto

**[Statistically Validated Networks in Bipartite Complex Systems (Tumminello, Miccichè, Lillo,
Piilo, Mantegna; PLOS ONE 2011, 6(3):e17994)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017994)**
`✅ verificado` (leído el artículo completo).

No es un paper de fraude, y por eso no ha salido en las rondas anteriores — pero es
**metodológicamente el más cercano a lo que este proyecto ya hace** (`enrichment_test`,
`density_vs_configuration_model`, modelo nulo hipergeométrico) y ataca de frente el problema
estructural diagnosticado: `net_rtr`/`net_rsr` son uniones de cliques exactos sin varianza interna.

Mecanismo: dado un sistema bipartito (aquí: reviewer ↔ negocio), en vez de proyectar y conectar
*todo* par que comparte un vecino, **se valida cada arista de la proyección contra un nulo
hipergeométrico** de co-ocurrencia aleatoria, con corrección por comparaciones múltiples
(Bonferroni, más conservador; FDR, menos). Detalle clave que lo hace válido aquí: descomponen el
sistema en subsistemas homogéneos por grado antes de testear, para que la heterogeneidad de grado
(unos reviewers con 1 review, otros con 200) no genere falsos positivos.

Cuánto filtra, en su caso de actores/películas: solo el **1% (Bonferroni) / 7% (FDR)** de las
aristas de la red de adyacencia sobreviven. Y el resultado que nos interesa: al aplicar detección
de comunidades **sobre la red validada** aparecen clusters con interpretación directa, cosa que no
pasa sobre la red de adyacencia cruda.

**Por qué encaja con nuestros datos**: el grafo resultante **ya no es una unión de cliques
exactos** — es una red ponderada con varianza topológica real, que es justo lo que faltaba para que
Louvain/Leiden/OddBall/FRAUDAR tuvieran algo que morder (las cuatro fracasaron por la misma causa).
Y es **completamente sin etiquetas**: el nulo es de co-ocurrencia, no de fraude.

Coste: **bajo**. `scipy.stats.hypergeom` + matrices sparse que ya existen; el propio proyecto ya
demostró que `A @ A` de scipy sobre `net_rsr` de Yelp-NYC termina en ~286s con pico 6,45GB. Sin
GPU. **Riesgo real y honesto**: si en Yelp-NYC la mayoría de reviewers tienen 1-2 reviews, casi
ninguna co-ocurrencia sobrevivirá a Bonferroni y la red validada saldrá vacía o casi. Se sabe en
una tarde midiendo la distribución de reviews por reviewer antes de implementar nada.

Complemento útil: **[A primer on statistically validated
networks](https://www.researchgate.net/publication/331221143_A_primer_on_statistically_validated_networks)**
`⚠️ solo localizado por búsqueda, no leído`.

### B.3. SIN GPU (al inferir) — HUGE/HALO: corrección a la ronda 1, el código SÍ existe

La ronda 1 citaba HALO diciendo "sí necesita entrenar una red". Sigue siendo cierto, pero faltaba
un dato: **el código oficial está publicado**.

**[A Label-Free Heterophily-Guided Approach for Unsupervised Graph Fraud Detection (AAAI 2025,
arXiv 2502.13308)](https://arxiv.org/abs/2502.13308)** — código:
**[github.com/CampanulaBells/HUGE-GAD](https://github.com/CampanulaBells/HUGE-GAD)**, con comandos
de reproducción para **Amazon, Facebook, Reddit, YelpChi, AmazonFull y YelpChiFull**. Versión de
revista: [ojs.aaai.org/index.php/AAAI/article/view/33356](https://ojs.aaai.org/index.php/AAAI/article/view/33356).
`⚠️ el repo se ha localizado por búsqueda pero no se ha clonado ni ejecutado`.

Que traiga comandos listos para **YelpChi y Amazon** — los dos datasets que ya tenemos — reduce
mucho el coste de probarlo: no hay que adaptar el pipeline de datos. Entrenar la parte MLP-GNN pide
GPU en la práctica, pero la red es pequeña comparada con un DeBERTa; **probarlo en CPU sobre
Yelp-Chi (45.954 nodos) es plausible aunque lento**, y merece una prueba de humo antes de darlo por
bloqueado.

### B.4. SIN GPU — TAM: el "one-class homophily", complementario a lo ya probado

**[Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection (Qiao
& Pang, NeurIPS 2023, arXiv 2306.00006)](https://arxiv.org/abs/2306.00006)** — código:
**[github.com/mala-lab/TAM-master](https://github.com/mala-lab/TAM-master)**. `⚠️ verificado el
repo y el abstract por búsqueda, no leído el paper completo`.

Idea: en datasets reales de GAD se observa **"one-class homophily"** — los nodos normales tienen
afinidad fuerte entre sí, mientras que la homofilia de los anómalos es significativamente más
débil. El score no supervisado es la **afinidad local** del nodo con sus vecinos; TAM aprende
representaciones maximizando esa afinidad sobre grafos truncados (elimina iterativamente aristas no
homofílicas). Reportan >10% de mejora en AUROC/AUPRC sobre 7 competidores en 10 datasets reales.

**Aviso de escala verificado independientemente**: el propio paper de FreeGAD (B.1) reporta **TAM
con OOM (out of memory) en YelpChi**. Nuestro proyecto ya se ha chocado tres veces con el muro de
memoria en estos mismos grafos — asumir que TAM correrá en el portátil sería optimista. Prioridad
media, por detrás de FreeGAD.

### B.5. CON GPU (entrenamiento) / SIN GPU (inferencia) — UNPrompt: zero-shot a un grafo nunca visto

**[Zero-shot Generalist Graph Anomaly Detection with Unified Neighborhood Prompts (Niu et al.,
IJCAI 2025, arXiv 2410.14886)](https://arxiv.org/abs/2410.14886)** — PDF de actas:
[ijcai.org/proceedings/2025/0359.pdf](https://www.ijcai.org/proceedings/2025/0359.pdf); código
anunciado en **[github.com/mala-lab/UNPrompt](https://github.com/mala-lab/UNPrompt)**.
`✅ verificado` (leída la versión HTML completa).

Es la respuesta más directa al problema de producto "cliente nuevo, cero etiquetas", y más madura
que AnomalyGFM (ronda 3, del que solo se había leído el abstract). Mecanismo: unificación de
atributos entre grafos distintos (proyección + normalización por coordenadas) y **predictibilidad
de los atributos latentes de un nodo a partir de sus vecinos** como medida genérica de anomalía,
con prompts aprendidos compartidos.

Números (7 datasets: Facebook, Reddit, Weibo, **Amazon, YelpChi**, Amazon-all, YelpChi-all):

| Escenario | AUROC medio | AUPRC medio |
|---|---|---|
| Zero-shot generalista (entrenado en Facebook, testeado en los otros 6) | 0,6853 | **0,2219** |
| No supervisado (por dataset) | 0,7267 | — |

**Lectura honesta y muy relevante para nosotros**: 0,6853 de AUROC zero-shot es el estado del arte
publicado para "llegar a un grafo nuevo sin etiquetas", y **está por debajo de nuestro 0,9028 y
apenas por encima de nuestro 0,6231**. El AUPRC medio de 0,2219 está en el mismo orden que nuestro
0,1775. Es decir: **nuestro número honesto sin `net_rur` no está lejos del estado del arte no
supervisado del campo**. Eso cambia bastante la lectura de "estamos pochos con los grafos" — no
estamos pochos frente a lo que de verdad se puede hacer sin etiquetas; estábamos comparándonos
contra un target encoding supervisado nuestro y contra el ~0,78 de SpEagle, que es semi-supervisado.

Requisitos: entrenamiento en GPU (usaron una Nvidia A40). **Inferencia sobre un grafo nuevo solo
necesita los prompts aprendidos, la red preentrenada y la capa de transformación, sin reentrenar ni
etiquetas** — así que, si publican pesos, el uso en producto sería CPU-viable. No confirmado si hay
checkpoints publicados.

### B.6. CON GPU — CAMERA: texto + grafo, no supervisado (profundiza la ronda 1)

**[CAMERA: Adapting to Semantic Camouflage in Unsupervised Text-Attributed Graph Fraud Detection
(Pan, Liu, Zheng, Chi, Liew, Pan; arXiv 2605.20032, mayo 2026)](https://arxiv.org/abs/2605.20032)**
`✅ parcialmente verificado` (descargado el PDF y leídas secciones; no he podido extraer las tablas
numéricas concretas).

La ronda 1 lo listó como "solo leído el abstract". Lo confirmado ahora: **es no supervisado, sin
etiquetas**, modela conjuntamente estructura de grafo y atributos de texto, y evalúa sobre
**YelpChi y Amazon (ambos con texto de review)** con AUROC y AUPRC. El problema que ataca —
"camuflaje semántico": el estafador escribe texto que *parece* legítimo mientras mantiene el patrón
de red malicioso — es literalmente el escenario `hard_evasion` de nuestro corpus de `gpt-6-astra`
trasladado al grafo.

No he podido confirmar disponibilidad de código ni requisitos de GPU. Mismos autores que FreeGAD y
HUGE (grupo de Shirui Pan / mala-lab), que sí publican código habitualmente — señal indirecta
favorable, no una confirmación.

---

## Sección C — Señales de coordinación que no dependen de cliques

El problema estructural ya está diagnosticado y confirmado cuatro veces: `net_rtr`/`net_rsr` son
uniones de cliques exactos, sin varianza topológica. Ningún método topológico puede sacar nada.
Estas son las alternativas que **cambian la señal, no el algoritmo**.

### C.1. El survey que organiza todo este espacio

**[Detection and Characterization of Coordinated Online Behavior: A Survey (Mannocci, Mazza,
Monreale, Tesconi, Cresci; arXiv 2408.01257, v1 ago. 2024, v2 abril 2026)](https://arxiv.org/abs/2408.01257)**
`✅ verificado` (leída la versión HTML v2).

Es la referencia de conjunto que faltaba en las rondas 1-4. Define cuatro dimensiones de
coordinación: **autenticidad, nocividad, orquestación** (de centralizada a descentralizada) y
**varianza temporal**. Y — esto es lo que más nos sirve — su sección 4.1.3 clasifica los tres
enfoques de filtrado de redes de coordinación:

1. **Umbrales fijos** (descartar aristas con `w < w_th`). El survey los critica explícitamente:
   *"similarity and centrality thresholds are typically selected arbitrarily, without strong
   underlying theoretical motivation"*.
2. **Validación estadística** (backbone extraction, poda por centralidad de vector propio) — que es
   exactamente la vía de SVN (B.2).
3. **Filtrado temporal**: ventanas adyacentes, solapamiento uniformemente distribuido, solapamiento
   guiado por acción.

Sobre evaluación reconoce lo que a nosotros nos pasa: *"the field still lacks a formal and
statistically grounded definition of coordination, including well-defined null models"* (sec. 2.3)
y propone usar la **caracterización** de los grupos detectados para validar la **detección** cuando
no hay ground truth (sec. 3.2.2) — que es conceptualmente lo que ya hace `build_cluster_card`.

Detalle que confirma nuestra experiencia: en los métodos revisados, **Louvain aparece más de 15
veces y Leiden en varios trabajos recientes** como algoritmos estándar para redes de coordinación.
No estábamos usando una herramienta rara; el problema era la construcción de la red, no el
clustering. Eso refuerza la conclusión de la ronda anterior con Leiden.

### C.2. DeFrauder — detección de grupos de reviewers fraudulentos sin etiquetas

**[Spotting Collective Behaviour of Online Frauds in Customer Reviews (Dhawan, Gangireddy, Kumar,
Chakraborty; arXiv 1905.13649, 2019)](https://arxiv.org/abs/1905.13649)** `✅ verificado` (abstract
y metadatos).

No supervisado, pensado explícitamente para *"scarcity of labeled group-level spam data"*. Dos
fases: (1) detectar **grupos candidatos** sobre el grafo producto-review incorporando varias
señales de comportamiento que modelan colaboración multifacética entre reviewers, y (2) mapear
reviewers a un espacio de embedding y asignar un **spam score por grupo**, alto cuando los miembros
tienen trazas de comportamiento muy similares. Evalúan sobre 4 datasets reales (2 propios), con
**+17,11% NDCG@50 de media** sobre el mejor baseline.

Dos cosas que lo hacen relevante más allá del método:
- **La unidad de evaluación es el grupo, no la review**, y la métrica es **NDCG@50** — ranking, no
  AUC. Eso es exactamente lo que el README de este proyecto dice que quiere ("precisión en top-k
  bajo revisión manual, nunca un AUC inventado") y encaja con `profile_cluster`/`build_cluster_card`
  mucho mejor que el LOO-AUC por nodo que usamos ahora.
- Da un catálogo de "group spam indicators" conductuales que no dependen de topología.

⚠️ El abstract no menciona código disponible, y no he encontrado repo oficial. Es de 2019, así que
implementarlo desde el paper es asumible pero no trivial. Sin GPU.

### C.3. Sincronía temporal fina, no bins de semana

Dos líneas, ambas por encima de lo que hace `detect_bursts_yelpnyc` (z-score sobre bins fijos):

- **Burstiness por coeficiente de variación de tiempos entre llegadas**, normalizado al intervalo
  [-1, 1]. Es una fórmula cerrada, sin bins, sin entrenar nada — sustituto directo y barato del
  z-score por ventana. `⚠️ recogido de fuentes secundarias en búsqueda, no de un paper concreto
  leído`; es una medida clásica bien establecida (B de Goh & Barabási), pero no he verificado una
  referencia primaria en esta sesión.
- **[Temporal burstiness and collaborative camouflage aware fraud detection (TBCCA), Information
  Processing & Management](https://www.sciencedirect.com/science/article/abs/pii/S0306457322002710)**
  `⚠️ sin verificar a fondo, solo resumen de búsqueda; artículo de pago`. Descubre features de
  burstiness temporal ocultas *detrás de estrategias de camuflaje*, a partir de series temporales
  de reviews — combina precisamente los dos problemas que tenemos (camuflaje + tiempo).
- **[Graph embedding and clustering collaborative optimization model for fraudster group detection
  (IPM 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0306457325004030)**
  `⚠️ sin verificar, de pago`. Construye un grafo espaciotemporal de usuarios donde el peso de
  arista sale de una **divergencia de Jensen-Shannon** entre las distribuciones de cada par de
  usuarios (features temporales + rating), luego LINE para embeddings y **HDBSCAN** para clusters.
  Mecánicamente interesante para nosotros: la divergencia JS da un peso **continuo** en vez del
  match exacto, que es justo lo que rompe la estructura de clique.

### C.4. Pseudo-cliques con densidad ponderada, en vez de cliques exactos

De **[FairPlay: Fraud and Malware Detection in Google Play (arXiv 1703.02002)](https://arxiv.org/abs/1703.02002)**
`⚠️ verificado solo por extractos de búsqueda`: construyen un **co-review graph** donde los nodos
son usuarios y el peso de la arista es **el número de apps reseñadas en común**. Y relajan
explícitamente la búsqueda de cliques perfectos (NP-hard) a **"pseudo-cliques"**: subgrafos cuya
**densidad ponderada** (suma de pesos dividida por el número de pares posibles) supera un umbral.

Esto es exactamente el cambio de construcción que nos falta: pasar de "misma
`(business_id, rating)` → arista" a "**número de negocios co-reseñados** → peso". Y se combina de
forma natural con SVN (B.2), que es la versión con test estadístico del mismo grafo en vez de un
umbral arbitrario — que es justo lo que el survey de C.1 critica de los umbrales fijos.

---

## Sección D — Datasets con mejor verdad de terreno

### D.1. Confirmación publicada de que nuestras etiquetas de Yelp son un proxy débil

**[Social Fraud Detection Review: Methods, Challenges and Analysis (arXiv
2111.05645)](https://arxiv.org/abs/2111.05645)** `⚠️ citas tomadas de extractos de búsqueda, no del
PDF completo`. Dos afirmaciones recogidas que respaldan lo que el proyecto ya sospechaba:

- Los datasets etiquetados por el propio detector de la plataforma (Yelp) se llaman **"near
  groundtruth"**, y sus etiquetas se usan sobre todo en enfoques semi-supervisados o no
  supervisados.
- *"even human labeling accuracy is no better than a random classifier"* para etiquetar reviews
  fraudulentas.

Implicación honesta: **un techo de AUC contra Yelp no mide "detectar fraude", mide "reproducir el
filtro de Yelp"**. Un método que supere al filtro de Yelp en fraude real aparecería en esta métrica
como peor, no mejor. Vale la pena decirlo así en el README.

### D.2. Public Traces / Overgoor et al. — el ground truth más fuerte que he encontrado

**[Leveraging Public Traces to Monitor Fake-review Campaigns (Overgoor, Tosyali, Feldman,
Bhattacherjee; junio 2026)](https://scholar.smu.edu/business_marketing_research/60/)** —
[SSRN 5156231](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5156231) (la página de SSRN
devuelve 403 a fetch automático; el abstract se ha verificado vía la página de SMU).
Sitio del proyecto: **[publictraces.net](https://publictraces.net/)** `✅ verificado`.

Método que usa **solo datos públicos** (sin acceso interno a la plataforma) para detectar campañas
de reviews compradas, validado contra observación independiente de grupos de brokers en Facebook:

- Recuperan el **92,2%** de los reviewers reclutados observados de forma independiente.
- De los **top-100 bursts de reviews sospechosos** que produce el método, el **87%** corresponde a
  productos observados de forma independiente en campañas de reclutamiento en los grupos de
  Facebook.
- Escalado al corpus de Amazon 2023: **~1 de cada 8 reviews de 5 estrellas con compra verificada**
  y casi 1 de cada 5 de 5 estrellas caen en clusters de alta evidencia.

**Por qué es oro para nosotros**: ese 87% en top-100 es *precision@100 contra fraude confirmado por
observación externa*, no contra un filtro de plataforma. Es la métrica y el tipo de validación que
el README de este proyecto dice que quiere y que hasta ahora no existía en ninguna fuente
disponible. Y el enfoque "solo datos públicos, sin acceso a la plataforma" es literalmente el
escenario B2B del producto.

**Límite honesto, verificado**: el sitio **no publica el dataset**. Ofrece una herramienta de
investigación en el navegador, ejemplos trabajados, y la implementación en Python del working paper
sobre **registros generados** (no los datos reales). Dicen que *"shared submissions and reviewed
community datasets are planned for a later release"*. Así que hoy es una referencia metodológica y
de benchmark, no un dataset descargable. Merece la pena vigilar el sitio.

### D.3. El paper que hay detrás del dataset que ya usamos

**[Detecting fake review buyers using network structure: Direct evidence from Amazon (PNAS,
doi 10.1073/pnas.2211932119)](https://www.pnas.org/doi/abs/10.1073/pnas.2211932119)**, versión
abierta en [arXiv 2410.17507](https://arxiv.org/abs/2410.17507). `⚠️ no he podido extraer el texto
del PDF (fetch devolvió binario ilegible)` — lo dejo anotado pero **sin citar ninguna cifra suya**.

Relevante porque es el trabajo del que sale el dataset `bretthollenbeck` que el proyecto ya carga,
y porque su ground truth son los mismos grupos de brokers de Facebook de D.2 (con Ethan Feldman
como autor compartido). Leerlo de verdad daría (a) qué features de red usan exactamente y (b) cómo
reportan resultados — información directamente comparable con nuestros números en ese dataset. Es
una tarea de lectura pendiente, no un hallazgo cerrado.

---

## Sección E — 🔵 Hipótesis propias, no publicadas

Marcadas explícitamente como razonamiento mío a partir del material de arriba y del estado medido
del proyecto. Ninguna está publicada como tal; las tres son **baratas y decisivas**, en el sentido
de que un resultado negativo también informa.

### E.1. 🔵 El LOO target encoding invierte el orden *dentro* de cada grupo — y eso es medible hoy

Matemática elemental, verificable a mano: en un grupo de `n` miembros con `k` positivos, el score
leave-one-out vale `(k-1)/(n-1)` para un miembro positivo y `k/(n-1)` para uno negativo. Es decir,
**dentro de un mismo grupo, los negativos reciben SIEMPRE un score estrictamente mayor que los
positivos**. Todo el AUC de `net_rur`/`net_rsr` viene, por tanto, de la varianza *entre* grupos
(qué grupos tienen tasa de fraude alta), no de discriminar dentro de ninguno.

**Experimento decisivo, una tarde de trabajo, sin GPU**: calcular el AUC restringido a pares de
nodos que pertenecen a la **misma** comunidad. La predicción es que salga **por debajo de 0,5**. Si
sale así, queda demostrado con un número — no con un argumento — que la señal no distingue
miembros, solo etiqueta grupos enteros con su propia tasa de fraude. Es la confirmación más limpia
posible del diagnóstico del maestro, y es el tipo de resultado que da credibilidad en un README.

Nota: sospecho (sin poder afirmarlo) que este mismo efecto explica el **signo invertido de OddBall
sobre `net_rur`** ya medido (AUC 0,2797/0,2581) — merece una mirada, porque una señal
sistemáticamente invertida es información, no ruido.

### E.2. 🔵 La versión sin etiquetas de `net_rur` probablemente conserva casi toda la señal

`CONTEXTO.md` ya documenta que en Yelp-NYC la tasa de fraude cae monótonamente con el número de
reviews del reviewer (22% con 1 review → 0,66% con 64+), y en bretthollenbeck sube monótonamente
(4,6% con 1 → 100% con 32+). **Si la relación es monótona, entonces el simple conteo de reviews por
reviewer — que no necesita NINGUNA etiqueta — debería reproducir casi todo el AUC de `net_rur`.**

**Experimento**: calcular el AUC de `log(n_reviews_del_reviewer)` como score, sin tocar etiquetas, y
compararlo con el 0,9045 del `net_rur` con target encoding. Dos desenlaces, ambos valiosos:

- Si sale cerca de 0,90: **el producto está salvado**. La señal fuerte del grafo es label-free
  después de todo, y lo único que había que hacer era dejar de calcularla con etiquetas. Habría que
  reescribir la narrativa de "no tenemos señal de grafo usable en producto" — que es la conclusión
  actual del proyecto y que podría estar equivocada.
- Si sale mucho más bajo: confirma que el 0,90 era fuga casi pura y que el número honesto de todo el
  grafo es el 0,6231.

Coste: **una hora**. Sin GPU. Es, con diferencia, el experimento con mejor relación
información/esfuerzo de todo este documento, y no aparece como tal en ningún paper porque es
específico de nuestro montaje. **La dirección del signo cambia entre datasets**, así que habría que
ajustar el signo por dominio — cosa que ya está documentada como necesaria.

### E.3. 🔵 SVN sobre el co-review multi-negocio como sustituto de `net_rsr`

Combinación de B.2 (SVN), C.4 (co-review ponderado) y el diagnóstico propio de cliques exactos.
Construcción propuesta: arista reviewer↔reviewer **ponderada por el número de negocios distintos
co-reseñados** (no por compartir uno solo), validada con test hipergeométrico + FDR contra el nulo
de co-ocurrencia aleatoria, con descomposición por grado como hace Tumminello.

Por qué debería funcionar en *nuestros* datos concretamente: dos cuentas legítimas pueden coincidir
en un restaurante popular por azar, pero coincidir en **cinco negocios distintos** es
estadísticamente improbable y es la firma exacta de una granja que ejecuta un lote de encargos. El
grafo resultante tiene varianza topológica real (grados heterogéneos, comunidades solapadas), que es
justo lo que faltaba para que Louvain/Leiden tuvieran algo que optimizar.

**Riesgo honesto, cuantificable antes de implementar**: si la distribución de reviews por reviewer
en Yelp-NYC está dominada por cuentas de 1-2 reviews, casi ningún par tendrá co-ocurrencias
suficientes para superar Bonferroni y la red validada saldrá vacía. Se comprueba en 10 minutos con
un `value_counts()` antes de escribir una línea del método. Si sale vacía en Yelp-NYC, probar en
bretthollenbeck, donde las campañas están concentradas en 3.389 productos.

---

## Resumen: qué necesita GPU y qué no

El usuario está hoy en un portátil sin GPU. Esta es la separación limpia.

### Se puede hacer HOY, sin GPU

| Qué | Coste | Sección |
|---|---|---|
| AUC/AP intra-comunidad (demostrar la inversión del LOO) | ~1 tarde | E.1 |
| AUC de `log(n_reviews)` sin etiquetas vs. `net_rur` | ~1 hora | E.2 |
| Reportar AP / precision@k / recall@FPR fijo en el lado de grafo | ~1 tarde | A.4 |
| Split temporal o `GroupKFold` por `business_id` en vez de aleatorio | ~1 día | A.3 |
| Target encoding cross-fitted out-of-fold + suavizado bayesiano | ~1 día | A.5 |
| Línea base trivial (grado del nodo) al lado de cualquier método | ~1 hora | A.2 |
| SVN sobre co-review multi-negocio | ~2-3 días | B.2 / E.3 |
| Burstiness por coef. de variación de tiempos entre llegadas | ~1 día | C.3 |
| FreeGAD sobre Yelp-Chi (tiene atributos de nodo) | ~2-3 días | B.1 |
| Densidad ponderada / pseudo-cliques estilo FairPlay | ~2 días | C.4 |
| DeFrauder (implementar desde el paper, sin repo oficial) | ~1 semana | C.2 |
| HUGE/HALO sobre Yelp-Chi — prueba de humo en CPU, puede ser lento | incierto | B.3 |

### Requiere GPU (o el ordenador de casa)

| Qué | Sección |
|---|---|
| UNPrompt — entrenamiento (la inferencia zero-shot no, si publican pesos) | B.5 |
| HUGE/HALO con entrenamiento completo sobre YelpChiFull/AmazonFull | B.3 |
| CAMERA (texto + grafo no supervisado) | B.6 |
| TAM (además, con riesgo confirmado de OOM en YelpChi) | B.4 |
| Todo lo de la ronda 4 (BWGNN/GAGA/PC-GNN/GTAN, TGN/DySAT, DiffGAD) | ronda 4 |

---

## Priorización: las 3 cosas que implementaría primero

### 1. El paquete de honestidad de evaluación (E.2 + E.1 + A.4), medio día en total

**Qué**: (a) AUC de `log(n_reviews_del_reviewer)` sin usar ninguna etiqueta, comparado con el 0,9045
de `net_rur`; (b) AUC restringido a pares dentro de la misma comunidad; (c) añadir AP, precision@k y
recall@FPR fijo a todo lo que hoy solo reporta ROC-AUC.

**Por qué primero**: porque (a) puede cambiar la conclusión estratégica del proyecto en una hora. La
narrativa actual — "el grafo no sirve en producto porque su única señal fuerte necesita etiquetas" —
**podría ser falsa**, si el conteo de reviews sin etiquetas reproduce el grueso del AUC. Es el único
experimento de todo el documento que puede revertir una decisión de producto ya tomada, y es el más
barato. Y (b) convierte el diagnóstico del maestro en un número publicable en vez de un argumento.

**Incertidumbre honesta**: alta sobre el desenlace de (a) — puede salir 0,88 o puede salir 0,62, no
lo sé. Baja sobre (b): la matemática dice que saldrá por debajo de 0,5, y si no sale es que hay algo
que no entendemos del pipeline, lo cual también sería valioso descubrir.

### 2. SVN sobre co-review multi-negocio (B.2 + E.3), 2-3 días

**Qué**: reconstruir el grafo reviewer-reviewer ponderado por negocios co-reseñados y filtrarlo con
test hipergeométrico + FDR, en vez de las uniones de cliques exactos actuales. Luego pasar el mismo
`evaluate_communities` que ya existe.

**Por qué segundo**: es la única propuesta de esta ronda que **ataca la causa raíz ya confirmada
cuatro veces** (cliques exactos sin varianza topológica) en vez de probar otro algoritmo sobre la
misma estructura rota. Es label-free, CPU-only, y encaja con la cultura estadística ya establecida
del proyecto (modelo nulo hipergeométrico, z-scores, modelo de configuración) — no es una caja
negra. Además el survey de C.1 dice explícitamente que la validación estadística es la alternativa
correcta a los umbrales arbitrarios, así que no es una ocurrencia aislada.

**Incertidumbre honesta**: media-alta. El riesgo concreto es que en Yelp-NYC la mayoría de reviewers
tengan 1-2 reviews y no sobreviva casi ninguna arista. Eso se comprueba en 10 minutos antes de
invertir los 2-3 días — no es un riesgo ciego. Y si Yelp-NYC sale vacío, bretthollenbeck es el plan
B natural.

### 3. FreeGAD sobre Yelp-Chi (B.1), 2-3 días

**Qué**: correr el código público de FreeGAD sobre Yelp-Chi (45.954 nodos, que ya tiene los 32
atributos de nodo que el método necesita) y comparar contra nuestros números.

**Por qué tercero**: sería la **primera cifra no supervisada real del proyecto** producida por un
método publicado con código propio, sin etiquetas y sin entrenar nada. Nos da a la vez un baseline
externo honesto y una comparación limpia. Y el dato de B.5 (UNPrompt zero-shot: AUROC 0,6853,
AUPRC 0,2219) sugiere que **nuestro 0,6231 / 0,1775 podría estar ya en la zona del estado del arte
no supervisado** — si FreeGAD lo confirma sobre nuestros propios datos, eso reencuadra por completo
el "estamos pochos con los grafos": no estamos pochos, estábamos comparándonos contra un target
encoding supervisado propio y contra SpEagle (semi-supervisado), que no son rivales justos.

**Incertidumbre honesta**: media. El código es público, pero (a) hay que adaptar el formato de
datos, (b) el propio paper reporta AUPRC 15,80% en YelpChi, o sea que **no espero que gane** — lo
espero como referencia, no como solución, y hay que decirlo así antes de correrlo para no
reinterpretar el resultado a posteriori. Y ojo con elegir α/β/L/K mirando el AUC: eso sería
cometer exactamente el pecado de A.1.

---

## Lo que NO recomiendo, y por qué

- **Seguir probando algoritmos de clustering/densidad sobre `net_rtr`/`net_rsr` tal cual están.**
  Cuatro resultados negativos independientes con el mismo diagnóstico. El problema es la
  construcción del grafo, no el algoritmo.
- **Usar inyección de anomalías sintéticas como validación de calidad** (matiz a la ronda 3, ver
  A.2). Como test de suelo mínimo sí, siempre reportando al lado la línea base de grado.
- **Presentar cualquier AUC de grafo sin su AP al lado.** Con base rate 10,27%, el AUC comprime
  diferencias que el AP amplifica — y el propio GADBench lo cuantifica (A.4).
- **Citar el ~0,78 de SpEagle como "el listón"** sin decir que es semi-supervisado. El listón
  honesto para nuestro caso de uso (cliente sin etiquetas) es más bien el ~0,685 AUROC / ~0,222
  AUPRC zero-shot de UNPrompt.

---

## Recursos nuevos (no cubiertos en rondas 1-4)

- **[github.com/yunf-zhao/FreeGAD](https://github.com/yunf-zhao/FreeGAD)** — código del método sin
  entrenamiento.
- **[github.com/CampanulaBells/HUGE-GAD](https://github.com/CampanulaBells/HUGE-GAD)** — código de
  HALO/HUGE con comandos para YelpChi y Amazon.
- **[github.com/mala-lab/TAM-master](https://github.com/mala-lab/TAM-master)** y
  **[github.com/mala-lab/UNPrompt](https://github.com/mala-lab/UNPrompt)** — mismo laboratorio
  (mala-lab) que ya mantiene el `Awesome-Deep-Graph-Anomaly-Detection` citado en la ronda 4; es la
  fuente más consistente de código real de este subcampo.
- **[publictraces.net](https://publictraces.net/)** — proyecto a vigilar; anuncian datasets
  comunitarios revisados "for a later release".
