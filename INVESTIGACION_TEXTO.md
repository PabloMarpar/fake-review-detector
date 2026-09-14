# Investigación: cómo mejorar la señal de texto (T1/T2/T3) (2026-09-14)

> Documento de investigación puntual, no de bitácora — sesión de búsqueda web hecha por
> `agente-maestro` mientras corría en paralelo la tarea de grafo delegada a `agente-codigo`
> (ver `INVESTIGACION_GRAFOS.md`). Recoge papers y técnicas encontrados sobre la otra mitad del
> producto: detección de texto generado por IA. Nada de esto está implementado todavía — es
> material de partida, no una tarea completada. Motivo de por qué surge: T1 (Binoculars) está
> documentado en `CONTEXTO.md`/README.md como casi inútil contra LLMs modernos (TPR@5%FPR
> 0,02-0,17 contra Claude/Qwen/OpenAI, ver "Hallazgo crítico de sesión" en CONTEXTO.md), y T2
> depende de regenerar corpus y reentrenar cada vez que aparece un generador nuevo (patrón ya
> repetido dos veces: 2 generadores → 3 generadores).

## Lo más accionable

**Test-time adaptation para distribution shift continuo** — [Hitting a Moving Target:
Test-Time Adaptation for AI Text Detection under Continual Distribution Shift
(2026)](https://arxiv.org/abs/2606.25152). Va directo al patrón de sesión ya vivido: T2 con
Claude+Qwen2.5 detectaba la mitad de OpenAI, con Qwen3 añadido subió a 0,825 — cada vez que
aparece un generador nuevo, la respuesta hasta ahora ha sido generar más corpus y reentrenar
desde cero. Este método es una capa de adaptación **no supervisada** (sin etiquetas en test)
que se engancha a un clasificador ya afinado tipo DeBERTa — compatible en principio con T2 tal
cual está, sin rediseñar arquitectura (verificado leyendo el paper vía WebFetch, no solo el
abstract: dice explícitamente que funciona como "plug-in layer" sobre clasificadores DeBERTa ya
existentes). No sustituye la necesidad de datos diversos en train, pero podría amortiguar la
caída cuando aparezca el próximo generador sin depender de generar corpus nuevo el mismo día.
No confirmado el coste computacional exacto (el paper no lo detalla en las secciones leídas) —
a verificar antes de comprometerse, dado que esta máquina de trabajo es CPU-only.

**CPD (Cumulative Probability Density) para reviews específicamente** — [AI-generated fake
review detection, Decision Support Systems
2026](https://www.sciencedirect.com/science/article/abs/pii/S0167923626000175). Idea de fondo
distinta a T1/T2/T3: en vez de perplejidad o un clasificador entrenado, mide si una review es
un **outlier estadístico "por ser demasiado normal"** — longitud, sentimiento y vocabulario
todos cerca de la media, justo lo que un LLM tiende a producir. Entrenan un AdaBoost simple
sobre esos valores de densidad acumulada. Encaja mejor con el rol actual de T3 (explicabilidad,
no la señal fuerte) que con T1/T2, pero es CPU-friendly (AdaBoost, no una red) y ataca el
problema desde un ángulo que ninguna de las tres señales actuales usa hoy — candidato a "T3 v2"
o a feature adicional de la fusión, no a reemplazar T2. No verificado sobre qué dataset evalúan
ni si comparan contra generadores modernos o solo GPT-2-like — a comprobar antes de invertir
tiempo, mismo criterio de cautela que ya aplica el proyecto a cifras de terceros.

## Para robustez adversarial (conecta con el propio `hard_evasion` ya generado)

Hay una línea de investigación activa justo sobre lo que ya se está explorando con los prompts
`hard_evasion` del corpus de `gpt-6-astra`: **[Adversarial Paraphrasing: A Universal Attack for
Humanizing AI-Generated Text](https://arxiv.org/abs/2506.07001)** confirma que parafrasear con
otro LLM guiado por el propio detector baja la detección a niveles casi aleatorios — y
**[PADBen](https://arxiv.org/pdf/2511.00416)** es un benchmark dedicado a esto. Del lado
defensivo, dos ideas concretas y baratas:

- **Aumentar el corpus de entrenamiento con back-translation** (traducir ida y vuelta como
  augmentación) — mismo espíritu que el `prompt_style="translate"` que ya existe en
  `own_corpus/generate_local_corpus.py`, pero aplicado como defensa de entrenamiento
  sistemática, no solo como un estilo de generación más entre otros.
- **Defensas basadas en retrieval** (comparar contra un índice de reviews conocidas) —
  reutilizaría el mismo índice MinHash+LSH ya planeado para el Nivel C de `profile_cluster.py`
  (near-duplicates, ver README), doble uso del mismo trabajo en vez de construir dos índices
  separados.

## Contexto de mercado, no técnico directamente accionable

RAID (el benchmark ya citado en el README) sigue activo y confirma, con generadores más
modernos (GPT-4o, o1, o3-mini, familia GPT-5), el mismo patrón ya visto con datos propios: los
detectores no generalizan a modelos no vistos, y perturbaciones simples (cambiar decoding,
penalización de repetición) ya bajan mucho el rendimiento. Es una confirmación externa e
independiente del propio hallazgo de Fase 0, útil para citar con más peso que solo las cifras
propias si hace falta justificar el enfoque honesto (bandas de abstención, no prometer 99%) de
cara a la landing (S1).

## Nivel de confianza de este documento

Todo lo de arriba viene de abstracts/resúmenes vía búsqueda web y, en el caso de la
test-time-adaptation, una lectura más detallada del PDF vía WebFetch — nada se ha leído línea a
línea ni se ha probado código propio. Antes de comprometer tiempo de implementación, verificar
igual que ya hace el proyecto con cualquier cifra de tercero (ver `print_reference_comparison()`
en `features_graph.py` como ejemplo del nivel de escrutinio esperado).
