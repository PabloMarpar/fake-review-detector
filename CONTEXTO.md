# Contexto del proyecto (documento vivo)

> Este fichero se actualiza en cada sesión de trabajo. Sirve para que, al abrir un chat
> nuevo, pegando/compartiendo este documento (o abriendo Claude Code directamente en este
> repo) se pueda retomar el trabajo sin tener que releer todo el histórico de conversación.
> El README.md es la narrativa pulida para quien visita el repo; este documento es el
> cuaderno de bitácora de trabajo — qué se ha hecho, qué falta, y por qué se decidió cada
> cosa que no sea obvia leyendo el código.

## Qué es esto, en una frase

Detector de reviews falsas/generadas por IA para webs/startups con reviews propias (B2B) —
combina detección de texto-IA con detección de redes de cuentas coordinadas. Arquitectura,
postura legal/ética y roadmap completos en `README.md`.

## Estado actual (última actualización: 2026-09-11, sesión desde el ordenador de casa)

**Fase 0 — Baseline de texto**, en progreso:

- ✅ Documentación inicial (`README.md`, `.gitignore`), repo creado y subido a
  [github.com/PabloMarpar/fake-review-detector](https://github.com/PabloMarpar/fake-review-detector).
- ✅ `data.py` — descarga y unifica los dos datasets académicos (Ott 2013 + Salminen/GPT-2
  2022) en `reviews_baseline.csv` (42.006 reviews tras deduplicar). Fuentes directas sin
  cuenta de Kaggle (ver sección "Fuentes de datos" abajo).
- ✅ `features_text.py` — T1 (zero-shot cross-perplexity, estilo Binoculars, par
  SmolLM2-360M/Instruct) y T3 (estilometría con spaCy + textstat). Probado en una muestra de
  4 reviews: **funciona y la dirección del score T1 es la esperada** (reviews humanas
  puntúan alto, GPT-2 puntúa bajo).
- ✅ `python train.py t3` — completado. Resultado real (42.006 reviews, split 80/20):
  - TPR@5%FPR global: 0.667, TPR@1%FPR: 0.449, AUC≈0.73.
  - **Desglosado por dataset, la cosa cambia mucho**: TPR@5%FPR = 0.684 en Salminen/GPT-2,
    pero solo **0.075 en Ott** (casi al nivel del azar). Hallazgo honesto esperado: la
    estilometría (superlativos, longitud de frase, etc.) distingue bien humano-vs-GPT-2,
    pero casi nada humano-real-vs-humano-intentando-mentir (Ott son ambas clases escritas
    por personas). Confirma lo que ya decía el README: T3 no es la señal fuerte, es para
    explicabilidad.
  - Modelo guardado en `outputs/models/t3_lightgbm.txt`, métricas en `outputs/metrics.json`,
    features cacheadas en `outputs/stylometric_features.csv`.
- ✅ `python train.py t1` — completado en la máquina de casa (GPU, ver más abajo). Resultado
  real (1.200 reviews, muestra estratificada 300/grupo):
  - TPR@5%FPR global: 0.46, TPR@1%FPR: 0.312.
  - **Desglosado por dataset, mismo patrón que T3**: TPR@5%FPR = 0.80 (TPR@1%FPR = 0.47) en
    Salminen/GPT-2, pero solo **0.08 (TPR@1%FPR = 0.01, es decir, nivel de azar)** en Ott.
    Hallazgo esperado y coherente con el de T3: Binoculars detecta "esto lo escribió un LLM",
    no "esto es una mentira" — así que en Ott (humano real vs. humano mintiendo, sin LLM de
    por medio) no tiene ninguna señal que capturar. El 0.46/0.312 agregados los está tirando
    casi en solitario Salminen/GPT-2.
  - **Lectura importante para priorizar próximos pasos**: GPT-2 (2019) es un generador viejo
    y estilísticamente reconocible — que T1 vaya bien aquí no dice gran cosa sobre si
    funcionará contra un LLM de 2026. La cifra que de verdad importa para el producto todavía
    no existe: hace falta el corpus propio con LLMs modernos (ver más abajo) para saber si T1
    generaliza al caso de uso real, no solo a GPT-2.
  - Métricas completas (con desglose) en `outputs/metrics.json`, scores cacheados en
    `outputs/binoculars_sample_scores.csv`.
  - **Bug real encontrado y corregido** (sesión anterior, en CPU): el checkpointing
    incremental no vaciaba el acumulador `results` tras cada guardado, así que cada guardado
    re-concatenaba TODO lo acumulado con TODO el CSV ya guardado — a las ~1100 iteraciones el
    fichero tenía 1.820 filas para solo 260 reviews únicas de verdad. No corrompía el
    resultado final (el `drop_duplicates` de cierre lo disimulaba), pero cada guardado
    reescribía un CSV cada vez más grande, ralentizando la ejecución progresivamente.
    Corregido en `train.py` — importante para cualquier sesión futura: **si el conteo de
    líneas del CSV de checkpoint no cuadra con el número de reviews de la muestra, hay que
    mirar `text.nunique()`, no `wc -l`** (un review con saltos de línea internos también hace
    que `wc -l` no cuadre aunque no haya ningún bug — pasó en esta sesión, era una falsa
    alarma).
- ✅ **Selección automática de dispositivo (CPU/GPU)**: `features_text.py` tiene
  `get_device()` (usa `cuda` si `torch.cuda.is_available()`, si no `cpu`) — el mismo código
  corre sin cambios en la máquina de trabajo (CPU-only) y en la de casa (GPU), sin flags ni
  configuración manual. `_load_binoculars_models()` mueve performer/observer al device
  detectado e imprime en qué device ha cargado.
- ✅ **Corpus propio con LLMs modernos** — generado (2026-09-11, misma sesión). Tres
  generadores independientes en `own_corpus/`, cada uno con su script reproducible:
  - **Generador A — Claude Sonnet 5** (`claude_generated.csv`, 100 reviews): autoría directa
    de Claude en esta sesión, sin API key. 10 categorías de negocio × 10, mezcla naive/adversarial.
  - **Generador B — Qwen2.5-1.5B-Instruct** (`qwen_generated.csv`, 500 reviews): local en la
    GPU, gratis. Script: `generate_qwen_corpus.py`.
  - **Generador C — OpenAI gpt-4o / gpt-4o-mini** (`openai_generated.csv`, creciendo): con
    prompts basados en técnicas reales encontradas en foros (BlackHatWorld) sobre cómo se
    generan reviews falsas de verdad — no solo "naive" e "adversarial" (roleplay/imaginary
    framing) sino también "fewshot" (pegar una review real como referencia de estilo, la
    técnica más avanzada del foro). Script: `generate_openai_corpus.py`.
  - **Hallazgo de sesión, cuenta OpenAI**: el límite de 50 peticiones/día es una **ventana
    móvil de 24h** (24h/50 = 28m48s exactos), no un reset a hora fija — un hueco se libera
    cada ~29 min, no 50 de golpe. El script ahora espera y reintenta solo (parsea el tiempo
    exacto del error), pensado para dejarlo corriendo sin relanzarlo a mano.
  - `T2_HELD_OUT_GENERATOR_FILE` en `train.py` reserva OpenAI al 100% como "generador nunca
    visto" — ni entrena ni calibra nada, solo evaluación.
- ✅ **T2 (DeBERTa-v3-base) — entrenado, y es la mejor noticia de la sesión.** Entrena con
  Ott (ambas clases) + Claude + Qwen; evalúa en val propio, en el generador held-out (OpenAI,
  nunca visto) y en Salminen/GPT-2 completo (generador viejo, dominio distinto).

  | Evaluación | TPR@1%FPR | TPR@5%FPR |
  |---|---|---|
  | Val (misma distribución de train) | 0.99 | 1.00 |
  | **Held-out — OpenAI, nunca visto** | **0.17** | **0.50** |
  | Salminen/GPT-2 (generador viejo, dominio Amazon) | 0.002 | 0.02 |

  **T2 sí generaliza entre LLMs modernos que nunca ha visto** (entrenado con Claude+Qwen,
  detecta la mitad de las reviews de OpenAI a 5%FPR — muy por encima del 0.02-0.17 que sacaban
  T1/T3 contra ese mismo generador) **pero no generaliza a GPT-2** (2019, generador mucho más
  débil, con "firma" de IA distinta) **ni al dominio Amazon** (solo vio hoteles de Ott en
  train). Conclusión honesta de la Fase 0: la detección de texto de IA en reviews
  generaliza entre generadores modernos, no de forma universal a cualquier generador o
  dominio — hay que decir esto explícitamente en el README/S1, no simplificarlo a "funciona"
  o "no funciona".
  - **Bug real encontrado y corregido**: el checkpoint de `deberta-v3-base` descargado está en
    fp16, y `AutoModelForSequenceClassification.from_pretrained` preserva ese dtype por
    defecto — DeBERTa-v2/v3 es conocido por dar NaN en entrenamiento con fp16 (issue
    documentado del propio modelo, atención "disentangled" desborda). El loss se iba a NaN en
    la primera época hasta forzar `dtype=torch.float32` explícito al cargar. Cualquier
    entrenamiento futuro de un modelo DeBERTa en este repo necesita ese mismo cuidado.
  - Modelo guardado en `outputs/models/t2_deberta/` (737MB, **no está en git** —
    supera el límite de GitHub de 100MB/fichero. `.gitignore` lo excluye; se reproduce
    corriendo `python train.py t2` de nuevo, no hace falta commitearlo).
- ✅ **Fusión T1+T2+T3 probada (`python train.py fusion`) — y descartada.** Regresión
  logística simple sobre las tres señales, mismo protocolo de held-out que T2. Resultado
  contraintuitivo pero real: **la fusión generaliza peor que T2 solo**, no mejor.

  | | T2 solo | Fusión T1+T2+T3 |
  |---|---|---|
  | Held-out (OpenAI) TPR@5%FPR | 0.37 | 0.017 |
  | Held-out (OpenAI) TPR@1%FPR | 0.15 | 0.0 |

  Los coeficientes de la regresión (T1: -0.59, T3: -1.03, T2: +8.7) muestran por qué: T1/T3
  no aportan señal real contra LLMs modernos (ver hallazgo de abajo), pero sí tienen
  correlaciones espurias dentro de la distribución de entrenamiento (Ott+Claude+Qwen) que la
  regresión aprende y que no se sostienen frente a un generador nuevo — al fusionar,
  contaminan la única señal que sí generaliza. **Decisión para el resto del proyecto: usar T2
  solo como señal de texto de Fase 0, no una fusión de las tres** — fusionar por rutina sin
  medir habría dado peor resultado y una falsa sensación de "más señales = mejor".

## Sesión nocturna de mejora de T2 (2026-09-11 noche → 2026-09-12), en marcha

El 0.50 TPR@5%FPR de T2 en el held-out es bueno pero mejorable. Plan acordado con el usuario
antes de irse a dormir, dejándome trabajando de forma autónoma:

1. **Más "voces" de IA en train**: añadir Qwen3-8B (generación actual, no Qwen2.5) como
   tercer generador de entrenamiento, además de Claude+Qwen2.5-1.5B ya existentes.
2. **Segundo held-out**: DeepSeek-R1-Distill-Llama-8B (destilado de DeepSeek-R1, familia
   distinta a todo lo demás), reservado al 100%, para no depender solo del goteo lento de
   OpenAI (ventana móvil 24h/50 ≈ 1 petición nueva cada 29 min — a este ritmo, de 60 a 70 en
   ~7h de la noche del 11 al 12).
3. **Modelo más grande**: `deberta-v3-large` en vez de `base`, con `weight_decay`.
4. **Técnica "innovadora"**: cabeza adversarial de "qué generador es" con Gradient Reversal
   Layer, inspirada en el paper real ACL 2026 "Breaking the Generator Barrier: Disentangled
   Representation for Generalizable AI-Text Detection" (arXiv 2604.13692) — el paper reporta
   que fuerza al encoder a dejar de usar tics de un generador concreto y aprender "IA-nidad"
   genérica. Implementación simplificada en `train_t2_v2.py` (una sola cabeza auxiliar +
   GRL, no la versión completa del paper con doble cuello de botella + cross-view).

**Ambos modelos nuevos verificados por búsqueda web antes de usarlos** (mi conocimiento tiene
corte en enero 2026 y esto se mueve rápido) — `Qwen/Qwen3-8B` y
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, ambos repos reales confirmados en HuggingFace.
Descargados con `own_corpus/_download_model_direct.py` (el mismo bypass de
`huggingface_hub` que ya nos hizo falta para Qwen2.5 y deberta-v3-base — sigue haciendo
falta, `snapshot_download` se sigue colgando en esta red).

**`own_corpus/generate_local_corpus.py`** generaliza el generador de Qwen a cualquier modelo
local, y añade dos estilos de prompt nuevos sobre naive/adversarial: `persona` (persona de
cliente detallada) y `translate` (pide la review en otro idioma y que la traduzca — imita a
un no-nativo real). Longitud mucho más variable (10-150 palabras) en vez de rangos fijos.

**Primer resultado de la cabeza adversarial — honesto, no es lo esperado todavía**: prueba de
humo en `deberta-v3-base` con los datos ya existentes (solo Claude+Qwen2.5, sin Qwen3 aún)
sale **peor** que el T2 baseline, no mejor:

| | T2 baseline (plano) | T2 + adversarial (solo 2 generadores) |
|---|---|---|
| Held-out OpenAI TPR@5%FPR | 0.50 | **0.043** |
| Held-out OpenAI TPR@1%FPR | 0.17 | **0.0** |

No se descarta la técnica todavía: el propio paper dice explícitamente que su ventaja "crece
con la diversidad de generadores de entrenamiento" — con solo 2 generadores estamos en el
peor escenario posible para este método (la cabeza adversarial de 3 clases -humano/claude/
qwen2.5- es demasiado fácil de despistar sin aprender nada útil, o puede estar tirando señal
real junto con los tics). Hipótesis a comprobar en cuanto Qwen3-8B esté listo: con 3
generadores de IA en train, ¿mejora sobre el 0.50, o se descarta también esta técnica como se
descartó la fusión? Se documentará el resultado sea cual sea, no solo si sale bien.

**Actualización — Qwen3-8B generado (500/500, 497 únicas)**. Bug real encontrado antes de
generar el lote completo: Qwen3 razona (`<think>...</think>`) por defecto, y con
`max_new_tokens=150` el pensamiento nunca llegaba a terminar — **cada generación de la
primera prueba era 100% traza de razonamiento cortada, 0% review real** (confirmado
manualmente: 26.7s por llamada, salida a medio pensar). Arreglado pasando
`enable_thinking=False` en `apply_chat_template` (la propia plantilla de Qwen3 lo soporta) —
baja a 6.6s por llamada y el texto sale limpio. Sin este fix, se habría generado un corpus
entero de basura sin darnos cuenta hasta evaluarlo.

**Conclusión de la cabeza adversarial (GRL) — descartada, con tres puntos de datos, no uno.**
Con Qwen3-8B ya generado, se repitió el experimento con 3 generadores de IA en train
(Claude+Qwen2.5+Qwen3) en vez de 2, y además se probó bajar la fuerza de la presión
adversarial (`aux_weight`) por si 0.3 era demasiado agresivo:

| Configuración | Held-out OpenAI TPR@5%FPR |
|---|---|
| T2 baseline (clasificador plano, sin cabeza adversarial) | **0.50** |
| Adversarial, aux_weight=0.3, 2 generadores (Claude+Qwen2.5) | 0.043 |
| Adversarial, aux_weight=0.3, 3 generadores (+Qwen3) | 0.063 |
| Adversarial, aux_weight=0.1, 3 generadores | 0.075 |

Tendencia consistente en la dirección esperada (más generadores ayuda un poco, menos presión
adversarial ayuda un poco) pero la magnitud no se acerca ni de lejos al baseline — no es
ruido de un solo experimento, son tres intentos razonables que convergen en la misma
conclusión. **Se descarta esta técnica para el proyecto, igual que se descartó la fusión T1+T2+T3
— T2 baseline (DeBERTa-v3-base, clasificador plano, sin cabeza adversarial) sigue siendo la
mejor señal de texto medida hasta ahora.**

Posible motivo, para quien retome esto más adelante (no investigado a fondo por tiempo):
el clasificador de este experimento usa mean-pooling propio en vez del `ContextPooler`
entrenado que trae DeBERTa por dentro (necesario para poder engancharle la cabeza auxiliar de
generador a la misma representación) — es un cambio de arquitectura además del adversarial, no
solo "adversarial sí/no" de forma aislada, así que parte de la caída podría venir de ahí y no
solo de la técnica en sí. Y el paper original (arXiv 2604.13692) usa un método bastante más
rico (doble cuello de botella + regularización cross-view + adaptación guiada por
discriminador) — esta es una simplificación de una sola pieza de esa idea, no la
reproducción completa, así que no generalizar bien no dice necesariamente que la idea
del paper esté mal, solo que esta versión simplificada no basta.

## 🟢 Resultado central de la noche: T2 baseline + 3er generador, salto real

Mientras la técnica adversarial se descartaba, se probó lo más simple y con más razón para
funcionar: reentrenar el **T2 baseline de siempre** (clasificador plano, `deberta-v3-base`,
sin ningún truco) añadiendo Qwen3-8B como tercer generador de entrenamiento junto a
Claude+Qwen2.5. Resultado:

| | T2 con 2 generadores (sesión anterior) | T2 con 3 generadores (Claude+Qwen2.5+Qwen3) |
|---|---|---|
| Held-out OpenAI TPR@5%FPR | 0.50 | **0.825** |
| Held-out OpenAI TPR@1%FPR | 0.17 | **0.525** |
| Val (misma distribución de train) | ~1.0 | 1.0 |
| Salminen/GPT-2 (dominio distinto) | 0.02 | 0.03 (sigue sin generalizar ahí, esperado) |

**Confirma la hipótesis original de la forma más limpia posible**: la palanca que de verdad
importa es la diversidad de generadores en entrenamiento, no la sofisticación de la
arquitectura — la misma arquitectura simple que ya funcionaba, con un tercer generador
(que además usa las técnicas de prompt nuevas: persona, traducción, más rango de longitud),
pasa de detectar la mitad de las reviews de un generador nunca visto a detectar más del 80%
dejando solo 5% de falsos positivos. Es la mejora real de la noche — mucho más que cualquier
cambio de arquitectura probado. Modelo en `outputs/models/t2_deberta/` (sobrescribe la
versión de 2 generadores; no está en git por tamaño, reproducible con `python train.py t2`).

**DeepSeek-R1-Distill-Llama-8B — objetivo reducido de 300 a 100 por tiempo real.** A
diferencia de Qwen3, este modelo **siempre** razona, sin toggle para desactivarlo (confirmado
por su documentación). Prueba manual: el pensamiento sí termina dentro de 350 tokens
(`</think>` presente), pero tarda **~5.2 minutos por review** — 300 habrían tardado ~26h,
inviable en una noche. Se lanzó con objetivo 100 (`--strip-think`, que descarta todo lo
anterior a `</think>`) — a este ritmo son ~8-9h, así que probablemente no esté completo al
despertar; se reanuda solo (mismo patrón de checkpoint que los demás generadores) si hace
falta seguir después. Con lo que haya generado, aunque sean 20-40 reviews, ya sirve como
segundo held-out junto a OpenAI — no hace falta esperar al objetivo completo para empezar a
medir.
- 🔴 **Hallazgo crítico de sesión — T1 y T3 evaluados contra el corpus propio
  (`python train.py eval_own_corpus`)**: **fallan casi por completo contra LLMs modernos.**

  | Señal | GPT-2 (2019) | Claude | Qwen | OpenAI |
  |---|---|---|---|---|
  | T1 TPR@5%FPR | 0.80 | 0.16 | 0.086 | 0.017 |
  | T3 TPR@5%FPR | 0.68 | 0.01 | 0.02 | 0.10 |

  Confirma con cifras reales el riesgo que el README ya avisaba citando RAID (96%/7%): T1
  (perplejidad cruzada con SmolLM2-360M) y T3 (estilometría) estaban detectando artefactos
  específicos de GPT-2, no "IA-nidad" en general. No es un problema de umbral — TPR@FPR ya usa
  la curva ROC completa, así que no hay recalibración que lo arregle; es que las señales en sí
  no separan las clases para estos generadores. **Consecuencia directa: T2 deja de ser "el
  siguiente paso" y pasa a ser la única vía real de esta fase** — si tampoco generaliza bien al
  generador held-out (OpenAI), la conclusión honesta de la Fase 0 sería que detectar reviews de
  IA moderna es mucho más difícil de lo que sugería el baseline con GPT-2, y tocaría decir eso
  claramente en vez de maquillarlo.
- ⏳ `docs/index.html` (landing S1) — pendiente, se hace al cerrar la Fase 0 con cifras
  reales.
- 🔴 **Bug real encontrado y corregido — `git autocrlf` corrompía el modelo T3**: esta
  máquina tiene `core.autocrlf=true`, y `outputs/models/t3_lightgbm.txt` no tenía marca de
  binario, así que el `git pull` inicial de la sesión convirtió sus saltos de línea LF a CRLF
  — el formato de texto de LightGBM se rompe con eso (cada árbol/hoja queda desalineado),
  así que cualquier carga del modelo commiteado producía predicciones basura ("Model format
  error, expect a tree here"). Las métricas cacheadas de la sesión anterior en `metrics.json`
  no se vieron afectadas (se calcularon antes de este checkout), pero reutilizar el fichero
  para predicciones nuevas sí. Arreglado con `.gitattributes` (`outputs/models/** -text`) +
  reentrenar T3 para regenerar un fichero limpio (cifras iguales a las de antes: TPR@5%FPR
  0.667 global). **Importante para cualquier fichero de modelo futuro** (T2 en
  `outputs/models/t2_deberta/`): ya cubierto por el mismo patrón en `.gitattributes`.
- ✅ Infraestructura de BBDD (`db.py`, PostgreSQL en Neon vía SQLAlchemy 2.0) montada:
  tabla `reviews` con el esquema de `reviews_baseline.csv` + `id` autoincremental,
  conexión perezosa (no rompe el import sin `.env`), `pool_pre_ping` para el
  auto-suspend del free tier de Neon. Sin datos migrados todavía ni tablas de
  clusters/grafo (ese código no existe aún) — deliberado, ver `.claude/agents/agente-datos.md`.

**Nada de esto se ha desplegado ni tiene modelos de producción todavía** — es todo
entrenamiento/evaluación local.

## Decisiones y hallazgos que no son obvios leyendo el código

- **Entorno: dos máquinas, dos venv distintos.** Máquina de trabajo: se reutiliza el venv
  compartido de todo `portfolio-projects/` (`C:\Users\pablo.mparera\Desktop\portfolio-projects\.venv\`),
  `torch==2.14.0+cpu`, sin GPU — condiciona T1 (de ahí el muestreo en vez de las 42k
  completas) y limitaría T2 a un modelo pequeño si se entrenara ahí. Máquina de casa: venv
  propio en `fake-review-detector/.venv/`, `torch==2.14.0+cu130` con GPU real (RTX 5060 Ti,
  16GB) — ver detalle de montaje en "Ordenador de casa" más abajo. El código no necesita
  saber en cuál está corriendo: `features_text.get_device()` detecta el device solo. En
  ambas, `python` del PATH de Git Bash es un stub de Microsoft Store — hay que invocar el
  `.exe` del venv directamente.
- **Fuentes de datos sin cuenta de Kaggle** (mejor que lo que decía el plan original):
  - Ott et al.: `https://myleott.com/op_spam_v1.4.zip` — descarga directa, confirmado que
    funciona (zip real, estructura `op_spam_v1.4/{positive,negative}_polarity/
    {deceptive_from_MTurk,truthful_from_*}/foldN/*.txt`).
  - Salminen et al. (OR/CG, GPT-2): OSF, `https://osf.io/download/3vds7/` — CSV directo,
    columnas confirmadas `category, rating, label (CG/OR), text_`.
- **Dedup real encontrado**: al combinar y deduplicar por texto exacto, Ott pasó de 800 a
  796 truthful, y Salminen de 20216/20216 a 20195 CG / 20215 OR — había duplicados exactos
  dentro de los propios datasets académicos.
- **T1 (Binoculars) es lento en CPU**: primera carga de modelos ~110s (descarga+carga en
  memoria), luego ~2-12s por review. Con 42k reviews eso son horas — de ahí el muestreo
  estratificado (300 por grupo) para la Fase 0, con la idea de escalar más adelante si hace
  falta.
- **Convención de nombres de fichero**: se sigue la de los otros proyectos del usuario en
  `portfolio-projects/` (`data.py` → `train.py` → `predict.py` → `app.py`, plano, sin
  paquete `src/`) — NO la de `Book_BBDD` (que es un pipeline ETL con estructura de paquete,
  un tipo de proyecto distinto).
- **Stack de S2 (self-serve) decidido antes de implementarlo: FastAPI + Jinja2 + HTMX, no
  Streamlit.** Streamlit era la opción inicial en el diseño original, pero se descarta porque
  es incómodo de cara a comercializar el producto más adelante (marca de terceros visible en
  la UI, control limitado de branding/dominio, modelo de precio que no encaja bien con un
  producto de pago). FastAPI+Jinja2+HTMX mantiene exactamente la misma regla ya fijada en
  `agente-web` ("`app.py` llama a `predict.py` directamente, un único proceso, sin servicio
  API aparte") — aquí FastAPI sirve HTML renderizado con Jinja2 y usa HTMX para la
  interactividad, no se usa como servicio API separado. Gratis de licencia, desplegable en
  free tier de Render/Fly.io/Railway sin decidir cuál todavía. Actualizado en `README.md` y
  `.claude/agents/agente-web.md`.

## Equipo de agentes de este proyecto

En `.claude/agents/` hay dos niveles, pensados para escalar con el proyecto:

- **`fake-review-detector`**: agente único, es el que se usa activamente ahora mismo para el
  día a día.
- **`agente-maestro` + `agente-datos` + `agente-codigo` + `agente-web`**: equipo
  especializado por dominio (datos, detección/ML, producto/web), listo para cuando el
  proyecto tenga suficiente superficie en marcha a la vez como para que repartir compense.
  Todos responden en español y leen este `CONTEXTO.md` al arrancar.

## Ordenador de casa (RTX 5060 Ti) — ya montado y en uso (2026-09-11)

GPU real: **RTX 5060 Ti, 16GB, driver con CUDA 13.1** (no una 4060 Ti como se apuntó antes de
verla). Entorno montado desde cero en esta sesión:

1. **Python 3.13.15** instalado con `winget install --id Python.Python.3.13` (la máquina no
   tenía más que el stub de Microsoft Store). Queda en
   `C:\Users\pablo\AppData\Local\Programs\Python\Python313\`.
2. **venv propio del proyecto** en `C:\Users\pablo\repos\fake-review-detector\.venv\` —
   aquí, a diferencia de la máquina de trabajo, NO se comparte el venv de
   `portfolio-projects/` porque esta máquina no lo tenía montado.
3. `torch` instalado con build CUDA, no la CPU-only del `requirements.txt` pelado:
   `pip install torch --index-url https://download.pytorch.org/whl/cu130` → instaló
   `torch-2.14.0+cu130`. El índice `cuXXX` correcto se eligió mirando qué wheels de Windows +
   Python 3.13 existían en `download.pytorch.org/whl/<tag>/torch/` (cu130 es el más alto que
   no excede el CUDA 13.1 que reporta el driver).
4. Resto de `requirements.txt` instalado normal (`pip install -r requirements.txt` no toca
   `torch` porque no está version-pinned, así que respeta la build CUDA ya instalada). Más
   `python -m spacy download en_core_web_sm` (necesario para T3, no viene con `pip install
   spacy`).
5. **Gotcha real encontrado**: al primer `import torch`, Windows lo bloqueaba con
   `OSError: [WinError 4551]` — **Smart App Control** rechazaba cargar `shm.dll` de torch por
   no tener reputación reconocida (típico con wheels de PyTorch+CUDA recién bajadas). Se
   resolvió desactivando Smart App Control a mano desde Configuración de Windows → Privacidad
   y seguridad → Seguridad de Windows → App y control del navegador (el usuario lo confirmó
   explícitamente sabiendo que, una vez apagado, Windows no deja reactivarlo sin reinstalar
   el sistema — no es un ajuste que se pueda ni se deba tocar por script sin esa confirmación).
6. **Detección automática de GPU ya en el código**: no hace falta ningún flag ni pasar
   `device=...` a mano — `features_text.get_device()` decide solo (ver más arriba). Mismo
   `train.py t1`/`t3` en ambas máquinas.
7. Resultado real de usarla: T1 sobre 1.200 reviews pasó de ~2-12s/review en CPU a
   **~0.04-0.07s/review en GPU (14-24 reviews/s)** — el resto de las 820 reviews pendientes
   del checkpoint se completó en 1-2 minutos en vez de la hora+ que habría tardado en CPU.

Para trabajar desde el ordenador del trabajo mientras el cómputo corre en casa (no probado
todavía, queda documentado por si hace falta): Claude Code **Remote Control** — se arranca
`claude` + `/remote-control` (o `claude --remote-control`) en la carpeta del proyecto en la
máquina de casa, y desde el navegador del trabajo (claude.ai/code) o el móvil se conecta a
esa sesión por su URL. Requiere la máquina de casa encendida y con la sesión ya arrancada.

## Próximos pasos inmediatos (en orden)

1. ~~Ver resultado de `train.py t3`~~ ✅ y ~~lanzar `train.py t1`~~ ✅ — ambos completados,
   con desglose por dataset. Ver "Estado actual" arriba.
2. **Prioridad número 1 ahora mismo**: generar el corpus propio de reviews con LLMs
   modernos. Es el paso que más impacta en la precisión real del proyecto — T1 y T3 ya han
   demostrado el mismo patrón dos veces (bien en GPT-2, azar en Ott/engaño humano), así que
   entrenar/calibrar T2 contra GPT-2 sin más daría una cifra bonita pero poco honesta. Sigue
   pendiente la decisión del usuario: con qué proveedor(es) generarlo (¿Claude vía esta misma
   sesión, complementado con otro proveedor para tener ≥2 familias de generador distintas? ¿o
   API keys propias de OpenAI/otros?) — necesario para el split "generador nunca visto" que
   el README promete como la única cifra honesta de generalización del proyecto.
3. Con el corpus propio en marcha (o al menos decidido), escribir el fine-tuning de T2
   (DeBERTa-v3). Con GPU ya disponible en la máquina de casa, se puede valorar `base` en vez
   de limitarse a `small`/`xsmall` por tiempo de cómputo — a decidir con cifras, no a ciegas.
4. Commitear y pushear cada hito con progreso real (no cada fichero suelto).

**Nota de criterio para toda la Fase 0 en adelante** (pedido explícito del usuario,
2026-09-11): el objetivo es que este proyecto sea técnicamente sólido de verdad — la métrica
que importa es la precisión real del detector, no cifras agregadas que queden bien. Eso
significa: desglosar siempre por dataset/generador antes de dar un número por bueno,
desconfiar de resultados fuertes contra generadores viejos (GPT-2) como indicador de que
funcionará contra LLMs modernos, y priorizar el corpus propio con generadores actuales por
encima de pulir T2 contra datos que no representan el caso de uso real.

## Cierre de sesión (2026-09-12, ~11:30) — apagado del ordenador de casa

Estado real al cortar, sin maquillar:

- **T2 con 3 generadores (Claude+Qwen2.5+Qwen3), `deberta-v3-base`**: ✅ resultado válido y
  documentado arriba — TPR@5%FPR 0.825, TPR@1%FPR 0.525 en held-out OpenAI. Este es el modelo
  bueno de la sesión, en `outputs/models/t2_deberta/`.
- **`deberta-v3-large` con los mismos 3 generadores**: ⚠️ **no terminó** — el entrenamiento
  (loss bajó bien, sin NaN: 0.21→0.03→0.01) se completó, pero la evaluación posterior contra
  Salminen completo (40k reviews) se quedó casi 50 minutos sin terminar con la GPU al
  96-99% sostenido. No estaba claro si era solo lento por el tamaño de `large` o si estaba
  realmente atascado, y no había tiempo para averiguarlo con el apagado inminente — se mató
  el proceso limpio en vez de arriesgar perderlo a medias. **Pendiente**: retomar con más
  margen de tiempo, y si se repite el parón, sospechar primero del batch de evaluación sobre
  40k filas con `large` (podría convenir bajarlo o trocearlo en vez de un único predict).
- **DeepSeek-R1-Distill-Llama-8B (held-out nuevo)**: 70/100, parado a medias (se le dio
  prioridad a `large` y luego al tier nuevo de OpenAI). Reanudable con el mismo comando de
  siempre.
- **Mistral-7B-Instruct-v0.3 (4º generador de train, propuesto)**: descargado del todo
  (`_model_cache/mistral-7b-instruct-v0.3/`), pero **la generación de reviews nunca se
  lanzó** — quedó pendiente. `train.py` ya tiene `T2_TRAIN_GENERATOR_FILES` listo para
  añadir `mistral_generated.csv` en cuanto exista.
- **OpenAI — tier de cuenta subió a mitad de sesión** (usuario añadió algo en el dashboard):
  de 50 RPD a **10.000 RPD / 500 RPM** para `gpt-4o-mini` y `gpt-4o`. Debería haber disparado
  la generación, pero se quedó parado en 90 reviews (70 gpt-4o-mini + 20 gpt-4o) sin dar
  tiempo a diagnosticar por qué antes del apagado — **revisar el log
  `outputs/openai_corpus_fasttier.log` y relanzar `own_corpus/generate_openai_corpus.py` la
  próxima vez que haya sesión**, ahora sí debería volar con el tier nuevo.

**Todo lo de arriba está commiteado y pusheado** salvo, por diseño, los checkpoints de
modelo grandes (gitignored, reproducibles) — no hay trabajo sin guardar en este apagado.

## Próximos días sin GPU (portátil de trabajo) — qué se puede hacer

El usuario va a trabajar los próximos días desde el portátil del curro, sin GPU — nada de
entrenar/generar con modelos locales grandes (Qwen3-8B, DeepSeek, Mistral-7B). Eso deja en
pausa: completar el corpus de Mistral (descargado, generación sin lanzar), terminar DeepSeek
(70/100), reintentar `deberta-v3-large` con más generadores. Retomar eso en el ordenador de
casa cuando vuelva.

Mientras tanto, trabajo con sentido que sí es CPU-only:

1. **Fase 1 (grafo + coordinación de cuentas)** — la siguiente fase del roadmap y no necesita
   GPU en absoluto:
   - `data.py`: descargar Yelp-NYC / Yelp-ZIP (Rayana & Akoglu) — dataset con grafo real
     reviewer-negocio + etiqueta proxy de spam, a escala.
   - `features_graph.py` (no existe aún): grafo bipartito reviewer↔negocio → proyección
     reviewer-reviewer, burst detection, clustering con Louvain, embeddings.
   - `profile_cluster.py` (no existe aún): ficha de cluster con Niveles A (estructural/
     temporal) y B (cohorte de cuenta) — diseño ya cerrado en README.md.
   - Comparación contra el ~0.78 AUC publicado de SpEagle.
2. **`docs/index.html` (landing S1)** — ya hay cifras reales de Fase 0 que enseñar, buen
   momento para cerrarlo (pendiente según el roadmap).
3. **Más corpus de Claude** (Generador A) — se genera escribiéndolo yo directamente en
   sesión, no necesita GPU ni API. Útil para cuando se retome el entrenamiento en casa.

## Fuentes de referencia rápida

- Arquitectura completa, roadmap por fases, líneas rojas sobre atribución, y las ideas
  descartadas (holehe, Seon, Google Custom Search): `README.md`.
- Histórico completo de decisiones y del proceso de planificación: conversación original en
  Claude Code (no reproducida aquí para no duplicar).
