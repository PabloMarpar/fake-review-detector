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
- ⏳ T2 (DeBERTa-v3 afinado) — todavía no empezado. Ahora que hay GPU disponible en la
  máquina de casa, ya no hace falta limitarse a small/xsmall por tiempo de CPU — se puede
  valorar directamente `base` si el resultado lo justifica.
- ⏳ Corpus propio con LLMs modernos (la pieza crítica según el README) — todavía no
  generado. **Requiere una decisión pendiente del usuario**: con qué proveedor(es) de LLM
  generarlo (¿usar Claude directamente vía esta misma sesión, que no necesita API key
  propia, complementado con algún otro proveedor para tener diversidad de generador? ¿o
  usar API keys propias de OpenAI/otros?). No se ha preguntado todavía porque no habíamos
  llegado a ese punto del pipeline.
- ⏳ `docs/index.html` (landing S1) — pendiente, se hace al cerrar la Fase 0 con cifras
  reales.
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

## Fuentes de referencia rápida

- Arquitectura completa, roadmap por fases, líneas rojas sobre atribución, y las ideas
  descartadas (holehe, Seon, Google Custom Search): `README.md`.
- Histórico completo de decisiones y del proceso de planificación: conversación original en
  Claude Code (no reproducida aquí para no duplicar).
