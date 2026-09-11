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

## Estado actual (última actualización: 2026-09-11)

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
- 🔄 `train.py` — en ejecución en segundo plano cuando se escribió este documento:
  - `python train.py t3`: entrena LightGBM sobre estilometría en las 42k reviews completas
    (rápido, sin red neuronal).
  - `python train.py t1`: pendiente de lanzar tras t3 — Binoculars sobre una muestra
    estratificada (300 por grupo dataset×label), porque en CPU cada review tarda ~2-12s
    (no es viable a las 42k reviews completas sin GPU). Guarda resultados incrementalmente
    en `outputs/binoculars_sample_scores.csv` (reanudable si se interrumpe).
- ⏳ T2 (DeBERTa-v3 afinado) — todavía no empezado. Pendiente de decidir tamaño del modelo
  (small vs. xsmall) según cuánto tarde el entrenamiento en esta máquina (CPU only, sin
  GPU — ver "Entorno técnico").
- ⏳ Corpus propio con LLMs modernos (la pieza crítica según el README) — todavía no
  generado. **Requiere una decisión pendiente del usuario**: con qué proveedor(es) de LLM
  generarlo (¿usar Claude directamente vía esta misma sesión, que no necesita API key
  propia, complementado con algún otro proveedor para tener diversidad de generador? ¿o
  usar API keys propias de OpenAI/otros?). No se ha preguntado todavía porque no habíamos
  llegado a ese punto del pipeline.
- ⏳ `docs/index.html` (landing S1) — pendiente, se hace al cerrar la Fase 0 con cifras
  reales.

**Nada de esto se ha desplegado ni tiene modelos de producción todavía** — es todo
entrenamiento/evaluación local.

## Decisiones y hallazgos que no son obvios leyendo el código

- **Entorno**: se reutiliza el venv compartido de todo `portfolio-projects/`
  (`C:\Users\pablo.mparera\Desktop\portfolio-projects\.venv\`), no uno propio de este
  proyecto — sigue la convención ya existente de los otros proyectos del usuario. `python`
  real está en `AppData\Local\Programs\Python\Python313` (el `python` del PATH de Git Bash
  no resuelve, es un stub de Microsoft Store — hay que invocar el `.exe` del venv
  directamente).
- **Torch es CPU-only en esta máquina** (`torch==2.14.0+cpu`) — no hay GPU disponible. Esto
  condiciona todo: T1 no se puede correr a escala completa (de ahí el muestreo), T2 se hará
  con un modelo DeBERTa-v3 pequeño (small/xsmall, no base) y probablemente también con
  subsample o pocas épocas — se documentará como decisión de ingeniería explícita, no como
  limitación escondida (mismo estilo que `spain-energy-forecast`).
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

## Próximos pasos inmediatos (en orden)

1. Ver resultado de `train.py t3` (LightGBM + estilometría, dataset completo).
2. Lanzar `train.py t1` (Binoculars, muestra) en segundo plano.
3. Con esos dos resultados, decidir tamaño de modelo para T2 y escribir el fine-tuning.
4. **Preguntar al usuario** cómo generar el corpus propio de reviews con LLMs modernos
   (qué proveedor(es), para tener al menos 2 familias de generador distintas — necesario
   para el split "generador nunca visto" que el README promete como la única cifra honesta
   de generalización del proyecto).
5. Commitear y pushear cada hito con progreso real (no cada fichero suelto).

## Fuentes de referencia rápida

- Arquitectura completa, roadmap por fases, líneas rojas sobre atribución, y las ideas
  descartadas (holehe, Seon, Google Custom Search): `README.md`.
- Histórico completo de decisiones y del proceso de planificación: conversación original en
  Claude Code (no reproducida aquí para no duplicar).
