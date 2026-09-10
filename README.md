# Detector de reviews falsas / generadas por IA

> **Estado: en fase de diseño, sin implementar todavía.** Este README documenta la
> arquitectura, las fuentes de datos y las decisiones ya tomadas *antes* de escribir una sola
> línea de código de producción — incluidas varias ideas que se evaluaron en serio y se
> descartaron con motivo. Se mantiene así de honesto a propósito: es más útil documentar el
> razonamiento mientras la arquitectura todavía se puede cambiar que maquillarlo después.

Un sistema que puntúa la autenticidad de reviews y, sobre todo, de **redes de cuentas
coordinadas**, combinando detección de texto generado por IA con análisis de grafos. Pensado
desde el diseño como un producto B2B para webs/startups que alojan sus propias reviews — no
como un scraper de reviews ajenas.

## Por qué este proyecto

Mozilla cerró **Fakespot** en julio de 2025: heurísticas pensadas para antes de que cualquiera
pudiera generar una review convincente con un LLM en dos segundos, "sin modelo de negocio
viable" y "caro de operar a escala", según sus propias palabras. El hueco que deja es real —
nadie lo ha llenado bien todavía en la era de los LLMs — y las tres razones de su fracaso
(lado consumidor sin comprador, rastrear webs de terceros sin permiso, coste que escala con
todo internet) son evitables si se invierten los tres ejes desde el principio: el cliente es
la web que aloja sus propias reviews, los datos los aporta ese cliente, y el coste escala con
clientes de pago, no con la web entera.

Además hay un motivo con fecha concreta, no solo una intuición de mercado: desde 2024-2025 hay
**obligación legal real** de vigilar reviews falsas — la norma de la FTC en EEUU (multas de
hasta ~53.000$ por infracción), la DMCC en Reino Unido (hasta el 10% de la facturación global,
con el deber explícito de tomar "pasos razonables y proporcionados"), y la DSA europea (hasta
el 6% de facturación global). El pitch no es "tengo un modelo majo", es: "tienes una
obligación legal nueva y probablemente no tienes equipo de datos para cumplirla".

## Sobre qué se puede prometer de verdad (y qué no)

Es tentador diseñar esto como "identificar de dónde son las granjas de bots" — nacionalidad,
origen geográfico. La metodología pública de quien investiga esto en serio (Stanford Internet
Observatory, Graphika, DFRLab, los informes trimestrales de Meta sobre "coordinated
inauthentic behavior") es clara en un punto: **nadie serio afirma nacionalidad de una cuenta a
partir de cómo escribe o cuándo se creó**. Meta lo dice explícitamente — actúan sobre
comportamiento, nunca sobre contenido, y solo publican vínculos con actores conocidos cuando
hay evidencia externa.

Hay motivos de fondo, no solo prudencia: identificar el idioma nativo de alguien por cómo
escribe en inglés tiene un techo real de ~85% en textos de 300-400 palabras y clasificación
cerrada — con reviews de 20-80 palabras el problema es mucho más duro, y "mala redacción =
extranjero" es exactamente el razonamiento flojo que desacreditaría la herramienta en vez de
darle fiabilidad. Y una nacionalidad/etnia inferida y asociada a una cuenta identificable es
un dato de categoría especial bajo RGPD — el precedente de las multas a Clearview AI
(~100M€ combinadas en varios países europeos por acumular identidad a partir de datos
"públicos") es el ejemplo de a dónde lleva industrializar esto.

**Lo que sí es real, útil y defendible**: un "perfil de cluster" construido con evidencia
medible, no con veredictos. Ráfagas de creación de cuenta comparadas contra la tasa de alta
normal de la plataforma. Huella de "lenguaje mediado por traducción automática" (campo de
investigación real, con papers recientes) reportada como indicador escalar, nunca como
etiqueta de idioma. Ventana horaria de actividad comparada contra la línea base de la propia
plataforma del cliente — un desfase de horas, nunca un país. Reutilización de texto, de
avatar, de patrones de username dentro del cluster. Es una versión más fiable de la misma
idea: en vez de "esto es un bot ruso", "estos 14 usuarios se registraron en una ventana de 6
días cuando lo esperable serían 1-2, y su texto tiene una huella de traducción automática
elevada" — más defendible y, para un comprador B2B con abogado, mucho más creíble.

## Estrategia de datos

**Postura sobre scraping**: nada de rastrear Google Maps, Amazon o Trustpilot como motor de
datos, ni siquiera para la demo pública. Expone a riesgo legal real a escala comercial y,
además, Google prohíbe cachear el contenido de sus reviews — es un callejón sin salida incluso
dejando la legalidad a un lado. Dos rutas legalmente limpias en su lugar: texto pegado o CSV
subido por el propio usuario en la demo (riesgo cero, y es literalmente lo que haría un
cliente B2B real el día 1), y Google Business Profile API / Trustpilot Business API con
autorización OAuth del propio negocio para producción (el negocio da acceso a *sus propios*
datos).

**Datasets para entrenar/validar** (todos públicos y sin restricción de scraping):

| Dataset | Qué aporta | Rol |
|---|---|---|
| Ott et al. *Deceptive Opinion Spam Corpus* | 1.600 reviews de hoteles, mitad reales/mitad engañosas escritas por humanos (MTurk) | Baseline de "estilo de engaño" — con el matiz honesto de que es de antes de los LLMs |
| Kaggle *Fake Reviews Dataset* (GPT-2) | 40k reviews reales vs. generadas por GPT-2 | Tratado como **suelo, no techo**, de dificultad: un detector puede sacar 96% aquí y 7% en texto de GPT-4 (benchmark RAID) |
| Yelp-NYC / Yelp-ZIP (Rayana & Akoglu) | Texto + grafo real reviewer-negocio + etiqueta proxy de spam de Yelp, a escala (359k/608k reviews) | El dataset bisagra — el único con texto y grafo y etiquetas, se entrena la fusión aquí, se compara contra el ~0.78 AUC publicado de SpEagle |
| Amazon-Reviews-2023 (McAuley Lab) | 571M reviews, sin etiqueta de fake | Solo referencia de volumen/estilo "normal" |
| Corpus propio (por generar) | Reviews escritas por 2-3 LLMs baratos sobre productos/negocios reales de los metadatos de Yelp/Amazon | La pieza que no existe en ningún dataset público — crítica para no repetir el 96%/7% de RAID |

## Arquitectura de detección de texto

Reviews cortas (20-80 palabras) están cerca del límite teórico de detectabilidad: en
investigación reciente (arXiv 2506.13313), ni humanos ni LLMs distinguen reviews falsas de
LLMs mejor que el azar a nivel de review individual. **El texto por sí solo no es decisivo —
la señal de grafo/cluster es la que de verdad carga el peso**, porque coordinar cuentas reales
cuesta recursos de verdad (cuentas envejecidas, paciencia, dispersión), mientras que un texto
mejor solo cuesta un prompt mejor.

Tres señales de texto, con billing honesto sobre cuál pesa más:

- **T1 — zero-shot cross-perplexity (estilo Binoculars), señal principal desde el día 1.**
  Ratio de perplejidad cruzada entre un modelo base pequeño y su versión instruct, con el
  umbral recalibrado sobre datos propios de reviews. Punto a favor decisivo: en el paper
  original mantiene ~99.7% de acierto igual en ensayos de estudiantes con inglés corregido que
  sin corregir, mientras otros detectores fallan 48-76% de esos mismos textos — no penaliza a
  hablantes no nativos, que es justo el sesgo a evitar.
- **T2 — DeBERTa-v3 afinado** (no DistilBERT — sistemáticamente el más flojo en las
  comparativas). Entrenado con Ott + el corpus propio; el dataset de GPT-2 de Kaggle se deja
  en un split de evaluación estrictamente separado para no maquillar el resultado con fuga de
  datos.
- **T3 — estilometría + LightGBM**, sin pretender competir con T1/T2 en detección — su función
  real es explicabilidad (alimenta SHAP) y diversidad de ensemble.

La fusión incorpora la longitud del texto como feature explícita y produce **tres salidas**:
`auténtica` / `inconcluyente` / `sospechosa`, con la banda "inconcluyente" fijada por una tasa
de falsos positivos objetivo — un detector que dice "no lo sé" en el 30% de reviews de 25
palabras es más útil y más vendible que uno que adivina.

## Perfilado de clusters ("granjas de bots")

El módulo de grafo implementa el método académico estándar para esto (Pacheco et al., ICWSM
2021: red bipartita reviewer↔negocio → proyección a reviewer-reviewer → cluster). Cada
comunidad detectada por Louvain genera una ficha con evidencia en niveles, de más a menos
fiable:

- **Nivel A — estructural/temporal**: sincronía de co-reviews frente a un modelo nulo,
  participación en ráfagas, densidad del grafo frente a un modelo de configuración aleatoria
  (la señal más importante — separa una granja real de "doce personas que fueron el finde de
  apertura del restaurante nuevo"), ratio de cuentas con una sola review en toda su vida.
- **Nivel B — cohorte de cuenta**: concentración de fechas de creación frente a la tasa de
  alta real de la plataforma, edad de la cuenta en su primera review, ratio de cero
  interacción recibida.
- **Nivel C — interno de texto**: near-duplicates/plantillas (MinHash+LSH) contra el propio
  CSV del cliente y contra un índice offline construido con los datasets públicos ya
  descargados; media y varianza del score T1/T2 dentro del cluster; huella de "lenguaje
  mediado por traducción" como indicador escalar, nunca como etiqueta de idioma.
- **Nivel D — metadatos**, solo si el cliente los tiene: reutilización de avatar (hash
  perceptual), homogeneidad de username, ventana horaria de actividad frente a la línea base
  de la plataforma, y **dominio de email desechable** — resuelto por DNS (registros MX) +
  lista abierta de dominios de usar-y-tirar, procesando solo el dominio (nunca el email
  completo), reportado agregado por cluster.

La ficha usa lenguaje estimativo tipo ICD-203 (probabilidad y confianza, nunca mezcladas en la
misma frase), muestra siempre sus modelos nulos y termina con un bloque fijo que dice
explícitamente qué **no** se afirma:

```
CLUSTER #7 · 14 cuentas · 41 reviews · 3 negocios
─────────────────────────────────────────────────
COORDINACIÓN     fuerte     densidad de co-review 8,4× el modelo nulo
INAUTENTICIDAD   moderada   9/14 cuentas de una sola review; 12/14 sin interacción
COHORTE          moderada   11/14 registradas en 6 días (esperado ~1,3)
TEXTO            débil      score IA μ=0,71 σ=0,06 (plataforma μ=0,22 σ=0,31)
VENTANA HORARIA  débil      82% de reviews en una banda de 5h, desfase +7h vs. la plataforma

Es muy probable (80-95%) que estas 14 cuentas actuaran de forma coordinada.
No se afirma nacionalidad, origen geográfico, identidad de personas ni quién se beneficia.
Esto es una pista para revisión humana, no un veredicto.
```

**Límite honesto**: no existe ningún dataset con verdad de terreno de "este cluster es una
granja de bots" — así que esta pieza no se evalúa con un AUC inventado, sino con
precisión-en-top-k bajo revisión manual y calibración de los modelos nulos. Y Yelp-NYC/ZIP no
trae fecha de creación de cuenta, así que el Nivel B se implementa y se documenta, pero solo
es validable de verdad con datos reales de un cliente futuro.

## Producto: webs/startups con reviews propias como cliente

- **S1 — landing page** (estática, sin backend): el gancho regulatorio, una ficha de cluster
  real como pieza central, y una sección explícita de **"qué esta herramienta no te va a
  decir"** — nada de nacionalidad, nada de acusaciones, nada de identificar personas, **nada
  de comprobar en qué otras webs tiene cuenta tu cliente** (ver "Ideas evaluadas y
  descartadas"). Para este comprador, en este contexto regulatorio, la honestidad calibrada es
  el diferenciador frente a cualquier "IA detector 99.9% preciso" de la competencia.
- **S2 — escaneo self-serve**: sube un CSV con esquema documentado, mapeo de columnas con
  degradación honesta ("sin `reviewer_created_at` las señales de cohorte no están
  disponibles"), informe con fichas de cluster ordenadas, banda de "inconcluyente" visible,
  exportación a JSON+PDF como informe de evidencia.
- **S3 — conexión OAuth** (más adelante, solo si hay un piloto real): Google Business Profile
  API / Trustpilot Business API.
- **S4 — página de "Trust & Safety" publicable**, no un badge embebible: un badge convierte
  una pista probabilística interna en una afirmación pública de confianza, y con un detector
  que falla más de la mitad de las veces en texto de GPT-4 (ver arriba) eso es una promesa que
  el modelo no puede sostener.

## Roadmap

| Fase | Contenido |
|---|---|
| **0 — Baseline de texto + tesis pública** | Ott + Kaggle OR/CG, T1 (zero-shot) y T2 (DeBERTa-v3) con splits sin fuga de datos, arranque del corpus propio con un generador reservado como held-out. Se cierra publicando S1 con solo las cifras de esta fase. |
| **1 — Grafo + fusión + perfilado** | Yelp-Chi → NYC → ZIP, Louvain, fusión con banda de abstención, comparación contra SpEagle. Perfilado de clusters, Niveles A y B. |
| **2 — Producto, no demo** | S2 completo (esquema CSV, informe, fichas de cluster con Niveles C y D, exportación de evidencia). Landing actualizada con métricas reales. |
| **3 — Piloto y endurecimiento** | Robustez adversarial, deriva del detector, S3 (OAuth, solo si hay piloto), S4 (Trust & Safety). API tipo FastAPI solo si un piloto real la necesita. |

## Ideas evaluadas y descartadas

Se consideraron tres enriquecimientos por email antes de fijar la arquitectura de arriba —
documentado porque el porqué importa tanto como el qué:

- **Reputación de email (Abstract API, ZeroBounce, DeBounce) → entra, pero sin ninguna de esas
  APIs.** Ninguna es gratis-para-siempre-sin-fricción (ZeroBounce exige dominio de empresa,
  DeBounce solo da créditos una vez). En su lugar: DNS directo (registros MX) + una lista
  abierta de dominios desechables mantenida a diario — gratis de verdad, sin cuenta, y más
  limpio de cara al RGPD porque solo se procesa el dominio, nunca el email completo.
- **"Social footprint" (Seon.io / `holehe`) → descartado, sin matices.** Choca de frente con
  la postura de "nada de perfilar personas más allá de lo defendible". `holehe` explota una
  categoría de vulnerabilidad catalogada por OWASP (enumeración de cuentas), viola los ToS de
  las plataformas que consulta, lleva años sin mantenimiento activo, y no se le podría sacar
  nunca una cifra de fiabilidad honesta. Bajo RGPD, perfilar en qué otras webs tiene cuenta el
  email de alguien sin su consentimiento es el mismo terreno que le costó ~100M€ en multas a
  Clearview AI. Se convierte en un punto a favor: es la tercera línea de "esto no te lo vamos
  a decir" en la landing.
- **Búsqueda de texto duplicado vía Google Custom Search → descartado, sustituido por una
  versión offline.** La API está cerrada a clientes nuevos desde 2025 y deja de funcionar del
  todo en 2027; las alternativas (Bing, Brave, proxies tipo SerpApi) están cerradas, de pago o
  siendo demandadas por Google por elusión de medidas técnicas — meter una de esas en el
  pipeline contradice la postura de "nada de scraping de terceros como motor de datos". En su
  lugar, la detección de near-duplicates (MinHash+LSH, Nivel C) compara contra un índice
  offline construido con los propios datasets públicos ya descargados.

## Riesgos clave

- El corpus propio generado con LLMs es ruta crítica, no un extra — sin él, cualquier cifra de
  detección corre el riesgo del 96%/7% de RAID (fiable solo contra su propio generador de
  entrenamiento).
- El Nivel B del perfilado no es validable con los datasets académicos disponibles — se
  documenta como no validado hasta un piloto real.
- El perfilado de clusters no tiene ningún dataset con verdad de terreno — se evalúa con
  precisión-en-top-k, nunca con un AUC inventado.

## Estructura prevista del repo

```
fake-review-detector/
├── .gitignore
├── requirements.txt      # (siguiente sesión)
├── README.md
├── data.py               # descarga/prepara datasets académicos + genera el corpus propio con LLMs
├── features_text.py      # T1 zero-shot + T2 DeBERTa-v3 afinado + T3 estilometría
├── features_graph.py     # grafo reviewer-reviewer, burst detection, Louvain, embeddings
├── profile_cluster.py    # perfil de cada cluster: señales por nivel + modelos nulos + lenguaje estimativo
├── train.py               # entrena T1/T2/T3, el scorer de grafo y el modelo de fusión
├── predict.py              # scoring end-to-end (texto pegado / CSV) con los modelos ya entrenados
├── report.py                # exporta el informe de evidencia (JSON + PDF)
├── app.py                    # Streamlit: el flujo self-serve (S2)
├── docs/index.html            # landing page (S1)
└── outputs/
    ├── models/
    ├── metrics.json
    └── *.png / *.json
```

Ninguno de estos ficheros existe todavía — es el plan para cuando arranque la implementación.
