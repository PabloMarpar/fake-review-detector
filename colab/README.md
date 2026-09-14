# Frente GPU vía Google Colab

Esta carpeta contiene el trabajo que necesita GPU y que, mientras el ordenador de casa
(RTX 5060 Ti) no esté accesible, se puede hacer en el tier gratuito de Google Colab (T4).

| Notebook | Qué hace | Tiempo en T4 |
|---|---|---|
| `bwgnn_yelpchi.ipynb` | Entrena **BWGNN** (Beta Wavelet GNN, ICML 2022) sobre Yelp-Chi y lo compara con un MLP sin grafo, un GCN paso-bajo y las referencias que ya tiene el proyecto | **~7–20 min** |
| `bwgnn_v2.ipynb` | **Parte A**: mejora BWGNN hetero sobre Yelp-Chi (más épocas con early stopping real, búsqueda de hiperparámetros, ensemble de semillas) partiendo del 0,9120 AUC / 0,6927 AP ya medido. **Parte B**: porta BWGNN hetero a **Yelp-NYC** (359.052 reviews, el dataset real del producto) — antes inviable por memoria, resuelto con un truco de agregación exacta por scatter | **~30–65 min** |

## `bwgnn_v2.ipynb` — cómo ejecutarlo (5 pasos, sin depurar nada)

Es el notebook a correr ahora mismo. Mismo procedimiento que el anterior:

1. Abre <https://colab.research.google.com> y entra con tu cuenta de Google.
2. `Archivo` → `Abrir cuaderno` → pestaña `GitHub` → pega la URL del repo
   (`PabloMarpar/fake-review-detector`) → elige `colab/bwgnn_v2.ipynb`.
   (Alternativa: `Archivo` → `Subir cuaderno` si lo tienes descargado a mano.)
3. **Activa la GPU**: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` →
   `T4 GPU` → `Guardar`.
4. `Entorno de ejecución` → `Ejecutar todo`.
5. Al terminar, el navegador descarga **`bwgnn_v2_results.json`**. Ese es el entregable:
   pásamelo de vuelta (o solo los números que salgan) y lo incorporo al proyecto.

**No hace falta subir ni descargar ningún dato a mano**: la Parte A descarga `YelpChi.zip`
igual que el notebook anterior, y la Parte B descarga `data_bundles/yelpnyc_bundle.npz`
(12 MB, las 24 features de Yelp-NYC ya calculadas y empaquetadas) directamente del repo. Si
la descarga automática del bundle falla (por ejemplo, si el fichero cambió y aún no está
pusheado), el notebook lo dice y basta con subir `yelpnyc_bundle.npz` a mano al panel de
archivos de la izquierda.

Este notebook se ha probado de punta a punta en CPU local con datos sintéticos antes de
subirlo (mismo código, parámetros reducidos) para detectar errores de antemano — no debería
hacer falta depurar nada en Colab, pero si algo falla, la sección "Si algo va mal" de abajo
también aplica a este notebook.

La **primera celda comprueba que hay GPU** y lo dice en grande si no la hay. Si ves el aviso,
vuelve al paso 3 antes de seguir: en CPU funciona igual, pero pasa de ~10-15 minutos a unas
3-4 horas.

## Qué esperar mientras corre

| Paso | Qué verás | Tiempo |
|---|---|---|
| 1. Entorno | Versiones de torch/numpy/scipy y el nombre de la GPU (`Tesla T4, ~15 GB`) | instantáneo |
| 2. Dataset | Descarga de `YelpChi.zip` (17 MB) y `loadmat` del `.mat` (198 MB) | 1–2 min |
| 3. Laplacianos | Cuatro grafos esparsos construidos y una comprobación numérica | 10–30 s |
| 4–6. Modelos | Definiciones y splits, sin cómputo | instantáneo |
| 7. Rejilla | 8 entrenamientos de 200 épocas con traza de `loss` y `val_AP` | 5–15 min |
| 8. Resultados | Tabla comparativa + comparación contra Louvain/CARE-GNN/SpEagle/el propio paper | instantáneo |
| 9. JSON | Se guarda y se descarga `bwgnn_yelpchi_results.json` | instantáneo |
| 10. Opcional | Barrido de hiperparámetros — **viene desactivado** (`RUN_SWEEP = False`); ponlo a `True` y reejecuta esas dos celdas si lo quieres | +10–20 min |

**De dónde sale la estimación**: la rejilla completa se cronometró en CPU (3 épocas × 8 configs
= 184 s), que extrapolado a 200 épocas son ~3,4 h en CPU. El rango asume que una T4 acelera la
multiplicación esparsa entre 15x y 40x, lo típico para un grafo de este tamaño — **no está
cronometrado en una T4 real**, así que es un orden de magnitud, no una promesa.

Todo cabe de sobra en una sesión gratuita de Colab (límite ~12 h, o ~90 min de inactividad).
Aun así, **no cierres la pestaña ni dejes el portátil dormirse** mientras corre, o Colab
desconecta el runtime y hay que volver a empezar.

## No hay que instalar ni subir nada

- **El dataset se descarga solo** desde el repo de CARE-GNN
  (`https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip`, 17 MB), el mismo fichero
  que usa `data.py::load_yelpchi_graph_dataset()` en este repo.
- **No se instala ninguna librería.** El notebook usa solo lo que Colab ya trae: `torch`,
  `numpy`, `scipy`, `scikit-learn`, `pandas`.

Esto último es deliberado y es lo que hace que el notebook no se rompa. El código oficial de
BWGNN depende de **DGL**, que lleva sin release desde la 2.4.0 y soporta como mucho PyTorch
2.4, mientras que Colab hoy trae **PyTorch 2.11** — instalar DGL ahí es la causa número uno de
que un notebook de este tipo no arranque. PyTorch Geometric 2.8 sí soportaría 2.11, pero
tampoco hace falta: BWGNN es multiplicación esparsa por potencias del Laplaciano normalizado, y
eso `torch.sparse` lo hace de forma nativa. Así que el notebook lleva una **implementación
propia de BWGNN en PyTorch puro**, con los coeficientes de las wavelets Beta verificados contra
la función `calculate_theta2` del repo oficial de los autores.

## Qué pregunta responde

La de la ronda 4 de `INVESTIGACION_GRAFOS.md`: **¿una GNN entrenada de verdad supera a los
métodos baratos sin entrenamiento (Louvain, OddBall, FRAUDAR, Leiden), y por cuánto?**

Dos cosas que el notebook hace a propósito y que conviene tener presentes al leer el JSON:

- **Protocolo limpio, sin *target encoding*.** Las señales de grafo que ya tiene el proyecto
  puntúan cada comunidad de Louvain con la tasa de fraude de sus propios miembros, o sea usando
  las etiquetas que luego evalúan. Eso infla el AUC y no es desplegable en un cliente real. El
  número de este notebook no tiene ese problema. **Consecuencia: el 0,814 de Louvain sobre
  `net_rur` no es un rival justo, juega con ventaja** — si BWGNN se queda cerca, ya es mejor
  resultado de lo que el número sugiere.
- **Métricas que no engañan con 14,5 % de positivos.** El ROC-AUC es optimista con clases
  desbalanceadas, así que la métrica principal es **Average Precision (PR-AUC)**, leída contra
  la base rate (0,145 = azar), más precisión/recall/lift en el top 1 % / 5 % / 10 %, que es lo
  que un cliente nota de verdad al revisar una cola priorizada. El F1-macro se calcula con el
  umbral que lo maximiza **en validación**, no con 0,5 — con cross-entropy ponderada el 0,5 no
  significa nada y no sería comparable con el número del paper.

Además de BWGNN se entrenan dos controles que cuestan casi nada:

- **MLP sin grafo** — las mismas 32 features, ninguna propagación. Dice cuánto aporta el grafo.
- **GCN paso-bajo** — misma arquitectura, filtro clásico en vez del banco Beta. Contrasta la
  tesis del paper: si GCN ≈ MLP pero BWGNN supera a ambos, la heterofilia diagnosticada en la
  ronda 1 de `INVESTIGACION_GRAFOS.md` queda confirmada.

Y dos regímenes de etiquetas: **40 %** y **1 %** — que son exactamente los dos que usa el paper,
así que hay un ancla de cordura directa. El 1 % es además el escenario de producto (un cliente
nuevo casi no tiene etiquetas) y sale bastante peor, que es justo el dato honesto que interesa.

Cifras publicadas (Tabla 2 del paper, sobre este mismo dataset) que el notebook imprime y
compara automáticamente al final:

| | 40 % train | 1 % train |
|---|---|---|
| BWGNN (homo) | AUC 84,03 / F1-macro 71,00 | AUC 72,01 / F1-macro 61,15 |
| BWGNN (hetero) | AUC 90,54 / F1-macro 76,96 | AUC 76,95 / F1-macro 67,02 |

Si la implementación del notebook se queda a más de 5 puntos de AUC por debajo, lo dice en
grande: en ese caso el sospechoso es la implementación, no el dataset.

## Qué pregunta responde `bwgnn_v2.ipynb`

Dos, una por parte:

- **Parte A**: en la sesión anterior, `best_epoch` fue 196 de 200 en BWGNN hetero — el modelo
  seguía mejorando cuando se acabó el entrenamiento. ¿Cuánto hay en la mesa si se entrena más
  (con early stopping real) y se afinan `wavelet_order`/`hidden`/`lr`/`weight_decay`?
- **Parte B**: ¿se sostiene la ventaja de BWGNN hetero (0,9120 AUC / 0,6927 AP en YelpChi) al
  portarlo al dataset real del producto? Y, más importante: ¿aguanta bajo un protocolo
  estricto — split agrupado por reviewer, agrupado por negocio (cliente nuevo), y temporal —
  en vez del split aleatorio que sabemos que infla la cifra de LightGBM (0,8448/0,3866)?

**El truco que hace viable la Parte B**: `net_rsr` de Yelp-NYC tiene decenas de millones de
aristas si se construyen explícitamente — inviable en una T4. Pero como la relación es una
unión de cliques disjuntos (cada nodo conectado a todos los demás de su grupo y a nadie más,
ya verificado en sesiones anteriores del proyecto), la propagación normalizada de BWGNN se
reduce matemáticamente a una media por grupo, calculable con `index_add_` (scatter) sin
construir ni una sola arista. Verificado antes de escribir el notebook que este método da
resultados numéricamente idénticos (diferencia <1e-6) a construir la matriz sparse real y
multiplicar.

## Si algo va mal

| Síntoma | Qué hacer |
|---|---|
| La celda 1 avisa de que no hay GPU | `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `T4 GPU`, y después `Ejecutar todo` otra vez |
| `No hay GPUs disponibles` al cambiar el entorno | El tier gratuito tiene cuota diaria. Espera unas horas, o ejecútalo en CPU (~3-4 h) |
| Falla la descarga del dataset | Reejecuta solo esa celda: si el `.mat` ya existe no lo vuelve a bajar |
| Un `[BAJO]` en el control de reproducción del final | La implementación se ha quedado >5 puntos de AUC por debajo del paper. Sube `EPOCHS`, o prueba otro `d` con la celda opcional de barrido |
| Un `assert` salta en la celda de carga | El fichero descargado no es el esperado. Borra `/content/data` y reejecuta — es mejor que romper ahí que publicar métricas de otro dataset |
| El runtime se desconecta a media rejilla | Vuelve a `Ejecutar todo`. No hay estado que recuperar, el notebook es reproducible con `random_state=42` |
| No se descarga el JSON al final | Está en el panel izquierdo de Colab (icono de carpeta) como `bwgnn_yelpchi_results.json` / `bwgnn_v2_results.json`; descárgalo a mano |
| (`bwgnn_v2.ipynb`) falla la descarga de `yelpnyc_bundle.npz` | Sube el fichero a mano al panel de archivos de la izquierda, en la ruta que indica el aviso, y reejecuta esa celda |
| (`bwgnn_v2.ipynb`) el aviso de "Reproducido" no coincide con la referencia | Revisar antes de seguir: las mejoras de las celdas siguientes no son comparables si la reproducción del baseline falla |

## Referencias

- Tang et al., *Rethinking Graph Neural Networks for Anomaly Detection*, ICML 2022 —
  <https://arxiv.org/abs/2205.15508> (código oficial: `squareRoot3/Rethinking-Anomaly-Detection`).
  **Ojo**: el enlace que da `INVESTIGACION_GRAFOS.md` para BWGNN (`arxiv.org/pdf/2312.06441`) no
  es este paper; el identificador bueno es **2205.15508**, verificado en esta sesión.
- Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged
  Fraudsters* (CARE-GNN), CIKM 2020 — origen del `YelpChi.mat` preprocesado
- Rayana & Akoglu, *Collective Opinion Spam Detection* (SpEagle), KDD 2015 — la referencia
  de ~0,78 AUC que cita el proyecto
- Tang et al., *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection*,
  NeurIPS 2023 D&B — <https://arxiv.org/abs/2306.12251>. Encuentra que árboles con agregación
  de vecindario ("XGB-Graph") baten a la mejor GNN en +12,9 puntos de AUPRC de media sobre 10
  datasets (YelpChi 70% train: XGB-Graph 91,11 AUPRC vs. BWGNN 61,53) — motivo por el que la
  Fase 2 del plan (agregación de vecindario sobre árboles, sin GNN) sigue siendo la comparación
  pendiente más importante después de este notebook.
