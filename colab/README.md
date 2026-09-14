# Frente GPU vía Google Colab

Esta carpeta contiene el trabajo que necesita GPU y que, mientras el ordenador de casa
(RTX 5060 Ti) no esté accesible, se puede hacer en el tier gratuito de Google Colab (T4).

| Notebook | Qué hace | Tiempo en T4 |
|---|---|---|
| `bwgnn_yelpchi.ipynb` | Entrena **BWGNN** (Beta Wavelet GNN, ICML 2022) sobre Yelp-Chi y lo compara con un MLP sin grafo, un GCN paso-bajo y las referencias que ya tiene el proyecto | **~7–20 min** |

## Cómo ejecutarlo (5 pasos, sin depurar nada)

1. Abre <https://colab.research.google.com> y entra con tu cuenta de Google.
2. `Archivo` → `Subir cuaderno` → arrastra `bwgnn_yelpchi.ipynb`.
   (Alternativa: `Archivo` → `Abrir cuaderno` → pestaña `GitHub` → pega la URL del repo
   y elige el fichero, si ya lo has pusheado.)
3. **Activa la GPU**: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` →
   en `Acelerador por hardware` elige **`T4 GPU`** → `Guardar`.
4. `Entorno de ejecución` → `Ejecutar todo`.
5. Al terminar, el navegador descarga solo un fichero **`bwgnn_yelpchi_results.json`**.
   Ese es el entregable: pásamelo de vuelta y lo integro en el repo.

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

## Si algo va mal

| Síntoma | Qué hacer |
|---|---|
| La celda 1 avisa de que no hay GPU | `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `T4 GPU`, y después `Ejecutar todo` otra vez |
| `No hay GPUs disponibles` al cambiar el entorno | El tier gratuito tiene cuota diaria. Espera unas horas, o ejecútalo en CPU (~3-4 h) |
| Falla la descarga del dataset | Reejecuta solo esa celda: si el `.mat` ya existe no lo vuelve a bajar |
| Un `[BAJO]` en el control de reproducción del final | La implementación se ha quedado >5 puntos de AUC por debajo del paper. Sube `EPOCHS`, o prueba otro `d` con la celda opcional de barrido |
| Un `assert` salta en la celda de carga | El fichero descargado no es el esperado. Borra `/content/data` y reejecuta — es mejor que romper ahí que publicar métricas de otro dataset |
| El runtime se desconecta a media rejilla | Vuelve a `Ejecutar todo`. No hay estado que recuperar, el notebook es reproducible con `random_state=42` |
| No se descarga el JSON al final | Está en el panel izquierdo de Colab (icono de carpeta) como `bwgnn_yelpchi_results.json`; descárgalo a mano |

## Referencias

- Tang et al., *Rethinking Graph Neural Networks for Anomaly Detection*, ICML 2022 —
  <https://arxiv.org/abs/2205.15508> (código oficial: `squareRoot3/Rethinking-Anomaly-Detection`).
  **Ojo**: el enlace que da `INVESTIGACION_GRAFOS.md` para BWGNN (`arxiv.org/pdf/2312.06441`) no
  es este paper; el identificador bueno es **2205.15508**, verificado en esta sesión.
- Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged
  Fraudsters* (CARE-GNN), CIKM 2020 — origen del `YelpChi.mat` preprocesado
- Rayana & Akoglu, *Collective Opinion Spam Detection* (SpEagle), KDD 2015 — la referencia
  de ~0,78 AUC que cita el proyecto
