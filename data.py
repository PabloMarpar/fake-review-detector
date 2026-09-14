"""Descarga y prepara los datasets académicos de la Fase 0 y de la Fase 1.

Fuentes de texto (Fase 0, ambas de acceso directo, sin cuenta ni API key):
- Ott et al., Deceptive Opinion Spam Corpus v1.4 (1.600 reviews de hoteles,
  mitad reales/mitad escritas por humanos en MTurk para parecer falsas).
- Salminen et al., Fake Reviews Dataset (40.432 reviews de Amazon, mitad
  reales, mitad generadas con GPT-2) — alojado en OSF, es el dataset detrás
  del "Fake Reviews Dataset" que circula también en Kaggle.

Ninguno de los dos contiene reviews escritas por LLMs modernos: Ott es de
2011-2013 (pre-LLM) y Salminen usa GPT-2 (2019). Sirven de baseline, no de
techo de dificultad — ver README, sección "Estrategia de datos".

Fuente de grafo (Fase 1):
- Yelp-Chi preprocesado, distribuido sin gate de acceso en el repo de
  CARE-GNN (Dou et al., CIKM 2020): https://github.com/YingtongDou/CARE-GNN,
  fichero `data/YelpChi.zip`. Se eligió esta versión, en vez del Yelp-NYC/ZIP
  original de Rayana & Akoglu (que sí trae texto), porque ese último exige
  pedirlo por email a la autora vía https://odds.cs.stonybrook.edu/yelpnyc-dataset/
  — sin descarga automatizable ni plazo garantizado. **Limitación real de
  esta alternativa, no disimulada**: el `.mat` de Yelp-Chi NO trae el texto
  original de la review, solo grafo + 32 features numéricas ya vectorizadas
  por los autores de CARE-GNN + etiqueta de fraude — sirve para grafo/
  clustering/perfilado estructural (Nivel A del perfilado, README), pero no
  para fusionar con la señal de texto T2 ni para Nivel C (near-duplicates de
  texto). Si llega el email de Yelp-NYC/ZIP más adelante, ese sí trae texto y
  habrá que revisar si conviene migrar.
- Amazon (Dou et al., CIKM 2020 / PC-GNN, WWW 2021), dataset hermano de
  Yelp-Chi, mismo estilo de preprocesado pero grafo usuario-usuario (fraude
  en reviews de producto, no de negocio). Copia oficial en el hosting de DGL
  (proyecto open source de AWS/Amazon), `https://data.dgl.ai/dataset/
  FraudAmazon.zip` — verificado en esta sesión con una descarga real: 200,
  `application/zip`, servido por S3/CloudFront, sin cuenta ni email.
  **Misma limitación que Yelp-Chi, no disimulada**: tampoco trae texto de
  review, solo grafo + 25 features numéricas + etiqueta.
- **Yelp-NYC (Rayana & Akoglu, KDD 2015)** — el dataset que sí trae texto +
  identificadores para construir el grafo reviewer-negocio (a diferencia de
  Yelp-Chi/Amazon de arriba, que solo dan grafo ya proyectado). La fuente
  oficial (https://odds.cs.stonybrook.edu/yelpnyc-dataset/) sigue exigiendo
  pedirlo por email a la autora (ya enviado, sin respuesta todavía) — **esta
  sesión usa mientras tanto un mirror de Kaggle** (`ahtxham/
  yelp-nyc-labelled-dataset`), única fuente de esta lista que rompe el patrón
  "sin cuenta" del resto — necesita `KAGGLE_USERNAME`/`KAGGLE_KEY` en `.env`.
  Ver docstring de `load_yelpnyc_dataset` para los hallazgos de verificación
  (cabeceras de columna mal etiquetadas en el mirror, pero datos reales y
  correctos una vez identificado el mapeo real).
- **bretthollenbeck/fake-reviews-data** — reviews de Amazon con etiqueta de
  campaña de fraude conocida, grafo reviewer↔producto real (a diferencia de
  Yelp-Chi/Amazon de arriba) y, novedad en este fichero, una fecha real de
  inicio de campaña (no un proxy de spam). Repo en GitHub
  (https://github.com/bretthollenbeck/fake-reviews-data, MIT, solo
  README+LICENSE) pero el CSV real está alojado aparte, en el WordPress
  personal del autor — descarga directa sin cuenta. Ver docstring de
  `load_bretthollenbeck_dataset` para los hallazgos de verificación (dos
  tipos de etiqueta de fraude que no hay que confundir, y por qué se usa la
  manual/curada como label principal en vez de la del clasificador
  automático).
"""

import collections
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import scipy.io
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_raw"

load_dotenv(BASE_DIR / ".env")

OTT_URL = "https://myleott.com/op_spam_v1.4.zip"
ORCG_URL = "https://osf.io/download/3vds7/"
YELPCHI_URL = "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip"
AMAZON_URL = "https://data.dgl.ai/dataset/FraudAmazon.zip"
YELPNYC_KAGGLE_DATASET = "ahtxham/yelp-nyc-labelled-dataset"
BRETTHOLLENBECK_URL = (
    "https://bretthollenbeckcom.wordpress.com/wp-content/uploads/2026/03/"
    "public_reviews_dataset_cleaned.csv_.zip"
)

OUTPUT_CSV = BASE_DIR / "reviews_baseline.csv"


def _download(url: str, dest: Path) -> Path:
    """Descarga `url` a `dest` si no existe ya en disco (cache simple)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_ott_corpus() -> pd.DataFrame:
    """Parsea el corpus de Ott a partir de la estructura real del zip:
    {positive,negative}_polarity / {deceptive_from_MTurk,truthful_from_*} / foldN / *.txt
    """
    zip_path = _download(OTT_URL, RAW_DIR / "op_spam_v1.4.zip")
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            parts = Path(name).parts
            if "deceptive_from_MTurk" in parts:
                label = "deceptive"
            elif any(p.startswith("truthful") for p in parts):
                label = "truthful"
            else:
                continue
            polarity = "positive" if "positive_polarity" in parts else "negative"
            fold = next((p for p in parts if p.startswith("fold")), None)
            filename = parts[-1]
            hotel_match = re.match(r"[a-z]_([a-z]+)_\d+\.txt", filename)
            hotel = hotel_match.group(1) if hotel_match else None
            text = zf.read(name).decode("utf-8", errors="replace").strip()
            rows.append(
                {
                    "text": text,
                    "is_fake": label == "deceptive",
                    "polarity": polarity,
                    "fold": fold,
                    "hotel": hotel,
                    "source_dataset": "ott_2013",
                    "generator": "human_mturk" if label == "deceptive" else "human_real",
                }
            )
    return pd.DataFrame(rows)


def load_orcg_dataset() -> pd.DataFrame:
    """Parsea el dataset OR/CG (columnas confirmadas: category, rating, label, text_)."""
    csv_path = _download(ORCG_URL, RAW_DIR / "fake_reviews_dataset.csv")
    df = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "text": df["text_"].astype(str).str.strip(),
            "is_fake": df["label"] == "CG",
            "category": df["category"],
            "rating": df["rating"],
            "source_dataset": "salminen_2022_gpt2",
            "generator": df["label"].map({"CG": "gpt2", "OR": "human_real"}),
        }
    )


def load_yelpchi_graph_dataset() -> dict:
    """Descarga y carga el grafo Yelp-Chi preprocesado de CARE-GNN.

    Devuelve un dict con:
    - `net_rur`, `net_rtr`, `net_rsr`: matrices sparse (`scipy.sparse.csc_matrix`)
      de (45.954 x 45.954), un grafo homogéneo review-review cada una (R-U-R =
      mismo usuario, R-T-R = mismo negocio y misma valoración en la misma
      semana/mes, R-S-R = mismo negocio y mismo rating).
    - `net_homo`: matriz sparse adicional presente en el `.mat` bajo la clave
      `homo`, no documentada en la descripción original que se nos dio —
      verificado en esta sesión que es exactamente la unión booleana de las
      tres redes anteriores (`homo[i,j] != 0` si y solo si al menos una de
      `net_rur`/`net_rtr`/`net_rsr` tiene una arista ahí). Se expone tal cual
      por si conviene usarla como grafo combinado en `features_graph.py`, sin
      recalcularla.
    - `features`: matriz sparse (45.954 x 32) de features numéricas ya
      calculadas por los autores de CARE-GNN (no son features propias de este
      proyecto), valores normalizados en [0, 1] (comprobado: min=0.0, max=1.0).
    - `label`: array 1D de longitud 45.954, `0` = genuina, `1` = fraudulenta/
      spam (etiqueta original de Rayana & Akoglu). Distribución real
      comprobada: 39.277 genuinas / 6.677 fraude (14,5% positivos).

    Hallazgo honesto importante: el `.mat` tiene 45.954 filas/nodos, no las
    ~67.395 reviews que reporta el paper original de Yelp-Chi (Rayana &
    Akoglu) ni las que cita habitualmente la literatura sobre este dataset —
    el preprocesado de CARE-GNN filtra el dataset original (probablemente a
    reviewers/reviews con suficiente conectividad en el grafo, no
    verificado más allá de constatar la diferencia). No se ha encontrado
    ninguna nota de los autores de CARE-GNN que documente el criterio exacto
    del filtrado; quien retome esto debería tratar 45.954 como el tamaño real
    de este dataset, no como un error de carga.
    """
    zip_path = _download(YELPCHI_URL, RAW_DIR / "YelpChi.zip")
    mat_path = RAW_DIR / "YelpChi.mat"
    if not mat_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("YelpChi.mat", RAW_DIR)

    # Comprobado en esta sesión: es MATLAB 5.0 clásico (cabecera "MATLAB 5.0
    # MAT-file"), no el formato v7.3/HDF5 — `scipy.io.loadmat` funciona
    # directamente, no hace falta `h5py`.
    mat = scipy.io.loadmat(mat_path)

    return {
        "net_rur": mat["net_rur"],
        "net_rtr": mat["net_rtr"],
        "net_rsr": mat["net_rsr"],
        "net_homo": mat["homo"],
        "features": mat["features"],
        "label": mat["label"].reshape(-1),
    }


def load_amazon_graph_dataset() -> dict:
    """Descarga y carga el grafo de fraude Amazon (Dou et al., CIKM 2020 /
    PC-GNN WWW 2021), dataset hermano de Yelp-Chi pero con grafo
    usuario-usuario (fraude en reviews de producto) en vez de review-review.

    Devuelve un dict con:
    - `net_upu`, `net_usu`, `net_uvu`: matrices sparse binarias
      (`scipy.sparse.csc_matrix`, valores comprobados: solo `1.0`, sin pesos)
      de (11.944 x 11.944), tres grafos homogéneos usuario-usuario. Los
      nombres de clave coinciden exactamente con lo esperado de la
      literatura (U-P-U, U-S-U, U-V-U); el significado semántico exacto de
      cada relación (mismo producto / mismo patrón de rating / misma
      review-voto) se toma de la documentación de CARE-GNN/PC-GNN, no se ha
      verificado de forma independiente más allá de confirmar claves, forma
      y que son binarias.
    - `net_homo`: matriz sparse adicional bajo la clave `homo`, misma forma
      (11.944 x 11.944). **Hallazgo distinto al de Yelp-Chi, no disimulado**:
      aquí `homo` NO es la unión booleana exacta de `net_upu`/`net_usu`/
      `net_uvu` — comprobado en esta sesión que hay 2.048.630 aristas en la
      unión que no están en `homo`, y 2.010.262 aristas en `homo` que no
      están en la unión de las tres (de un total de ~8,8M aristas cada una).
      En Yelp-Chi `homo` sí era exactamente esa unión; en Amazon no se ha
      identificado qué combinación exacta produce `homo` (no es la unión
      simple, ni coincide con ninguna de las tres por separado) — se expone
      tal cual, sin asumir que sea equivalente a construirla a mano.
    - `features`: matriz sparse (11.944 x 25) de features numéricas ya
      calculadas por los autores de CARE-GNN/PC-GNN (no propias de este
      proyecto). **25 dimensiones, no 32 como Yelp-Chi** (confirmado).
      **A diferencia de Yelp-Chi, aquí NO están normalizadas a [0, 1]**:
      comprobado min=-1.0, max=5525.0, y 16 de las 25 columnas superan 1.0 —
      quien use estas features junto a las de Yelp-Chi en el mismo pipeline
      necesita normalizar antes, no asumir la misma escala.
    - `label`: array 1D de longitud 11.944, `0` = genuina, `1` = fraudulenta
      (en el `.mat` viene como fila `(1, 11944)`, se aplana con
      `.reshape(-1)` igual que en Yelp-Chi). Distribución real comprobada:
      11.123 genuinas / 821 fraude (6,9% positivos). **A diferencia de
      Yelp-Chi (que tenía 45.954 nodos frente a los ~67.395 publicados), aquí
      el conteo de 11.944 nodos / 821 fraudsters (~6,9%) sí coincide con lo
      que reporta habitualmente la literatura sobre este dataset** — no hay
      discrepancia de tamaño que documentar en este caso.

    Sin texto de review, igual que Yelp-Chi: solo grafo + features + etiqueta.
    Sirve para Nivel A del perfilado (estructural), no para fusión con T2 ni
    Nivel C (near-duplicates de texto).
    """
    zip_path = _download(AMAZON_URL, RAW_DIR / "FraudAmazon.zip")
    mat_path = RAW_DIR / "Amazon.mat"
    if not mat_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("Amazon.mat", RAW_DIR)

    # Comprobado en esta sesión: igual que YelpChi.mat, es MATLAB 5.0 clásico
    # (cabecera "MATLAB 5.0 MAT-file"), no v7.3/HDF5 — `scipy.io.loadmat`
    # funciona directamente, no hace falta `h5py`.
    mat = scipy.io.loadmat(mat_path)

    return {
        "net_upu": mat["net_upu"],
        "net_usu": mat["net_usu"],
        "net_uvu": mat["net_uvu"],
        "net_homo": mat["homo"],
        "features": mat["features"],
        "label": mat["label"].reshape(-1),
    }


def load_yelpnyc_dataset() -> pd.DataFrame:
    """Descarga (vía API de Kaggle) y carga Yelp-NYC (Rayana & Akoglu, KDD
    2015) con texto de review + identificadores de reviewer/negocio, para
    poder construir el grafo bipartito reviewer-negocio en
    `features_graph.py` Y correr las señales de texto (T1/T2/T3) sobre el
    mismo dato — a diferencia de Yelp-Chi/Amazon, que solo dan el grafo ya
    proyectado sin texto.

    Requiere `KAGGLE_USERNAME` y `KAGGLE_KEY` en `.env` (única fuente de
    `data.py` que necesita cuenta; ver docstring del módulo). Usa la API de
    Kaggle (`kaggle.api.dataset_download_files`), no el CLI.

    **Hallazgo real de esta sesión — las cabeceras de columna del mirror de
    Kaggle están mal etiquetadas, pero el dato subyacente es correcto una vez
    identificado el mapeo real.** El mirror declara:
    - `Yelp NYC Metadata.csv`: columnas `Product_id, Product_id2, Rating,
      Label, Date` — pero `Product_id` tiene 160.225 valores únicos y
      `Product_id2` solo 923, justo al revés de lo que dicen sus nombres
      (923 negocios / 160.225 reviewers es la cifra publicada del paper
      original). `Product_id` es en realidad el id de reviewer y
      `Product_id2` el id de negocio.
    - `yelp.csv`: columnas `Review_id, Product_id, Date, Review` — mismo
      problema, `Review_id` es el id de reviewer (mismo rango que el
      `Product_id` mal llamado así en el otro fichero) y `Product_id` aquí sí
      es el id de negocio.
    - **Verificado con un join real, no solo por rango de valores**: al
      renombrar con esta hipótesis y cruzar ambos ficheros por
      (reviewer_id, business_id, date), los 359.052 registros de metadata
      encajan 1:1 con los 359.052 de `yelp.csv` sin ninguno huérfano — si el
      mapeo estuviera mal, el cruce no habría dado 100% de coincidencia.
    - **Cifras finales, coinciden con lo publicado por Rayana & Akoglu**
      (a diferencia de Yelp-Chi, donde el `.mat` de CARE-GNN no cuadraba con
      el paper): 923 negocios, 160.225 reviewers, 359.052 reviews. Label -1 =
      filtered (proxy de fake, 36.885 filas, 10,3%) / 1 = recommended (proxy
      de genuina, 322.167 filas, 89,7%) — el 10,3% de filtered cuadra con la
      tasa que reporta la literatura para Yelp-NYC.
    - Texto sin nulos ni mojibake tras el cruce completo; 896 duplicados
      exactos de texto (0,25%), **no eliminados aquí a propósito** — a
      diferencia de Ott/Salminen en `build_baseline_dataset`, quitar
      duplicados de texto aquí borraría aristas reviewer-negocio reales del
      grafo, no solo texto redundante.

    Devuelve un DataFrame con: `reviewer_id`, `business_id`, `rating`,
    `is_fake` (bool, `label == -1`), `date`, `text`, `source_dataset` fijo a
    `"yelpnyc_rayana_akoglu"`. Sin columna `generator` (no aplica: estas son
    reviews humanas reales, con o sin intención de engañar, no generadas por
    ningún modelo).
    """
    import kaggle

    dest_dir = RAW_DIR / "yelpnyc_kaggle"
    meta_path = dest_dir / "Yelp NYC Metadata.csv"
    text_path = dest_dir / "yelp.csv"
    if not meta_path.exists() or not text_path.exists():
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(YELPNYC_KAGGLE_DATASET, path=dest_dir, unzip=True)

    meta = pd.read_csv(meta_path)
    meta.columns = ["reviewer_id", "business_id", "rating", "label", "date"]
    text = pd.read_csv(text_path)
    text.columns = ["reviewer_id", "business_id", "date", "text"]

    merged = meta.merge(text, on=["reviewer_id", "business_id", "date"], how="inner")
    return pd.DataFrame(
        {
            "reviewer_id": merged["reviewer_id"],
            "business_id": merged["business_id"],
            "rating": merged["rating"],
            "is_fake": merged["label"] == -1,
            "date": merged["date"],
            "text": merged["text"],
            "source_dataset": "yelpnyc_rayana_akoglu",
        }
    )


def load_bretthollenbeck_dataset() -> pd.DataFrame:
    """Descarga y carga el dataset de reviews de Amazon de
    `bretthollenbeck/fake-reviews-data`
    (https://github.com/bretthollenbeck/fake-reviews-data, MIT, sin cuenta).
    El repo de GitHub solo trae README+LICENSE — el CSV real está alojado
    aparte, en el WordPress personal del autor (descarga directa, sin cuenta
    ni gate de acceso).

    **Verificado en esta sesión descargando el zip real y cargando el CSV con
    pandas, no solo leyendo el README del repo** (que además no lista
    `reviewer_id`/`asin` entre sus columnas — sin bajar el CSV de verdad no
    habría quedado claro que este dataset sí trae grafo reviewer↔producto
    construible). Comprobado: 64.369.322 bytes descargados, coinciden exacto
    con el `Content-Length` del servidor. Mismo patrón de "zip con basura de
    macOS" que ya usa `load_yelpchi_graph_dataset`: el zip contiene
    `public_reviews_dataset_cleaned.csv` (280MB descomprimido) +
    `__MACOSX/._public_reviews_dataset_cleaned.csv` (277 bytes, metadata de
    Finder sin datos), que se ignora al extraer.

    **Es un dataset centrado en productos con campaña de fraude conocida, no
    una muestra aleatoria de Amazon** — comprobado: 381.734 reviews pero solo
    3.389 `asin` únicos (334.342 `reviewer_id` únicos), o sea que cada
    producto concentra muchísimas reviews/reviewers. Igual que ya se
    documentó la rareza de tamaño/distribución de Yelp-Chi y Amazon (Dou et
    al.) en sus propios docstrings de este fichero, no tratar esto como
    representativo de "Amazon en general". Rango real de `review_date`:
    2000-06-12 a 2021-01-19.

    **Dos tipos de etiqueta de fraude, no hay que confundirlos**:
    - `reviewer_classified_fake` / `reviewer_classified_honest` (columnas
      `bool`, sin nulos): salida de un **clasificador automático de un paper
      externo** citado en el README del repo original — es una *predicción*,
      no ground truth. Comprobado: `classified_fake` True en 57.667 filas,
      `classified_honest` True en 22.614.
    - `reviewer_labeled_fake` / `reviewer_labeled_honest` (columnas
      `float64`, con `NaN`): etiqueta **manual/curada — esta sí es ground
      truth real**, pero mucho más escasa: solo 80.281 de 381.734 filas
      (21%) la tienen. Comprobado cruzando ambas columnas dentro de esas
      80.281: 22.138 tienen `fake=1`, 10.802 tienen `honest=1`, y **47.341
      tienen las dos a 0** (ni fake ni honesto confirmado — no es lo mismo
      que "sin etiquetar": es un caso que el curador sí miró y no marcó en
      ningún extremo). Cero filas con las dos a 1 a la vez.
    - **Este loader usa `reviewer_labeled_fake` (la manual) como label
      principal para `is_fake`, no la automática** — mismo criterio que ya
      prioriza la etiqueta de Rayana & Akoglu sobre cualquier proxy en
      `load_yelpnyc_dataset`. `is_fake` es un booleano *nullable*
      (`pandas.BooleanDtype`): `True` donde `reviewer_labeled_fake == 1`,
      `False` donde `reviewer_labeled_fake == 0` (incluye tanto "honesto
      confirmado" como "ni fake ni honesto confirmado" — usar la columna
      extra `reviewer_labeled_honest` si hace falta separar esos dos casos),
      y `pandas.NA` en las 301.453 filas sin etiqueta manual — deliberadamente
      **no se rellena con `False`**, porque eso diría "no es fake" donde en
      realidad no hubo ningún juicio manual. `reviewer_classified_fake` y
      `reviewer_classified_honest` se exponen como columnas extra, sin usarse
      para calcular `is_fake`.

    **No se pierde la fecha de campaña**: columna extra `campaign_start_date`
    (de `fake_review_campaign_start_date`, no nula en 143.427 filas) — es la
    primera fecha de evento real de este proyecto (no un proxy de spam como
    en Yelp-Chi/Amazon/Yelp-NYC), útil para burst detection en
    `features_graph.py` más adelante aunque este loader no la usa todavía.

    **No trae fecha de alta de cuenta**: Nivel B del perfilado (README) sigue
    sin resolverse aquí, igual que en Yelp-Chi/Amazon/Yelp-NYC — no se ha
    inventado ningún proxy para aparentar que sí.

    9.329 duplicados exactos de `review_text` (2,4%), **no eliminados aquí a
    propósito**, mismo criterio que `load_yelpnyc_dataset`: quitarlos borraría
    aristas reviewer-producto reales del grafo, y en este dataset en
    concreto texto repetido puede ser en sí una señal de campaña coordinada,
    no solo ruido.

    Devuelve un DataFrame con las columnas estándar de este fichero
    (`reviewer_id`, `business_id` [de `asin`], `rating`, `date`, `text`,
    `is_fake`, `source_dataset` fijo a `"bretthollenbeck_amazon"`) más
    columnas específicas de este dataset que no encajan en el esquema
    estándar sin perder información: `review_id`, `review_title` (926
    nulos), `campaign_start_date`, `fake_review_product` (bool: True si el
    producto tiene una campaña de fraude conocida, en 160.553 filas),
    `reviewer_classified_fake`, `reviewer_classified_honest`,
    `reviewer_labeled_honest`, `review_is_removed_by_amazon` (float64 con
    NaN: False en 256.614, True en 38.945, NaN en 86.175 — el propio dataset
    no documenta el motivo de esos nulos). Sin columna `generator`: son
    reviews humanas (reales o de reviewers pagados dentro de una campaña),
    no generadas por ningún LLM conocido.
    """
    zip_path = _download(BRETTHOLLENBECK_URL, RAW_DIR / "bretthollenbeck_reviews.zip")
    csv_path = RAW_DIR / "public_reviews_dataset_cleaned.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("public_reviews_dataset_cleaned.csv", RAW_DIR)

    df = pd.read_csv(csv_path, low_memory=False)

    is_fake = df["reviewer_labeled_fake"].map({1.0: True, 0.0: False}).astype("boolean")

    return pd.DataFrame(
        {
            "reviewer_id": df["reviewer_id"],
            "business_id": df["asin"],
            "rating": df["review_rating"],
            "date": df["review_date"],
            "text": df["review_text"],
            "is_fake": is_fake,
            "source_dataset": "bretthollenbeck_amazon",
            "review_id": df["review_id"],
            "review_title": df["review_title"],
            "campaign_start_date": df["fake_review_campaign_start_date"],
            "fake_review_product": df["fake_review_product"],
            "reviewer_classified_fake": df["reviewer_classified_fake"],
            "reviewer_classified_honest": df["reviewer_classified_honest"],
            "reviewer_labeled_honest": df["reviewer_labeled_honest"],
            "review_is_removed_by_amazon": df["review_is_removed_by_amazon"],
        }
    )


def build_baseline_dataset() -> pd.DataFrame:
    ott = load_ott_corpus()
    orcg = load_orcg_dataset()
    combined = pd.concat([ott, orcg], ignore_index=True)
    combined = combined[combined["text"].str.len() > 0].drop_duplicates(subset="text")
    return combined


if __name__ == "__main__":
    df = build_baseline_dataset()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total reviews: {len(df)}")
    print(df.groupby("source_dataset")["is_fake"].value_counts())
    print("\nGenerador:")
    print(collections.Counter(df["generator"]))
    print(f"\nGuardado en {OUTPUT_CSV}")

    print("\n--- Yelp-Chi (grafo, Fase 1) ---")
    yelpchi = load_yelpchi_graph_dataset()
    n_nodes = yelpchi["label"].shape[0]
    n_fraud = int((yelpchi["label"] == 1).sum())
    n_genuine = int((yelpchi["label"] == 0).sum())
    print(f"Nodos/reviews: {n_nodes} (publicado ~67.395; ver docstring de "
          f"load_yelpchi_graph_dataset sobre esta diferencia)")
    print(f"Genuinas: {n_genuine} ({n_genuine / n_nodes:.1%}) | "
          f"Fraude: {n_fraud} ({n_fraud / n_nodes:.1%})")
    print(f"net_rur: {yelpchi['net_rur'].shape}, {yelpchi['net_rur'].nnz} aristas no-cero")
    print(f"net_rtr: {yelpchi['net_rtr'].shape}, {yelpchi['net_rtr'].nnz} aristas no-cero")
    print(f"net_rsr: {yelpchi['net_rsr'].shape}, {yelpchi['net_rsr'].nnz} aristas no-cero")
    print(f"net_homo (unión de las tres): {yelpchi['net_homo'].shape}, {yelpchi['net_homo'].nnz} aristas no-cero")
    print(f"features: {yelpchi['features'].shape} (32 features numéricas de CARE-GNN, no propias)")

    print("\n--- Amazon (grafo, Fase 1) ---")
    amazon = load_amazon_graph_dataset()
    n_nodes = amazon["label"].shape[0]
    n_fraud = int((amazon["label"] == 1).sum())
    n_genuine = int((amazon["label"] == 0).sum())
    print(f"Nodos/usuarios: {n_nodes} (coincide con lo publicado, a diferencia de Yelp-Chi)")
    print(f"Genuinas: {n_genuine} ({n_genuine / n_nodes:.1%}) | "
          f"Fraude: {n_fraud} ({n_fraud / n_nodes:.1%})")
    print(f"net_upu: {amazon['net_upu'].shape}, {amazon['net_upu'].nnz} aristas no-cero")
    print(f"net_usu: {amazon['net_usu'].shape}, {amazon['net_usu'].nnz} aristas no-cero")
    print(f"net_uvu: {amazon['net_uvu'].shape}, {amazon['net_uvu'].nnz} aristas no-cero")
    print(f"net_homo (NO es la unión exacta de las tres, ver docstring): {amazon['net_homo'].shape}, {amazon['net_homo'].nnz} aristas no-cero")
    print(f"features: {amazon['features'].shape} (25 features numéricas, NO normalizadas a [0,1], ver docstring)")

    print("\n--- Yelp-NYC (texto + grafo, Fase 1) ---")
    yelpnyc = load_yelpnyc_dataset()
    print(f"Filas: {len(yelpnyc)} (publicado: 359.052)")
    print(f"Negocios: {yelpnyc['business_id'].nunique()} (publicado: 923) | "
          f"Reviewers: {yelpnyc['reviewer_id'].nunique()} (publicado: 160.225)")
    print(yelpnyc["is_fake"].value_counts(normalize=True))

    print("\n--- bretthollenbeck/fake-reviews-data (texto + grafo, Fase 1) ---")
    bh = load_bretthollenbeck_dataset()
    print(f"Filas: {len(bh)}")
    print(f"Productos (asin): {bh['business_id'].nunique()} | "
          f"Reviewers: {bh['reviewer_id'].nunique()}")
    print("is_fake (label manual, con NA = sin etiquetar):")
    print(bh["is_fake"].value_counts(dropna=False))
    print(f"Con campaign_start_date: {bh['campaign_start_date'].notna().sum()}")
