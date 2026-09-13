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
"""

import collections
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import scipy.io

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_raw"

OTT_URL = "https://myleott.com/op_spam_v1.4.zip"
ORCG_URL = "https://osf.io/download/3vds7/"
YELPCHI_URL = "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip"

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
