"""Descarga y prepara los datasets académicos de la Fase 0.

Fuentes (ambas de acceso directo, sin cuenta ni API key):
- Ott et al., Deceptive Opinion Spam Corpus v1.4 (1.600 reviews de hoteles,
  mitad reales/mitad escritas por humanos en MTurk para parecer falsas).
- Salminen et al., Fake Reviews Dataset (40.432 reviews de Amazon, mitad
  reales, mitad generadas con GPT-2) — alojado en OSF, es el dataset detrás
  del "Fake Reviews Dataset" que circula también en Kaggle.

Ninguno de los dos contiene reviews escritas por LLMs modernos: Ott es de
2011-2013 (pre-LLM) y Salminen usa GPT-2 (2019). Sirven de baseline, no de
techo de dificultad — ver README, sección "Estrategia de datos".
"""

import collections
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_raw"

OTT_URL = "https://myleott.com/op_spam_v1.4.zip"
ORCG_URL = "https://osf.io/download/3vds7/"

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
