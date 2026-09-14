"""Empaqueta las 24 features label-free de Yelp-NYC en un .npz compacto para
que el notebook de Colab (colab/bwgnn_v2.ipynb) pueda descargarlas desde git
sin necesitar credenciales de Kaggle ni recalcular ~18 min de features.

Las cachés fuente (`outputs/behavior_block1_features.csv`,
`outputs/behavior_text_similarity.csv`) pesan 76 MB + 9 MB y están excluidas
de git (ver .gitignore) -- igual que los modelos grandes del proyecto, son
reproducibles con `python features_behavior.py`, no hace falta commitearlas.
Este bundle es distinto: es pequeño a propósito (float32, sin las columnas de
grafo con fuga de etiquetas) para que SÍ se pueda commitear.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import data
from features_behavior import BLOCK1_CACHE, SIMILARITY_CACHE, BUSINESS_COLS, REVIEW_COLS, REVIEWER_COLS, SIMILARITY_COLS

OUT_DIR = Path(__file__).parent.parent / "data_bundles"
OUT_PATH = OUT_DIR / "yelpnyc_bundle.npz"

FEATURE_ORDER = REVIEW_COLS + REVIEWER_COLS + BUSINESS_COLS + SIMILARITY_COLS


def build() -> None:
    if not BLOCK1_CACHE.exists() or not SIMILARITY_CACHE.exists():
        raise SystemExit(
            f"Faltan cachés ({BLOCK1_CACHE} / {SIMILARITY_CACHE}). "
            "Correr primero `python features_behavior.py all`."
        )

    df = data.load_yelpnyc_dataset().reset_index(drop=True)
    block1 = pd.read_csv(BLOCK1_CACHE)
    similarity = pd.read_csv(SIMILARITY_CACHE)
    assert len(df) == len(block1) == len(similarity), "Las tres fuentes deben tener el mismo nº de filas (alineadas por posición)"

    X_df = pd.concat([block1[REVIEW_COLS + REVIEWER_COLS + BUSINESS_COLS], similarity[SIMILARITY_COLS]], axis=1)
    assert list(X_df.columns) == FEATURE_ORDER

    X = X_df.to_numpy(dtype=np.float32)
    y = df["is_fake"].to_numpy(dtype=np.uint8)

    reviewer_codes, _ = pd.factorize(df["reviewer_id"])
    business_codes, _ = pd.factorize(df["business_id"])
    rating = df["rating"].to_numpy(dtype=np.int8)

    dates = pd.to_datetime(df["date"])
    date_days = (dates - dates.min()).dt.days.to_numpy(dtype=np.int32)

    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X=X,
        feature_names=np.array(FEATURE_ORDER),
        y=y,
        reviewer_id=reviewer_codes.astype(np.int32),
        business_id=business_codes.astype(np.int32),
        rating=rating,
        date_days=date_days,
    )

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Guardado {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"X: {X.shape}, features: {FEATURE_ORDER}")
    print(f"y: {y.sum()} fraude / {len(y)} ({y.mean():.4f})")
    print(f"reviewers únicos: {reviewer_codes.max() + 1}, negocios únicos: {business_codes.max() + 1}")
    print(f"date_days: min={date_days.min()}, max={date_days.max()}")
    if size_mb > 80:
        print("AVISO: el bundle supera ~80 MB, considerar recortar a float16 las features menos sensibles antes de commitear.")


if __name__ == "__main__":
    build()
