"""Infraestructura de base de datos del proyecto (PostgreSQL en Neon).

De momento solo define el esquema de `reviews`, reflejando exactamente las
columnas de `reviews_baseline.csv` (ver `data.py`) -- no hay datos migrados
todavía ni tablas para clusters/grafo (ese código no existe aún).

Requiere `DATABASE_URL` en `.env` (ver `.env.example`), apuntando a un
proyecto de Neon con el prefijo `postgresql+psycopg://`.

Uso: python db.py
    Crea las tablas si no existen y muestra el resultado (idempotente).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Boolean, Float, String, Text, create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_fake: Mapped[bool] = mapped_column(Boolean, nullable=False)
    polarity: Mapped[str | None] = mapped_column(String(20))
    fold: Mapped[str | None] = mapped_column(String(20))
    hotel: Mapped[str | None] = mapped_column(String(100))
    source_dataset: Mapped[str] = mapped_column(String(50), nullable=False)
    generator: Mapped[str | None] = mapped_column(String(50))
    category: Mapped[str | None] = mapped_column(String(100))
    rating: Mapped[float | None] = mapped_column(Float)


def get_engine():
    """Crea el engine bajo demanda (nunca a nivel de módulo, para que un
    simple `import db` no falle si `.env` todavía no está configurado)."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Falta DATABASE_URL. Copia .env.example a .env y rellena la "
            "connection string de tu proyecto de Neon."
        )
    # pool_pre_ping evita fallos en la primera query tras el auto-suspend
    # del cómputo en el free tier de Neon.
    return create_engine(database_url, pool_pre_ping=True)


def get_session_factory(engine=None):
    return sessionmaker(bind=engine or get_engine())


def init_db():
    """Crea las tablas que falten (no toca las que ya existen)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    engine = init_db()
    print("Conectado a:", engine.url.render_as_string(hide_password=True))
    print("Tablas:", inspect(engine).get_table_names())
