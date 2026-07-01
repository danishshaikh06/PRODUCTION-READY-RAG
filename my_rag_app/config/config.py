"""
Database connection setup — engine, session factory, and table creation.
Reads connection details from .env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from my_rag_app.entity.models import Base
from my_rag_app.exception.db_connection import MissingDBCredentialsError
from my_rag_app.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


# Connection
def _build_database_url() -> str:
    host = os.getenv("DB_HOST", "")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")

    missing = [
        var
        for var, val in [
            ("DB_HOST", host),
            ("DB_NAME", name),
            ("DB_USER", user),
            ("DB_PASSWORD", password),
        ]
        if not val
    ]
    if missing:
        logger.error("Missing required DB env vars: %s", ", ".join(missing))
        raise MissingDBCredentialsError(missing)

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


DATABASE_URL = _build_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Session:
    """Returns a new SQLAlchemy session. Caller is responsible for closing it
    (use as a context manager: `with get_session() as session:`)."""
    return SessionLocal()


def init_db() -> None:
    """Creates all tables defined in entity/models.py if they don't already exist.
    Safe to call multiple times — does not drop or modify existing tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully")
    except Exception:
        logger.exception("Failed to create database tables")
        raise


if __name__ == "__main__":
    init_db()
