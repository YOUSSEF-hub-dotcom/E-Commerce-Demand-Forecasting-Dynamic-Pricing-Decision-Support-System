"""Engine and store for the inference log database.

Production deployments use PostgreSQL; SQLite is supported for local runs and
tests. Transactions are short and failures surface as ``PersistenceError`` up
to the caller, never silently.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.errors import PersistenceError
from app.db.models import Base, InferenceRecord

logger = logging.getLogger("deploy.db")


def build_engine(url: str, timeout_s: int = 5) -> Engine:
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        connect_args["connect_timeout"] = timeout_s
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args=connect_args,
        future=True,
    )


class InferenceStore:
    def __init__(self, url: str, timeout_s: int = 5):
        self.url = url
        self.timeout_s = timeout_s
        self.engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def _ensure_engine(self) -> Engine:
        if self.engine is None:
            self.engine = build_engine(self.url, timeout_s=self.timeout_s)
            self._session_factory = sessionmaker(
                bind=self.engine, autoflush=False, expire_on_commit=False
            )
        return self.engine

    def create_schema(self) -> None:
        """Create tables if they do not exist (idempotent)."""
        engine = self._ensure_engine()
        Base.metadata.create_all(engine)

    def ping(self) -> None:
        """Raise if the database is unreachable."""
        engine = self._ensure_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def insert(self, record: InferenceRecord) -> None:
        if self._session_factory is None:
            self._ensure_engine()
        assert self._session_factory is not None
        try:
            with self._session_factory() as session:
                session.add(record)
                session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            raise PersistenceError(f"inference log write failed: {exc}") from exc