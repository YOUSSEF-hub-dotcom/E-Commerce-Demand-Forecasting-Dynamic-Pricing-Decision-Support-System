"""SQLAlchemy ORM models for the inference log.

Only the production inference log lives in the database; the analytics
datasets and the MLflow registry are never handled here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InferenceRecord(Base):
    __tablename__ = "inference_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True, nullable=True)
    week: Mapped[str | None] = mapped_column(String(10), index=True, nullable=True)
    prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_supplied: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pricing_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    elasticity_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    elasticity_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recommended_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_computable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)