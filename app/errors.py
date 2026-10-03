from __future__ import annotations

from dataclasses import dataclass, field

# --- canonical error codes ---------------------------------------------------
PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
INVALID_WEEK = "INVALID_WEEK"
WEEK_OUT_OF_SUPPORTED_RANGE = "WEEK_OUT_OF_SUPPORTED_RANGE"
TEST_BOUNDARY_VIOLATION = "TEST_BOUNDARY_VIOLATION"
INVALID_COST = "INVALID_COST"
MODEL_NOT_READY = "MODEL_NOT_READY"
FEATURE_BUILD_FAILED = "FEATURE_BUILD_FAILED"
ELASTICITY_UNAVAILABLE = "ELASTICITY_UNAVAILABLE"
NO_ELASTICITY = "NO_ELASTICITY"
NO_PRICE_OBSERVATION = "NO_PRICE_OBSERVATION"
PROFIT_NOT_COMPUTABLE = "PROFIT_NOT_COMPUTABLE"
MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
VALIDATION_ERROR = "VALIDATION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
RATE_LIMITED = "RATE_LIMITED"

# Business outcomes that are returned *within* a 200 response are not raised
# as HTTP errors. They are documented in the pricing report as well.


@dataclass
class AppError(Exception):
    """Structured application error mapped to an HTTP response.

    ``status_code`` is the HTTP status, ``code`` one of the canonical error
    codes above and ``extra`` optional structured detail (never stack traces).
    """

    status_code: int
    code: str
    message: str
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debugging aid only
        return f"{self.code}: {self.message}"


class PersistenceError(Exception):
    """Raised when the inference log cannot be written."""


def business_rule(status_code: int, code: str, message: str, **extra) -> AppError:
    return AppError(status_code=status_code, code=code, message=message, extra=extra)