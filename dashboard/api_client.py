"""Thin HTTP client for the deployment API (business-level contract only)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from dashboard.config import API_BASE_URL

TIMEOUT_S = 600.0


@dataclass
class ApiError(Exception):
    detail: str
    code: str | None = None
    status_code: int | None = None

    def __str__(self) -> str:
        return f"[{self.code or 'ERROR'}] {self.detail}"


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(self.base_url + path, json=payload, timeout=TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code >= 400:
            raise ApiError(
                detail=body.get("detail", "request failed"),
                code=body.get("code"),
                status_code=resp.status_code,
            )
        return body

    def health(self) -> dict:
        return httpx.get(self.base_url + "/health", timeout=30).json()

    def ready(self) -> dict:
        return httpx.get(self.base_url + "/ready", timeout=30).json()

    def predict(self, product_id: str, week: str | None) -> dict:
        return self._post("/predict", {"product_id": product_id, "week": week})

    def pricing(self, product_id: str, week: str | None, cost: float | None) -> dict:
        return self._post("/pricing", {"product_id": product_id, "week": week, "cost": cost})

    def explain(self, product_id: str, week: str | None) -> dict:
        return self._post("/explain", {"product_id": product_id, "week": week})