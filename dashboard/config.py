"""Streamlit UI configuration (env-driven)."""

from __future__ import annotations

import os

API_BASE_URL = os.environ.get("STREAMLIT_API_BASE_URL", "http://localhost:8000")
PAGE_TITLE = "E-Commerce Demand & Pricing Manager"
PAGE_ICON = "📈"
DEFAULT_WEEK = "2018-05-14"
DOCS_RELATIVE = "docs"