"""AppTest fixture that renders a raw /pricing API response via the presenter.

Mirrors dashboard/app.py's Pricing section: given a response dict (injected as
JSON via the streamlit session state), it calls ``render_pricing_response`` —
the exact same function the deployed Pricing tab renders. The API is never
contacted; the response is a fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.pricing_presenter import render_pricing_response

st.set_page_config(page_title="probe", layout="wide")

raw = st.session_state.get("pricing_response")
if raw is not None:
    res = json.loads(raw) if isinstance(raw, str) else raw
    render_pricing_response(res)
else:
    st.write("NO_RESPONSE")