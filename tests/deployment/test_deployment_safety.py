"""Static safety scans: the deployment code must never touch protected data."""

from __future__ import annotations

import ast
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = ["app", "dashboard", "deployment"]

FORBIDDEN_LITERALS = [
    "test.csv",
    "splits/test.csv",
    "data/splits",
    "splits\\",
    ".trash",
]
FORBIDDEN_CALLS = [
    "register_model",
    "transition_model_version_stage",
    "log_model",
    "create_model_version",
    "delete_model_version",
    "update_model_version",
    "set_stage",
    "set_aliases",
]
FORBIDDEN_IMPORTS_PREFIX = [
    "optuna",
    "xgboost",
]


def _py_files(dirs):
    for d in dirs:
        base = PROJECT_ROOT / d
        if not base.is_dir():
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".py"):
                    yield Path(root) / f


def _code_text(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def test_deployment_code_never_references_test_split():
    offenders = []
    for path in _py_files(SCAN_DIRS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                for tok in FORBIDDEN_LITERALS:
                    if tok.lower() in low:
                        offenders.append((str(path), node.lineno, node.value))
    assert offenders == [], f"forbidden literals found: {offenders}"


def test_deployment_code_never_registers_or_transitions_models():
    offenders = []
    for path in _py_files(SCAN_DIRS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = ""
                if isinstance(fn, ast.Name):
                    name = fn.id
                elif isinstance(fn, ast.Attribute):
                    name = fn.attr
                if name in FORBIDDEN_CALLS:
                    offenders.append((str(path), node.lineno, name))
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in FORBIDDEN_IMPORTS_PREFIX):
                        offenders.append((str(path), node.lineno, alias.name))
    assert offenders == [], f"forbidden training/registry writes found: {offenders}"


def test_deployment_code_does_not_fit_models():
    """No `.fit(` calls inside the served application/SHAP/dashboard code."""
    offenders = []
    for path in _py_files(["app", "dashboard"]):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "fit":
                    offenders.append((str(path), node.lineno, "fit()"))
    assert offenders == [], f"fit() calls in served code: {offenders}"


def test_deployment_uses_only_mlflow_client_reads():
    """mlflow is only used for read-only registry access in the app."""
    offenders = []
    for path in _py_files(["app", "dashboard"]):
        text = path.read_text(encoding="utf-8")
        for bad in ["mlflow.models.", "mlflow.register_model", "mlflow.pytorch", "mlflow.lightgbm."]:
            if bad in text:
                offenders.append((str(path), bad))
    assert offenders == []


def test_docker_compose_and_dockerignore_exclude_protected_data():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "data/splits" in dockerignore
    assert "data/raw" in dockerignore
    assert "test.csv" in dockerignore or "splits" in dockerignore
    compose = (PROJECT_ROOT / "deployment" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "postgres:16-alpine" in compose
    assert "demand_forecasting_final" in compose


def test_dashboard_ui_does_not_reference_feature_names():
    content = (PROJECT_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "demand_lag_" not in content
    assert "rolling_demand" not in content
    assert "test.csv" not in content


def _dockerignore_rules() -> list[str]:
    text = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerignore_ships_only_the_part4_pricing_artifact():
    """Reports stay out of the image except the artifact the dashboard reads.

    A bare ``reports`` entry cannot be re-included selectively: Docker skips the
    excluded directory, and a ``!reports/forecasting/`` style re-include would
    un-exclude that whole subtree, shipping every other report. The only correct
    shape is ``reports/**`` plus a single file-level negation.
    """
    rules = _dockerignore_rules()
    assert "reports/**" in rules, "reports must be excluded by content, not by directory"
    assert "!reports/forecasting/part4_pricing_engine.json" in rules
    assert "reports" not in rules, "a bare `reports` entry would ship every report"
    assert not any(
        r.startswith("!") and r.rstrip("/") == "reports/forecasting"
        for r in rules
    ), "re-including the forecasting directory would ship its other reports"


def test_dockerfile_copies_the_part4_pricing_artifact():
    dockerfile = (PROJECT_ROOT / "deployment" / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "COPY reports/forecasting/part4_pricing_engine.json "
        "./reports/forecasting/part4_pricing_engine.json" in dockerfile
    ), "the dashboard image must bake in the artifact it reads at runtime"
    assert "COPY reports " not in dockerfile
    assert "COPY reports/." not in dockerfile
