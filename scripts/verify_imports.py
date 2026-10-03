import importlib, sys, traceback

runtime = {
    "numpy", "pandas", "sklearn", "lightgbm", "shap", "pyarrow", "joblib",
    "matplotlib", "mlflow", "fastapi", "uvicorn", "pydantic",
    "pydantic_settings", "dotenv", "streamlit", "sqlalchemy",
    "psycopg2", "httpx",
}
dev = {"pytest", "xgboost", "optuna"}

print(f"python: {sys.version.split()[0]}  executable: {sys.executable}")
print()

failed = []
for name in sorted(runtime):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"[runtime] OK  {name} == {ver}")
    except Exception as e:
        failed.append((name, e))
        print(f"[runtime] FAIL {name}: {e!r}")

for name in sorted(dev):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"[dev]    OK  {name} == {ver}")
    except Exception as e:
        failed.append((name, e))
        print(f"[dev]    FAIL {name}: {e!r}")

project_modules = [
    "app.config", "app.main", "app.dependencies", "app.errors", "app.schemas",
    "app.api.health", "app.api.prediction", "app.api.pricing", "app.api.explain",
    "app.services.model_service", "app.services.prediction_service",
    "app.services.pricing_service", "app.services.explainability_service",
    "app.services.state", "app.db.database", "app.db.models", "app.db.repository",
    "app.middleware.request_id", "app.middleware.rate_limit",
    "app.middleware.payload_limit", "app.middleware.logging",
]

import os
os.environ.setdefault("PYTHONPATH", r"D:\ALL Projects\E-Commerce")
sys.path.insert(0, r"D:\ALL Projects\E-Commerce")

for name in project_modules:
    try:
        importlib.import_module(name)
        print(f"[app]    OK  {name}")
    except Exception as e:
        failed.append((name, e))
        print(f"[app]    FAIL {name}: {e!r}")
        traceback.print_exc()

print()
print("RESULT:", "ALL IMPORTS OK" if not failed else f"{len(failed)} FAILURES")
for n, e in failed:
    print(" -", n, e)
sys.exit(1 if failed else 0)