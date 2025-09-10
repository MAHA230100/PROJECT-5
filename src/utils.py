import os
import json
from typing import Any, Dict
import joblib

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
OUTPUTS_DIR = os.path.join(ARTIFACTS_DIR, "outputs")


def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_joblib(obj: Any, name: str) -> str:
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(obj, path)
    return path


def load_joblib(name: str) -> Any:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def save_json(data: Dict[str, Any], name: str) -> str:
    ensure_dirs()
    path = os.path.join(OUTPUTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(name: str) -> Dict[str, Any]:
    path = os.path.join(OUTPUTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

