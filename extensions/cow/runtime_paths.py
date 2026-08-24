"""Portable runtime path resolution for the public YY repository.

All defaults are derived from this file's location, never from the current
working directory or the original author's drive layout.  Environment values
may be absolute or repository-relative.  Runtime data lives under
``<repo>/.runtime`` by default and is excluded from version control.
"""
from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
EXTENSIONS_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = EXTENSIONS_ROOT.parent


def env_path(name: str, default: Path) -> Path:
    """Return an absolute path from *name*, resolving relative values to repo."""
    raw = os.environ.get(name, "").strip()
    path = Path(os.path.expandvars(os.path.expanduser(raw))) if raw else Path(default)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


RUNTIME_ROOT = env_path("COW_RUNTIME_ROOT", REPO_ROOT / ".runtime")
WORKSPACE_ROOT = env_path("COW_WORKSPACE_ROOT", REPO_ROOT)
MODEL_ROOT = env_path("COW_MODEL_ROOT", REPO_ROOT / "models")
BASE_MODEL_PATH = env_path(
    "COW_BASE_MODEL_PATH", MODEL_ROOT / "bge-base-zh-v1.5"
)

MEMORY_DATA_DIR = env_path(
    "COW_MEMORY_DATA_DIR", RUNTIME_ROOT / "memory_engine"
)
INITIATIVE_DATA_DIR = env_path(
    "INITIATIVE_DATA_DIR", RUNTIME_ROOT / "initiative_engine"
)
TEMPORAL_DATA_DIR = env_path(
    "COW_TEMPORAL_DATA_DIR", RUNTIME_ROOT / "temporal_cognition"
)
SELF_AWARENESS_DATA_DIR = env_path(
    "COW_SELF_AWARENESS_DATA_DIR", RUNTIME_ROOT / "self_awareness"
)
TEMP_DIR = env_path("COW_TEMP_DIR", RUNTIME_ROOT / "tmp")


def base_model_reference(model_id: str = "BAAI/bge-base-zh-v1.5") -> str:
    """Prefer an explicitly configured/local model, otherwise allow HF lookup."""
    explicit = os.environ.get("COW_BASE_MODEL_PATH", "").strip()
    if explicit or BASE_MODEL_PATH.exists():
        return str(BASE_MODEL_PATH)
    return os.environ.get("COW_BASE_MODEL_ID", model_id).strip() or model_id
