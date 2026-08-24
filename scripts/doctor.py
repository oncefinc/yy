"""Read-only deployment diagnostics; never prints secrets or private data."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from launch import REPO_ROOT, load_dotenv


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("COW_EXTENSION_ROOT", str(REPO_ROOT / "extensions"))
    os.environ.setdefault("COW_RUNTIME_ROOT", str(REPO_ROOT / ".runtime"))
    os.environ.setdefault("COW_DATA_DIR", str(REPO_ROOT / ".runtime" / "cowagent"))
    for path in (REPO_ROOT / "extensions", REPO_ROOT / "cowagent"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from cow.runtime_paths import (
        BASE_MODEL_PATH,
        INITIATIVE_DATA_DIR,
        MEMORY_DATA_DIR,
        RUNTIME_ROOT,
        TEMPORAL_DATA_DIR,
        base_model_reference,
    )
    from cow.memory_engine.config import BASE_LANCE_DIR, V2_LANCE_DIR

    checks = {
        "python_3_11_or_newer": sys.version_info >= (3, 11),
        "cowagent_config_present": (
            Path(os.environ.get("COW_DATA_DIR", REPO_ROOT / "cowagent"))
            / "config.json"
        ).exists(),
        "base_model_local": BASE_MODEL_PATH.exists(),
        "v2_index_present": V2_LANCE_DIR.exists(),
        "base_index_present": BASE_LANCE_DIR.exists(),
    }
    payload = {
        "ok_to_start": checks["python_3_11_or_newer"] and checks["cowagent_config_present"],
        "checks": checks,
        "paths": {
            "repo_root": str(REPO_ROOT),
            "runtime_root": str(RUNTIME_ROOT),
            "memory_data": str(MEMORY_DATA_DIR),
            "initiative_data": str(INITIATIVE_DATA_DIR),
            "temporal_data": str(TEMPORAL_DATA_DIR),
            "base_model": base_model_reference(),
        },
        "note": "Missing indexes/model disable related features; no secret values were inspected.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok_to_start"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
