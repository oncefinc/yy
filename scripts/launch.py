"""Cross-platform YY launcher with repository-relative defaults."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {
    "COW_EXTENSION_ROOT", "COW_DATA_DIR", "COW_RUNTIME_ROOT",
    "COW_WORKSPACE_ROOT", "COW_BASELINE_ROOT", "COW_MODEL_ROOT",
    "COW_BASE_MODEL_PATH", "COW_MEMORY_DATA_DIR", "COW_V1_LANCE_DIR",
    "COW_V2_LANCE_DIR", "COW_BASE_LANCE_DIR", "COW_MIGRATION_REPORTS_DIR",
    "COW_SCENE_STORE_PATH", "INITIATIVE_DATA_DIR", "INITIATIVE_SHADOW_DIR",
    "COW_TEMPORAL_DATA_DIR", "COW_SELF_AWARENESS_DATA_DIR", "COW_TEMP_DIR",
}


def load_dotenv(path: Path) -> None:
    """Load the simple KEY=VALUE format used by this repository."""
    if not path.exists():
        return
    for raw_line in path.read_text("utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    for key in PATH_KEYS:
        value = os.environ.get(key, "").strip()
        if not value:
            continue
        path_value = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path_value.is_absolute():
            path_value = REPO_ROOT / path_value
        os.environ[key] = str(path_value.resolve(strict=False))


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("COW_EXTENSION_ROOT", str(REPO_ROOT / "extensions"))
    os.environ.setdefault("COW_RUNTIME_ROOT", str(REPO_ROOT / ".runtime"))
    os.environ.setdefault("COW_DATA_DIR", str(REPO_ROOT / ".runtime" / "cowagent"))

    for path in (REPO_ROOT / "extensions", REPO_ROOT / "cowagent"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    app_dir = REPO_ROOT / "cowagent"
    os.chdir(app_dir)
    runpy.run_path(str(app_dir / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
