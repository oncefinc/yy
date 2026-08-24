"""Public-clone portability contracts (no model, network, or private data)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _python(code: str, cwd: Path, env: dict[str, str] | None = None):
    merged = os.environ.copy()
    for key in (
        "COW_RUNTIME_ROOT", "COW_MEMORY_DATA_DIR", "INITIATIVE_DATA_DIR",
        "COW_TEMPORAL_DATA_DIR", "COW_SELF_AWARENESS_DATA_DIR",
    ):
        merged.pop(key, None)
    merged["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "extensions"), str(REPO_ROOT / "cowagent"))
    )
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=merged,
        text=True, capture_output=True, check=False,
    )


def test_defaults_do_not_depend_on_current_working_directory(tmp_path):
    result = _python(
        "from cow.runtime_paths import REPO_ROOT,RUNTIME_ROOT;"
        "print(REPO_ROOT);print(RUNTIME_ROOT)",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert Path(lines[0]) == REPO_ROOT
    assert Path(lines[1]) == REPO_ROOT / ".runtime"


def test_relative_environment_paths_are_repo_relative(tmp_path):
    result = _python(
        "from cow.runtime_paths import MEMORY_DATA_DIR;print(MEMORY_DATA_DIR)",
        tmp_path,
        {"COW_MEMORY_DATA_DIR": "portable/memory"},
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT / "portable" / "memory"


def test_memory_indexes_can_be_overridden_independently(tmp_path):
    result = _python(
        "from cow.memory_engine.config import V2_LANCE_DIR,BASE_LANCE_DIR;"
        "print(V2_LANCE_DIR);print(BASE_LANCE_DIR)",
        tmp_path,
        {
            "COW_V2_LANCE_DIR": "portable/v2",
            "COW_BASE_LANCE_DIR": "portable/base",
        },
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert Path(lines[0]) == REPO_ROOT / "portable" / "v2"
    assert Path(lines[1]) == REPO_ROOT / "portable" / "base"


def test_runtime_python_has_no_author_drive_literal():
    roots = (REPO_ROOT / "extensions" / "cow", REPO_ROOT / "cowagent" / "bridge")
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "tests" in path.parts or path.name == "conftest.py":
                continue
            text = path.read_text("utf-8", errors="ignore").casefold()
            if "d:/cow" in text or "d:\\cow" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_public_launcher_uses_repository_location():
    text = (REPO_ROOT / "scripts" / "launch.py").read_text("utf-8")
    assert "Path(__file__).resolve().parents[1]" in text
    assert "d:/cow" not in text.casefold()


def test_dotenv_relative_paths_are_normalized(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from launch import load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("COW_DATA_DIR=./portable/data\n", encoding="utf-8")
        monkeypatch.delenv("COW_DATA_DIR", raising=False)
        load_dotenv(env_file)
        assert Path(os.environ["COW_DATA_DIR"]) == REPO_ROOT / "portable" / "data"
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
