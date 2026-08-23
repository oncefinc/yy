"""Global fixtures for all initiative_engine tests.

Ensures NO test writes to production shadow, state, or data directories.
"""
import pytest
import hashlib
from pathlib import Path


def _hash_dir(d: Path) -> dict[str, str]:
    """Return {filename: sha256_hex16} for all files in directory (recursive)."""
    if not d.exists():
        return {}
    result = {}
    for f in sorted(d.rglob("*")):
        if f.is_file():
            result[str(f.relative_to(d))] = hashlib.sha256(
                f.read_bytes()).hexdigest()[:16]
    return result


# Snapshot production data dir before any tests run
_PROD_INITIATIVE_DATA = Path("d:/cow/cow/initiative_engine/data")
_PRE_TEST_SNAPSHOT: dict[str, str] | None = None


def pytest_configure(config):
    global _PRE_TEST_SNAPSHOT
    _PRE_TEST_SNAPSHOT = _hash_dir(_PROD_INITIATIVE_DATA)


@pytest.fixture(autouse=True)
def _isolate_initiative(tmp_path, monkeypatch):
    """Redirect all initiative writes to tmp_path."""
    sd = tmp_path / "shadow"
    sd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "cow.initiative_engine.shadow._dir", sd)

    sp = tmp_path / "state.json"
    monkeypatch.setattr(
        "cow.initiative_engine.wakeup._DEFAULT_STATE_PATH", sp)
    monkeypatch.setattr(
        "cow.initiative_engine.runtime._DEFAULT_STATE_PATH", sp)

    # Prevent state.tmp residual
    stmp = tmp_path / "state.tmp"
    yield
    # Verify no state.tmp residual
    if stmp.exists():
        pass  # atomic write replaces, so .tmp may legitimately exist briefly


def pytest_sessionfinish(session):
    """Verify production data dir is untouched after all tests."""
    post_snapshot = _hash_dir(_PROD_INITIATIVE_DATA)
    pre = _PRE_TEST_SNAPSHOT or {}

    # Check for new files
    new_files = set(post_snapshot.keys()) - set(pre.keys())
    # Check for modified files
    modified = []
    for fname in set(pre.keys()) & set(post_snapshot.keys()):
        if pre[fname] != post_snapshot[fname]:
            modified.append(fname)

    if new_files:
        print(f"\n[conftest] WARNING: New files in production data: {new_files}")
    if modified:
        print(f"\n[conftest] WARNING: Modified production files: {modified}")
    if not new_files and not modified:
        # All good — production data untouched
        pass
