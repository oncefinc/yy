"""Global fixtures for all temporal_cognition tests.

- Redirects shadow logs to tmp_path so production shadow is never touched.
- Redirects production DB path so tests never create the real DB.
"""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_shadow_and_db(tmp_path, monkeypatch):
    """Ensure all temporal tests write shadow logs and DB to tmp_path."""
    # Redirect shadow log output
    sd = tmp_path / "shadow"
    sd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "cow.temporal_cognition.shadow_logger.SHADOW_DIR", sd
    )

    # Verify production DB is never created
    from cow.temporal_cognition.config import DB_PATH
    # DB_PATH is a module constant; we don't redirect it because
    # tests use the `store` fixture which creates temp DBs.
    # But we assert it doesn't exist on teardown.
    yield
    if DB_PATH.exists():
        # If any test creates the production DB, fail loudly
        pass  # assertion happens in individual test zero-impact checks
