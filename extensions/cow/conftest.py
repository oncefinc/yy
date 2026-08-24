"""Public-release pytest safeguards.

Tests that inspect the author's live V1/V2 databases are useful deployment
guards, but a clean clone neither has nor should read those private stores.
Set COW_TEST_PRODUCTION_INTEGRITY=1 only inside an isolated production audit.
"""
from __future__ import annotations

import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


# A clean public test run must never create state under the checkout or inspect
# a developer's live deployment. Production-integrity audits opt out explicitly.
_TEST_RUNTIME_ROOT: Path | None = None
if os.environ.get("COW_TEST_PRODUCTION_INTEGRITY") != "1":
    _TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="yy-public-tests-"))
    os.environ["COW_RUNTIME_ROOT"] = str(_TEST_RUNTIME_ROOT)


_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "cowagent", _REPO_ROOT / "extensions"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def pytest_collection_modifyitems(items):
    if os.environ.get("COW_TEST_PRODUCTION_INTEGRITY") == "1":
        return
    skip = pytest.mark.skip(
        reason="requires an explicitly enabled local production-memory audit"
    )
    production_markers = (
        'Path("C:/Users/',
        "== 2691",
        "!= 2691",
        "== 709",
        "!= 709",
        "count >= 200",
        "V1 should still have data",
        "def test_returns_results",
    )
    for item in items:
        cls_name = getattr(getattr(item, "cls", None), "__name__", "")
        if "Production" in cls_name and "Integrity" in cls_name:
            item.add_marker(skip)
            continue
        try:
            source = inspect.getsource(item.obj).replace("\\", "/")
        except (OSError, TypeError):
            continue
        if any(marker in source for marker in production_markers):
            item.add_marker(skip)


def pytest_sessionfinish(session, exitstatus):
    if _TEST_RUNTIME_ROOT is not None:
        shutil.rmtree(_TEST_RUNTIME_ROOT, ignore_errors=True)
