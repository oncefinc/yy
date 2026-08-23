from __future__ import annotations

from bridge.optional_hook_health import (
    get_optional_failure_snapshot,
    record_optional_failure,
    reset_optional_failures_for_testing,
)


def setup_function():
    reset_optional_failures_for_testing()


def teardown_function():
    reset_optional_failures_for_testing()


def test_failure_is_counted_without_exception_message():
    record_optional_failure(
        "initiative.on_user_message",
        RuntimeError("private message text and receiver id"),
    )

    snapshot = get_optional_failure_snapshot()
    assert snapshot["initiative.on_user_message"]["count"] == 1
    assert snapshot["initiative.on_user_message"]["last_error_type"] == "RuntimeError"
    assert "private message" not in repr(snapshot)


def test_repeated_failure_increments_same_hook():
    record_optional_failure("temporal.ingest", ValueError("one"))
    record_optional_failure("temporal.ingest", OSError("two"))

    value = get_optional_failure_snapshot()["temporal.ingest"]
    assert value["count"] == 2
    assert value["last_error_type"] == "OSError"


def test_hook_name_is_sanitized_and_bounded():
    record_optional_failure("hook user@example.com / secret" * 10, KeyError("x"))
    name = next(iter(get_optional_failure_snapshot()))
    assert "@" not in name
    assert "/" not in name
    assert len(name) <= 80


def test_snapshot_is_detached():
    record_optional_failure("a", RuntimeError("x"))
    snapshot = get_optional_failure_snapshot()
    snapshot["a"]["count"] = 999
    assert get_optional_failure_snapshot()["a"]["count"] == 1
