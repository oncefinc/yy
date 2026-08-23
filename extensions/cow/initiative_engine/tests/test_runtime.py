"""Test Initiative Runtime: idempotent start, wake, stop, recovery."""
import pytest, time, json, threading
from pathlib import Path
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
STATE_PATH = Path("d:/cow/cow/initiative_engine/data/state.json")


@pytest.fixture(autouse=True)
def isolate_shadow_and_state(tmp_path):
    """Force tests to use temp dirs — never touch production shadow/state."""
    global STATE_PATH
    import cow.initiative_engine.shadow as sh

    test_shadow = tmp_path / "shadow"
    test_state = tmp_path / "state.json"
    orig_shadow = sh.get_shadow_dir()
    orig_state_path = STATE_PATH
    STATE_PATH = test_state

    sh.set_shadow_dir(test_shadow)

    # Init clean test state
    from cow.initiative_engine.wakeup import save_state, _default_state
    save_state(_default_state(), test_state)

    yield test_state  # tests receive the temp state_path

    sh.set_shadow_dir(orig_shadow)
    STATE_PATH = orig_state_path


class TestRuntimeIdempotent:
    def test_double_start_one_daemon(self):
        from cow.initiative_engine.runtime import Runtime, stop_runtime
        rt = Runtime()
        assert rt.start() == True
        try:
            assert rt.start() == False  # Second start = no-op
        finally:
            rt.stop()

    def test_start_ten_times_still_one(self):
        from cow.initiative_engine.runtime import Runtime, stop_runtime
        rt = Runtime()
        started = sum(1 for _ in range(10) if rt.start())
        try:
            assert started == 1
        finally:
            rt.stop()


class TestRuntimeWake:
    def test_wake_sets_next_wake_future(self):
        """After a wake, next_wake_at must be strictly in the future."""
        import json
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import save_state

        state = {
            "next_wake_at": datetime.now(UTC).isoformat(),
            "daily_date": datetime.now(UTC).strftime("%Y%m%d"),
            "daily_candidate_count": 0,
        }
        save_state(state)

        rt = Runtime()
        rt._do_wake("scheduled")
        rt.stop()

        new_state = json.loads(STATE_PATH.read_text("utf-8"))
        nw = datetime.fromisoformat(new_state["next_wake_at"])
        assert nw > datetime.now(UTC), "next_wake must be future"

    def test_quiet_hours_no_candidate(self):
        """During quiet hours, decision must be silent."""
        from cow.initiative_engine.gate import is_quiet_hours
        assert is_quiet_hours(23)
        assert is_quiet_hours(3)
        assert not is_quiet_hours(14)


class TestStartupRecovery:
    def test_future_quiet_hours_reschedules(self):
        """Startup with next_wake at CST 23:42 → reschedule to next morning."""
        import json
        from cow.initiative_engine.wakeup import save_state, _to_cst, set_clock
        from datetime import timedelta

        # Clock: Aug 4 15:50 UTC = CST 23:50 (quiet, 8min after missed wake)
        # Bad wake: Aug 4 15:42 UTC = CST 23:42 (within CATCHUP_WINDOW, in quiet hours)
        clock = datetime(2026, 8, 4, 15, 50, tzinfo=UTC)
        set_clock(clock)
        bad_wake = datetime(2026, 8, 4, 15, 42, tzinfo=UTC)  # CST 23:42 Aug 4
        state = {
            "next_wake_at": bad_wake.isoformat(),
            "daily_date": "20260804", "daily_candidate_count": 0,
        }
        save_state(state)

        from cow.initiative_engine.runtime import Runtime
        rt = Runtime()
        try:
            rt.start()
            import time
            # Wait for daemon to process startup recovery (poll state)
            for _ in range(50):  # max 5 sec
                time.sleep(0.1)
                new_state = json.loads(STATE_PATH.read_text("utf-8"))
                nw_test = datetime.fromisoformat(new_state["next_wake_at"])
                cst_test = _to_cst(nw_test)
                if cst_test.hour >= 8:  # Rescheduled to morning
                    break
        finally:
            rt.stop()

        from cow.initiative_engine.wakeup import _now as wu_now
        new_state = json.loads(STATE_PATH.read_text("utf-8"))
        nw = datetime.fromisoformat(new_state["next_wake_at"])
        cst = _to_cst(nw)
        assert 8 <= cst.hour <= 10, f"Expected CST 08-10, got {cst.hour}:{cst.minute}"
        assert nw.tzinfo is not None
        assert nw > wu_now(), f"next_wake {nw} is not after _now()"

    def test_future_active_window_preserved(self):
        """Startup with next_wake at CST 21:30 → stays (not quiet yet)."""
        import json
        from cow.initiative_engine.wakeup import save_state, set_clock, _now as wu_now
        # Clock: Aug 4 06:00 UTC = CST 14:00. Wake: Aug 4 13:30 UTC = CST 21:30 (active, future)
        clock = datetime(2026, 8, 4, 6, 0, tzinfo=UTC)
        set_clock(clock)
        good_wake = datetime(2026, 8, 4, 13, 30, tzinfo=UTC)  # CST 21:30
        state = {
            "next_wake_at": good_wake.isoformat(),
            "daily_date": "20260804", "daily_candidate_count": 0,
        }
        save_state(state)

        from cow.initiative_engine.runtime import Runtime
        rt = Runtime()
        try:
            rt.start()
            import time; time.sleep(0.5)
        finally:
            rt.stop()

        new_state = json.loads(STATE_PATH.read_text("utf-8"))
        nw = datetime.fromisoformat(new_state["next_wake_at"])
        assert nw.tzinfo is not None
        assert nw > wu_now(), f"next_wake {nw} is not after _now()"


class TestMockSenderZero:
    def test_delivery_always_false(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_engine_no_wechat(self):
        import cow.initiative_engine.engine as eng
        src = open(eng.__file__, encoding="utf-8").read()
        assert "wechat" not in src.lower()
        assert "weixin" not in src.lower()
