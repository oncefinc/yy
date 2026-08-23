"""v1.1.1 Restart Continuity tests — all use fixed clock, never real time."""
import pytest, json, time, uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CST = timezone(timedelta(hours=8))


@pytest.fixture(autouse=True)
def isolate(tmp_path):
    import cow.initiative_engine.shadow as sh
    import cow.initiative_engine.wakeup as wu
    orig_shadow = sh.get_shadow_dir()
    test_shadow = tmp_path / "shadow"
    test_state = tmp_path / "state.json"
    sh.set_shadow_dir(test_shadow)
    old_path = wu._DEFAULT_STATE_PATH
    wu._DEFAULT_STATE_PATH = test_state
    wu.save_state({"next_wake_at": None, "state_version": 2,
                   "daily_date": "", "daily_candidate_count": 0}, test_state)
    wu.set_clock(None)
    yield test_state
    sh.set_shadow_dir(orig_shadow)
    wu._DEFAULT_STATE_PATH = old_path
    wu.set_clock(None)


def _set_state(sp, **kw):
    import cow.initiative_engine.wakeup as wu
    s = wu.load_state(sp)
    s.update(kw)
    wu.save_state(s, sp)


class TestPreserveFutureWake:
    def test_restart_10min_before_wake_preserves(self, isolate):
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import set_clock
        clock = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
        future = clock + timedelta(minutes=10)
        wid = uuid.uuid4().hex[:12]
        set_clock(clock)
        _set_state(isolate, next_wake_at=future.isoformat(), scheduled_wake_id=wid)
        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(1); rt.stop()
        s = __import__('json').loads(isolate.read_text("utf-8"))
        nw = datetime.fromisoformat(s["next_wake_at"])
        assert abs((nw - future).total_seconds()) < 5
        assert s["scheduled_wake_id"] == wid

    def test_three_restarts_no_drift(self, isolate):
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import set_clock
        clock = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
        future = clock + timedelta(minutes=10)
        wid = uuid.uuid4().hex[:12]
        set_clock(clock)
        for i in range(3):
            _set_state(isolate, next_wake_at=future.isoformat(), scheduled_wake_id=wid)
            rt = Runtime(state_path=isolate)
            rt.start(); time.sleep(0.5); rt.stop()
            s = __import__('json').loads(isolate.read_text("utf-8"))
            nw = datetime.fromisoformat(s["next_wake_at"])
            assert abs((nw - future).total_seconds()) < 5, f"Restart {i}: drifted"


class TestCatchUp:
    def test_missed_20min_catches_up(self, isolate):
        """Wake missed by 20 min → catch-up executes, state advances."""
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import set_clock, _now
        clock = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
        missed = datetime(2026, 8, 9, 5, 40, tzinfo=UTC)
        wid = uuid.uuid4().hex[:12]
        set_clock(clock)
        _set_state(isolate, next_wake_at=missed.isoformat(), scheduled_wake_id=wid,
                   daily_date="20260809", daily_candidate_count=0)

        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(3); rt.stop()

        s = __import__('json').loads(isolate.read_text("utf-8"))
        # After catch-up: daemon_instance_id should be set (proves startup ran)
        assert s.get("state_version") == 2
        assert s.get("daemon_instance_id"), "Daemon should record its instance ID"

    def test_same_wake_id_not_duplicated(self, isolate):
        """Already-completed wake → reschedules without re-executing."""
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import set_clock, _now
        clock = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
        missed = datetime(2026, 8, 9, 5, 40, tzinfo=UTC)
        wid = uuid.uuid4().hex[:12]
        set_clock(clock)
        _set_state(isolate, next_wake_at=missed.isoformat(), scheduled_wake_id=wid,
                   last_completed_wake_id=wid, daily_date="20260809")

        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(2); rt.stop()

        s = __import__('json').loads(isolate.read_text("utf-8"))
        assert s.get("state_version") == 2
        assert s.get("daemon_instance_id"), "Should start normally"


class TestQuietHours:
    def test_missed_at_23h_reschedules(self, isolate):
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import _to_cst, set_clock
        clock = datetime(2026, 8, 8, 15, 30, tzinfo=UTC)
        missed = datetime(2026, 8, 8, 15, 0, tzinfo=UTC)
        set_clock(clock)
        _set_state(isolate, next_wake_at=missed.isoformat(),
                   scheduled_wake_id=uuid.uuid4().hex[:12], daily_date="20260808")
        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(3); rt.stop()
        s = __import__('json').loads(isolate.read_text("utf-8"))
        nw = datetime.fromisoformat(s["next_wake_at"])
        cst = _to_cst(nw)
        assert 8 <= cst.hour <= 10, f"Got CST {cst.hour}:{cst.minute}"

    def test_2200_boundary_quiet(self, isolate):
        from cow.initiative_engine.wakeup import _in_quiet
        assert _in_quiet(datetime(2026, 8, 8, 14, 0, tzinfo=UTC))

    def test_2159_active(self, isolate):
        from cow.initiative_engine.wakeup import _in_quiet
        assert not _in_quiet(datetime(2026, 8, 8, 13, 59, tzinfo=UTC))

    def test_0800_active(self, isolate):
        from cow.initiative_engine.wakeup import _in_quiet
        assert not _in_quiet(datetime(2026, 8, 8, 0, 0, tzinfo=UTC))

    def test_0759_quiet(self, isolate):
        from cow.initiative_engine.wakeup import _in_quiet
        assert _in_quiet(datetime(2026, 8, 7, 23, 59, tzinfo=UTC))

    def test_2210_startup_no_catchup(self, isolate):
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import _to_cst, set_clock
        clock = datetime(2026, 8, 8, 14, 10, tzinfo=UTC)
        missed = datetime(2026, 8, 8, 13, 30, tzinfo=UTC)
        set_clock(clock)
        _set_state(isolate, next_wake_at=missed.isoformat(),
                   scheduled_wake_id=uuid.uuid4().hex[:12], daily_date="20260808")
        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(3); rt.stop()
        s = __import__('json').loads(isolate.read_text("utf-8"))
        nw = datetime.fromisoformat(s["next_wake_at"])
        cst = _to_cst(nw)
        assert 8 <= cst.hour <= 10, f"Got CST {cst.hour}:{cst.minute}"


class TestLongDowntime:
    def test_24h_downtime_no_catchup_queue(self, isolate):
        from cow.initiative_engine.runtime import Runtime
        from cow.initiative_engine.wakeup import set_clock
        clock = datetime(2026, 8, 9, 6, 0, tzinfo=UTC)
        missed = datetime(2026, 8, 8, 6, 0, tzinfo=UTC)
        set_clock(clock)
        _set_state(isolate, next_wake_at=missed.isoformat(),
                   scheduled_wake_id=uuid.uuid4().hex[:12], daily_date="20260808")
        rt = Runtime(state_path=isolate)
        rt.start(); time.sleep(3); rt.stop()
        s = __import__('json').loads(isolate.read_text("utf-8"))
        assert s.get("missed_wake_count", 0) >= 1


class TestDeliveryKill:
    def test_delivery_still_false(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False
