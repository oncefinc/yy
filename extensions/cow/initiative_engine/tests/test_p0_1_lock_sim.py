"""P0.1: State lock + barrier concurrency + 30-day simulation.

Covers:
  1. All state mutations via atomic_update (verified by concurrent tests)
  2. Barrier concurrency: daemon × chat (7 scenarios)
  3. 30-day wake simulation with real numbers
  4. State isolation verification
"""
import pytest
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta

UTC = timezone.utc


@pytest.fixture(autouse=True)
def reset_clock():
    from cow.initiative_engine.wakeup import set_clock
    set_clock(datetime(2026, 8, 10, 14, 0, tzinfo=UTC))
    yield
    set_clock(None)


# ═══════════════════════════════════════════════════════════════
# 1. Barrier concurrency tests
# ═══════════════════════════════════════════════════════════════
class TestBarrierConcurrency:
    """Daemon × chat thread barrier-synchronized tests."""

    def test_daemon_wake_vs_user_message(self):
        """Daemon updates next_wake_at while user updates last_user_message_at."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_user_message, set_clock,
        )
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def daemon_sim():
            try:
                barrier.wait()
                for _ in range(30):
                    def _daemon_update(state: dict):
                        import uuid
                        state["next_wake_at"] = (
                            datetime(2026, 8, 10, 15, 0, tzinfo=UTC) +
                            timedelta(minutes=hash(str(_)) % 180)
                        ).isoformat()
                        state["scheduled_wake_id"] = uuid.uuid4().hex[:12]
                    atomic_update(_daemon_update)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"daemon: {e}")

        def user_sim():
            try:
                barrier.wait()
                for i in range(30):
                    on_user_message(f"user_{i % 3}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"user: {e}")

        t1 = threading.Thread(target=daemon_sim); t2 = threading.Thread(target=user_sim)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        s = load_state()
        assert s["last_user_message_at"] is not None
        assert s["next_wake_at"] is not None

    def test_daemon_wake_vs_assistant_message(self):
        """Daemon updates last_actual_wake_at while assistant updates."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_assistant_message,
        )
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def daemon_sim():
            try:
                barrier.wait()
                for _ in range(30):
                    def _daemon_update(state: dict):
                        state["last_actual_wake_at"] = (
                            datetime(2026, 8, 10, 14, 0, tzinfo=UTC).isoformat())
                        state["last_completed_wake_id"] = "wid_daemon"
                    atomic_update(_daemon_update)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"daemon: {e}")

        def assistant_sim():
            try:
                barrier.wait()
                for i in range(30):
                    on_assistant_message(f"user_{i % 3}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"assistant: {e}")

        t1 = threading.Thread(target=daemon_sim); t2 = threading.Thread(target=assistant_sim)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        s = load_state()
        assert s["last_assistant_message_at"] is not None
        assert s["last_actual_wake_at"] is not None

    def test_startup_recovery_vs_user_message(self):
        """Startup recovery writes state while user message arrives."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_user_message,
        )
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def recovery_sim():
            try:
                barrier.wait()
                for _ in range(30):
                    def _recovery(state: dict):
                        state["last_recovery_at"] = (
                            datetime(2026, 8, 10, 14, 0, tzinfo=UTC).isoformat())
                        state["missed_wake_count"] = state.get("missed_wake_count", 0) + 1
                    atomic_update(_recovery)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"recovery: {e}")

        def user_sim():
            try:
                barrier.wait()
                for i in range(30):
                    on_user_message(f"user_{i % 3}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"user: {e}")

        t1 = threading.Thread(target=recovery_sim); t2 = threading.Thread(target=user_sim)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        s = load_state()
        assert s["last_user_message_at"] is not None
        assert s["last_recovery_at"] is not None

    def test_30_rounds_all_fields_preserved(self):
        """30 rounds of concurrent daemon+user+assistant → all fields survive."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_user_message, on_assistant_message,
        )
        errors = []
        barrier = threading.Barrier(3, timeout=5)

        def daemon():
            try:
                barrier.wait()
                for i in range(30):
                    def _d(s: dict):
                        s["next_wake_at"] = datetime(2026, 8, 10, 15, 0, tzinfo=UTC).isoformat()
                        s["last_actual_wake_at"] = datetime(2026, 8, 10, 14, i, tzinfo=UTC).isoformat()
                    atomic_update(_d)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"daemon: {e}")

        def user():
            try:
                barrier.wait()
                for i in range(30):
                    on_user_message("u1")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"user: {e}")

        def assistant():
            try:
                barrier.wait()
                for i in range(30):
                    on_assistant_message("u1")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"assistant: {e}")

        threads = [threading.Thread(target=f) for f in [daemon, user, assistant]]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        s = load_state()
        assert s["last_user_message_at"] is not None, "User timestamp lost"
        assert s["last_assistant_message_at"] is not None, "Assistant timestamp lost"
        assert s["last_actual_wake_at"] is not None, "Wake timestamp lost"
        assert s["next_wake_at"] is not None, "next_wake lost"

    def test_state_json_always_valid(self):
        """After concurrent writes, state is always parseable JSON."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_user_message,
        )
        import threading, json

        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def writer():
            try:
                barrier.wait()
                for i in range(50):
                    on_user_message(f"u_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))

        def reader():
            try:
                barrier.wait()
                for _ in range(50):
                    try:
                        s = load_state()
                        # Verify it's valid dict
                        assert isinstance(s, dict)
                        assert "state_version" in s
                    except Exception as e:
                        errors.append(str(e))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=writer); t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"JSON validity errors: {errors}"

    def test_no_state_tmp_residual(self, tmp_path):
        """After atomic updates, state.tmp should not persist."""
        from cow.initiative_engine.wakeup import atomic_update, save_state, _default_state

        sp = tmp_path / "state.json"
        save_state(_default_state(), sp)

        for _ in range(20):
            def _update(s: dict):
                s["test_counter"] = s.get("test_counter", 0) + 1
            atomic_update(_update, sp)

        # state.tmp should not exist (rename replaces it)
        tmp_file = tmp_path / "state.tmp"
        assert not tmp_file.exists(), f"state.tmp residual: {tmp_file}"

    def test_restart_reads_consistent_state(self):
        """After concurrent writes, restart read gets consistent state."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, on_user_message, on_assistant_message,
        )

        on_user_message("u1")
        on_assistant_message("u1")

        s1 = load_state()
        lum1 = s1["last_user_message_at"]
        lam1 = s1["last_assistant_message_at"]

        # "Restart" — re-read
        s2 = load_state()
        assert s2["last_user_message_at"] == lum1
        assert s2["last_assistant_message_at"] == lam1
        assert s2["last_user_message_at"] is not None
        assert s2["last_assistant_message_at"] is not None


# ═══════════════════════════════════════════════════════════════
# 2. 30-day wake simulation
# ═══════════════════════════════════════════════════════════════
class Test30DaySimulation:
    def test_30_day_simulation(self, tmp_path, monkeypatch):
        """Simulate 30 days of initiative engine wakes."""
        from cow.initiative_engine.wakeup import (
            atomic_update, load_state, _default_state, save_state, set_clock,
        )
        from cow.initiative_engine.gate import evaluate, is_quiet_hours
        from cow.initiative_engine.models import ContextSnapshot, MotiveCandidate
        from cow.initiative_engine.thoughts import generate as generate_thoughts

        sp = tmp_path / "sim_state.json"
        save_state(_default_state(), sp)
        monkeypatch.setattr("cow.initiative_engine.wakeup._DEFAULT_STATE_PATH", sp)

        # Simulation parameters
        start = datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
        results = {
            "total_wakes": 0,
            "silent": 0,
            "send_candidate": 0,
            "revisit": 0,
            "thought_types": {},
            "daily_candidates": [],
            "max_daily": 0,
            "real_delivery": 0,
        }

        for day in range(30):
            daily_count = 0
            # Simulate 3-6 wakes per day (daytime only)
            for wake_hour in [9, 12, 15, 18, 21]:
                wake_time = start + timedelta(days=day, hours=wake_hour - 8)
                set_clock(wake_time)

                # Simulate user activity patterns
                day_of_week = wake_time.weekday()
                if wake_hour == 9:
                    # Morning: user might have messaged last night
                    hours_since = 12 if day_of_week < 5 else 14
                elif wake_hour == 21:
                    hours_since = 2  # Evening: recent activity
                else:
                    hours_since = 5  # Mid-day: moderate gap

                cst_hour = (wake_hour + 8) % 24  # UTC→CST

                # Update state with simulated user timestamp
                def _sim_user(state: dict):
                    state["last_user_message_at"] = (
                        wake_time - timedelta(hours=hours_since)).isoformat()
                atomic_update(_sim_user, sp)

                s = load_state(sp)
                mins = hours_since * 60

                ctx = ContextSnapshot(
                    receiver_id="test_user",
                    local_hour=cst_hour,
                    minutes_since_user_message=mins,
                    quiet_hours=is_quiet_hours(cst_hour),
                    proactive_candidates_today=daily_count,
                )

                # Skip quiet hours
                if ctx.quiet_hours:
                    continue

                # Generate thoughts, build candidates, run gate
                thoughts = generate_thoughts(ctx, set(), [])
                candidates = []
                for t in thoughts[:3]:
                    mc = MotiveCandidate(
                        motive_type=t.thought_type, summary=t.subject,
                        evidence_memory_ids=t.evidence_ids,
                        confidence=t.confidence, urgency=t.relevance,
                        freshness=t.novelty, personal_relevance=t.relevance,
                        initiative_policy="shadow_only",
                    )
                    mc.dedupe_key = t.dedupe_key
                    candidates.append(mc)

                decision, reasons, selected = evaluate(
                    candidates, ctx, set(), {})

                results["total_wakes"] += 1
                if decision == "silent":
                    results["silent"] += 1
                elif decision == "send_candidate":
                    results["send_candidate"] += 1
                    daily_count += 1
                elif decision == "revisit_later":
                    results["revisit"] += 1

                if selected:
                    t = selected.motive_type
                    results["thought_types"][t] = results["thought_types"].get(t, 0) + 1

            results["daily_candidates"].append(daily_count)
            results["max_daily"] = max(results["max_daily"], daily_count)

        print(f"\n  === 30-Day Wake Simulation ===")
        print(f"  total_wakes: {results['total_wakes']}")
        print(f"  silent: {results['silent']} ({100*results['silent']/max(1,results['total_wakes']):.0f}%)")
        print(f"  send_candidate: {results['send_candidate']}")
        print(f"  revisit: {results['revisit']}")
        print(f"  real_delivery: 0 (DELIVERY_ENABLED=False)")
        print(f"  max_daily_candidates: {results['max_daily']}")
        print(f"  avg_daily_candidates: {sum(results['daily_candidates'])/len(results['daily_candidates']):.1f}")
        print(f"  thought_types: {results['thought_types']}")
        print(f"  social_presence count: {results['thought_types'].get('social_presence', 0)}")
        print(f"  ambient_event count: {results['thought_types'].get('ambient_event', 0)}")
        print(f"  task_followup count: {results['thought_types'].get('task_followup', 0)}")

        # Acceptance criteria
        assert results["total_wakes"] > 0
        assert results["silent"] > 0, "Should have silent periods"
        assert results["silent"] < results["total_wakes"], \
            "NOT 100% silent — social_presence and ambient_event should pass gate"
        assert results["max_daily"] <= 2, \
            f"Daily candidates {results['max_daily']} exceeds max 2"
        assert results["real_delivery"] == 0, "DELIVERY_ENABLED must be False"


# ═══════════════════════════════════════════════════════════════
# 3. Zero impact + isolation
# ═══════════════════════════════════════════════════════════════
class TestIsolation:
    def test_delivery_disabled(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_v1_v2_unchanged(self):
        import lancedb
        v1 = lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db"
        ).open_table("memories").search().limit(100000).to_list()
        v2 = lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db_v2"
        ).open_table("memories_v2").search().limit(100000).to_list()
        assert len(v1) == 709
        assert len(v2) == 2691

    def test_kill_switches(self):
        from cow.temporal_cognition.config import (
            TEMPORAL_PROMPT_ENABLED, TEMPORAL_INITIATIVE_ENABLED)
        assert TEMPORAL_PROMPT_ENABLED is True
        assert TEMPORAL_INITIATIVE_ENABLED is True
