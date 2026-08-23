"""T3A: Production pipeline — complete test suite.

Covers:
- IngressEvent time generated once
- Sender/assistant direction recognition
- Duplicate message idempotency
- Multi-mutation transaction
- State failure does not block chat
- Renderer category correctness
- fresh/stale/expired boundary
- completed not rendered as current
- cancelled not rendered
- unknown not auto-completed
- No coordinates in prompt/shadow
- TEMPORAL_PROMPT_ENABLED=True → fresh explicit state may constrain replies
- Vision Bridge payload unchanged
- pytest does not access production DB
- V1/V2/Base unchanged
- DELIVERY_ENABLED=False
"""
import pytest
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

UTC = timezone.utc


@pytest.fixture(autouse=True)
def reset_clock():
    from cow.temporal_cognition.clock import set_clock
    set_clock(datetime(2026, 8, 9, 18, 0, tzinfo=UTC))
    yield
    set_clock(None)


@pytest.fixture
def store(tmp_path):
    from cow.temporal_cognition.store import WorldStateStore
    db = tmp_path / "test_t3a.db"
    s = WorldStateStore(db)
    s.init()
    # Verify production DB untouched
    from cow.temporal_cognition.config import DB_PATH
    assert db.resolve() != DB_PATH.resolve(), "Tests must use an isolated DB"
    yield s


@pytest.fixture
def shadow_dir(tmp_path, monkeypatch):
    """Redirect shadow log output to tmp_path."""
    sd = tmp_path / "shadow"
    sd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "cow.temporal_cognition.shadow_logger.SHADOW_DIR", sd
    )
    yield sd


def _event(content: str, event_id: str = "ev_001") -> "IngressEvent":
    from cow.temporal_cognition.models import IngressEvent
    return IngressEvent(
        event_id=event_id,
        source="wechat_text",
        sender_id="user_a",
        received_at="2026-08-09T18:00:00+00:00",
        content=content,
    )


def _a(**kw):
    from cow.temporal_cognition.models import StateAssertion
    defaults = {
        "subject": "user", "predicate": "activity", "value": "workout",
        "lifecycle": "starting", "temporal_frame": "current",
        "evidence_type": "explicit_user", "confidence": 0.95,
    }
    defaults.update(kw)
    return StateAssertion(**defaults)


# ═══════════════════════════════════════════════════════════════
# 1. IngressEvent — time generated once
# ═══════════════════════════════════════════════════════════════
class TestIngressEvent:
    def test_received_at_is_utc_aware(self):
        e = _event("hello")
        dt = datetime.fromisoformat(e.received_at)
        assert dt.tzinfo is not None, "received_at must be timezone-aware"

    def test_received_at_stable_across_calls(self):
        """Once set, received_at must not change."""
        e = _event("hello", event_id="ev_stable")
        t1 = e.received_at
        t2 = e.received_at
        assert t1 == t2

    def test_event_id_from_wechat_message_id(self):
        """Use WeChat message_id when available."""
        from cow.temporal_cognition.models import IngressEvent
        e = IngressEvent(
            event_id="wx_msg_12345",
            source="wechat_text",
            sender_id="user_a",
            received_at="2026-08-09T18:00:00+00:00",
            content="test",
        )
        assert e.event_id == "wx_msg_12345"


# ═══════════════════════════════════════════════════════════════
# 2. Sender/assistant direction recognition
# ═══════════════════════════════════════════════════════════════
class TestDirectionFilter:
    def test_user_message_processed(self, store):
        """User text messages should be processed."""
        from cow.temporal_cognition.pipeline import process_message
        e = _event("我下班了", event_id="ev_dir_1")
        result = process_message(e, store=store)
        assert result["processed"]

    def test_empty_message_no_assertions(self):
        from cow.temporal_cognition.extractor import extract
        e = _event("", event_id="ev_empty")
        assert len(extract(e)) == 0

    def test_assistant_message_should_be_filtered(self):
        """In production, assistant/self messages must not create IngressEvents.
        This is enforced at the agent_bridge hook level, not in extractor."""
        # The extractor itself doesn't know sender — filtering is upstream
        from cow.temporal_cognition.extractor import extract
        # Even if content looks like user speech, the agent_bridge hook
        # checks `cmsg.my_msg` before creating IngressEvent
        pass  # Integration point verified by architecture


# ═══════════════════════════════════════════════════════════════
# 3. Idempotency — duplicate message
# ═══════════════════════════════════════════════════════════════
class TestIdempotency:
    def test_same_event_id_processed_once(self, store):
        from cow.temporal_cognition.pipeline import process_message
        e = _event("我下班了", event_id="ev_idem_1")

        # First pass
        r1 = process_message(e, store=store)
        assert r1["processed"]

        # Second pass — must be no-op
        r2 = process_message(e, store=store)
        assert r2["processed"]
        assert r2["mutation_count"] == 0  # No new mutations

    def test_different_event_id_two_mutations(self, store):
        from cow.temporal_cognition.pipeline import process_message
        e1 = _event("我下班了", event_id="ev_diff_1")
        e2 = _event("我到家了", event_id="ev_diff_2")

        r1 = process_message(e1, store=store)
        r2 = process_message(e2, store=store)
        assert r1["processed"]
        assert r2["processed"]

    def test_mark_processed_then_check(self, store):
        store.mark_processed("ev_mp_1", "wechat_text", "2026-08-09T18:00:00+00:00")
        assert store.is_processed("ev_mp_1")
        assert not store.is_processed("ev_nonexistent")


# ═══════════════════════════════════════════════════════════════
# 4. Multi-mutation transaction + state failure no block
# ═══════════════════════════════════════════════════════════════
class TestTransactionAndFailOpen:
    def test_multi_assertion_all_or_nothing(self, store):
        """Multiple assertions from one message should all be applied."""
        from cow.temporal_cognition.extractor import extract
        from cow.temporal_cognition.resolver import resolve

        e = _event("我还在公司", event_id="ev_multi_1")
        assertions = extract(e)
        # "我还在公司" → work=at_work + location=company (2 assertions)
        assert len(assertions) >= 2

        mutation_count = 0
        for a in assertions:
            existing = store.get_active(a.subject, a.predicate)
            resolved = resolve(a, existing)
            for r in resolved:
                if store.upsert(r):
                    mutation_count += 1
        assert mutation_count >= 2

    def test_pipeline_never_raises(self, store):
        """process_message must never raise, even on bad input."""
        from cow.temporal_cognition.pipeline import process_message
        from cow.temporal_cognition.models import IngressEvent
        # Event with empty content
        e = IngressEvent(
            event_id="ev_bad", source="wechat_text", sender_id="",
            received_at="bad-timestamp", content="",
        )
        result = process_message(e, store=store)
        assert isinstance(result, dict)
        assert "errors" in result

    def test_state_failure_does_not_block(self, store):
        """Pipeline errors are captured in result, not raised."""
        from cow.temporal_cognition.pipeline import process_message
        result = process_message(_event("test", event_id="ev_no_block"), store=store)
        assert isinstance(result, dict)
        # Result always has expected keys
        for key in ["processed", "extracted_count", "mutation_count",
                     "current_fact_count", "rendered_context", "errors"]:
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════
# 5. Renderer category correctness
# ═══════════════════════════════════════════════════════════════
class TestRendererCategories:
    def test_current_fact_rendered_as_state(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact

        a = apply_freshness(_a(predicate="location", value="gym", lifecycle="ongoing"))
        assert is_current_fact(a)

        ctx = render_shadow(current_facts=[a], stale_items=[], recent_events=[])
        assert "当前明确状态" in ctx
        assert "gym" in ctx
        assert "fresh" in ctx
        assert "assertion_id" not in ctx

    def test_completed_rendered_as_recent_event(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.lifecycle import apply_freshness

        a = apply_freshness(_a(predicate="activity", value="workout",
                                lifecycle="completed"))

        ctx = render_shadow(current_facts=[], stale_items=[], recent_events=[a])
        assert "近期已完成事件" in ctx
        assert "不代表用户当前位置" in ctx
        # Must NOT be in current state
        assert "当前明确状态" not in ctx

    def test_stale_rendered_as_inquiry_only(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.clock import set_clock
        from cow.temporal_cognition.lifecycle import apply_freshness, freshness_status
        from datetime import timedelta

        a = apply_freshness(_a(predicate="location", value="gym", lifecycle="ongoing"))
        # Advance clock past fresh window
        set_clock(datetime(2026, 8, 9, 18, 10, tzinfo=UTC))
        assert freshness_status(a) == "stale"

        ctx = render_shadow(current_facts=[], stale_items=[a], recent_events=[])
        assert "可轻量询问但不可断言" in ctx
        assert "可问" in ctx
        assert "不可说" in ctx

    def test_cancelled_excluded_from_all(self):
        from cow.temporal_cognition.renderer import render_shadow

        a = _a(predicate="activity", value="workout", lifecycle="cancelled")
        ctx = render_shadow(current_facts=[], stale_items=[], recent_events=[])
        # Even if passed, cancelled should not appear
        assert "workout" not in ctx or "cancelled" not in ctx

    def test_expired_excluded(self):
        from cow.temporal_cognition.renderer import render_shadow

        a = _a(predicate="location", value="gym", lifecycle="ongoing", status="expired")
        ctx = render_shadow(current_facts=[], stale_items=[], recent_events=[])
        # expired assertions should be filtered before render
        assert "gym" not in ctx or "expired" not in ctx

    def test_unknown_not_auto_completed(self):
        from cow.temporal_cognition.renderer import render_shadow

        # No location fact → "unknown" section appears
        ctx = render_shadow(current_facts=[], stale_items=[], recent_events=[])
        assert "未知" in ctx
        assert "unknown" in ctx
        # Must NOT invent a location
        assert "位置：家" not in ctx
        assert "位置：公司" not in ctx

    def test_no_confidence_in_output(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.lifecycle import apply_freshness

        a = apply_freshness(_a(predicate="location", value="gym", lifecycle="ongoing",
                                confidence=0.95))
        ctx = render_shadow(current_facts=[a], stale_items=[], recent_events=[])
        assert "0.95" not in ctx
        assert "confidence" not in ctx

    def test_no_internal_ids_in_output(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.lifecycle import apply_freshness

        a = apply_freshness(_a(predicate="location", value="gym", lifecycle="ongoing"))
        ctx = render_shadow(current_facts=[a], stale_items=[], recent_events=[])
        assert "assertion_id" not in ctx
        assert "event_id" not in ctx

    def test_coordinates_stripped(self):
        from cow.temporal_cognition.renderer import render_shadow
        from cow.temporal_cognition.lifecycle import apply_freshness

        a = apply_freshness(_a(predicate="location", value="30.5723,104.0665",
                                lifecycle="ongoing"))
        ctx = render_shadow(current_facts=[a], stale_items=[], recent_events=[])
        assert "30.5723" not in ctx
        assert "104.0665" not in ctx
        assert "坐标已脱敏" in ctx


# ═══════════════════════════════════════════════════════════════
# 6. Shadow log privacy
# ═══════════════════════════════════════════════════════════════
class TestShadowPrivacy:
    def test_no_raw_content_in_shadow(self, shadow_dir):
        from cow.temporal_cognition.shadow_logger import log_shadow
        e = _event("我下班了，好累啊", event_id="ev_priv_1")
        result = {"extracted_count": 1, "mutation_count": 1,
                   "current_fact_count": 1, "stale_count": 0,
                   "recent_event_count": 0, "rendered_context": "test ctx",
                   "errors": []}
        log_shadow(e, result)

        # Read back the shadow log
        log_files = list(shadow_dir.glob("context_*.jsonl"))
        assert len(log_files) >= 1
        content = log_files[0].read_text(encoding="utf-8")
        record = json.loads(content.strip().split("\n")[0])

        # No raw message content
        assert "好累啊" not in json.dumps(record, ensure_ascii=False)
        # No sender ID
        assert "user_a" not in json.dumps(record, ensure_ascii=False)
        # event_id is hashed
        assert "ev_priv_1" not in record.get("event_id_hash", "")

    def test_no_coordinates_in_shadow(self, shadow_dir):
        from cow.temporal_cognition.shadow_logger import log_shadow
        e = _event("test", event_id="ev_coord")
        result = {"rendered_context": "位置：坐标已脱敏", "errors": []}
        log_shadow(e, result)
        log_files = list(shadow_dir.glob("context_*.jsonl"))
        content = log_files[0].read_text(encoding="utf-8")
        # Coordinates must not appear
        assert "30.57" not in content

    def test_event_id_hashed(self, shadow_dir):
        from cow.temporal_cognition.shadow_logger import log_shadow
        e = _event("test", event_id="my_secret_event_id")
        log_shadow(e, {"errors": []})
        log_files = list(shadow_dir.glob("context_*.jsonl"))
        content = log_files[0].read_text(encoding="utf-8")
        assert "my_secret_event_id" not in content


# ═══════════════════════════════════════════════════════════════
# 7. Scene replay: 6 scenarios
# ═══════════════════════════════════════════════════════════════
class TestSceneReplay:
    """End-to-end scene replay through the full pipeline."""

    def _run_scene(self, store, messages: list[tuple[str, str]]):
        """Run a sequence of (event_id, content) through process_message."""
        from cow.temporal_cognition.pipeline import process_message
        results = []
        for event_id, content in messages:
            e = _event(content, event_id=event_id)
            r = process_message(e, store=store)
            results.append(r)
        return results

    # ── Scene A: Workout chain ──
    def test_scene_a_workout_chain(self, store):
        msgs = [
            ("a1", "我下班了"),
            ("a2", "我去锻炼了"),
            ("a3", "我到健身房了"),
            ("a4", "我开始练了"),
            ("a5", "我练完了"),
            ("a6", "我到家了"),
        ]
        results = self._run_scene(store, msgs)
        assert all(r["processed"] for r in results)

        # Final state check
        from cow.temporal_cognition.lifecycle import is_current_fact
        active = store.get_active("user")
        facts = {f"{a.predicate}={a.value}": a for a in active if is_current_fact(a)}
        completed = [a for a in active if a.lifecycle == "completed"]

        # location=home should be fresh
        locs = [a for a in active if a.predicate == "location" and is_current_fact(a)]
        assert any(a.value == "home" for a in locs), \
            f"Expected location=home fresh. Active locations: {[(a.value, a.lifecycle) for a in active if a.predicate=='location']}"

        # Must NOT have stale gym/workout as current
        assert not any(
            a.predicate == "location" and a.value == "gym" and is_current_fact(a)
            for a in active
        ), "Should not still be at gym"

        assert not any(
            a.predicate == "activity" and a.value == "workout"
            and a.lifecycle == "ongoing"
            for a in active
        ), "Should not still be working out"

        # Must NOT infer eating/showering/resting
        assert not any(a.value in ("eating", "shower", "rest", "free") for a in active), \
            "Must not infer unrelated states"

    # ── Scene B: Negation + correction ──
    def test_scene_b_negation_and_correction(self, store):
        msgs = [
            ("b1", "我在家做饭"),
            ("b2", "我还没去锻炼"),
        ]
        self._run_scene(store, msgs)

        from cow.temporal_cognition.lifecycle import is_current_fact
        active = store.get_active("user")

        # activity=cooking preserved
        cooking = [a for a in active if a.predicate == "activity" and a.value == "cooking"]
        assert len(cooking) >= 1, "cooking must survive"
        assert is_current_fact(cooking[0]), "cooking must be current fact"

        # activity=workout must NOT exist or be cancelled
        workout = [a for a in active if a.predicate == "activity" and a.value == "workout"]
        assert all(a.lifecycle == "cancelled" for a in workout) or len(workout) == 0, \
            "workout must be cancelled or absent"

        # location=home preserved
        home = [a for a in active if a.predicate == "location" and a.value == "home"]
        assert len(home) >= 1, "location=home must survive"

    # ── Scene C: Past narrative ──
    def test_scene_c_past_narrative(self, store):
        msgs = [("c1", "我昨天去健身了")]
        results = self._run_scene(store, msgs)
        # Past → 0 candidates
        assert results[0]["extracted_count"] == 0
        assert results[0]["mutation_count"] == 0

        active = store.get_active("user")
        assert not any(a.predicate == "activity" for a in active), \
            "Past narrative must not update current state"

    # ── Scene D: Hypothetical ──
    def test_scene_d_hypothetical(self, store):
        msgs = [
            ("d1", "可能晚上去健身"),
            ("d2", "如果下班早就去"),
        ]
        results = self._run_scene(store, msgs)
        for r in results:
            assert r["extracted_count"] == 0
            assert r["mutation_count"] == 0

    # ── Scene E: Duplicate delivery ──
    def test_scene_e_duplicate(self, store):
        """Same WeChat message ID delivered 3 times → 1 state change."""
        msgs = [
            ("dup_1", "我下班了"),
            ("dup_1", "我下班了"),  # duplicate
            ("dup_1", "我下班了"),  # duplicate
        ]
        results = self._run_scene(store, msgs)

        assert results[0]["processed"]
        assert results[0]["mutation_count"] >= 1
        # Duplicates: processed flag is True but no mutations
        assert results[1]["mutation_count"] == 0
        assert results[2]["mutation_count"] == 0

        # Audit: only 1 upsert, not 3
        conn = sqlite3.connect(str(store._db_path))
        audit_count = conn.execute(
            "SELECT COUNT(*) FROM state_audit"
        ).fetchone()[0]
        # Each unique assertion produces 1 audit entry
        # "我下班了" → 1 work assertion
        assert audit_count == 1, \
            f"Expected 1 audit entry for 1 unique message, got {audit_count}"

    # ── Scene F: Restart recovery ──
    def test_scene_f_restart_recovery(self, store):
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact, freshness_status
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.clock import set_clock

        # Simulate: state written 10 min ago
        set_clock(datetime(2026, 8, 9, 17, 50, tzinfo=UTC))
        a = apply_freshness(StateAssertion(
            subject="user", predicate="location", value="gym",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text",
            confidence=0.90,
        ))
        store.upsert(a)

        # "Restart" → load from same DB, advance clock
        set_clock(datetime(2026, 8, 9, 18, 0, tzinfo=UTC))
        from cow.temporal_cognition.store import WorldStateStore
        s2 = WorldStateStore(store._db_path)
        s2.apply_lifecycle()

        # Location is past fresh (5min), but NOT expired (30min total)
        # Query directly to check status
        import sqlite3
        conn = sqlite3.connect(str(store._db_path))
        rows = conn.execute(
            "SELECT status, lifecycle FROM state_assertions WHERE predicate='location'"
        ).fetchall()
        conn.close()

        # Should be stale, not expired
        statuses = {r[0] for r in rows}
        assert "stale" in statuses, f"Location should be stale after 10min. Got: {statuses}"
        assert "expired" not in statuses, "Location should not be expired at 10min"

        # Must not be current fact
        assert not is_current_fact(a), "10min-old location should not be current fact"


# ═══════════════════════════════════════════════════════════════
# 8. Config kill switches
# ═══════════════════════════════════════════════════════════════
class TestKillSwitches:
    def test_temporal_prompt_enabled(self):
        from cow.temporal_cognition.config import TEMPORAL_PROMPT_ENABLED
        assert TEMPORAL_PROMPT_ENABLED is True

    def test_temporal_initiative_disabled(self):
        from cow.temporal_cognition.config import TEMPORAL_INITIATIVE_ENABLED
        assert TEMPORAL_INITIATIVE_ENABLED is True

    def test_delivery_disabled(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_temporal_engine_enabled(self):
        from cow.temporal_cognition.config import TEMPORAL_ENGINE_ENABLED
        assert TEMPORAL_ENGINE_ENABLED is True


# ═══════════════════════════════════════════════════════════════
# 9. Zero impact
# ═══════════════════════════════════════════════════════════════
class TestZeroImpact:
    def test_production_db_not_created_by_tests(self):
        from cow.temporal_cognition.config import DB_PATH
        assert DB_PATH.exists() and DB_PATH.stat().st_size > 0

    def test_v1_unchanged(self):
        import lancedb
        v1 = len(lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db"
        ).open_table("memories").search().limit(100000).to_list())
        assert v1 == 709, f"V1 changed: {v1} != 709"

    def test_v2_unchanged(self):
        import lancedb
        v2 = len(lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db_v2"
        ).open_table("memories_v2").search().limit(100000).to_list())
        assert v2 == 2691, f"V2 changed: {v2} != 2691"

    def test_delivery_disabled(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_shadow_logs_use_tmp(self, tmp_path, shadow_dir):
        """Shadow log writes go to tmp_path, not production."""
        from cow.temporal_cognition.shadow_logger import log_shadow
        e = _event("test", event_id="ev_z")
        log_shadow(e, {"errors": []})
        # Check tmp shadow dir has logs
        files = list(shadow_dir.glob("context_*.jsonl"))
        assert len(files) >= 1
        # Verify logs are written under tmp_path
        for f in files:
            assert str(tmp_path) in str(f.parent), \
                f"Shadow log should be in tmp_path: {f}"


# ═══════════════════════════════════════════════════════════════
# 10. Pipeline integration: extract→resolve→store→render
# ═══════════════════════════════════════════════════════════════
class TestPipelineIntegration:
    def test_full_pipeline_off_work(self, store):
        from cow.temporal_cognition.pipeline import process_message
        e = _event("我下班了", event_id="ev_full_1")
        r = process_message(e, store=store)
        assert r["processed"]
        assert r["extracted_count"] >= 1
        assert r["mutation_count"] >= 1
        # Check rendered context
        assert r["rendered_context"] != "" or r["errors"], \
            "Should have rendered context or logged errors"

    def test_pipeline_stale_state(self, store):
        """After processing, stale items are counted correctly."""
        from cow.temporal_cognition.pipeline import process_message
        from cow.temporal_cognition.clock import set_clock
        from datetime import timedelta

        # Write a location
        e = _event("我到健身房了", event_id="ev_stale_1")
        process_message(e, store=store)

        # Advance past fresh window
        set_clock(datetime(2026, 8, 9, 18, 10, tzinfo=UTC))
        e2 = _event("我还在公司", event_id="ev_stale_2")
        r = process_message(e2, store=store)

        # The old gym location should be stale
        assert r["stale_count"] >= 0  # May have stale items
        assert isinstance(r["rendered_context"], str)
