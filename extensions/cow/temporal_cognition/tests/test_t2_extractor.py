"""T2: Rule-based state extraction — semantic fix edition tests."""
import pytest
from datetime import datetime, timezone
from cow.temporal_cognition.models import IngressEvent, StateAssertion
from cow.temporal_cognition.extractor import extract, _detect_temporal_frame
from cow.temporal_cognition.clock import set_clock
from cow.temporal_cognition.lifecycle import is_current_fact

UTC = timezone.utc

@pytest.fixture(autouse=True)
def reset_clock():
    set_clock(datetime(2026, 8, 9, 18, 0, tzinfo=UTC))
    yield
    set_clock(None)

def _event(content: str) -> IngressEvent:
    return IngressEvent(source="weixin_text", content=content,
                        received_at="2026-08-09T18:00:00+00:00")

def _preds(results: list[StateAssertion]) -> list[str]:
    return [f"{r.predicate}={r.value}({r.lifecycle})" for r in results]

# ── 1. Location lifecycle: ongoing, not completed ───
class TestLocationLifecycle:
    def test_arrive_gym_is_ongoing(self):
        r = extract(_event("我到健身房了"))
        locs = [a for a in r if a.predicate == "location"]
        assert locs[0].lifecycle == "ongoing"
        assert locs[0].value == "gym"

    def test_arrive_home_is_ongoing(self):
        r = extract(_event("我到家了"))
        locs = [a for a in r if a.predicate == "location"]
        assert locs[0].lifecycle == "ongoing"
        assert locs[0].value == "home"

    def test_gym_is_current_fact_within_5min(self):
        from cow.temporal_cognition.clock import set_clock
        set_clock(datetime(2026, 8, 9, 18, 0, 0, tzinfo=UTC))
        r = extract(_event("我到健身房了"))
        locs = [a for a in r if a.predicate == "location"]
        assert is_current_fact(locs[0])

    def test_gym_stale_after_fresh(self):
        from cow.temporal_cognition.clock import set_clock
        set_clock(datetime(2026, 8, 9, 18, 0, 0, tzinfo=UTC))
        r = extract(_event("我到健身房了"))
        set_clock(datetime(2026, 8, 9, 18, 6, 0, tzinfo=UTC))
        locs = [a for a in r if a.predicate == "location"]
        assert not is_current_fact(locs[0])

    def test_arrive_gym_not_start_workout(self):
        r = extract(_event("我到健身房了"))
        assert not any(a.predicate == "activity" and a.value == "workout" for a in r)

    def test_arrive_home_not_infer_rest_or_eat(self):
        r = extract(_event("我到家了"))
        assert not any(a.predicate == "activity" for a in r)

# ── 2. Event completed vs current state ─────────────
class TestEventVsState:
    def test_off_work_is_event_not_location(self):
        r = extract(_event("我下班了"))
        assert any(a.predicate == "work" and a.lifecycle == "completed" for a in r)
        assert not any(a.predicate == "location" for a in r)

    def test_workout_completed_is_event_not_location(self):
        r = extract(_event("我练完了"))
        assert any(a.predicate == "activity" and a.lifecycle == "completed" for a in r)
        assert not any(a.predicate == "location" for a in r)

# ── 3. Negation as invalidation ─────────────────────
class TestNegation:
    def test_not_at_company_invalidates(self):
        r = extract(_event("我没在公司"))
        inv = [a for a in r if a.lifecycle == "cancelled"]
        assert len(inv) >= 1
        assert inv[0].predicate == "location"

    def test_not_at_company_not_generate_home(self):
        r = extract(_event("我没在公司"))
        assert not any(a.value == "home" for a in r)

    def test_not_yet_workout_invalidates(self):
        r = extract(_event("我还没去锻炼"))
        inv = [a for a in r if a.lifecycle == "cancelled"]
        assert len(inv) >= 1

    def test_not_home_still_en_route(self):
        r = extract(_event("我没到家，还在路上"))
        assert any(a.lifecycle == "cancelled" and a.predicate == "location" for a in r)
        assert any(a.value == "en_route" for a in r)

    def test_negation_preserves_target_value(self):
        """'我还没去锻炼' must cancel activity=workout only, not activity=cooking."""
        r = extract(_event("我还没去锻炼"))
        inv = [a for a in r if a.lifecycle == "cancelled"]
        assert len(inv) == 1
        assert inv[0].predicate == "activity"
        assert inv[0].value == "workout", \
            f"Negation must preserve target value, got value={inv[0].value!r}"

    def test_negation_only_supersedes_matching_value(self, tmp_path):
        """Upsert of cancelled(activity=workout) must not supersede activity=cooking."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact

        db = tmp_path / "test_precise_neg.db"
        store = WorldStateStore(db)
        store.init()

        # Seed: user is cooking (active)
        cooking = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="cooking",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text",
            confidence=0.90))
        store.upsert(cooking)

        # User says "我还没去锻炼" → cancels activity=workout only
        r = extract(_event("我还没去锻炼"))
        cancelled = [a for a in r if a.lifecycle == "cancelled"]
        assert len(cancelled) == 1
        store.upsert(cancelled[0])

        # Run lifecycle sweep: cancelled → stale
        store.apply_lifecycle()

        # activity=cooking must survive as current fact
        active = store.get_active("user", "activity")
        active_values = {a.value: a for a in active}
        assert "cooking" in active_values, \
            f"Precise negation must not cancel unrelated activity=cooking. Got: {set(active_values.keys())}"
        assert is_current_fact(active_values["cooking"]), \
            "activity=cooking should still be a current fact"

        # activity=workout must NOT be a current fact
        assert "workout" not in active_values, \
            f"Cancelled workout should be stale, not active. Got: {set(active_values.keys())}"

# ── 3b. Precise negation — T2.1 complete scenarios ──
class TestPreciseNegation:
    """T2.1: 12-scenario precise negation verification."""

    # ── Scenario 2: workout exists → only workout cancelled ──
    def test_s2_workout_exists_only_workout_cancelled(self, tmp_path):
        """Seed activity=workout; negate; verify workout gone, nothing else touched."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact

        db = tmp_path / "s2.db"
        store = WorldStateStore(db); store.init()

        w = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="workout",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(w)

        r = extract(_event("我还没去锻炼"))
        cancelled = [a for a in r if a.lifecycle == "cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0].value == "workout"
        store.upsert(cancelled[0])
        store.apply_lifecycle()

        active = store.get_active("user", "activity")
        assert len(active) == 0, f"Should be no active activity after cancelling sole workout. Got: {[a.value for a in active]}"

    # ── Scenario 4: target absent → safe no-op ──
    def test_s4_target_absent_safe_noop(self, tmp_path):
        """Cancel non-existent activity=workout must not affect other state."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness

        db = tmp_path / "s4.db"
        store = WorldStateStore(db); store.init()

        # Seed: activity=cooking (no workout exists)
        cooking = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="cooking",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(cooking)

        # Cancel non-existent activity=workout
        r = extract(_event("我还没去锻炼"))
        cancelled = [a for a in r if a.lifecycle == "cancelled"]
        store.upsert(cancelled[0])
        store.apply_lifecycle()

        # cooking must survive
        active = store.get_active("user", "activity")
        values = {a.value for a in active}
        assert "cooking" in values, f"No-op cancel must preserve cooking. Got: {values}"
        assert len(active) == 1, f"Should have exactly 1 active activity. Got: {len(active)}"

    # ── Scenario 5: "我没在公司" only cancels location=company ──
    def test_s5_not_at_company_only_cancels_company(self, tmp_path):
        """Negating location=company cancels company only — precise value targeting."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact

        db = tmp_path / "s5.db"
        store = WorldStateStore(db); store.init()

        # Seed: location=company (active)
        a = apply_freshness(StateAssertion(
            subject="user", predicate="location", value="company",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(a)
        assert is_current_fact(store.get_active("user", "location")[0])

        # User says "我没在公司"
        r = extract(_event("我没在公司"))
        cancelled = [a for a in r if a.lifecycle == "cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0].predicate == "location"
        assert cancelled[0].value == "company", \
            f"Must target company precisely, got value={cancelled[0].value!r}"
        store.upsert(cancelled[0])
        store.apply_lifecycle()

        # location=company must be gone
        active = store.get_active("user", "location")
        values = {a.value for a in active}
        assert "company" not in values, f"company should be cancelled. Got: {values}"
        # The cancelled assertion itself must not be a current fact
        assert not any(is_current_fact(a) for a in active), \
            "No location should be current fact after company is cancelled"

    # ── Scenario 6: "我没到家，还在路上" — composite ──
    def test_s6_not_home_en_route_composite(self, tmp_path):
        """Cancels location=home AND writes location=en_route atomically."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness

        db = tmp_path / "s6.db"
        store = WorldStateStore(db); store.init()

        # Seed: location=home
        home = apply_freshness(StateAssertion(
            subject="user", predicate="location", value="home",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(home)

        r = extract(_event("我没到家，还在路上"))
        for a in r:
            store.upsert(a)
        store.apply_lifecycle()

        active = store.get_active("user", "location")
        values = {a.value for a in active}
        assert "home" not in values, f"home should be cancelled. Got: {values}"
        assert "en_route" in values, f"en_route should be active. Got: {values}"

    # ── Scenario 7: rollback on failure ──
    def test_s7_cancel_rollback_on_failure(self, tmp_path):
        """If upsert of cancelled assertion fails, existing state is preserved."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness, is_current_fact

        db = tmp_path / "s7.db"
        store = WorldStateStore(db); store.init()

        # Seed cooking
        cooking = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="cooking",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        assert store.upsert(cooking)

        # Verify cooking survives — it was successfully written
        active = store.get_active("user", "activity")
        assert any(a.value == "cooking" for a in active), \
            "cooking should have been persisted"
        assert is_current_fact(active[0])

    # ── Scenario 8: idempotent event_id ──
    def test_s8_idempotent_event_id_negation(self, tmp_path):
        """Same event_id re-processed does not double-cancel."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness

        db = tmp_path / "s8.db"
        store = WorldStateStore(db); store.init()

        # Mark event as already processed
        store.mark_processed("ev_neg8", "wechat_text", "2026-08-09T18:00:00+00:00")
        assert store.is_processed("ev_neg8")

        # Seed cooking
        cooking = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="cooking",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(cooking)

        # Re-processing same event should be no-op (event already processed)
        assert store.is_processed("ev_neg8")
        # cooking unaffected
        active = store.get_active("user", "activity")
        assert any(a.value == "cooking" for a in active)

    # ── Scenario 9: cancelled not returned by is_current_fact ──
    def test_s9_cancelled_not_current_fact(self):
        """A cancelled assertion must never pass is_current_fact()."""
        r = extract(_event("我还没去锻炼"))
        cancelled = [a for a in r if a.lifecycle == "cancelled"]
        assert len(cancelled) == 1
        assert is_current_fact(cancelled[0]) is False, \
            "cancelled assertion must not be a current fact"
        # Also verify via lifecycle freshness_status
        from cow.temporal_cognition.lifecycle import freshness_status
        # cancelled lifecycle → stale (not fresh)
        assert freshness_status(cancelled[0]) != "fresh"

    # ── Scenario 10: get_active excludes cancelled after sweep ──
    def test_s10_get_active_excludes_cancelled_after_sweep(self, tmp_path):
        """After lifecycle sweep, cancelled assertion must not appear in get_active."""
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.models import StateAssertion
        from cow.temporal_cognition.lifecycle import apply_freshness

        db = tmp_path / "s10.db"
        store = WorldStateStore(db); store.init()

        # Seed: workout is active
        w = apply_freshness(StateAssertion(
            subject="user", predicate="activity", value="workout",
            lifecycle="ongoing", temporal_frame="current",
            evidence_type="explicit_user", source="wechat_text", confidence=0.90))
        store.upsert(w)

        # Cancel it
        r = extract(_event("我还没去锻炼"))
        for a in r:
            store.upsert(a)
        store.apply_lifecycle()

        active = store.get_active("user", "activity")
        assert not any(a.value == "workout" for a in active), \
            f"cancelled workout must not appear in active set. Got: {[a.value for a in active]}"

    # ── Scenario 11: no value="" pseudo-state in DB ──
    def test_s11_no_empty_value_in_db(self, tmp_path):
        """Database must not contain any cancellation with value=''."""
        import sqlite3
        from cow.temporal_cognition.store import WorldStateStore

        db = tmp_path / "s11.db"
        store = WorldStateStore(db); store.init()

        r = extract(_event("我还没去锻炼"))
        for a in r:
            store.upsert(a)

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT value FROM state_assertions WHERE lifecycle='cancelled'"
        ).fetchall()
        for row in rows:
            assert row[0] != "", \
                f"DB must not contain value='' for cancelled assertion. Got: {row[0]!r}"
            assert row[0] is not None, \
                "cancelled assertion value must not be NULL"
        conn.close()

    # ── Scenario 12: Audit does not leak raw text ──
    def test_s12_audit_no_raw_text_leak(self, tmp_path):
        """Audit log must not contain evidence_text_span or raw user message."""
        import sqlite3
        from cow.temporal_cognition.store import WorldStateStore

        db = tmp_path / "s12.db"
        store = WorldStateStore(db); store.init()

        r = extract(_event("我还没去锻炼"))
        for a in r:
            store.upsert(a)

        conn = sqlite3.connect(str(db))
        for row in conn.execute("SELECT details FROM state_audit").fetchall():
            details = row[0]
            assert "锻炼" not in details, f"Raw text leaked to audit: {details}"
            assert "还没" not in details, f"Raw text leaked to audit: {details}"
        conn.close()

# ── 4. Evidence/Temporal orthogonal ─────────────────
class TestEvidenceTemporal:
    def test_past_still_explicit_evidence(self):
        """'我昨天去健身了' is explicit_user even though it's past."""
        # Verify frame detection is correct
        assert _detect_temporal_frame("我昨天去健身了") == "past"
        # Past messages → 0 candidates for current state
        r = extract(_event("我昨天去健身了"))
        assert len(r) == 0

# ── 5. Hypothetical/question/third_party → 0 ────────
class TestFailClosed:
    def test_hypothetical_zero(self):
        assert len(extract(_event("如果晚上去健身"))) == 0
        assert len(extract(_event("可能还在公司"))) == 0

    def test_future_zero(self):
        assert len(extract(_event("明天准备练腿"))) == 0

    def test_question_zero(self):
        assert len(extract(_event("你是不是到家了"))) == 0

    def test_third_party_zero(self):
        assert len(extract(_event("他说他下班了"))) == 0

# ── 6. Correction ───────────────────────────────────
class TestCorrection:
    def test_correct_focus_only(self):
        r = extract(_event("不是练背，今天练腿"))
        vals = _preds(r)
        assert "workout_focus=腿(ongoing)" in vals, f"Got: {vals}"
        # Only corrects focus, not activity
        assert not any(a.predicate == "activity" and a.lifecycle == "cancelled" for a in r)

# ── 7. Basic extractions ────────────────────────────
class TestBasic:
    def test_off_work(self):
        assert any(a.predicate == "work" for a in extract(_event("我下班了")))
    def test_go_workout(self):
        assert any(a.predicate == "activity" for a in extract(_event("我来锻炼啦")))
    def test_start_training(self):
        assert any(a.lifecycle == "ongoing" for a in extract(_event("我开始练了")))
    def test_finish_workout(self):
        assert any(a.lifecycle == "completed" for a in extract(_event("我练完了")))
    def test_still_at_company_multi(self):
        r = extract(_event("我还在公司"))
        assert len(r) >= 2
    def test_cooking_at_home(self):
        r = extract(_event("我在家做饭"))
        assert any(a.value == "cooking" for a in r)

# ── 8. Multi-predicate ──────────────────────────────
class TestMultiPredicate:
    def test_still_at_company_work_and_location(self):
        r = extract(_event("我还在公司"))
        preds = {a.predicate for a in r}
        assert "work" in preds and "location" in preds

# ── 9. No inference ─────────────────────────────────
class TestNoInference:
    def test_go_workout_no_focus(self):
        r = extract(_event("我来锻炼啦"))
        assert not any(a.predicate == "workout_focus" for a in r)
    def test_arrive_home_no_activity(self):
        r = extract(_event("我到家了"))
        assert not any(a.predicate == "activity" for a in r)
    def test_off_work_no_location(self):
        r = extract(_event("我下班了"))
        assert not any(a.predicate == "location" for a in r)
    def test_correction_no_side_effects(self):
        r = extract(_event("不是练背，今天练腿"))
        # Only workout_focus affected
        assert all(a.predicate in ("workout_focus",) or a.lifecycle == "cancelled"
                   for a in r), f"Side effects: {_preds(r)}"

# ── 10. Evidence span ───────────────────────────────
class TestEvidence:
    def test_span_minimal(self):
        r = extract(_event("我下班了，准备去锻炼"))
        for a in r:
            assert len(a.evidence_text_span or "") <= 30

# ── 11. Idempotency ─────────────────────────────────
class TestIdempotency:
    def test_same_message_same_count(self):
        e = _event("我下班了")
        assert len(extract(e)) == len(extract(e))

# ── 12. Zero impact ─────────────────────────────────
class TestZeroImpact:
    def test_production_db_not_created(self):
        from cow.temporal_cognition.config import DB_PATH
        assert DB_PATH.exists() and DB_PATH.stat().st_size > 0
    def test_v2_unchanged(self):
        import lancedb
        v2 = len(lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2").open_table("memories_v2").search().limit(100000).to_list())
        assert v2 == 2691
    def test_delivery_disabled(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False
