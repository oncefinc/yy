"""Production delivery adapter tests (no real network or WeChat calls)."""
from cow.initiative_engine.config import DELIVERY_ENABLED
from cow.initiative_engine.shadow import assert_no_delivery

class TestDeliveryKillSwitch:
    def test_delivery_disabled_by_default_in_public_release(self):
        assert DELIVERY_ENABLED is False

    def test_delivery_can_be_explicitly_enabled_from_environment(self, monkeypatch):
        from cow.initiative_engine.config import _env_flag
        monkeypatch.setenv("INITIATIVE_DELIVERY_ENABLED", "true")
        assert _env_flag("INITIATIVE_DELIVERY_ENABLED", False) is True

    def test_legacy_probe_reports_shadow_default(self):
        assert assert_no_delivery() is True

    def test_decision_delivery_allowed_always_false(self):
        """Every decision must have delivery_allowed=False."""
        from cow.initiative_engine.models import InitiativeDecision
        d = InitiativeDecision(decision="send_candidate")
        assert d.delivery_allowed == False

    def test_audit_log_does_not_perform_delivery(self):
        from cow.initiative_engine.shadow import log_decision
        from cow.initiative_engine.models import InitiativeDecision
        d = InitiativeDecision(decision="send_candidate",
                               candidate_message="test message should not be sent")
        log_decision(d)
        assert d.delivery_allowed == False

    def test_adapter_delivers_exactly_once(self):
        from cow.initiative_engine.delivery import configure_delivery, deliver
        from cow.initiative_engine.models import InitiativeDecision
        calls = []
        configure_delivery(lambda receiver, message: calls.append((receiver, message)) or True)
        try:
            d = InitiativeDecision(receiver_id="user", decision="send_candidate",
                                   candidate_message="晚上好")
            assert deliver(d) is True
            assert calls == [("user", "晚上好")]
        finally:
            configure_delivery(None)

    def test_adapter_without_channel_fails_closed(self):
        from cow.initiative_engine.delivery import configure_delivery, deliver
        from cow.initiative_engine.models import InitiativeDecision
        configure_delivery(None)
        d = InitiativeDecision(receiver_id="user", decision="send_candidate",
                               candidate_message="晚上好")
        assert deliver(d) is False

    def test_no_wechat_import_in_engine(self):
        """Engine must not import wechat/channel modules."""
        import cow.initiative_engine.engine as eng
        src = open(eng.__file__, encoding="utf-8").read()
        assert "wechat" not in src.lower(), "Engine must not touch wechat!"
        assert "channel" not in src.lower(), "Engine must not touch channel!"
