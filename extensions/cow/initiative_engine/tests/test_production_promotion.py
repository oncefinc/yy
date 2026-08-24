"""Production promotion tests; all channel/API interactions are fakes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from cow.initiative_engine.delivery import configure_delivery, deliver
from cow.initiative_engine.models import InitiativeDecision


def test_host_binds_weixin_channel_with_reply_prefix(monkeypatch):
    import app
    import cow.initiative_engine.config as initiative_config

    # Public builds are fail-closed. This test explicitly opts into the
    # production branch while keeping all channel interactions fake.
    monkeypatch.setattr(initiative_config, "DELIVERY_ENABLED", True)

    sent = []
    remembered = []

    class FakeChannel:
        def send_proactive_text(self, receiver, text):
            sent.append((receiver, text))
            return True

    class FakeManager:
        def get_channel(self, name):
            return FakeChannel() if name == "weixin" else None

    class FakeAgentBridge:
        def remember_proactive_output(self, receiver, text, channel_type=""):
            remembered.append((receiver, text, channel_type))

    class FakeBridge:
        def get_agent_bridge(self):
            return FakeAgentBridge()

    monkeypatch.setattr(app, "conf", lambda: {
        "single_chat_reply_prefix": "[银月] "
    })
    import bridge.bridge as bridge_module
    monkeypatch.setattr(bridge_module, "Bridge", FakeBridge)
    app._configure_initiative_delivery(FakeManager())
    try:
        d = InitiativeDecision(receiver_id="teacher", decision="send_candidate",
                               candidate_message="在干嘛呀～")
        assert deliver(d) is True
        assert sent == [("teacher", "[银月] 在干嘛呀～")]
        assert remembered == [("teacher", "在干嘛呀～", "weixin")]
    finally:
        configure_delivery(None)


def test_channel_rejection_is_not_counted_as_delivery():
    configure_delivery(lambda receiver, message: False)
    try:
        d = InitiativeDecision(receiver_id="teacher", decision="send_candidate",
                               candidate_message="晚上好")
        assert deliver(d) is False
        assert d.delivery_allowed is False
    finally:
        configure_delivery(None)


def test_confirmed_proactive_output_is_persisted_as_assistant_turn(monkeypatch):
    from bridge.agent_bridge import AgentBridge

    stored = []
    bridge = object.__new__(AgentBridge)
    monkeypatch.setattr(
        bridge, "_persist_messages",
        lambda session_id, messages, channel_type="": stored.append(
            (session_id, messages, channel_type)
        ),
    )
    monkeypatch.setattr(
        bridge, "sync_session_messages_from_store", lambda session_id: None,
    )
    bridge.remember_proactive_output("teacher", "今天怎么样？", "weixin")
    assert stored[0][0] == "teacher"
    assert stored[0][1] == [{
        "role": "assistant",
        "content": [{"type": "text", "text": "今天怎么样？"}],
    }]


def test_temporal_loader_exposes_only_fresh_semantic_state(monkeypatch):
    import cow.temporal_cognition.store as store_mod
    import cow.temporal_cognition.lifecycle as lifecycle_mod
    from cow.initiative_engine.context_builder import _load_temporal_current_state

    @dataclass
    class Assertion:
        predicate: str
        value: str
        observed_at: str = datetime.now(timezone.utc).isoformat()
        lifecycle: str = "ongoing"
        evidence_type: str = "explicit_user"

    class FakeStore:
        def __init__(self, *args, **kwargs): pass
        def apply_lifecycle(self): return {}
        def get_active(self, subject):
            return [Assertion("location", "gym"),
                    Assertion("activity", "workout"),
                    Assertion("location", "30.1,104.1")]

    monkeypatch.setattr(store_mod, "WorldStateStore", FakeStore)
    monkeypatch.setattr(lifecycle_mod, "is_current_fact", lambda a: True)
    state = _load_temporal_current_state()
    assert state["activity"]["value"] == "workout"
    # Exact coordinates are excluded; the semantic location remains.
    assert state["location"]["value"] == "gym"


def test_non_candidate_is_never_delivered():
    calls = []
    configure_delivery(lambda receiver, message: calls.append(message) or True)
    try:
        d = InitiativeDecision(receiver_id="teacher", decision="silent",
                               candidate_message="should not send")
        assert deliver(d) is False
        assert calls == []
    finally:
        configure_delivery(None)
