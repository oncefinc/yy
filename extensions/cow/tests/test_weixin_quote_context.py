"""Regression tests for Weixin quoted-message context propagation."""

import sys

sys.path.insert(0, "cowagent")

from channel.weixin.weixin_message import WeixinMessage
from channel.weixin.weixin_reference_store import WeixinReferenceStore


def _message(text="看一下这个", **extra):
    msg = {
        "message_id": "quote-test",
        "from_user_id": "user",
        "to_user_id": "bot",
        "item_list": [{"type": 1, "text_item": {"text": text}}],
    }
    msg.update(extra)
    return msg


def test_documented_item_level_quote_is_injected():
    msg = _message()
    msg["item_list"][0]["ref_msg"] = {
        "title": "Once: synthetic development index",
        "message_item": {
            "type": 1,
            "text_item": {"text": "Synthetic source message"},
        },
    }
    parsed = WeixinMessage(msg)
    assert "Synthetic source message" in parsed.content
    assert parsed.content.endswith("看一下这个")
    assert parsed.reference_resolved is True


def test_nested_and_camel_case_quote_layouts_are_supported():
    msg = _message()
    msg["item_list"][0]["text_item"]["refMsg"] = {
        "messageItem": {"type": 1, "textItem": {"text": "Nested quote"}}
    }
    assert "Nested quote" in WeixinMessage(msg).content


def test_id_only_quote_exposes_reference_id():
    msg = _message()
    msg["item_list"][0]["ref_msg"] = {
        "message_item": {"type": 1, "msg_id": "item-original"}
    }
    parsed = WeixinMessage(msg)
    assert parsed.reference_ids == ["item-original"]
    assert parsed.reference_resolved is False


def test_reference_store_survives_reopen(tmp_path):
    path = tmp_path / "weixin_refs.db"
    store = WeixinReferenceStore(str(path))
    assert store.remember("user", ["envelope-1", "item-1"], "Synthetic text") == 2
    reopened = WeixinReferenceStore(str(path))
    assert reopened.resolve("user", ["item-1"]) == "Synthetic text"


def test_channel_recovers_id_only_quote_from_local_index(tmp_path):
    from channel.weixin.weixin_channel import _restore_quoted_text_from_store

    store = WeixinReferenceStore(str(tmp_path / "refs.db"))
    store.remember("user", ["original-item"], "Synthetic indexed source")
    msg = _message()
    msg["item_list"][0]["ref_msg"] = {
        "message_item": {"type": 1, "msg_id": "original-item"}
    }
    parsed = WeixinMessage(msg)
    assert _restore_quoted_text_from_store(parsed, "user", store) is True
    assert "Synthetic indexed source" in parsed.content


def test_original_text_is_indexed_before_agent_execution(tmp_path):
    from channel.weixin.weixin_channel import _index_inbound_text_in_store

    store = WeixinReferenceStore(str(tmp_path / "refs.db"))
    msg = _message("Synthetic message retained before a model error")
    msg["item_list"][0]["msg_id"] = "source-item"
    parsed = WeixinMessage(msg)
    assert _index_inbound_text_in_store(parsed, "user", store) == 2
    assert store.resolve("user", ["source-item"]) == (
        "Synthetic message retained before a model error"
    )


def test_plain_text_behavior_is_unchanged():
    parsed = WeixinMessage(_message("plain text"))
    assert parsed.content == "plain text"
    assert parsed.has_reference is False
