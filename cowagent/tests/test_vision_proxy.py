from unittest.mock import patch

from bridge import vision_proxy


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}
TEXT = {"type": "text", "text": "原始问题"}


class FakeBot:
    def __init__(self, result=None):
        self.result = result or {"content": "画面里有一只黑猫。"}
        self.calls = []

    def call_vision(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _conf(values):
    return lambda: values


def test_no_blocks_is_noop():
    blocks, meta = vision_proxy.proxy_images_for_primary(None, "q", "enhanced")
    assert blocks is None
    assert meta["reason"] == "no_blocks"


def test_no_images_is_noop():
    blocks, meta = vision_proxy.proxy_images_for_primary([TEXT], "q", "enhanced")
    assert blocks == [TEXT]
    assert meta["reason"] == "no_images"


def test_disabled_keeps_image_and_uses_enhanced_query():
    with patch.object(vision_proxy, "conf", _conf({"deepseek_vision_proxy_enabled": False})):
        blocks, meta = vision_proxy.proxy_images_for_primary([IMAGE, TEXT], "raw", "enhanced")
    assert meta["reason"] == "disabled"
    assert blocks[0]["type"] == "image_url"
    assert blocks[-1] == {"type": "text", "text": "enhanced"}


def test_success_replaces_image_with_grounded_observation():
    bot = FakeBot()
    values = {
        "deepseek_vision_proxy_enabled": True,
        "deepseek_api_key": "configured",
        "deepseek_vision_model": "deepseek-v4-flash-vision-exp",
        "deepseek_vision_max_tokens": 600,
    }
    with patch.object(vision_proxy, "conf", _conf(values)), patch.object(
        vision_proxy, "_get_bot", return_value=bot
    ):
        blocks, meta = vision_proxy.proxy_images_for_primary([IMAGE, TEXT], "这是什么", "增强查询")
    assert meta["used"] is True
    assert len(blocks) == 1 and blocks[0]["type"] == "text"
    assert "增强查询" in blocks[0]["text"]
    assert "画面里有一只黑猫" in blocks[0]["text"]
    assert "不代表用户当前状态" in blocks[0]["text"]
    assert "base64" not in blocks[0]["text"]
    assert bot.calls[0]["model"] == "deepseek-v4-flash-vision-exp"


def test_multiple_images_are_all_observed():
    bot = FakeBot()
    values = {"deepseek_vision_proxy_enabled": True, "deepseek_api_key": "configured"}
    with patch.object(vision_proxy, "conf", _conf(values)), patch.object(
        vision_proxy, "_get_bot", return_value=bot
    ):
        blocks, meta = vision_proxy.proxy_images_for_primary([IMAGE, IMAGE, TEXT], "q", "e")
    assert meta["image_count"] == 2
    assert len(bot.calls) == 2
    assert "图1：" in blocks[0]["text"] and "图2：" in blocks[0]["text"]


def test_provider_error_falls_back_to_primary_image():
    bot = FakeBot({"error": True, "message": "failed"})
    values = {"deepseek_vision_proxy_enabled": True, "deepseek_api_key": "configured"}
    with patch.object(vision_proxy, "conf", _conf(values)), patch.object(
        vision_proxy, "_get_bot", return_value=bot
    ):
        blocks, meta = vision_proxy.proxy_images_for_primary([IMAGE, TEXT], "raw", "enhanced")
    assert meta["used"] is False
    assert meta["reason"] == "provider_error"
    assert blocks[0]["type"] == "image_url"
    assert blocks[-1]["text"] == "enhanced"


def test_empty_provider_content_falls_back():
    bot = FakeBot({"content": ""})
    values = {"deepseek_vision_proxy_enabled": True, "deepseek_api_key": "configured"}
    with patch.object(vision_proxy, "conf", _conf(values)), patch.object(
        vision_proxy, "_get_bot", return_value=bot
    ):
        blocks, meta = vision_proxy.proxy_images_for_primary([IMAGE], "raw", "enhanced")
    assert meta["reason"] == "provider_error"
    assert any(block.get("type") == "image_url" for block in blocks)


def test_prompt_forbids_current_state_inference():
    prompt = vision_proxy._observation_prompt("我在哪", 1, 1)
    assert "不等于用户当前所在位置" in prompt
    assert "不补全" in prompt
