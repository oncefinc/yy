"""Real Zhipu LLM adapter — called by llm_worker, not directly by engine."""
from __future__ import annotations
import json, logging, requests, time
from .models import ThoughtSeed, ContextSnapshot

logger = logging.getLogger("initiative.zhipu")

SYSTEM_PROMPT = """你是银月——用户的AI朋友。你的回复要像微信朋友聊天：1-2句话，自然、干脆、有主见。
不要像客服、HR、提醒软件或心理咨询师。不要用"根据记录""检测到""系统发现"等词。
不要编造你看见、听说或经历的事情。不确定时宁可不说。
只有念头中明确提供“已验证后台搜索 receipt”时，才可以说“我刚查了/我搜到”。
没有 receipt 时绝对不能声称自己刚刚浏览、查询或操作过。
可以有幽默感，可以吐槽，可以只表达一句感想不求回复。
不要每次都说"想你了"或"在干嘛"。不要固定用"呀、啦、哦、～"。
时间词必须与提供的北京时间一致。不要说"我们一起喝杯茶"等物理共处的话，除非是比喻。"""


def create_zhipu_generator(api_key: str, api_base: str, model: str = "glm-4-flash"):
    """Create a callable for llm_worker.configure()."""
    def generate(thought: ThoughtSeed, ctx: ContextSnapshot) -> dict | None:
        CST = __import__('datetime').timezone(__import__('datetime').timedelta(hours=8))
        local = __import__('datetime').datetime.now(CST)
        local_hour = local.hour

        evidence = thought.evidence_summary or "(无具体事实依据,只基于时间和情境)"
        current_state = json.dumps(ctx.current_state, ensure_ascii=False)
        if not ctx.current_state:
            current_state = "unknown（不得补全或猜测）"
        evidence = (
            f"{evidence}\n当前短期状态: {current_state}\n"
            "只有当前短期状态中的 fresh 明确信息可以当作‘现在’的事实；"
            "字段缺失就是未知。未知时可以自然地问一句，但不能用历史记忆、"
            "计划或作息表补全用户此刻状态。"
        )
        if thought.action_receipt_id:
            evidence += (
                f"\n已验证行为回执: {thought.action_receipt_id}。"
                "你可以如实表达刚做过搜索，但只能复述搜索证据中的信息。"
                f"事实 claims 的 evidence_id 必须填写 receipt:{thought.action_receipt_id}。"
            )
        user_prompt = f"""念头类型: {thought.thought_type}
念头: {thought.subject}
现在是北京时间{local_hour}点。距上次聊天约{max(1,ctx.minutes_since_user_message)//60}小时。
已知信息({thought.life_domain}): {evidence}

请生成一句自然的微信消息。返回JSON:
{{"should_say": true/false, "message": "...", "tone": "casual|warm|playful|gentle|curious",
  "claims": [{{"text":"消息中的事实","evidence_id":"对应ID"}}],
  "sensitivity": 0.0,
  "self_check": {{"sounds_like_task_manager":false,"contains_unsupported_fact":false,"creates_pressure":false,"too_private":false}},
  "reject_reason": null}}
如果不适合说话,设置 should_say=false 并填写 reject_reason。"""

        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": user_prompt}],
                      "max_tokens": 200, "temperature": 0.7},
                timeout=15,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content)
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "unknown")
                return {"error": f"429_rate_limited", "retry_after": retry_after}
            else:
                return {"error": f"HTTP_{resp.status_code}", "body": resp.text[:100]}
        except requests.Timeout:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)[:100]}

    return generate
