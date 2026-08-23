# 主动意识引擎：理论基础、工程模型与验证计划

> 文档状态：工程理论说明（v0.1）  
> 适用范围：`extensions/cow/initiative_engine` 及其依赖的 Memory、Temporal Cognition、Reality Grounding 模块

## 1. 先说明它不是什么

“主动意识引擎”是项目内部沿用的产品名和拟人化称呼。技术上，它是一个具有长期记忆、短期情境、候选念头生成、打扰控制和可审计决策记录的**主动交互决策引擎**。

本项目目前不主张以下结论：

- 不主张系统拥有主观体验、自我意识或人类意义上的意识；
- 不主张已经实现某一种神经科学意识理论；
- 不主张它能够准确读懂用户的内心、情绪或实时处境；
- 不主张单用户长期实验能够证明对其他用户同样有效；
- 不把“回复了一条自然的消息”当成意识存在的证据。

仓库里的“念头”“动机”“关心”等词，是便于产品设计和代码沟通的功能术语，不是对系统内在体验的本体论声明。

## 2. 问题定义

普通定时推送解决的是“什么时候执行任务”，却没有解决以下问题：

1. 此刻是否值得打扰用户；
2. 系统为什么想说这句话；
3. 它掌握的是事实、历史、习惯，还是不确定的推测；
4. 没有合适内容时能否选择沉默；
5. 同一类关心是否会机械重复；
6. 主动行为发生后，能否留下可追溯、可复盘的决策记录。

因此，本项目研究的问题不是“如何随机一个发送时间”，而是：

> 在用户状态不可完全观测、长期记忆可能过期、主动联系存在打扰成本的条件下，系统如何有限度地产生候选念头，并在表达、询问和沉默之间作出可解释的选择？

## 3. 理论来源

### 3.1 Mixed-Initiative Interaction：人和系统都可以发起交互

Eric Horvitz 在 Mixed-Initiative User Interfaces 中提出，界面代理的关键问题包括：对用户目标的错误猜测、没有充分权衡自动行动的收益与成本、行动时机不佳，以及缺少让用户纠正系统的机制。

这与本项目的直接关系是：

- 系统可以主动，但主动权不是无限的；
- 不确定性必须进入决策，而不是被语言模型悄悄补全；
- 用户最近是否活跃、是否处于安静时段，会改变行动成本；
- 轻量询问通常比无依据断言更合适；
- 用户纠正必须覆盖旧推断。

参考：Eric Horvitz, [Principles of Mixed-Initiative User Interfaces](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/chi99horvitz.pdf), CHI 1999.

### 3.2 Attention-Sensitive Alerting：主动消息是一项注意力决策

主动消息不是免费的。一次不合时宜的联系会消耗注意力、破坏正在进行的活动，并降低用户对后续消息的信任。Attention-Sensitive Alerting 将通知决策描述为：在“延迟信息的成本”与“立即打断的成本”之间权衡，并承认用户活动与消息价值都存在不确定性。

这构成以下机制的理论依据：

- quiet hours 与最近活动抑制；
- 每日候选预算和全局冷却；
- 话题去重、领域冷却和通用问候冷却；
- 大部分 wake 允许以 `silent` 结束；
- Shadow 模式先评估候选，再开放真实发送。

参考：Eric Horvitz, Andy Jacobs, David Hovel, [Attention-Sensitive Alerting](https://www.microsoft.com/en-us/research/publication/attention-sensitive-alerting/), UAI 1999.

### 3.3 BDI-like Control Loop：信念、愿望与意图的工程分离

BDI（Belief–Desire–Intention）架构强调将智能体所相信的世界状态、可能追求的目标和已经承诺执行的意图分离。

银月不是一个完整的形式化 BDI 系统，但可以作如下有限映射：

| BDI 概念 | 银月中的近似实现 | 重要边界 |
|---|---|---|
| Belief | Memory Scene、Temporal State、近期对话、可验证外部结果 | 检索结果不自动等于事实，历史不等于当前 |
| Desire | `ThoughtSeed`、`MotiveCandidate` | 候选念头只是可能性，不代表系统一定会行动 |
| Intention | 通过 Gate、准备生成和校验表达的候选 | 仍可因校验失败、预算或发送开关而沉默 |
| Action | 发送消息或保持沉默 | 发送必须经过独立 delivery 边界 |

这个映射的价值不在于给代码贴上 BDI 标签，而在于避免“检索到一条记忆就直接说出去”：记忆、候选和行动必须是三个不同阶段。

参考：Anand S. Rao, Michael P. Georgeff, [BDI Agents: From Theory to Practice](https://aaai.org/papers/icmas95-042-bdi-agents-from-theory-to-practice/), ICMAS 1995.

### 3.4 记忆类型：工程组织借鉴，而非人脑复刻

Memory V2 区分 core、episodic、semantic 和 prospective，并在 Scene Layer 中把原子记忆整理为历史背景、稳定模式、带日期事件和开放事项。这里借鉴了认知心理学对情景记忆、语义记忆和前瞻记忆的区分，但目的只是改善存储、检索和时间边界。

项目不主张这些数据结构等价于人类记忆，也不以“结构像人脑”作为正确性的证明。

### 3.5 Epistemic Grounding：知道什么，以及凭什么知道

Reality Grounding、Temporal Cognition、证据 ID 和 Action Receipt 共同实现一项工程原则：

> 相关不等于真实，历史不等于现在，计划不等于完成，图片场景不等于用户当前位置，执行意图不等于执行成功。

当证据不足时，系统的合理动作是降低置信度、改用问句或沉默，而不是利用语言流畅性补全剧情。

## 4. 设计公理

以下公理是当前实现应持续遵守的设计约束，也可以被外部审计：

1. **沉默是一种有效动作**：没有合适候选不是错误状态。
2. **候选不等于意图，意图不等于发送**：各阶段必须可独立拒绝。
3. **当前状态默认未知**：只有具有时效性的明确证据才能成为当前事实。
4. **历史和习惯只能提供背景**：不能单独证明用户此刻正在做什么。
5. **证据强度决定表达方式**：证据越弱，越应询问而不是断言。
6. **用户注意力是稀缺资源**：系统应承担“不打扰”的证明责任。
7. **新证据覆盖旧推断**：纠正只影响相关事实，不擅自重写其他状态。
8. **日常状态短命，重要事件可长期保存**：短期世界状态与长期记忆分层。
9. **所有主动行为必须可追溯**：至少能回答候选来源、门控结果和发送结果。
10. **拟人化服务于交互，不用于伪装能力**：自然语言不能掩盖不确定性和能力边界。

## 5. 形式化工程模型

### 5.1 状态

在时间 `t`，系统使用的状态可抽象为：

```text
S_t = {
  M_t,   # 长期原子记忆与 Scene
  W_t,   # 短期时间、位置、活动等世界状态
  H_t,   # 近期对话、上次用户/助手消息时间
  R_t,   # 互动节奏、近期话题与领域冷却
  B_t,   # 当日预算、全局冷却和发送开关
  X_t    # 时间、星期、外部只读信号
}
```

这里的 `S_t` 是不完全、带来源且会过期的观测集合，不是真实世界的完整状态。

### 5.2 候选生成

念头生成器从状态中产生有限候选集：

```text
C_t = G(S_t)
```

候选可以来自普通社交存在、生活兴趣、记忆关联、情绪关怀、对话连续性、环境情境、任务跟进或有界好奇心。生成候选只表示“可能值得考虑”，不表示应该发送。

### 5.3 门控与选择

对候选 `c`，先执行硬约束：

```text
Eligible(c, S_t) =
  active_window
  AND within_daily_budget
  AND cooldown_satisfied
  AND not_recently_active
  AND evidence_requirement_satisfied
  AND sensitivity_policy_satisfied
  AND not_duplicate
```

通过硬约束的候选，再按相关性、置信度、新颖性和侵入性进行有限排序。其抽象效用可以写成：

```text
U(c | S_t) =
    w_r * relevance
  + w_n * novelty
  + w_c * continuity
  + w_v * care_value
  - w_i * interruption_cost
  - w_u * unsupported_inference_risk
  - w_d * repetition_cost
```

当前代码不是一个经过数据学习的统一效用模型；它使用确定性门控、局部评分和优先级来近似上述决策。这里的公式是公开可讨论的设计模型，不应误报为已经拟合完成的算法。

最终动作是：

```text
A_t ∈ { silent, ask, express }
```

当前工程实现把 `ask` 和 `express` 都表现为候选消息，但要求不确定状态使用询问式表达；若没有候选通过全部阶段，则选择 `silent`。

### 5.4 表达与发送

Gate 通过后，语言模型只负责将已经选定、已有证据边界的念头表达得自然。它不应重新决定事实，也不应创造新的发送理由。

```text
selected candidate
    → bounded LLM draft
    → deterministic validator
    → delivery kill switch
    → channel send or silent
```

这使“说什么”和“是否应该说”保持分离。

## 6. 理论到代码的映射

| 工程职责 | 主要模块 | 对应原则 |
|---|---|---|
| 非固定唤醒与重启连续性 | `initiative_engine/wakeup.py`、`runtime.py` | 系统获得考虑机会，不等于必须发送 |
| 上下文构建 | `initiative_engine/context_builder.py` | 形成带边界的不完全状态 `S_t` |
| 候选念头 | `initiative_engine/thoughts.py`、`motives.py` | Desire-like candidate generation |
| 硬门控 | `initiative_engine/gate.py` | 打扰成本、证据、预算、冷却与去重 |
| 表达生成 | `initiative_engine/llm_worker.py`、`llm_adapter.py` | 语言模型只在候选通过后调用 |
| 确定性校验 | `initiative_engine/validator.py` | 无依据事实、语气、敏感与重复约束 |
| 发送边界 | `initiative_engine/delivery.py` | 意图与实际行动分离、默认关闭 |
| 决策审计 | `initiative_engine/shadow.py` | Shadow、原因码、观察计数和实际结果 |
| 长期记忆与 Scene | `memory_engine/schemas.py`、`scenes.py` | 原子事实、来源、历史背景和开放事项分层 |
| 当前世界状态 | `temporal_cognition/` | 新鲜、陈旧、过期；用户陈述优先 |
| 行为真实性 | `self_awareness/` | “准备做”与“已经做”的证据分离 |

## 7. 可证伪假设

理论说明如果不能指导失败判定，就只是事后解释。当前项目提出以下可被数据否定的工程假设：

### H1：门控优于定时推送

与固定 cron 或 cron + jitter 相比，Gate 应降低不合时宜消息和重复消息的比例，同时保留可接受的主动互动率。

若打扰率没有下降，或所有候选都被压成沉默，则该假设失败。

### H2：证据门控降低错误状态断言

引入 Temporal/Reality Grounding 后，“把历史当现在”“把计划当完成”“把图片地点当用户当前位置”等错误率应下降。

若错误断言率无显著改善，则标签与门控没有真正影响生成模型。

### H3：Scene 提高连续性和话题多样性

与只检索孤立 Atom 相比，Scene 应增加跨轮次生活主题的连续性，并降低通用问候长期占据主动候选的比例。

若 Scene 只产生更长的摘要、没有改善候选多样性，或引入更多过期状态，则该假设失败。

### H4：冷却和预算减少机械感

通用问候冷却、领域轮换和话题指纹应降低短时间内的意图重复率。

若系统只是换一种措辞重复同一意图，说明当前去重粒度不足。

### H5：不确定时轻问优于直接猜测

在证据不足的当前状态问题上，询问式表达应比直接断言获得更少的用户纠正和负面反馈。

若问句导致明显的查岗感或打扰感增加，则需要重新校准询问频率，而不是假设问句天然更好。

## 8. 评测设计

### 8.1 对照基线

至少比较以下策略：

| 基线 | 说明 |
|---|---|
| B0 固定 cron | 固定时间、固定主题 |
| B1 cron + jitter | 只改变时间，不判断内容和打扰成本 |
| B2 LLM-only | 把上下文直接交给模型决定说不说 |
| B3 Engine without Scene | 候选生成 + Gate，但只使用原子记忆 |
| B4 Full Engine | Scene + Temporal + Thought + Gate + Validator |

### 8.2 指标

**事实与时间质量**

- Unsupported Assertion Rate：无当前证据却陈述为当前事实的比例；
- Temporal Confusion Rate：过去、现在、计划、完成状态混淆的比例；
- Provenance Coverage：有事实主张的候选中，具备可追溯证据 ID 的比例。

**主动交互质量**

- Interruption Complaint Rate：被用户明确认为打扰、查岗或不合时宜的比例；
- Intent Repetition Rate：滚动时间窗内重复意图的比例，而非只比较字符串；
- Topic Diversity：主动候选覆盖的生活领域和主题分布；
- Candidate Acceptance：用户认为“可以发”的 Shadow 候选比例；
- Negative Correction Rate：主动消息触发事实纠正的比例；
- Reply Rate：用户回复比例，只作为行为指标，不单独等同于满意度；
- Silence Appropriateness：人工复核中，应沉默场景被正确抑制的比例。

**系统质量**

- wake、检索、LLM 和总链路延迟；
- 每日 LLM 调用量与 token 成本；
- 重启后重复 wake、状态丢失和重复发送次数；
- API 异常是否影响正常聊天。

### 8.3 消融实验

分别移除以下组件，观察指标变化：

- 去掉 recent-user-activity gate；
- 去掉 evidence requirement；
- 去掉 generic check-in cooldown；
- 去掉 Scene，只保留 Atom；
- 去掉 Temporal State；
- 让 LLM 直接决定发送，跳过确定性 Gate；
- 关闭 validator。

消融实验用于判断“哪个模块真的产生了价值”，防止用系统总体表现替每个组件背书。

### 8.4 Shadow 与真实发送的证据边界

Shadow 可以回答：系统醒了几次、生成了什么、为什么拒绝、候选是否重复、是否有依据。它不能直接证明用户会喜欢一条从未收到的消息。

因此证据分为四级：

1. **Implemented**：代码存在且单元测试通过；
2. **Shadow-observed**：真实运行中产生过相应决策记录；
3. **User-observed**：真实发送后得到用户反馈；
4. **Generalized**：在多个用户和较长时间内重复成立。

当前项目的大部分结论位于第 1～3 级，不应表述为第 4 级。

## 9. 当前证据与限制

### 已有

- 确定性测试覆盖 quiet hours、预算、证据、去重、重启连续性和 delivery kill switch；
- Shadow 日志能够记录 wake、候选、拒绝原因、选中类型和实际发送结果；
- 真实单用户使用暴露并推动修复了时间混淆、错误状态推断、机械重复和上游 API 故障；
- Memory、Temporal、Initiative 与 Action Truth 已拆分为独立模块。

### 尚缺

- 尚无多用户实验；
- 尚无严格随机化 A/B 测试；
- 尚无统一公开的人工标注集和指标报告；
- “自然”“像朋友”等体验指标仍含较强主观性；
- 用户回复率会受到时间、话题和关系本身影响，不能简单归因于引擎；
- 当前门控权重与规则主要来自单用户迭代，不保证迁移到其他关系类型；
- 尚未证明当前架构优于所有更简单的替代方案。

## 10. 开发纪律

后续新增功能应至少回答以下问题：

1. 它解决了哪一个已观察问题？
2. 它对应哪个状态变量、候选来源或约束？
3. 是否存在更简单的基线？
4. 如何在 Shadow 中观察它？
5. 哪项指标应该改善？
6. 什么结果意味着它无效，应当删除？
7. 是否增加了无依据推断、打扰、隐私或成本风险？

不能回答这些问题的功能，不应仅因为“更像人”就进入生产路径。

## 11. 一句话总结

银月主动意识引擎的研究对象不是“机器是否产生意识”，而是一个更有限、也更可验证的问题：

> 一个拥有长期记忆但无法完全知道用户当前状态的 AI，能否在不滥用推测、不频繁打扰的前提下，偶尔自然地发起有连续性的交流，并且为每一次开口或沉默留下可审计的理由？

这份文档给出的是该问题当前的理论起点，而不是最终答案。
