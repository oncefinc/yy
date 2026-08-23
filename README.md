# 银月 / YY

银月是一个基于 CowAgent 的个人 AI 伙伴实验项目。仓库保存可复用的源代码和测试，重点包括：

- 微信文字与多模态消息通道
- V2 结构化记忆、bge-base 检索投影和 Scene Layer
- 主动意识引擎：非固定唤醒、念头生成、Gate、好奇心搜索与主动消息
- Temporal Cognition：短期活动、位置和时间状态
- Action Receipt / Reality Grounding：可验证行为与事实约束
- 视觉桥接、接口异常恢复和长会话可靠性修复

“主动意识引擎”是项目的产品名，不代表本项目声称机器拥有主观意识。其技术定位、理论来源、形式化决策模型、可证伪假设与评测计划，见
[THEORY.md](THEORY.md)。

## 仓库结构

```text
cowagent/                 CowAgent 框架及微信、模型、视觉和可靠性改造
extensions/cow/           银月扩展包
  memory_engine/          记忆 V2、检索、增量同步和场景层
  initiative_engine/      主动意识与好奇心引擎
  temporal_cognition/     短期时间/位置/活动状态
  self_awareness/         能力快照、Action Receipt 和事实门控
  tests/                  跨模块回归测试
```

部署时将 `extensions` 加入 `PYTHONPATH`，使 `import cow` 指向
`extensions/cow`。公开版本使用仓库相对路径，并允许通过 `.env.example` 中的环境变量
覆盖数据、配置和扩展目录。

## 快速开始

1. 安装 Python 3.11。
2. 安装 `cowagent/requirements.txt` 与
   `extensions/cow/memory_engine/requirements.txt` 中的依赖。
3. 将 `cowagent/config-template.json` 复制为 `cowagent/config.json`，只在本机填写密钥。
4. 下载 bge-base 模型到本机模型目录；模型权重不存放在本仓库。
5. 设置 `PYTHONPATH=<repo>/extensions;<repo>/cowagent`。
6. 主动消息默认关闭。仅在完成 Shadow 验证后设置 `INITIATIVE_RECEIVER_ID`，并显式配置
   `INITIATIVE_DELIVERY_ENABLED=true`。
7. 在 `cowagent/` 下运行 `python app.py`。

建议先使用 Shadow 模式和测试账号验证，再打开真实主动发送。

## 隐私说明

这个仓库不会包含真实 API 密钥、微信凭据、用户 ID、聊天记录、长期记忆、知识库、
LanceDB/SQLite 数据、Shadow 日志、行为回执、图片、浏览器配置或本地模型权重。
详见 [PRIVACY.md](PRIVACY.md)。

## 测试

```powershell
$env:PYTHONPATH="<repo>\extensions;<repo>\cowagent"
py -3.11 -m pytest extensions\cow -q
```

涉及真实本地数据库的生产边界测试在普通克隆中会自动跳过。只有在隔离审计环境中，
才应显式设置 `COW_TEST_PRODUCTION_INTEGRITY=1`；不要在共享开发机上误读或改写私人记忆库。

## 上游与许可证

本仓库包含基于 [zhayujie/CowAgent](https://github.com/zhayujie/CowAgent)
v2.1.4 修改的框架代码，并非 CowAgent 官方发行版。上游版权与 MIT 许可证均已保留；
详细归属见 [NOTICE.md](NOTICE.md)。本仓库代码按根目录 [LICENSE](LICENSE) 发布。
