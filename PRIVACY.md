# Privacy policy for this repository

## Never commit

- `config.json`、`.env` 或任何真实 API key/token/password
- 微信登录凭据、receiver ID、二维码和联系人标识
- `MEMORY.md`、`USER.md`、人物档案、聊天记录与对话摘要
- `knowledge/`、`memory/` 及从私人对话生成的技术/生活文档
- LanceDB、SQLite、JSONL journal、Shadow 日志、Action Receipt 和运行 state
- 用户图片、文件、浏览器 profile、模型权重及 embedding 索引

## Safe contribution workflow

1. 在独立发布副本中工作，不要直接 `git add` 生产目录。
2. 提交前运行路径、密钥、微信 ID、邮箱、手机号和身份证模式扫描。
3. 使用合成测试数据；不要把真实聊天片段复制进测试 fixture。
4. 发现密钥进入历史后，先撤销/轮换密钥，再清理 Git 历史。
5. 将仓库改为公开前重新执行一次全历史扫描。
6. 主动消息在公开版本中默认关闭；只有显式设置
   `INITIATIVE_DELIVERY_ENABLED=true` 才允许进入发送路径。

本仓库中的空数据目录会在运行时自动创建。私人数据应始终保留在 Git 管理之外。
