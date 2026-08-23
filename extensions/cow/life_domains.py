"""Shared life-domain definitions for Memory Scene and Initiative retrieval.

This module intentionally contains data only.  It lives above both engines so
Memory does not depend on Initiative and Initiative does not own the taxonomy.
"""

LIFE_DOMAIN_CONFIG = {
    "work": {
        "query": "工作 项目 示例公司 最近忙 日报 同事",
        "keywords": ("工作", "项目", "示例公司", "日报", "同事", "上班", "下班", "面试", "离职", "公司", "职业", "岗位"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "fitness": {
        "query": "健身 训练 腰伤 恢复 身体感受",
        "keywords": ("健身", "训练", "练腿", "练背", "练胸", "腰伤", "骶尾骨", "深蹲", "减脂", "锻炼", "游泳", "晒伤"),
        "allowed_source_domains": ("personal", "general", "fitness"),
    },
    "family": {
        "query": "家人 家庭 家人 妈妈 姐姐",
        "keywords": ("家人", "家庭", "家人", "妈妈", "姐姐", "大姐", "亲属", "亲属"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "relationship": {
        "query": "朋友 相亲 关系 情绪 crush",
        "keywords": ("朋友", "相亲", "关系", "crush", "内耗", "喜欢的人", "小学同学"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "hardware": {
        "query": "电脑 显卡 RTX 4070 硬件 升级",
        "keywords": ("电脑", "显卡", "b580", "rx5700", "5700xt", "硬件", "升级", "台式机", "笔记本", "显存"),
        "allowed_source_domains": ("personal", "general", "hardware", "knowledge"),
    },
    "gaming": {
        "query": "示例游戏 游戏 最近玩 兴趣",
        "keywords": ("示例游戏", "游戏", "帧率", "玩游戏"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "food": {
        "query": "吃饭 做饭 菜 市场 饮食",
        "keywords": ("吃饭", "做饭", "饮食", "汉堡", "午饭", "示例食材", "家常菜", "外卖", "柴火鸡"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "travel": {
        "query": "旅游 出差 示例山区 旅行",
        "keywords": ("旅游", "出差", "示例山区", "旅行", "昆明", "大理"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "creation": {
        "query": "写作 视频 剪辑 创作 散文",
        "keywords": ("写作", "视频", "剪辑", "创作", "散文", "钓鱼随笔", "朋友圈文案"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
    "daily_life": {
        "query": "通勤 作息 日常生活 电瓶车",
        "keywords": ("通勤", "作息", "日常生活", "日常闲聊", "电瓶车", "到家", "睡觉", "午休", "下班回家"),
        "allowed_source_domains": ("personal", "general", "knowledge"),
    },
}

LIFE_DOMAIN_ORDER = tuple(LIFE_DOMAIN_CONFIG)
