"""
记忆引擎全局配置
"""
from pathlib import Path

# ── 路径 ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DICT_PATH = BASE_DIR / "jieba_dict.txt"
PENDING_POOL_PATH = DATA_DIR / "pending_pool.json"

# LanceDB 数据目录
LANCE_DIR = DATA_DIR / "lance_db"

# ── 记忆权威与派生索引（Memory 2.1 / M0）──────────────
# V2 MemoryRecordV2 是唯一权威 L1 Atom Store；Base 是 V2 的可重建
# 检索投影（绑定 bge-base 模型）；V1（lance_db）是尚未退出的兼容读取路径。
MEMORY_AUTHORITY_STORE = "memory_v2"
MEMORY_AUTHORITY_TABLE = "memories_v2"
MEMORY_SEARCH_INDEX = "memory_base"
MEMORY_SEARCH_INDEX_TABLE = "memories_base"

V2_LANCE_DIR = DATA_DIR / "lance_db_v2"
BASE_LANCE_DIR = DATA_DIR / "lance_db_base"
BASE_MANIFEST_PATH = BASE_LANCE_DIR / "index_manifest.json"

# Base 派生索引使用的嵌入模型（区别于 V1/V2 迁移时的 bge-small）
BASE_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
BASE_EMBEDDING_DIM = 768

# ── 嵌入模型 ───────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512
EMBEDDING_MAX_LENGTH = 512  # token

# ── LanceDB 表名 ───────────────────────────────────
TABLE_MAIN = "memories"
TABLE_ARCHIVE = "memories_archive"  # 已清除的记忆归档

# ── 衰减参数 ───────────────────────────────────────
# 分类 → 默认半衰期（天），同时也决定了检索时的权重系数
CATEGORY_HALF_LIFE: dict[str, int] = {
    "identity":      60,   # 姓名、生日、住址
    "preference":    45,   # 饮食口味、使用习惯
    "relationship":  45,   # crush、家人、朋友
    "work":          30,   # 项目进度、工作决策
    "decision":      30,   # 技术选型、重要决定
    "plan":          20,   # 待办、计划
    "event":         20,   # 发生过的事
    "fact":          25,   # 一般事实
    "lesson":        14,   # 教训——错误记忆更慢衰减
    "feeling":       10,   # 情绪——时效性强
}

# 默认半衰期（未知分类）
DEFAULT_HALF_LIFE = 20

# 置信度在衰减公式中的权重：越高 → 衰减越慢
CONFIDENCE_DECAY_WEIGHT = 0.6

# 每次检索命中给 strength 的加成系数
RETRIEVAL_BONUS = 0.15

# 建议好的反馈加成范围 (reward_factor)
REWARD_FACTOR_RANGE = (0.5, 1.5)

# ── 生存阈值 ───────────────────────────────────────
# strength 低于此值 → 待清除
PRUNE_THRESHOLD = 0.05
# 超过此天数未命中 → 标记为沉睡
DORMANT_DAYS = 30
# 超过此天数未命中 → 可直接归档
ARCHIVE_DAYS = 90

# ── 检索参数 ───────────────────────────────────────
# 混合检索融合权重
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# 默认返回条数
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# RRF 平滑参数
RRF_K = 60

# ── 写入参数 ───────────────────────────────────────
# 防重复：语义相似度阈值
DEDUP_SIMILARITY_THRESHOLD = 0.80
# 规则合并：同 category + 同 tags + 相似度
DEDUP_RULE_SIMILARITY = 0.75
DEDUP_RULE_TAG_OVERLAP = 1  # 至少 N 个公共标签

# 每轮对话最多自动提取条数
AUTO_EXTRACT_MAX_PER_SESSION = 5
# 同一话题冷却期（秒）
TOPIC_COOLDOWN_SECONDS = 30

# ── 整理调度 ───────────────────────────────────────
# 日整理：衰减更新 + 去重
DAILY_CONSOLIDATION_HOUR = 3  # 凌晨 3 点
# 周整理：沉睡标记 + 摘要压缩
WEEKLY_CONSOLIDATION_DAY = 0  # 周一（0=周一, 6=周日）
WEEKLY_CONSOLIDATION_HOUR = 4
