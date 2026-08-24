"""Initiative Engine configuration — all tunable values, no hardcoded magic numbers."""

import os
from pathlib import Path

from cow.runtime_paths import INITIATIVE_DATA_DIR, MEMORY_DATA_DIR, env_path


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Kill Switch ─────────────────────────────────────
DELIVERY_ENABLED = _env_flag("INITIATIVE_DELIVERY_ENABLED", False)
ENGINE_ENABLED = True     # Master switch for initiative engine

# ── Active/Quiet Windows ────────────────────────────
ACTIVE_WINDOW_START = "08:00"
ACTIVE_WINDOW_END = "22:00"
QUIET_HOURS_START = "22:00"  # 22:00–08:00 = quiet
QUIET_HOURS_END = "08:00"
TIMEZONE = "Asia/Shanghai"

# Note: quiet hours only block initiative candidates.
# Normal chat replies are never restricted by quiet hours.

# ── Timing Gates ────────────────────────────────────
MIN_MINUTES_AFTER_USER_MESSAGE = 45
MIN_MINUTES_AFTER_ASSISTANT_MESSAGE = 45
MIN_HOURS_BETWEEN_PROACTIVE = 4
MAX_PROACTIVE_CANDIDATES_PER_DAY = 2
MAX_REVISITS_PER_MOTIVE = 2
REVISIT_MIN_DELAY_MINUTES = 60
REVISIT_MAX_DELAY_MINUTES = 180
REVISIT_DEFAULT_DELAY_MINUTES = 120
REVISIT_POOL_MAX_ITEMS = 20

# Response-aware cadence consumes only receipt outcome categories. It affects
# proactive outreach only; direct user messages are always answered normally.
PROACTIVE_RESPONSE_POLICY_ENABLED = True
PROACTIVE_AWAIT_REPLY_HOURS = 24
PROACTIVE_NO_RESPONSE_COOLDOWN_HOURS = 24
PROACTIVE_BUSY_COOLDOWN_HOURS = 12
PROACTIVE_MINIMAL_ACK_COOLDOWN_HOURS = 24
PROACTIVE_REPEATED_LOW_ENGAGEMENT_COOLDOWN_HOURS = 72
PROACTIVE_BOUNDARY_COOLDOWN_HOURS = 168
PROACTIVE_REDUCED_MODE_HOURS = 72
PROACTIVE_REDUCED_DAILY_LIMIT = 1

# ── M0.5 Initiative Diversity ──────────────────────
# Generic check-ins are a useful fallback, but should not win every wake.
GENERIC_CHECK_IN_COOLDOWN_HOURS = 72
LIFE_DOMAIN_COOLDOWN_HOURS = 48
LIFE_DOMAIN_QUERIES_PER_WAKE = 2
LIFE_DOMAIN_TOP_K = 5

# ── Bounded Curiosity Loop ─────────────────────────
# A wake may continue one recent conversation topic with a read-only web
# search.  Search and proactive-message budgets are independent and persisted
# in state.json so restarts cannot multiply calls.
CURIOSITY_SEARCH_ENABLED = True
CURIOSITY_MAX_SEARCHES_PER_DAY = 3
CURIOSITY_MIN_TOPIC_AGE_MINUTES = 120
CURIOSITY_TOPIC_MAX_AGE_DAYS = 7
CURIOSITY_TOPIC_COOLDOWN_HOURS = 168
CURIOSITY_SEARCH_RESULT_COUNT = 5
CURIOSITY_EPHEMERAL_TTL_MINUTES = 90

# C1 is observational. Pool entries do not authorize search or delivery.
CURIOSITY_POOL_SHADOW_ENABLED = True
CURIOSITY_POOL_MAX_ITEMS = 30
CURIOSITY_POOL_TTL_DAYS = 7

# ── Memory 2.1 / M2 Scene Shadow ───────────────────
SCENE_SHADOW_ENABLED = True
SCENE_STORE_PATH = str(env_path(
    "COW_SCENE_STORE_PATH", MEMORY_DATA_DIR / "scenes" / "scenes_v1.json"
))
SCENE_CANDIDATES_PER_WAKE = 2

# One shared taxonomy for M0.5 live retrieval and M1 offline grouping.
from cow.life_domains import LIFE_DOMAIN_CONFIG

# Quick-fix scope agreed for M0.5. Other shared domains remain available to M1.
INITIATIVE_LIFE_DOMAINS = (
    "fitness", "gaming", "hardware", "work", "food", "travel", "relationship",
)

# ── Confidence ──────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.70

# ── Motive Sources ──────────────────────────────────
ENABLE_FOLLOW_UP = True
ENABLE_CARE = True
ENABLE_PROSPECTIVE = True
ENABLE_RELATIONSHIP = True
ENABLE_SHARE_MOTIVE = False  # Disabled per M1 spec

# ── Wake Scheduling ─────────────────────────────────
WAKE_DAY_MINUTES = 60       # min minutes between wakes (daytime)
WAKE_DAY_MAX = 180           # max minutes between wakes (daytime)
WAKE_AFTER_CHAT_MIN = 60     # min after conversation ends
WAKE_AFTER_CHAT_MAX = 150    # max after conversation ends
WAKE_BUDGET_EXHAUSTED_HOUR = 8  # push to next day after this hour
WAKE_BUDGET_NEXT_DAY_MIN = 30  # minutes after quiet_hours_end
WAKE_BUDGET_NEXT_DAY_MAX = 150
WAKE_ERROR_RETRY_MIN = 120
WAKE_ERROR_RETRY_MAX = 240

# ── Scheduler ───────────────────────────────────────
SCHEDULER_POLL_SECONDS = 30

# ── Restart Continuity ───────────────────────────────
CATCHUP_WINDOW_MINUTES = 120       # Max gap to perform startup catch-up
MAX_ACTIVE_WAKE_GAP_HOURS = 4      # Max silence before forced recovery check
MAX_ACTIVE_WAKE_GAP = MAX_ACTIVE_WAKE_GAP_HOURS * 3600  # seconds (only 08-22)

# ── LLM Judge ───────────────────────────────────────
LLM_JUDGE_MODEL = "glm-4-flash"  # Use cheaper model for judge
LLM_JUDGE_TIMEOUT_SEC = 15
LLM_JUDGE_MAX_TOKENS = 200

# ── Shadow ──────────────────────────────────────────
SHADOW_DIR = str(env_path("INITIATIVE_SHADOW_DIR", INITIATIVE_DATA_DIR / "shadow"))
SHADOW_MAX_QUEUE = 50
