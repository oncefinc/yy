"""Temporal Cognition configuration."""
import os
from pathlib import Path

# ── Kill Switches ───────────────────────────────────
TEMPORAL_ENGINE_ENABLED = True
TEMPORAL_INGEST_ENABLED = True     # T3A: production message ingest bypass
TEMPORAL_WRITE_ENABLED = True
TEMPORAL_PROMPT_ENABLED = True     # T3B: fresh facts enter normal reply context
TEMPORAL_INITIATIVE_ENABLED = True  # Fresh state is available to Initiative
AMBIGUITY_EXTRACTOR_ENABLED = False # T2
LOCATION_ENABLED = False           # T5
LOCATION_HISTORY_ENABLED = False   # T5
SCHEDULE_PRIOR_ENABLED = False     # T6

# ── Paths ───────────────────────────────────────────
DATA_DIR = Path(os.environ.get(
    "COW_TEMPORAL_DATA_DIR",
    str(Path(__file__).resolve().parent / "data"),
))
DB_PATH = DATA_DIR / "world_state.db"

# ── Clock ───────────────────────────────────────────
TIMEZONE = "Asia/Shanghai"

# ── Lifecycle: fresh / stale / expired ───────────────
# fresh_until: usable as current fact
# expires_at:  record fully expired
# Between fresh_until and expires_at = stale (inquiry only)
FRESH_SECONDS: dict[str, int] = {
    "activity":   7200,    # 2h fresh after starting/ongoing
    "location":   300,     # 5min fresh (manual location ages fast)
    "work":       3600,    # 1h fresh
    "availability":3600,   # 1h
    "meal":       1800,    # 30min
    "default":    3600,
}
STALE_SECONDS: dict[str, int] = {
    "activity":   7200,    # +2h inquiry window
    "location":   1500,    # +25min inquiry (total 30min: 5 fresh + 25 stale)
    "work":       7200,    # +2h
    "availability":3600,   # +1h
    "meal":       600,     # +10min
    "default":    3600,
}
# Data retention (separate from state validity):
# Coordinates retained max 7 days per privacy policy
# Semantic labels retained indefinitely
DATA_RETENTION_SECONDS: dict[str, int] = {
    "location": 604800,  # 7 days for precise coordinates
    "default": 2592000,  # 30 days default
}

# ── Store ───────────────────────────────────────────
MAX_ACTIVE_ASSERTIONS = 200
CLEANUP_INTERVAL_MINUTES = 60

# ── Resolver ────────────────────────────────────────
EVIDENCE_PRIORITY_ORDER = [
    "explicit_user",
    "manual_location",
    "schedule_prior",
    "memory",
    "inference",
]
