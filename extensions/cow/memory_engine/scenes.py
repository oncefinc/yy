"""Memory 2.1 / M1 deterministic Scene grouping (Shadow only).

Reads the authoritative V2 Atom store, assigns eligible personal memories to
one life domain, and writes a human-readable grouping report.  It never calls
an LLM, never writes V2/Base, and is not imported by production reply paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import lancedb

from cow.life_domains import LIFE_DOMAIN_CONFIG, LIFE_DOMAIN_ORDER
from .config import DATA_DIR, MEMORY_AUTHORITY_TABLE, V2_LANCE_DIR
from .index_manifest import stable_id_hash


SCENES_DIR = DATA_DIR / "scenes"
DEFAULT_GROUPS_PATH = SCENES_DIR / "scene_groups_v0.json"
DEFAULT_LABELS_PATH = SCENES_DIR / "source_domain_labels_v0.json"
DEFAULT_AUDIT_PATH = SCENES_DIR / "source_domain_audit_v0.json"
DEFAULT_SCENES_V1_PATH = SCENES_DIR / "scenes_v1.json"

GROUP_ATOM_LIMIT = 12
PREVIEW_CHARS = 40
UNASSIGNED_SAMPLES_PER_REASON = 5

SCENE_TITLES = {
    "work": "工作与职业",
    "fitness": "健身与身体状态",
    "family": "家人与家庭事件",
    "relationship": "关系与情感经历",
    "hardware": "电脑与硬件设备",
    "gaming": "游戏兴趣与体验",
    "food": "饮食偏好与做饭",
    "travel": "旅行与出差经历",
    "creation": "写作与创作项目",
    "daily_life": "通勤作息与日常生活",
}

ACTIVE_STATUSES = {"active", "open"}
INACTIVE_STATUSES = {"archived", "superseded", "cancelled", "expired", "closed"}

_TECH_PATH_PREFIXES = (
    "knowledge/tech/",
)
_TECH_EXACT_PATHS = {
    "agent.md", "rule.md", "knowledge/index.md", "knowledge/capabilities.md",
}
_PERSONAL_PATH_PREFIXES = (
    "memory/", "knowledge/life/", "knowledge/work/",
)
_PERSONAL_EXACT_PATHS = {"memory.md", "user.md", "knowledge/crush.md"}

_TECH_TERMS = (
    "api", "mcp", "llm", "github", "python", "docker", "lancedb",
    "embedding", "prompt", "token", "agent", "skill", "代码", "脚本",
    "接口", "模型", "部署", "架构", "数据库", "向量", "检索", "测试",
    "开发", "爬虫", "配置文件", "源码", "函数", "json", "yaml",
    "openclaw", "vpn", "梯子", "代理节点", "切节点", "ip地址",
)
_PERSONAL_MARKERS = (
    "用户", "用户", "我现在", "我今天", "我昨天", "我的", "目前在",
    "喜欢", "不喜欢", "住在", "到家", "下班", "上班", "同事", "家人",
    "朋友", "妈妈", "姐姐", "家人", "亲属", "相亲", "工作", "面试",
    "离职", "健身", "锻炼", "喜欢吃", "吃了", "吃过", "饮食", "旅游", "出差",
)
_HARDWARE_PERSONAL_TERMS = (
    "b580", "rx5700", "5700xt", "台式机", "笔记本", "显卡升级", "换上",
    "机箱", "购入", "买了", "当前使用",
)
_MIXED_TECH_PATHS = {
    "knowledge/tech/post-resignation-plan.md",
    "knowledge/tech/work-profile.md",
    "knowledge/tech/job-search.md",
    "knowledge/tech/pc-upgrade-plan.md",
}
_WORK_PERSONAL_TERMS = (
    "入职", "公司", "工作地点", "同事", "薪资", "考勤", "岗位", "职业",
    "面试", "离职", "上班", "下班", "出差", "领导", "offer",
)
_NON_USER_SUBJECT_TERMS = (
    "助手回答", "银月今晚", "银月也", "银月的官方形象", "助手未能",
    "记忆系统", "记忆关联教训", "归档：[", "*(可选)*", "时间状态感知模块",
)
_HARD_TECH_CONTENT_TERMS = ("openclaw", "梯子", "切节点", "模拟出差ip")
_SCENE_DIALOGUE_FRAGMENT_TERMS = (
    "用户:", "回复:", "助手解释", "助手分析", "助手识别",
    "帮用户", "技术讨论暂告一段落",
)
_NON_SCENE_META_TERMS = (
    "妈妈冒充用户跟claude谈判分手",
    "glm-4.6v-flash是免费模型",
    "日常图片/视频走免费版",
)
_AMBIGUOUS_MEDICAL_ADVICE_TERMS = (
    "消化功能极弱", "营养不良", "减缓恶化而非治疗",
)
_FAMILY_SUBJECT_TERMS = ("家人", "妈妈", "姐姐", "大姐", "亲属", "亲属")
_RELATIONSHIP_SUBJECT_TERMS = ("crush", "相亲", "前女友", "喜欢的女生")
_HARDWARE_SUBJECT_TERMS = ("b580", "rx 5700", "5700xt", "台式机", "笔记本")
_TRAVEL_SUBJECT_TERMS = ("出差", "旅游", "旅行", "示例山区", "大理", "海南", "兴隆湖")
_WORK_SUBJECT_TERMS = ("面试", "岗位", "简历", "薪资", "公司", "同事", "上班", "离职")
_FITNESS_SUBJECT_TERMS = ("健身", "锻炼", "训练", "腰伤", "腰痛", "骶尾骨")


@dataclass
class MemorySceneV1:
    scene_id: str
    receiver_id: str
    schema_version: int = 1
    title: str = ""
    life_domain: str = ""
    summary: str = ""
    summary_atom_ids: list[str] = field(default_factory=list)
    stable_patterns: list[dict] = field(default_factory=list)
    dated_events: list[dict] = field(default_factory=list)
    open_loops: list[dict] = field(default_factory=list)
    uncertainties: list[dict] = field(default_factory=list)
    conversation_hooks: list[dict] = field(default_factory=list)
    atom_ids: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    confidence: float = 0.5
    sensitivity: str = "normal"
    initiative_policy: str = "never"
    status: str = "active"
    temporal_start: Optional[str] = None
    temporal_end: Optional[str] = None
    last_event_at: Optional[str] = None
    last_selected_at: Optional[str] = None
    selected_count: int = 0
    cooldown_until: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    revision: int = 1

_CATEGORY_BOOSTS = {
    "work": "work",
    "relationship": "relationship",
}
_PATH_DOMAIN_HINTS = {
    "knowledge/work/": "work",
    "knowledge/crush.md": "relationship",
    "knowledge/life/fitness": "fitness",
}


def _norm(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm_path(value: object) -> str:
    return _norm(value).replace("\\", "/").casefold()


def _row_text(row: Mapping) -> str:
    return " ".join((
        _norm(row.get("content")),
        _norm(row.get("tags")),
        _norm(row.get("category")),
        _norm(row.get("source_file")),
    )).casefold()


def _contains(text: str, term: object) -> bool:
    return str(term).casefold() in text


def _domain_scores(row: Mapping) -> dict[str, int]:
    """Return deterministic domain evidence scores; no ML/LLM involved."""
    text = _row_text(row)
    scores = {
        domain: sum(1 for keyword in cfg["keywords"] if _contains(text, keyword))
        for domain, cfg in LIFE_DOMAIN_CONFIG.items()
    }
    category = _norm(row.get("category")).casefold()
    boosted = _CATEGORY_BOOSTS.get(category)
    if boosted:
        scores[boosted] += 3
    source_file = _norm_path(row.get("source_file"))
    for prefix, domain in _PATH_DOMAIN_HINTS.items():
        if source_file.startswith(prefix):
            scores[domain] += 4
    return scores


def infer_source_domain(row: Mapping) -> str:
    """Classify source eligibility as personal / technical / unknown.

    File paths are used for hard isolation.  Daily memory and MEMORY.md are
    mixed sources, so they additionally require content evidence.  This avoids
    treating every chat log as personal while keeping explicit life events.
    """
    source_file = _norm_path(row.get("source_file"))
    if source_file in _TECH_EXACT_PATHS:
        return "technical"
    # Dream summaries often mix assistant reflection and inferred facts.  M1
    # excludes them until a later evidence-aware stage can separate the two.
    if source_file.startswith("memory/dreams/"):
        return "technical"

    text = _row_text(row)
    scores = _domain_scores(row)
    life_score = max(scores.values(), default=0)
    tech_hits = sum(1 for term in _TECH_TERMS if _contains(text, term))
    personal_hits = sum(1 for term in _PERSONAL_MARKERS if _contains(text, term))
    hardware_personal = any(_contains(text, term) for term in _HARDWARE_PERSONAL_TERMS)
    work_personal = any(_contains(text, term) for term in _WORK_PERSONAL_TERMS)
    non_user_subject = any(_contains(text, term) for term in _NON_USER_SUBJECT_TERMS)
    hard_tech_content = any(_contains(text, term) for term in _HARD_TECH_CONTENT_TERMS)

    if hard_tech_content:
        return "technical"

    if source_file.startswith(_TECH_PATH_PREFIXES):
        if source_file not in _MIXED_TECH_PATHS:
            return "technical"
        # A few historical "tech" documents intentionally mix career/device
        # facts with research.  Admit only strongly grounded personal spans.
        if hardware_personal or (work_personal and max(scores.values(), default=0) >= 1):
            return "personal"
        return "technical"

    if source_file.startswith("knowledge/life/") or source_file == "knowledge/crush.md":
        return "personal"
    if source_file.startswith("knowledge/work/"):
        return "personal" if work_personal and not non_user_subject else "technical"
    if source_file == "user.md":
        return "personal" if life_score > 0 and not non_user_subject else "unknown"

    if source_file in _PERSONAL_EXACT_PATHS or source_file.startswith(_PERSONAL_PATH_PREFIXES):
        if non_user_subject:
            return "technical"
        if tech_hits >= 2 and not hardware_personal and not work_personal:
            return "technical"
        category = _norm(row.get("category")).casefold()
        explicit_personal_category = category in {"identity", "preference", "relationship", "work"}
        memory_summary_evidence = (
            source_file == "memory.md" and life_score >= 2 and tech_hits == 0
        )
        if life_score > 0 and (personal_hits > 0 or hardware_personal or work_personal
                               or explicit_personal_category or memory_summary_evidence):
            return "personal"
        return "unknown"

    if tech_hits >= 2:
        return "technical"
    if life_score > 0 and (personal_hits > 0 or hardware_personal):
        return "personal"
    return "unknown"


def classify_atom(row: Mapping) -> tuple[str, str]:
    """Return ``(life_domain, reason)``; exactly one domain at most."""
    status = _norm(row.get("status") or "active").casefold()
    if status in INACTIVE_STATUSES or status not in ACTIVE_STATUSES:
        return "", "inactive_status"
    if bool(row.get("dormant", False)):
        return "", "dormant"
    if not _norm(row.get("id")) or not _norm(row.get("content")):
        return "", "missing_content_or_id"

    content = _norm(row.get("content")).casefold()
    if any(term in content for term in _SCENE_DIALOGUE_FRAGMENT_TERMS):
        return "", "dialogue_fragment"
    if any(term in content for term in _NON_SCENE_META_TERMS):
        return "", "meta_discussion"
    # Advice fragments whose subject is no longer present must not be turned
    # into the user's own food/health Scene.  Keep them at L1 until provenance
    # can identify the subject explicitly.
    if any(term in content for term in _AMBIGUOUS_MEDICAL_ADVICE_TERMS):
        return "", "ambiguous_subject"

    source_domain = infer_source_domain(row)
    if source_domain == "technical":
        return "", "technical"
    if source_domain != "personal":
        return "", "no_personal_evidence"

    try:
        confidence = float(row.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.35:
        return "", "low_confidence"

    source_file = _norm_path(row.get("source_file"))
    # Explicit subject ownership beats incidental verbs such as "上班" or
    # food words.  This prevents "大姐在成都上班" from becoming the user's
    # work state and crush preferences from becoming the user's preferences.
    if source_file == "knowledge/crush.md" or any(
            term in content for term in _RELATIONSHIP_SUBJECT_TERMS):
        return "relationship", "assigned"
    if "女方" in content and any(place in content for place in ("昆明", "成都", "示例城市")):
        return "relationship", "assigned"
    if any(term in content for term in _FAMILY_SUBJECT_TERMS):
        return "family", "assigned"
    if any(term in content for term in _WORK_SUBJECT_TERMS):
        return "work", "assigned"
    if any(term in content for term in _HARDWARE_SUBJECT_TERMS):
        return "hardware", "assigned"
    if any(term in content for term in _FITNESS_SUBJECT_TERMS):
        return "fitness", "assigned"
    if any(term in content for term in _TRAVEL_SUBJECT_TERMS):
        return "travel", "assigned"

    scores = _domain_scores(row)
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return "", "no_stable_domain"
    # LIFE_DOMAIN_ORDER is the documented stable tie-breaker.
    domain = next(d for d in LIFE_DOMAIN_ORDER if scores[d] == best_score)
    return domain, "assigned"


def _rank_key(row: Mapping) -> tuple:
    def number(name: str) -> float:
        try:
            return float(row.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return (-number("importance"), -number("confidence"), _norm(row.get("id")))


def _preview(row: Mapping) -> str:
    return _norm(row.get("content"))[:PREVIEW_CHARS]


def _group_id(receiver_id: str, life_domain: str) -> str:
    material = f"{receiver_id}|{life_domain}".encode("utf-8")
    return "scene_group:" + hashlib.sha256(material).hexdigest()[:16]


def _atom_view(row: Mapping) -> dict:
    return {
        "atom_id": _norm(row.get("id")),
        "preview": _preview(row),
        "importance": float(row.get("importance", 0.0) or 0.0),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "category": _norm(row.get("category")),
        "source_file": _norm(row.get("source_file")),
    }


def build_scene_groups(rows: Iterable[Mapping], receiver_id: str) -> dict:
    """Build stable, deterministic Scene groups from already-read V2 rows."""
    all_rows = [dict(row) for row in rows]
    source_ids = {_norm(row.get("id")) for row in all_rows if _norm(row.get("id"))}
    receiver_rows = [
        row for row in all_rows
        if not _norm(row.get("receiver_id")) or _norm(row.get("receiver_id")) == receiver_id
    ]

    candidates: dict[str, list[dict]] = defaultdict(list)
    unassigned: dict[str, list[dict]] = defaultdict(list)
    for row in receiver_rows:
        domain, reason = classify_atom(row)
        if domain:
            candidates[domain].append(row)
        else:
            unassigned[reason].append(row)

    groups: list[dict] = []
    assigned_ids: list[str] = []
    overflow_count = 0
    overflowed_groups = 0
    for domain in LIFE_DOMAIN_ORDER:
        ordered = sorted(candidates.get(domain, []), key=_rank_key)
        if not ordered:
            continue
        selected = ordered[:GROUP_ATOM_LIMIT]
        overflow = ordered[GROUP_ATOM_LIMIT:]
        if overflow:
            overflowed_groups += 1
            overflow_count += len(overflow)
            unassigned["group_overflow"].extend(overflow)
        atom_views = [_atom_view(row) for row in selected]
        assigned_ids.extend(atom["atom_id"] for atom in atom_views)
        groups.append({
            "group_id": _group_id(receiver_id, domain),
            "life_domain": domain,
            "atoms": atom_views,
            "source_files": sorted({_norm(row.get("source_file")) for row in selected if _norm(row.get("source_file"))}),
            "excluded_atom_count": len(overflow),
            "single_fact_exception": len(selected) == 1 and float(selected[0].get("importance", 0.0) or 0.0) >= 0.8,
        })

    duplicate_count = len(assigned_ids) - len(set(assigned_ids))
    missing_ids = sorted(set(assigned_ids) - source_ids)
    selected_by_id = {str(row.get("id")): row for row in receiver_rows}
    missing_provenance = [
        atom_id for atom_id in assigned_ids
        if atom_id in selected_by_id and not (
            _norm(selected_by_id[atom_id].get("source_file"))
            or _norm(selected_by_id[atom_id].get("source_id"))
            or _norm(selected_by_id[atom_id].get("evidence_ids"))
        )
    ]
    technical_contamination = sum(
        1 for atom_id in assigned_ids
        if infer_source_domain(selected_by_id[atom_id]) == "technical"
    )

    reason_counts = {reason: len(items) for reason, items in sorted(unassigned.items())}
    reason_samples = []
    for reason in sorted(unassigned):
        for row in sorted(unassigned[reason], key=lambda item: _norm(item.get("id")))[:UNASSIGNED_SAMPLES_PER_REASON]:
            reason_samples.append({
                "atom_id": _norm(row.get("id")),
                "preview": _preview(row),
                "reason": reason,
            })

    report = {
        "group_count": len(groups),
        "assigned_atom_count": len(assigned_ids),
        "unassigned_atom_count": sum(reason_counts.values()),
        "domain_distribution": {g["life_domain"]: len(g["atoms"]) for g in groups},
        "technical_contamination_count": technical_contamination,
        "missing_evidence_count": len(missing_ids) + len(missing_provenance),
        "duplicate_atom_assignment_count": duplicate_count,
        "group_overflow_count": overflow_count,
        "overflowed_group_count": overflowed_groups,
        "source_id_set_hash": stable_id_hash(source_ids),
    }
    payload = {
        "schema_version": 0,
        "source_store": "memory_v2",
        "source_table": MEMORY_AUTHORITY_TABLE,
        "receiver_id_hash": hashlib.sha256(receiver_id.encode("utf-8")).hexdigest()[:16],
        "source_record_count": len(source_ids),
        "source_id_set_hash": report["source_id_set_hash"],
        "groups": groups,
        "unassigned_summary": {
            "total": report["unassigned_atom_count"],
            "by_reason": reason_counts,
            "samples": reason_samples,
        },
        "report": report,
    }
    payload["quality_gate"] = validate_scene_groups(payload, source_ids)
    return payload


def validate_scene_groups(payload: Mapping,
                          source_ids: Optional[set[str]] = None) -> dict:
    """Deterministically evaluate the M1→M1B entry gate."""
    issues: list[str] = []
    groups = list(payload.get("groups", []))
    if not 7 <= len(groups) <= 15:
        issues.append(f"group_count_out_of_range:{len(groups)}")

    seen: set[str] = set()
    for group in groups:
        atoms = list(group.get("atoms", []))
        domain = str(group.get("life_domain", ""))
        if len(atoms) > GROUP_ATOM_LIMIT:
            issues.append(f"group_over_limit:{domain}:{len(atoms)}")
        if len(atoms) < 2 and not group.get("single_fact_exception", False):
            issues.append(f"group_too_small:{domain}:{len(atoms)}")
        for atom in atoms:
            atom_id = str(atom.get("atom_id", ""))
            if not atom_id:
                issues.append(f"empty_atom_id:{domain}")
            elif atom_id in seen:
                issues.append(f"duplicate_atom_id:{atom_id}")
            seen.add(atom_id)
            if source_ids is not None and atom_id not in source_ids:
                issues.append(f"missing_atom_id:{atom_id}")
            if len(str(atom.get("preview", ""))) > PREVIEW_CHARS:
                issues.append(f"preview_too_long:{atom_id}")

    serialized = json.dumps(payload.get("groups", []), ensure_ascii=False)
    for forbidden in ('"summary"', '"stable_patterns"', '"current_state"'):
        if forbidden in serialized:
            issues.append(f"expression_field_present:{forbidden}")
    report = payload.get("report", {})
    if report.get("technical_contamination_count", 0):
        issues.append("technical_contamination")
    if report.get("missing_evidence_count", 0):
        issues.append("missing_evidence")
    if report.get("duplicate_atom_assignment_count", 0):
        issues.append("duplicate_assignment")
    return {"passed": not issues, "issues": issues}


def build_source_domain_audit(rows: Iterable[Mapping], labels: Mapping[str, str]) -> dict:
    """Compare deterministic source filtering with 100 hand-reviewed labels."""
    by_id = {_norm(row.get("id")): dict(row) for row in rows}
    samples = []
    confusion = Counter()
    missing = []
    for atom_id, manual in labels.items():
        row = by_id.get(atom_id)
        if row is None:
            missing.append(atom_id)
            continue
        inferred = infer_source_domain(row)
        predicted = "personal" if inferred == "personal" else "technical"
        confusion[f"{manual}_as_{predicted}"] += 1
        samples.append({
            "atom_id": atom_id,
            "source_file": _norm(row.get("source_file")),
            "preview": _preview(row),
            "manual_label": manual,
            "predicted_label": predicted,
            "match": manual == predicted,
        })
    correct = sum(1 for sample in samples if sample["match"])
    return {
        "schema_version": 0,
        "method": "100 deterministic hash-stratified atoms, manually reviewed as personal/technical",
        "sample_size": len(samples),
        "missing_labelled_atoms": missing,
        "accuracy": round(correct / len(samples), 4) if samples else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "false_personal_count": confusion.get("technical_as_personal", 0),
        "false_technical_count": confusion.get("personal_as_technical", 0),
        "samples": samples,
    }


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_v2_rows(source_dir: Path) -> list[dict]:
    table = lancedb.connect(str(source_dir)).open_table(MEMORY_AUTHORITY_TABLE)
    return table.search().limit(100000).to_list()


def _read_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text("utf-8"))
    labels = {}
    for item in payload.get("labels", []):
        label = item.get("manual_label")
        if label not in {"personal", "technical"}:
            raise ValueError(f"invalid manual_label for {item.get('atom_id')}: {label}")
        labels[str(item["atom_id"])] = label
    return labels


def rebuild_scene_groups(receiver_id: str,
                         source_dir: Optional[Path] = None,
                         output_path: Optional[Path] = None,
                         labels_path: Optional[Path] = None,
                         audit_output_path: Optional[Path] = None) -> dict:
    """Read V2 once, write deterministic Shadow outputs, and return report."""
    if not receiver_id:
        raise ValueError("receiver_id is required")
    source = Path(source_dir) if source_dir else V2_LANCE_DIR
    output = Path(output_path) if output_path else DEFAULT_GROUPS_PATH
    labels_file = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
    audit_output = Path(audit_output_path) if audit_output_path else DEFAULT_AUDIT_PATH

    rows = _read_v2_rows(source)
    groups = build_scene_groups(rows, receiver_id)
    if labels_file.exists():
        audit = build_source_domain_audit(rows, _read_labels(labels_file))
        _atomic_json_write(audit_output, audit)
        groups["source_domain_audit"] = {
            "path": str(audit_output),
            "sample_size": audit["sample_size"],
            "accuracy": audit["accuracy"],
            "false_personal_count": audit["false_personal_count"],
            "false_technical_count": audit["false_technical_count"],
        }
    _atomic_json_write(output, groups)
    return groups["report"]


# ── M1B: LLM expression over already-approved deterministic groups ──────────

_RELATIVE_TIME_TERMS = (
    "今天", "昨天", "昨晚", "刚刚", "刚才", "最近", "今晚", "今早",
    "今夜", "明天", "明早", "本周", "这周", "近期",
)
_CURRENT_STATE_PATTERNS = (
    re.compile(r"(?:当前|目前|此刻)"),
    re.compile(r"(?:当前|现在|此刻|今天)(?:正在|还在|位于|身处|住在|生活在|工作在|使用着)"),
    re.compile(r"(?:正在|还在)(?:健身|锻炼|上班|加班|出差|旅行|路上|公司|家里|健身房)"),
    re.compile(r"(?:用户|用户)(?:目前)?生活在"),
    re.compile(r"正在(?:恢复|使用|进行|居住|工作|旅行|出差)"),
)
_UNSUPPORTED_ABSTRACTION_PATTERNS = (
    re.compile(r"(?:用户|用户)(?:很|比较|非常)?擅长"),
    re.compile(r"(?:长期偏好或稳定模式|可自然联想的话题方向|历史事件)$"),
    re.compile(r"更换了?[^，。]{0,8}(?:接口|电源接口)"),
    re.compile(r"周日通常无安排"),
)
_SCENE_LIST_LIMITS = {
    "stable_patterns": 5,
    "dated_events": 8,
    "open_loops": 5,
    "uncertainties": 5,
    "conversation_hooks": 5,
}


def _scene_id(receiver_id: str, life_domain: str) -> str:
    title = SCENE_TITLES[life_domain]
    normalized = _norm(title).lower()
    raw = f"{receiver_id}|{normalized}|{life_domain}"
    return "scene:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _contains_current_state_claim(text: str) -> bool:
    compact = _norm(text)
    return any(pattern.search(compact) for pattern in _CURRENT_STATE_PATTERNS)


def _contains_relative_time(text: str) -> bool:
    compact = _norm(text)
    return any(term in compact for term in _RELATIVE_TIME_TERMS)


def _contains_unsupported_abstraction(text: str) -> bool:
    compact = _norm(text)
    return any(pattern.search(compact) for pattern in _UNSUPPORTED_ABSTRACTION_PATTERNS)


def _neutralize_summary_temporal_frame(summary: str,
                                       normalizations: list[str]) -> str:
    replacements = (
        ("当前", "相关记录中"), ("目前", "相关记录中"), ("此刻", "记录时"),
        ("近期", "相关记录中"), ("最近", "相关记录中"),
        ("今天", "记录当日"), ("今晚", "记录当晚"), ("今早", "记录当日早晨"),
        ("今夜", "记录当晚"), ("昨天", "记录前一日"), ("昨晚", "记录前一晚"),
        ("明天", "记录次日"), ("明早", "记录次日早晨"),
        ("刚刚", "记录不久前"), ("刚才", "记录不久前"),
        ("本周", "记录所在周"), ("这周", "记录所在周"),
        ("用户在玩", "相关记录显示用户曾玩"),
        ("仍有效", "记录时未过期"),
        ("暂未购买", "相关记录未显示已购买"),
        ("同时拥有GitHub项目", "相关记录包含GitHub项目"),
    )
    changed = []
    for old, new in replacements:
        if old in summary:
            summary = summary.replace(old, new)
            changed.append(old)
    if changed:
        normalizations.append("summary:temporal_frame_neutralized:" + ",".join(changed))
    sunday_pattern = r"周日通常无(?:具体)?安排"
    if re.search(sunday_pattern, summary):
        summary = re.sub(sunday_pattern, "记录中的那个周日无具体安排", summary)
        normalizations.append("summary:single_sunday_not_generalized")
    return summary


def _drop_sentences_with_unsupported_numbers(summary: str,
                                             rows: Iterable[Mapping],
                                             normalizations: list[str]) -> str:
    evidence = " ".join(
        " ".join((
            _norm(row.get("content")), _norm(row.get("source_excerpt")),
            _norm(row.get("source_file")), _norm(row.get("valid_from")),
            _norm(row.get("valid_until")),
        ))
        for row in rows
    )
    sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", summary)
    kept = []
    for index, sentence in enumerate(sentences):
        numbers = re.findall(r"\d+(?:\.\d+)?", sentence)
        unsupported = [number for number in numbers if number not in evidence]
        if unsupported:
            normalizations.append(
                f"summary:sentence_{index}_dropped_unsupported_numbers:{','.join(unsupported)}")
            continue
        kept.append(sentence)
    result = "".join(kept).strip()
    if not result:
        raise ValueError("summary has no evidence-supported sentences")
    if result.endswith(("；", ";", "，", ",")):
        result = result[:-1] + "。"
        normalizations.append("summary:trailing_punctuation_repaired")
    return result


def _json_from_llm(value: object) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("scene expression must be a JSON object")
    return parsed


def _validate_evidence_ids(value: object, allowed_ids: set[str], field_name: str,
                           normalizations: Optional[list[str]] = None) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name}.atom_ids must be a non-empty list")
    original_ids = [str(item) for item in value]
    ids = list(dict.fromkeys(original_ids))
    if len(ids) != len(original_ids) and normalizations is not None:
        normalizations.append(f"{field_name}:deduplicated_atom_ids")
    unknown = sorted(set(ids) - allowed_ids)
    if unknown:
        raise ValueError(f"{field_name}.atom_ids contains unknown IDs: {unknown}")
    return ids


def _validate_text_items(raw: object, field_name: str,
                         allowed_ids: set[str], normalizations: list[str]) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list")
    if len(raw) > _SCENE_LIST_LIMITS[field_name]:
        normalizations.append(f"{field_name}:truncated_to_{_SCENE_LIST_LIMITS[field_name]}")
        raw = raw[:_SCENE_LIST_LIMITS[field_name]]
    result = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be an object")
        text = _norm(item.get("text"))
        if not text:
            raise ValueError(f"{field_name}[{index}].text is empty")
        if _contains_current_state_claim(text):
            normalizations.append(f"{field_name}[{index}]:dropped_current_state_claim")
            continue
        if _contains_relative_time(text):
            normalizations.append(f"{field_name}[{index}]:dropped_relative_time")
            continue
        if _contains_unsupported_abstraction(text):
            normalizations.append(f"{field_name}[{index}]:dropped_placeholder_or_trait")
            continue
        if not isinstance(item.get("atom_ids"), list) or not item.get("atom_ids"):
            normalizations.append(f"{field_name}[{index}]:dropped_without_evidence")
            continue
        ids = _validate_evidence_ids(
            item.get("atom_ids"), allowed_ids, f"{field_name}[{index}]", normalizations)
        result.append({"text": text, "atom_ids": ids})
    return result


def _atom_supports_exact_date(row: Mapping, date: str) -> bool:
    year, month, day = date.split("-")
    variants = {
        date,
        f"{year}年{int(month)}月{int(day)}日",
        f"{year}/{int(month)}/{int(day)}",
    }
    evidence = " ".join((
        _norm(row.get("content")), _norm(row.get("source_excerpt")),
        _norm(row.get("valid_from")), _norm(row.get("valid_until")),
        _norm(row.get("source_file")),
    ))
    return any(variant in evidence for variant in variants)


def _normalize_summary_dates(summary: str, summary_ids: list[str],
                             atoms_by_id: Mapping[str, Mapping],
                             normalizations: list[str]) -> str:
    rows = [atoms_by_id[atom_id] for atom_id in summary_ids]

    def replace_chinese(match: re.Match) -> str:
        year, month, day = match.groups()
        iso = f"{year}-{int(month):02d}-{int(day):02d}"
        if any(_atom_supports_exact_date(row, iso) for row in rows):
            return match.group(0)
        short = f"{int(month)}月{int(day)}日"
        if any(short in _norm(row.get("content")) or short in _norm(row.get("source_excerpt"))
               for row in rows):
            normalizations.append(f"summary:removed_unsupported_year:{year}")
            return short
        normalizations.append(f"summary:removed_unsupported_date:{iso}")
        return "日期未明确"

    def replace_iso(match: re.Match) -> str:
        date = match.group(0)
        if any(_atom_supports_exact_date(row, date) for row in rows):
            return date
        normalizations.append(f"summary:removed_unsupported_date:{date}")
        return "日期未明确"

    summary = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日", replace_chinese, summary)
    return re.sub(r"\d{4}-\d{2}-\d{2}", replace_iso, summary)


def _atoms_support_historical_event(rows: Iterable[Mapping]) -> bool:
    future_markers = ("预计", "计划", "准备", "明天", "待办")
    for row in rows:
        kind = _norm(row.get("memory_kind")).casefold()
        category = _norm(row.get("category")).casefold()
        text = _norm(row.get("content"))
        if kind == "prospective" or category in {"plan", "task"}:
            continue
        if any(marker in text for marker in future_markers) and not any(
                marker in text for marker in ("已完成", "已经", "完成了", "已于")):
            continue
        return True
    return False


def _atoms_support_open_loop(rows: Iterable[Mapping]) -> bool:
    markers = ("待确认", "尚未", "未完成", "还没", "计划", "准备", "考虑", "待办")
    for row in rows:
        status = _norm(row.get("status")).casefold()
        kind = _norm(row.get("memory_kind")).casefold()
        category = _norm(row.get("category")).casefold()
        text = _norm(row.get("content"))
        if status == "open" or kind == "prospective" or category in {"plan", "task"}:
            if any(marker in text for marker in markers):
                return True
    return False


def validate_scene_expression(raw: object, allowed_atom_ids: Iterable[str],
                              atoms_by_id: Optional[Mapping[str, Mapping]] = None) -> dict:
    """Validate one model response without trusting any model-controlled IDs.

    The validator proves structural traceability and temporal boundaries.  It
    intentionally does not claim semantic entailment; that remains visible for
    human Shadow review through the cited Atom IDs.
    """
    allowed = {str(item) for item in allowed_atom_ids}
    if not allowed:
        raise ValueError("allowed_atom_ids is empty")
    data = _json_from_llm(raw)
    normalizations: list[str] = []

    summary = _norm(data.get("summary"))
    if not summary:
        raise ValueError("summary is empty")
    summary = _neutralize_summary_temporal_frame(summary, normalizations)
    if len(summary) > 300:
        cut = max(summary.rfind(mark, 0, 300) for mark in ("。", "！", "？", ";", "；"))
        summary = summary[:cut + 1] if cut >= 80 else summary[:300]
        normalizations.append("summary:truncated_to_300")
    if _contains_current_state_claim(summary):
        raise ValueError("summary asserts current state")
    if _contains_relative_time(summary):
        raise ValueError("summary uses relative time")
    if _contains_unsupported_abstraction(summary):
        raise ValueError("summary contains unsupported trait or placeholder")
    summary_ids = _validate_evidence_ids(
        data.get("summary_atom_ids"), allowed, "summary", normalizations)
    if atoms_by_id is not None:
        summary = _normalize_summary_dates(
            summary, summary_ids, atoms_by_id, normalizations)
        summary = _drop_sentences_with_unsupported_numbers(
            summary, (atoms_by_id[atom_id] for atom_id in summary_ids), normalizations)

    stable_patterns = _validate_text_items(
        data.get("stable_patterns", []), "stable_patterns", allowed, normalizations)
    uncertainties = _validate_text_items(
        data.get("uncertainties", []), "uncertainties", allowed, normalizations)
    conversation_hooks = _validate_text_items(
        data.get("conversation_hooks", []), "conversation_hooks", allowed, normalizations)

    dated_events_raw = data.get("dated_events", [])
    if not isinstance(dated_events_raw, list):
        raise ValueError("dated_events must be a list")
    if len(dated_events_raw) > 8:
        normalizations.append("dated_events:truncated_to_8")
        dated_events_raw = dated_events_raw[:8]
    dated_events = []
    for index, item in enumerate(dated_events_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"dated_events[{index}] must be an object")
        date = _norm(item.get("date"))
        text = _norm(item.get("summary"))
        if date != "unknown" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError(f"dated_events[{index}] has invalid date")
        if any(term in date or term in text for term in _RELATIVE_TIME_TERMS):
            normalizations.append(f"dated_events[{index}]:dropped_relative_time")
            continue
        if not text or _contains_current_state_claim(text):
            normalizations.append(f"dated_events[{index}]:dropped_current_state_claim")
            continue
        if not isinstance(item.get("atom_ids"), list) or not item.get("atom_ids"):
            normalizations.append(f"dated_events[{index}]:dropped_without_evidence")
            continue
        ids = _validate_evidence_ids(
            item.get("atom_ids"), allowed, f"dated_events[{index}]", normalizations)
        if atoms_by_id is not None and not _atoms_support_historical_event(
                atoms_by_id[atom_id] for atom_id in ids):
            normalizations.append(f"dated_events[{index}]:dropped_prospective_event")
            continue
        if date != "unknown" and atoms_by_id is not None:
            if not any(_atom_supports_exact_date(atoms_by_id[atom_id], date) for atom_id in ids):
                normalizations.append(f"dated_events[{index}]:date_downgraded_to_unknown")
                date = "unknown"
        dated_events.append({
            "date": date,
            "summary": text,
            "atom_ids": ids,
            "status": "historical",
        })

    open_loops_raw = data.get("open_loops", [])
    if not isinstance(open_loops_raw, list):
        raise ValueError("open_loops must be a list")
    if len(open_loops_raw) > 5:
        normalizations.append("open_loops:truncated_to_5")
        open_loops_raw = open_loops_raw[:5]
    open_loops = []
    for index, item in enumerate(open_loops_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"open_loops[{index}] must be an object")
        text = _norm(item.get("summary"))
        if not text or _contains_current_state_claim(text) or _contains_relative_time(text):
            normalizations.append(f"open_loops[{index}]:dropped_temporal_claim")
            continue
        status = _norm(item.get("status")) or "uncertain"
        if status not in {"open", "uncertain"}:
            raise ValueError(f"open_loops[{index}] has invalid status")
        if not isinstance(item.get("atom_ids"), list) or not item.get("atom_ids"):
            normalizations.append(f"open_loops[{index}]:dropped_without_evidence")
            continue
        ids = _validate_evidence_ids(
            item.get("atom_ids"), allowed, f"open_loops[{index}]", normalizations)
        if atoms_by_id is not None and not _atoms_support_open_loop(
                atoms_by_id[atom_id] for atom_id in ids):
            normalizations.append(f"open_loops[{index}]:dropped_without_open_evidence")
            continue
        open_loops.append({
            "summary": text,
            "status": status,
            "atom_ids": ids,
        })

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    return {
        "summary": summary,
        "summary_atom_ids": summary_ids,
        "stable_patterns": stable_patterns,
        "dated_events": dated_events,
        "open_loops": open_loops,
        "uncertainties": uncertainties,
        "conversation_hooks": conversation_hooks,
        "confidence": confidence,
        "_normalizations": normalizations,
    }


def build_scene_prompt(group: Mapping, atoms_by_id: Mapping[str, Mapping]) -> str:
    """Build the complete M1B prompt from one approved group only."""
    domain = str(group.get("life_domain", ""))
    if domain not in SCENE_TITLES:
        raise ValueError(f"unsupported life_domain: {domain}")
    refs = list(group.get("atoms", []))
    if not 1 <= len(refs) <= GROUP_ATOM_LIMIT:
        raise ValueError("group atom count must be between 1 and 12")

    input_atoms = []
    for ref in refs:
        atom_id = str(ref.get("atom_id", ""))
        row = atoms_by_id.get(atom_id)
        if row is None:
            raise ValueError(f"group references missing V2 atom: {atom_id}")
        input_atoms.append({
            "atom_id": atom_id,
            "content": _norm(row.get("content")),
            "category": _norm(row.get("category")),
            "memory_kind": _norm(row.get("memory_kind")),
            "source_file": _norm(row.get("source_file")),
            "valid_from": _norm(row.get("valid_from")) or None,
            "valid_until": _norm(row.get("valid_until")) or None,
            "status": _norm(row.get("status")),
        })

    schema = {
        "summary": "不超过300字，只概括证据中明确出现的内容",
        "summary_atom_ids": ["支撑summary的atom_id"],
        "stable_patterns": [{"text": "长期偏好或稳定模式", "atom_ids": ["..."]}],
        "dated_events": [{
            "date": "YYYY-MM-DD或unknown",
            "summary": "历史事件",
            "atom_ids": ["..."],
        }],
        "open_loops": [{
            "summary": "来源明确写明仍待确认/未完成的事项",
            "status": "open或uncertain",
            "atom_ids": ["..."],
        }],
        "uncertainties": [{"text": "证据不足、冲突或时间不明之处", "atom_ids": ["..."]}],
        "conversation_hooks": [{"text": "可自然联想的话题方向，不是发送文案", "atom_ids": ["..."]}],
        "confidence": 0.0,
    }
    return (
        "你是 Memory Scene 的只读整理器。只能整理下面给出的 V2 Atom，不得检索、补充常识或创造事实。\n"
        "Scene 是历史背景与长期模式，不是用户当前状态。严禁把习惯、计划、旧照片、历史事件写成"
        "现在正在发生；严禁生成最终聊天消息。没有明确证据就放入 uncertainties 或省略。\n"
        "角色不可互换：用户/用户是同一人，银月/助手是另一方；助手做过的事不能写成用户做过，"
        "其他家人、同事或 crush 的偏好也不能写成用户偏好。一次行为不得概括成擅长或性格。\n"
        "硬限制：stable_patterns最多5条、dated_events最多8条、open_loops最多5条、"
        "uncertainties最多5条、conversation_hooks最多5条；宁可少写，不得超出。\n"
        "dated_events 的日期只能是 Atom 明确支持的 YYYY-MM-DD，否则写 unknown；不要使用今天、昨天、"
        "昨晚、刚刚、刚才、最近。open_loops 只有来源明确显示仍未完成/待确认时才能填写。\n"
        "每个陈述必须引用下面存在的 atom_id；不要输出未提供的 ID。只返回一个 JSON 对象，不要代码围栏。\n"
        "summary_atom_ids 必须是非空数组，且至少引用一条真正支撑 summary 的 Atom。\n"
        "输出前逐字自检：任何字段都不得含今天、昨天、昨晚、刚刚、刚才、近期、最近、今晚、明天；"
        "不得含当前、目前、此刻或正在恢复/正在使用等当前状态；不得把兼容、建议或方案写成已经执行；"
        "单次事件不得概括为通常、总是或擅长。\n"
        f"生活领域：{domain}（{SCENE_TITLES[domain]}）\n"
        f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
        f"输入 Atom（最多12条）：{json.dumps(input_atoms, ensure_ascii=False)}"
    )


def create_zhipu_scene_generator(api_key: str,
                                  api_base: str = "https://open.bigmodel.cn/api/paas/v4",
                                  model: str = "glm-4-flash",
                                  timeout_seconds: int = 30):
    """Return a single-attempt provider callable for offline M1B builds."""
    if not api_key:
        raise ValueError("api_key is required")
    base = api_base.rstrip("/")

    def generate(prompt: str) -> object:
        import requests
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你只输出严格JSON，并忠实引用给定证据。"},
                    {"role": "user", "content": prompt},
                ],
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 3000,
            },
            timeout=timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(f"scene LLM HTTP {response.status_code}")
        return response.json()["choices"][0]["message"]["content"]

    return generate


def _scene_content_signature(scene: Mapping) -> str:
    ignored = {
        "created_at", "updated_at", "revision", "last_selected_at",
        "selected_count", "cooldown_until",
    }
    stable = {key: value for key, value in scene.items() if key not in ignored}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True)


def _scene_from_expression(receiver_id: str, group: Mapping,
                           expression: Mapping, atoms_by_id: Mapping[str, Mapping],
                           now_iso: str, previous: Optional[Mapping] = None) -> dict:
    domain = str(group["life_domain"])
    atom_ids = [str(item["atom_id"]) for item in group["atoms"]]
    source_files = sorted({
        _norm(atoms_by_id[atom_id].get("source_file"))
        for atom_id in atom_ids if _norm(atoms_by_id[atom_id].get("source_file"))
    })
    dates = sorted({
        event["date"] for event in expression["dated_events"]
        if event["date"] != "unknown"
    })
    scene = MemorySceneV1(
        scene_id=_scene_id(receiver_id, domain),
        receiver_id=receiver_id,
        title=SCENE_TITLES[domain],
        life_domain=domain,
        summary=expression["summary"],
        summary_atom_ids=expression["summary_atom_ids"],
        stable_patterns=expression["stable_patterns"],
        dated_events=expression["dated_events"],
        open_loops=expression["open_loops"],
        uncertainties=expression["uncertainties"],
        conversation_hooks=expression["conversation_hooks"],
        atom_ids=atom_ids,
        source_files=source_files,
        confidence=expression["confidence"],
        sensitivity="sensitive" if domain in {"family", "relationship"} else "normal",
        initiative_policy="never",
        temporal_start=dates[0] if dates else None,
        temporal_end=dates[-1] if dates else None,
        last_event_at=dates[-1] if dates else None,
        created_at=now_iso,
        updated_at=now_iso,
    )
    result = asdict(scene)
    if previous:
        result["created_at"] = str(previous.get("created_at") or now_iso)
        result["last_selected_at"] = previous.get("last_selected_at")
        result["selected_count"] = int(previous.get("selected_count", 0) or 0)
        result["cooldown_until"] = previous.get("cooldown_until")
        result["revision"] = int(previous.get("revision", 1) or 1) + 1
        if _scene_content_signature(result) == _scene_content_signature(previous):
            result["updated_at"] = str(previous.get("updated_at") or now_iso)
            result["revision"] = int(previous.get("revision", 1) or 1)
    return result


def validate_scenes_payload(payload: Mapping, source_ids: set[str]) -> dict:
    issues: list[str] = []
    scenes = list(payload.get("scenes", []))
    if not 7 <= len(scenes) <= 15:
        issues.append(f"scene_count_out_of_range:{len(scenes)}")
    seen_scene_ids: set[str] = set()
    seen_atom_ids: set[str] = set()
    for scene in scenes:
        scene_id = str(scene.get("scene_id", ""))
        if not scene_id or scene_id in seen_scene_ids:
            issues.append(f"duplicate_or_empty_scene_id:{scene_id}")
        seen_scene_ids.add(scene_id)
        atom_ids = {str(item) for item in scene.get("atom_ids", [])}
        missing = atom_ids - source_ids
        if missing:
            issues.append(f"missing_atom_ids:{scene_id}:{sorted(missing)}")
        overlap = atom_ids & seen_atom_ids
        if overlap:
            issues.append(f"duplicate_atom_assignment:{scene_id}:{sorted(overlap)}")
        seen_atom_ids.update(atom_ids)
        if len(str(scene.get("summary", ""))) > 300:
            issues.append(f"summary_too_long:{scene_id}")
        if _contains_current_state_claim(str(scene.get("summary", ""))):
            issues.append(f"current_state_claim:{scene_id}")
        if scene.get("initiative_policy") != "never":
            issues.append(f"initiative_policy_not_shadow:{scene_id}")
    return {"passed": not issues, "issues": issues}


def rebuild_scene_expressions(receiver_id: str, generator,
                              groups_path: Optional[Path] = None,
                              source_dir: Optional[Path] = None,
                              output_path: Optional[Path] = None,
                              now: Optional[datetime] = None,
                              generator_id: str = "custom") -> dict:
    """Build Shadow Scene expressions with exactly one LLM call per group."""
    if not receiver_id:
        raise ValueError("receiver_id is required")
    group_file = Path(groups_path) if groups_path else DEFAULT_GROUPS_PATH
    output = Path(output_path) if output_path else DEFAULT_SCENES_V1_PATH
    source = Path(source_dir) if source_dir else V2_LANCE_DIR
    groups_payload = json.loads(group_file.read_text("utf-8"))
    if not groups_payload.get("quality_gate", {}).get("passed", False):
        raise ValueError("M1 group quality gate has not passed")
    rows = _read_v2_rows(source)
    atoms_by_id = {_norm(row.get("id")): row for row in rows}
    source_ids = set(atoms_by_id)
    if groups_payload.get("source_id_set_hash") != stable_id_hash(source_ids):
        raise ValueError("M1 groups do not match current V2 authority snapshot")

    previous_by_id: dict[str, Mapping] = {}
    previous_generator_id = ""
    if output.exists():
        previous_payload = json.loads(output.read_text("utf-8"))
        previous_generator_id = str(previous_payload.get("generator_id", ""))
        previous_by_id = {
            str(item.get("scene_id")): item for item in previous_payload.get("scenes", [])
        }

    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_iso = now_dt.astimezone(timezone.utc).isoformat()
    scenes = []
    skipped = []
    attempted = 0
    reused = 0
    normalization_log: list[dict] = []
    for group in groups_payload.get("groups", []):
        domain = str(group.get("life_domain", ""))
        try:
            allowed = [str(item["atom_id"]) for item in group.get("atoms", [])]
            sid = _scene_id(receiver_id, domain)
            previous = previous_by_id.get(sid)
            if (previous and previous_generator_id == generator_id
                    and list(previous.get("atom_ids", [])) == allowed):
                previous_expression = {
                    key: previous.get(key, [] if key != "summary" else "")
                    for key in (
                        "summary", "summary_atom_ids", "stable_patterns", "dated_events",
                        "open_loops", "uncertainties", "conversation_hooks", "confidence",
                    )
                }
                try:
                    previous_validated = validate_scene_expression(
                        previous_expression, allowed, atoms_by_id)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                else:
                    actions = list(previous_validated.get("_normalizations", []))
                    if actions:
                        normalization_log.append({
                            "life_domain": domain,
                            "actions": actions,
                            "source": "existing_scene",
                        })
                        scenes.append(_scene_from_expression(
                            receiver_id, group, previous_validated, atoms_by_id,
                            now_iso, previous=previous,
                        ))
                    else:
                        scenes.append(dict(previous))
                    reused += 1
                    continue
            attempted += 1
            prompt = build_scene_prompt(group, atoms_by_id)
            raw = generator(prompt)
            expression = validate_scene_expression(raw, allowed, atoms_by_id)
            if expression.get("_normalizations"):
                normalization_log.append({
                    "life_domain": domain,
                    "actions": expression["_normalizations"],
                })
            scenes.append(_scene_from_expression(
                receiver_id, group, expression, atoms_by_id, now_iso,
                previous=previous,
            ))
        except Exception as exc:
            skipped.append({"life_domain": domain, "error": type(exc).__name__, "reason": str(exc)[:180]})

    scenes.sort(key=lambda item: LIFE_DOMAIN_ORDER.index(item["life_domain"]))
    report = {
        "group_count": len(groups_payload.get("groups", [])),
        "attempted_group_count": attempted,
        "reused_scene_count": reused,
        "scene_count": len(scenes),
        "skipped_group_count": len(skipped),
        "skipped_groups": skipped,
        "normalized_group_count": len(normalization_log),
        "normalizations": normalization_log,
        "source_record_count": len(source_ids),
        "source_id_set_hash": stable_id_hash(source_ids),
        "llm_retry_count": 0,
        "generator_id": generator_id,
    }
    payload = {
        "schema_version": 1,
        "source_store": "memory_v2",
        "source_groups": str(group_file),
        "source_id_set_hash": report["source_id_set_hash"],
        "built_at": now_iso,
        "generator_id": generator_id,
        "shadow_only": True,
        "scenes": scenes,
        "report": report,
    }
    payload["quality_gate"] = validate_scenes_payload(payload, source_ids)
    minimum_expected = max(7, report["group_count"] - 1)
    if report["scene_count"] < minimum_expected:
        payload["quality_gate"]["passed"] = False
        payload["quality_gate"]["issues"].append(
            f"scene_coverage_too_low:{report['scene_count']}/{report['group_count']}")
    _atomic_json_write(output, payload)
    return report | {"quality_gate": payload["quality_gate"]}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Memory 2.1 Scene Shadow artifacts")
    parser.add_argument("--receiver-id", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_GROUPS_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--express", action="store_true",
                        help="Build M1B expressions from an already-approved M1 groups file")
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS_PATH)
    parser.add_argument("--scenes-output", type=Path, default=DEFAULT_SCENES_V1_PATH)
    configured_path = os.environ.get("COWAGENT_CONFIG")
    parser.add_argument(
        "--cow-config",
        type=Path,
        default=Path(configured_path) if configured_path else None,
        help="CowAgent config.json path (or set COWAGENT_CONFIG)",
    )
    parser.add_argument("--model", default="glm-4-flash")
    args = parser.parse_args(argv)
    if args.express:
        if args.cow_config is None:
            parser.error("--cow-config or COWAGENT_CONFIG is required with --express")
        config = json.loads(args.cow_config.read_text("utf-8"))
        generator = create_zhipu_scene_generator(
            api_key=str(config.get("zhipu_ai_api_key", "")),
            model=args.model,
        )
        report = rebuild_scene_expressions(
            args.receiver_id, generator, groups_path=args.groups,
            output_path=args.scenes_output, generator_id=args.model,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = rebuild_scene_groups(
        args.receiver_id, output_path=args.output, audit_output_path=args.audit_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
