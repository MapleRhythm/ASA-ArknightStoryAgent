from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


STORYLINE_LABEL_PREFIX = "storyline:"
STORYLINE_SCOPE_EXCLUDED_ACTIVITY_IDS = frozenset(
    {
        "operator_voice",
        "operator_handbook",
        "moegirl_lore",
    }
)


@dataclass(frozen=True, slots=True)
class StorylineRule:
    key: str
    label: str
    activity_ids: frozenset[str] = frozenset()
    activity_names: frozenset[str] = frozenset()
    memory_prefixes: frozenset[str] = frozenset()


STORYLINE_RULES: tuple[StorylineRule, ...] = (
    StorylineRule(
        key="yan_sui",
        label="炎国/岁线",
        activity_ids=frozenset(
            {
                "act6d5",
                "act14mini",
                "act15side",
                "act16d5",
                "act19mini",
                "act23side",
                "act31side",
                "act40side",
                "act49side",
            }
        ),
        activity_names=frozenset({"洪炉示岁", "春分", "将进酒", "画中人", "镜中集", "登临意", "怀黍离", "相见欢", "辞岁行"}),
        memory_prefixes=frozenset({"story_nian_", "story_dusk_", "story_ling_", "story_shu_", "story_lin_"}),
    ),
    StorylineRule(
        key="rhine_columbia",
        label="莱茵生命/哥伦比亚线",
        activity_ids=frozenset({"act15d0", "act19side", "act25side", "act47side"}),
        activity_names=frozenset({"孤岛风云", "绿野幻梦", "孤星", "未许之地"}),
        memory_prefixes=frozenset({"story_saria_", "story_silence_", "story_ifrit_", "story_muelsyse_", "story_doroth_"}),
    ),
    StorylineRule(
        key="leithanien",
        label="莱塔尼亚线",
        activity_ids=frozenset({"act11d0", "act18side", "act29side"}),
        activity_names=frozenset({"沃伦姆德的薄暮", "尘影余音", "崔林特尔梅之金"}),
        memory_prefixes=frozenset({"story_eben_", "story_czerny_", "story_hibiscus2_"}),
    ),
    StorylineRule(
        key="abyssal",
        label="深海线",
        activity_ids=frozenset({"1stact", "act18d3", "act17side", "act34side", "act39side"}),
        activity_names=frozenset({"骑兵与猎人", "覆潮之下", "愚人号", "生路", "出苍白海"}),
        memory_prefixes=frozenset({"story_skadi_", "story_specter_", "story_gladiia_", "story_irene_", "story_lumen_"}),
    ),
    StorylineRule(
        key="main_rhodes",
        label="罗德岛/主线线",
        activity_ids=frozenset(
            {
                "main_0",
                "main_1",
                "main_2",
                "main_3",
                "main_4",
                "main_5",
                "main_6",
                "main_7",
                "main_8",
                "main_9",
                "main_10",
                "main_11",
                "main_12",
                "main_13",
                "main_14",
                "act8mini",
                "act9d0",
                "act18mini",
                "act18d0",
                "act33side",
            }
        ),
        activity_names=frozenset(
            {
                "黑暗时代·上",
                "黑暗时代·下",
                "急性衰竭",
                "二次呼吸",
                "靶向药物",
                "局部坏死",
                "苦难摇篮",
                "怒号光明",
                "风暴瞭望",
                "破碎日冕",
                "淬火尘霾",
                "惊霆无声",
                "恶兆湍流",
                "慈悲灯塔",
                "生于黑夜",
                "如我所见",
                "我们明日见",
                "遗尘漫步",
                "巴别塔",
            }
        ),
        memory_prefixes=frozenset({"story_kalts_", "story_amiya_", "story_rosmon_", "story_shining_"}),
    ),
    StorylineRule(
        key="kazimierz",
        label="卡西米尔线",
        activity_ids=frozenset({"act12mini", "act13d5", "act13side", "act9mini"}),
        activity_names=frozenset({"日暮寻路", "玛莉娅·临光", "长夜临光", "红松林"}),
        memory_prefixes=frozenset({"story_nearl_", "story_nearl2_", "story_blemsh_", "story_grani_"}),
    ),
    StorylineRule(
        key="siracusa",
        label="叙拉古线",
        activity_ids=frozenset({"act20mini", "act21side", "act38side"}),
        activity_names=frozenset({"十字路口", "叙拉古人", "揭幕者们"}),
        memory_prefixes=frozenset({"story_texas_", "story_lappland_"}),
    ),
    StorylineRule(
        key="tara",
        label="塔拉线",
        activity_ids=frozenset({"main_9", "main_10", "main_11", "main_12", "main_13", "main_14", "act22side", "act41side"}),
        activity_names=frozenset({"风暴瞭望", "破碎日冕", "淬火尘霾", "惊霆无声", "恶兆湍流", "慈悲灯塔", "照我以火", "挽歌燃烧殆尽"}),
        memory_prefixes=frozenset({"story_reed_", "story_reed2_"}),
    ),
    StorylineRule(
        key="kjerag",
        label="谢拉格线",
        activity_ids=frozenset({"act14side", "act46side"}),
        activity_names=frozenset({"风雪过境", "雪山降临1101"}),
        memory_prefixes=frozenset({"story_svrash_", "story_gnosis_", "story_kjera_", "story_aurora_", "story_pramanix_", "story_cliffheart_"}),
    ),
    StorylineRule(
        key="laterano",
        label="拉特兰线",
        activity_ids=frozenset({"act16side", "act26side", "act42side"}),
        activity_names=frozenset({"吾导先路", "空想花庭", "众生行记"}),
        memory_prefixes=frozenset({"story_executor_", "story_enforcer_", "story_archet_"}),
    ),
    StorylineRule(
        key="ursus",
        label="乌萨斯线",
        activity_ids=frozenset({"act10d5"}),
        activity_names=frozenset({"乌萨斯的孩子们"}),
        memory_prefixes=frozenset({"story_absin_", "story_zima_", "story_istina_", "story_glassb_"}),
    ),
)


def _storyline_scope(key: str) -> str:
    return f"{STORYLINE_LABEL_PREFIX}{key}"


def _activity_fallback_scope(activity_id: str) -> str:
    return f"{STORYLINE_LABEL_PREFIX}activity:{activity_id}"


def _memory_story_key(document: dict[str, Any]) -> str:
    story_id = str(document.get("story_id") or document.get("story_key") or document.get("id") or "").strip()
    match = re.search(r"(?:^|/)obt/memory/([^/#]+)", story_id)
    if match:
        return match.group(1)
    source_path = str(document.get("source_path") or "").strip()
    match = re.search(r"(?:^|/)obt/memory/([^/#.]+)", source_path)
    return match.group(1) if match else ""


def _normalized_activity_id(document: dict[str, Any]) -> str:
    activity_id = str(document.get("activity_id") or "").strip()
    if activity_id:
        return activity_id
    story_id = str(document.get("story_id") or document.get("story_key") or document.get("id") or "").strip()
    match = re.search(r"(?:^|/)activities/([^/]+)/", story_id)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|/)(?:level_)?main[_-](\d{1,2})(?:[-_/]|$)", story_id, flags=re.IGNORECASE)
    if match:
        return f"main_{int(match.group(1))}"
    return ""


def document_storyline_scopes(document: dict[str, Any]) -> list[str]:
    activity_id = _normalized_activity_id(document)
    activity_name = str(document.get("activity_name") or "").strip()
    memory_key = _memory_story_key(document)
    if activity_id in STORYLINE_SCOPE_EXCLUDED_ACTIVITY_IDS:
        return []

    scopes: list[str] = []
    for rule in STORYLINE_RULES:
        if activity_id and activity_id in rule.activity_ids:
            scopes.append(_storyline_scope(rule.key))
            continue
        if activity_name and activity_name in rule.activity_names:
            scopes.append(_storyline_scope(rule.key))
            continue
        if memory_key and any(memory_key.startswith(prefix) for prefix in rule.memory_prefixes):
            scopes.append(_storyline_scope(rule.key))

    if scopes:
        return list(dict.fromkeys(scopes))
    if activity_id:
        return [_activity_fallback_scope(activity_id)]
    if activity_name:
        safe_name = re.sub(r"\s+", "_", activity_name.strip())
        return [_activity_fallback_scope(safe_name)]
    return []


def document_primary_storyline_scope(document: dict[str, Any]) -> str:
    scopes = document_storyline_scopes(document)
    return scopes[0] if scopes else ""


def storyline_scope_label(scope: str) -> str:
    if not scope.startswith(STORYLINE_LABEL_PREFIX):
        return scope
    key = scope[len(STORYLINE_LABEL_PREFIX) :]
    for rule in STORYLINE_RULES:
        if key == rule.key:
            return f"{scope} ({rule.label})"
    if key.startswith("activity:"):
        return f"{scope} (单活动:{key.removeprefix('activity:')})"
    return scope
