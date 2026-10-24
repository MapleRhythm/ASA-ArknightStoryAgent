"""Narrow discourse bridges for a retrieval tokenizer's broken phrase spans.

These rules excuse *framing fragments*, not substantive factual claims. They
are not an entailment classifier. No character-bag coverage threshold is used.
"""
from __future__ import annotations

import re


def compact(text: str) -> str:
    return re.sub(r"[\s，。！？；：、,.!?;:·“”‘’\"'（）()的地得了着]", "", text)


def discourse_support(token: str, claim: str, question: str, cited_texts: list[str]) -> str | None:
    normalized = compact(token)
    # Resolve an explicitly plural antecedent within ONE speaker turn, with
    # an explicit universal quantifier and the same following predicate.
    # Do not infer "all" from merely mentioning a group elsewhere in a chunk.
    group = re.fullmatch(r"(兄弟姐妹|同伴|队友|工人|成员|学生|士兵|村民)(?:们)?都", normalized)
    if group:
        full_claim = compact(claim)
        group_at = full_claim.find(normalized)
        # Preserve clause boundaries while isolating the asserted predicate.
        raw_group = re.search(re.escape(group[1]) + r"们?都([^，。；！？,.;!?]+)", claim)
        if group_at >= 0 and raw_group:
            predicate = compact(raw_group[1])
            predicate = re.sub(r"^(?:已经|已)", "", predicate)
            predicate = predicate.replace("死去", "死").replace("死亡", "死")
            if len(predicate) >= 1:
                for text in cited_texts:
                    turns = re.split(r"(?:^|[\s。！？])[\u4e00-\u9fff·]{2,12}[：:]", text)
                    for turn in turns:
                        antecedent = re.search(
                            re.escape(group[1]) + r".{0,65}?他们都([^，。；！？,.;!?]+)", turn,
                        )
                        if antecedent:
                            # Ambiguous intervening group blocks resolution.
                            span = antecedent[0].split("他们都", 1)[0]
                            groups = re.findall(r"兄弟姐妹|同伴|队友|工人|成员|学生|士兵|村民", span)
                            source_predicate = compact(antecedent[1])
                            source_predicate = re.sub(r"^(?:已经|已)", "", source_predicate)
                            source_predicate = source_predicate.replace("死去", "死").replace("死亡", "死")
                            if groups == [group[1]] and predicate == source_predicate:
                                return "explicit_same_turn_plural_antecedent"
    # Pronouns shift in indirect speech. Only license an epistemic/reporting
    # fragment whose predicate is literally present; never drop a pronoun
    # from arbitrary actions such as "她杀了他".
    mental = re.fullmatch(r"[我你他她](想知道|想了解|需要知道)", normalized)
    if mental and any(mental[1] in compact(text) for text in cited_texts):
        return "reported_epistemic_predicate"
    if normalized == "前提条件" and any("前提" in text for text in cited_texts):
        return "prerequisite_wording"
    for text in cited_texts:
        speakers = re.findall(r"(?:^|[\s。！？])([\u4e00-\u9fff·]{2,12})[：:]", text)
        for speaker in speakers:
            # The token ends inside a reporting frame. Content following it
            # is still independently checked by the original validator.
            if re.fullmatch(
                re.escape(speaker) + r"(?:表示|提到|指出|说道|说)(?:只|其|自己|他|她)?",
                normalized,
            ):
                return "visible_speaker_reporting_frame"
    # Repeated temporal scope comes from the question, not an answer fact.
    # Limit to scope fragments; do not exempt arbitrary copied questions,
    # relationships, numbers, motives or negations.
    if (
        re.search(r"(?:交战|战斗|会面|讨论|训练|会议)(?:时|期间)$", normalized)
        and normalized in compact(question)
        and not re.search(r"[不没未无否零一二三四五六七八九十0-9]", normalized)
    ):
        return "question_temporal_scope"
    # Example class (not an answer template): a rule's name is repeated from
    # the question and the complete rule after "规定" is literal source text.
    # This avoids the tokenizer joining the last topic characters with the
    # first character of the factual clause.
    scoped = re.match(r"^(.{2,60}?(?:法案|条例|规章|制度|规则|通知))(规定|要求)(.+)", compact(claim))
    if scoped:
        topic, predicate, body = scoped.groups()
        kind = re.search(r"(法案|条例|规章|制度|规则|通知)$", topic)[1]
        frame = topic + predicate
        fragment_start = compact(claim).find(normalized)
        if (
            topic in compact(question)
            and fragment_start >= 0 and fragment_start < len(frame)
            and len(body) >= 8
            and any(kind in text and body in compact(text) for text in cited_texts)
        ):
            return "question_topic_with_literal_rule_body"
    return None
