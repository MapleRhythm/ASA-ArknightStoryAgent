from __future__ import annotations

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    ACTION_HINT_TERMS,
    MOTIVE_HINT_TERMS,
    OUTCOME_HINT_TERMS,
    TARGET_CONTEXT_HINT_TERMS,
)


class HybridEvidenceChainRolesMixin:
    def _document_chain_terms(self, document: dict, bridge_terms: list[str]) -> set[str]:
        text = self._document_text(document)
        terms = {term for term in bridge_terms if term and term in text}
        for segment in document.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            if not speaker:
                continue
            terms.add(speaker)
            if len(speaker) > 2 and speaker.endswith(("刺客", "主教", "成员", "干员")):
                terms.add(speaker[-2:])
        return terms

    def _classify_evidence_roles(
        self,
        query: str,
        document: dict,
        *,
        query_terms: list[str],
        bridge_terms: list[str],
    ) -> set[str]:
        text = self._document_text(document)
        roles: set[str] = set()
        query_overlap = sum(1 for term in query_terms if term in text)
        bridge_overlap = sum(1 for term in bridge_terms if term in text)
        query_action_terms = [term for term in ACTION_HINT_TERMS if term in query]
        source_type = self._document_source_type(document)

        if source_type == "moegirl_background" and self._is_story_detail_query(query):
            if query_overlap or bridge_overlap:
                return {"context"}
            return set()

        if query_action_terms and any(term in text for term in query_action_terms):
            roles.add("action")
        if query_overlap >= 3:
            roles.add("action")
        if bridge_overlap >= 2:
            roles.add("context")
        if any(term in text for term in MOTIVE_HINT_TERMS) and (bridge_overlap or query_overlap):
            roles.add("motive")
        if any(term in text for term in OUTCOME_HINT_TERMS) and (bridge_overlap or query_overlap):
            roles.add("outcome")
        if any(term in text for term in TARGET_CONTEXT_HINT_TERMS) and bridge_overlap:
            roles.add("target")
        if self._is_reveal_query(query):
            answerability = self._compute_answerability_bonus(query, document)
            if answerability >= 1.5:
                roles.add("outcome")
            if answerability >= 2.5:
                roles.add("motive")
            if answerability >= 3.5:
                roles.add("action")
        if "本舰" in text and (bridge_overlap or query_overlap):
            roles.add("context")
        if "[uc]info" in str(document.get("source_path") or "") and (bridge_overlap or query_overlap):
            roles.add("context")
            if any(term in text for term in OUTCOME_HINT_TERMS | TARGET_CONTEXT_HINT_TERMS):
                roles.add("outcome")
        return roles

    def _document_link_score(self, left: dict, right: dict, *, bridge_terms: list[str]) -> float:
        if left is right:
            return 1.0
        score = 0.0
        left_story = str(left.get("story_id") or "")
        right_story = str(right.get("story_id") or "")
        if left_story and left_story == right_story:
            score += 0.45

        left_activity = str(left.get("activity_id") or "")
        right_activity = str(right.get("activity_id") or "")
        left_source_type = self._document_source_type(left)
        right_source_type = self._document_source_type(right)
        generic_background_pair = (
            left_source_type == "moegirl_background"
            and right_source_type == "moegirl_background"
            and left_story != right_story
        )
        if left_activity and left_activity == right_activity:
            if not generic_background_pair:
                score += 0.2
            left_stage = str(left.get("stage_code") or "")
            right_stage = str(right.get("stage_code") or "")
            if left_stage and left_stage == right_stage and not generic_background_pair:
                score += 0.25

            left_sort = left.get("story_sort")
            right_sort = right.get("story_sort")
            if isinstance(left_sort, int) and isinstance(right_sort, int) and not generic_background_pair:
                distance = abs(left_sort - right_sort)
                if distance <= 1:
                    score += 0.25
                elif distance <= 3:
                    score += 0.12

            left_stage_number = self._document_stage_number(left)
            right_stage_number = self._document_stage_number(right)
            if left_stage_number is not None and right_stage_number is not None and not generic_background_pair:
                distance = abs(left_stage_number - right_stage_number)
                if distance == 0:
                    score += 0.22
                elif distance == 1:
                    score += 0.14

        shared_terms = self._document_chain_terms(left, bridge_terms) & self._document_chain_terms(right, bridge_terms)
        if shared_terms:
            score += min(len(shared_terms), 3) * 0.08
        return min(score, 1.0)
