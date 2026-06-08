from __future__ import annotations

from typing import Any


class HybridEvidenceChainBuildingMixin:
    def _build_evidence_chains(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> list[dict[str, Any]]:
        query_terms = self._extract_chain_query_terms(query)
        scored_items: list[dict[str, Any]] = []
        for item in candidates:
            document = item["document"]
            roles = self._classify_evidence_roles(
                query,
                document,
                query_terms=query_terms,
                bridge_terms=bridge_terms,
            )
            text = self._document_text(document)
            scored_items.append(
                {
                    "item": item,
                    "roles": roles,
                    "query_overlap": sum(1 for term in query_terms if term in text),
                    "bridge_overlap": sum(1 for term in bridge_terms if term in text),
                    "base_score": float(item.get("rerank_score") or item.get("fusion_score") or 0.0),
                }
            )

        seeds = [
            scored
            for scored in scored_items
            if scored["roles"] or scored["query_overlap"] >= 2 or scored["bridge_overlap"] >= 2
        ]
        seeds = sorted(seeds, key=lambda scored: scored["base_score"], reverse=True)[:24]
        chains: list[dict[str, Any]] = []
        for seed in seeds:
            members: list[dict[str, Any]] = [seed]
            for scored in scored_items:
                if scored is seed:
                    continue
                link_score = self._document_link_score(
                    seed["item"]["document"],
                    scored["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                if link_score < 0.4:
                    continue
                member = dict(scored)
                member["link_score"] = link_score
                members.append(member)

            members = sorted(
                members,
                key=lambda member: (
                    len(member["roles"]),
                    member.get("link_score", 1.0),
                    member["bridge_overlap"],
                    member["base_score"],
                ),
                reverse=True,
            )[:8]
            role_union = set().union(*(member["roles"] for member in members))
            if len(role_union) < 2:
                continue
            if "action" in role_union and not ({"outcome", "target"} & role_union):
                continue
            if "action" not in role_union and not {"motive", "outcome"}.issubset(role_union):
                continue
            link_scores = [
                self._document_link_score(
                    left["item"]["document"],
                    right["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                for left in members
                for right in members
                if left is not right
            ]
            average_link_score = sum(link_scores) / len(link_scores) if link_scores else 0.0
            role_score = 0.0
            role_score += 0.35 if "action" in role_union else 0.0
            role_score += 0.3 if "outcome" in role_union else 0.0
            role_score += 0.25 if "motive" in role_union else 0.0
            role_score += 0.18 if "target" in role_union else 0.0
            role_score += 0.12 if "context" in role_union else 0.0
            if {"action", "outcome"}.issubset(role_union):
                role_score += 0.25
            if {"action", "motive"}.issubset(role_union):
                role_score += 0.18
            query_coverage = len(
                {
                    term
                    for term in query_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in members)
                }
            )
            bridge_coverage = len(
                {
                    term
                    for term in bridge_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in members)
                }
            )
            chain_score = (
                role_score
                + min(query_coverage, 8) * 0.04
                + min(bridge_coverage, 6) * 0.06
                + average_link_score * 0.45
                + min(len(members), 5) * 0.03
            )
            chains.append(
                {
                    "score": chain_score,
                    "roles": role_union,
                    "members": members,
                }
            )
        return sorted(chains, key=lambda chain: chain["score"], reverse=True)

    def _build_generic_evidence_chains(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> list[dict[str, Any]]:
        query_terms = self._extract_chain_query_terms(query)
        scored_items: list[dict[str, Any]] = []
        for item in candidates:
            document = item["document"]
            text = self._document_text(document)
            roles = self._classify_evidence_roles(
                query,
                document,
                query_terms=query_terms,
                bridge_terms=bridge_terms,
            )
            query_overlap = sum(1 for term in query_terms if term in text)
            bridge_overlap = sum(1 for term in bridge_terms if term in text)
            scored_items.append(
                {
                    "item": item,
                    "roles": roles or ({"context"} if query_overlap or bridge_overlap else set()),
                    "query_overlap": query_overlap,
                    "bridge_overlap": bridge_overlap,
                    "base_score": float(item.get("rerank_score") or item.get("fusion_score") or 0.0),
                }
            )

        seeds = [
            scored
            for scored in scored_items
            if scored["query_overlap"] > 0 or scored["bridge_overlap"] > 0 or scored["base_score"] > 0
        ]
        seeds = sorted(
            seeds,
            key=lambda scored: (
                scored["query_overlap"],
                scored["bridge_overlap"],
                len(scored["roles"]),
                scored["base_score"],
            ),
            reverse=True,
        )[:32]

        chains: list[dict[str, Any]] = []
        seen_chain_keys: set[tuple[int, ...]] = set()
        for seed in seeds:
            linked_members: list[dict[str, Any]] = [seed]
            for scored in scored_items:
                if scored is seed:
                    continue
                link_score = self._document_link_score(
                    seed["item"]["document"],
                    scored["item"]["document"],
                    bridge_terms=bridge_terms,
                )
                if link_score < 0.25:
                    continue
                member = dict(scored)
                member["link_score"] = link_score
                linked_members.append(member)

            linked_members = sorted(
                linked_members,
                key=lambda member: (
                    member is seed,
                    member.get("link_score", 1.0),
                    member["query_overlap"],
                    member["bridge_overlap"],
                    len(member["roles"]),
                    member["base_score"],
                ),
                reverse=True,
            )[:5]
            if len(linked_members) < 2:
                continue
            chain_key = tuple(
                dict.fromkeys(
                    int(member["item"]["doc_index"])
                    for member in sorted(linked_members, key=self._chain_member_sort_key)
                )
            )
            if chain_key in seen_chain_keys:
                continue
            seen_chain_keys.add(chain_key)
            role_union = set().union(*(member["roles"] for member in linked_members))
            query_coverage = len(
                {
                    term
                    for term in query_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in linked_members)
                }
            )
            bridge_coverage = len(
                {
                    term
                    for term in bridge_terms
                    if any(term in self._document_text(member["item"]["document"]) for member in linked_members)
                }
            )
            average_base_score = sum(member["base_score"] for member in linked_members) / len(linked_members)
            chains.append(
                {
                    "score": (
                        min(query_coverage, 8) * 0.12
                        + min(bridge_coverage, 6) * 0.08
                        + min(len(role_union), 4) * 0.05
                        + average_base_score * 0.1
                    ),
                    "roles": role_union,
                    "members": linked_members,
                }
            )
        return sorted(chains, key=lambda chain: chain["score"], reverse=True)

    def _chain_member_sort_key(self, member: dict[str, Any]) -> tuple[str, int, int, int]:
        document = member["item"]["document"]
        return (
            str(document.get("story_id") or document.get("activity_id") or ""),
            int(document.get("story_sort") if isinstance(document.get("story_sort"), int) else 10**9),
            self._document_stage_number(document) or 10**9,
            int(member["item"].get("doc_index") or 0),
        )

    def _render_chain_text(self, chain: dict[str, Any]) -> str:
        members = sorted(chain["members"], key=self._chain_member_sort_key)
        lines: list[str] = []
        seen: set[int] = set()
        for index, member in enumerate(members, start=1):
            item = member["item"]
            doc_index = int(item["doc_index"])
            if doc_index in seen:
                continue
            seen.add(doc_index)
            document = item["document"]
            prefix_parts = [
                str(document.get("activity_name") or ""),
                str(document.get("story_name") or ""),
                str(document.get("stage_code") or ""),
            ]
            prefix = " / ".join(part for part in prefix_parts if part)
            text = str(document.get("clean_text") or document.get("search_text") or "").strip()
            if prefix:
                lines.append(f"[E{index}] {prefix}\n{text}")
            else:
                lines.append(f"[E{index}] {text}")
        return "\n".join(lines)
