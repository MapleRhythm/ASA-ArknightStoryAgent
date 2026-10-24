from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import LOW_RERANK_QUERY_TYPES


class HybridEvidenceChainScoringMixin:
    def _apply_chain_model_scores(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
        batch_size: int,
        query_mode: str | None = None,
    ) -> bool:
        if not self.reranker:
            return False
        chains = self._build_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            chains = self._build_generic_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            return False

        chain_payloads: list[dict[str, Any]] = []
        seen_chain_keys: set[tuple[int, ...]] = set()
        for chain in chains[: max(24, min(80, len(chains)))]:
            member_items = sorted(chain["members"], key=self._chain_member_sort_key)
            member_indices = tuple(dict.fromkeys(int(member["item"]["doc_index"]) for member in member_items))
            if len(member_indices) < 2 or member_indices in seen_chain_keys:
                continue
            seen_chain_keys.add(member_indices)
            chain_text = self._render_chain_text(chain)
            if not chain_text:
                continue
            chain_structure = {
                "chain_length": len(member_indices),
                "causal_order": "model_candidate",
                "evidence_types": sorted(chain["roles"]) if chain.get("roles") else ["context"],
            }
            chain_text = (
                f"[CHAIN_LEN={chain_structure['chain_length']}] "
                f"[CAUSAL_ORDER={chain_structure['causal_order']}] "
                f"[EVIDENCE_TYPES=({'|'.join(chain_structure['evidence_types'])})]\n"
                + chain_text
            )
            chain_payloads.append(
                {
                    "chain": chain,
                    "member_indices": member_indices,
                    "chain_text": chain_text,
                }
            )
        if not chain_payloads:
            return False

        chain_scores = self.reranker.score(
            query=query,
            documents=[payload["chain_text"] for payload in chain_payloads],
            batch_size=batch_size,
        )
        doc_index_to_item = {int(item["doc_index"]): item for item in candidates}
        for rank, (payload, score) in enumerate(
            sorted(
                zip(chain_payloads, chain_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ):
            chain = payload["chain"]
            normalized_rank_bonus = 1.0 / (rank + 1)
            chain_score = float(score)
            for member in chain["members"]:
                item = doc_index_to_item.get(int(member["item"]["doc_index"]))
                if item is None:
                    continue
                document = item["document"]
                role_bonus = 0.07 * len(member["roles"])
                if "action" in member["roles"]:
                    role_bonus += 0.12
                if "outcome" in member["roles"]:
                    role_bonus += 0.11
                if "motive" in member["roles"]:
                    role_bonus += 0.1
                member_relevance = self._compute_query_overlap_bonus(
                    query,
                    document,
                    bridge_terms,
                    query_mode=query_mode,
                )
                adjusted_chain_score = self._adjust_chain_score_for_query_type(
                    query,
                    document,
                    chain_score,
                    query_mode,
                )
                if query_mode in LOW_RERANK_QUERY_TYPES:
                    chain_member_score = (
                        adjusted_chain_score * 0.35
                        + normalized_rank_bonus * 0.25
                        + role_bonus * 0.25
                        + member_relevance
                    )
                else:
                    chain_member_score = (
                        adjusted_chain_score * 0.45
                        + normalized_rank_bonus * 0.45
                        + role_bonus
                        + member_relevance
                    )
                item["evidence_chain_score"] = max(
                    float(item.get("evidence_chain_score", float("-inf"))),
                    chain_member_score,
                )
                item["evidence_chain_model_score"] = max(
                    float(item.get("evidence_chain_model_score", float("-inf"))),
                    float(score),
                )
                item["evidence_chain_roles"] = sorted(member["roles"])
                item["evidence_chain_text"] = payload["chain_text"]
        return True

    def _final_rerank_score(self, item: dict[str, Any]) -> float:
        chain_score = item.get("evidence_chain_score")
        rerank_score = float(item.get("rerank_score") or 0.0)
        if chain_score is not None:
            chain_bonus = max(min(float(chain_score), 3.5), -1.5)
            return rerank_score + chain_bonus
        return rerank_score

    def rerank_with_evidence_chains(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
        batch_size: int,
        bridge_terms: list[str] | None = None,
        query_mode: str | None = None,
        fallback_to_document_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        if query_mode is None:
            query_mode = self._infer_query_mode(query)
        candidates = [dict(item) for item in hits]
        if self.reranker:
            scores = self.reranker.score(
                query=query,
                documents=[item["document"]["search_text"] for item in candidates],
                batch_size=batch_size,
            )
            for item, score in zip(candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            resolved_bridge_terms = bridge_terms or self.extract_bridge_terms(query, candidates)
            chain_scored = self._apply_chain_model_scores(
                query,
                candidates,
                bridge_terms=resolved_bridge_terms,
                batch_size=batch_size,
                query_mode=query_mode,
            )
            if not chain_scored:
                self._apply_evidence_chain_rerank(query, candidates, bridge_terms=resolved_bridge_terms)
            self._apply_main_chapter_focus_adjustment(query, candidates)
            self._apply_source_type_focus_adjustment(query, candidates, query_mode=query_mode)
            return sorted(
                candidates,
                key=self._final_rerank_score,
                reverse=True,
            )[:top_k]

        for item in candidates:
            item["rerank_score"] = float(item.get("fusion_score") or 0.0)
        resolved_bridge_terms = bridge_terms or self.extract_bridge_terms(query, candidates)
        self._apply_evidence_chain_rerank(query, candidates, bridge_terms=resolved_bridge_terms)
        self._apply_main_chapter_focus_adjustment(query, candidates)
        self._apply_source_type_focus_adjustment(query, candidates, query_mode=query_mode)
        return sorted(
            candidates,
            key=self._final_rerank_score,
            reverse=True,
        )[:top_k]

    def _apply_evidence_chain_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        bridge_terms: list[str],
    ) -> None:
        chains = self._build_evidence_chains(query, candidates, bridge_terms=bridge_terms)
        if not chains:
            self._apply_causal_focus_adjustment(query, candidates)
            return
        query_terms = self._extract_chain_query_terms(query)
        for chain_rank, chain in enumerate(chains[:3]):
            chain_bonus = min(chain["score"], 1.25) * (0.85 ** chain_rank)
            for member in chain["members"]:
                item = member["item"]
                text = self._document_text(item["document"])
                roles = member["roles"]
                long_exact_overlap = sum(1 for term in query_terms if len(term) >= 4 and term in text)
                bridge_overlap = sum(1 for term in bridge_terms if term in text)
                contributes_target_or_outcome = bool({"target", "outcome"} & roles)
                contributes_exact_action = "action" in roles and long_exact_overlap > 0
                contributes_motive = "motive" in roles and bridge_overlap >= 2 and any(
                    term in text for term in ("杀死", "约定", "交易", "计划", "目的", "决定")
                )
                if not (contributes_target_or_outcome or contributes_exact_action or contributes_motive):
                    continue
                role_bonus = 0.08 * len(member["roles"])
                if "action" in member["roles"]:
                    role_bonus += 0.12
                if "outcome" in member["roles"]:
                    role_bonus += 0.11
                if "motive" in member["roles"]:
                    role_bonus += 0.1
                bonus = chain_bonus + role_bonus
                item["chain_bonus"] = max(float(item.get("chain_bonus") or 0.0), bonus)
                item["evidence_chain_roles"] = sorted(member["roles"])
        for item in candidates:
            item["rerank_score"] = float(item.get("rerank_score") or 0.0) + float(item.get("chain_bonus") or 0.0)
        self._apply_causal_focus_adjustment(query, candidates)
