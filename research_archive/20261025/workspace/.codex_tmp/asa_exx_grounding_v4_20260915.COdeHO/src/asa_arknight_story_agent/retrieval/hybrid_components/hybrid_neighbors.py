from __future__ import annotations

from typing import Any


class HybridNeighborExpansionMixin:
    def expand_hits_with_neighbors(
        self,
        hits: list[dict[str, Any]],
        *,
        max_seed_docs: int = 24,
        story_window: int = 2,
        activity_story_sort_window: int = 1,
        same_story_sweep: bool = False,
        same_story_max_seed_docs: int = 8,
        same_story_max_docs_per_story: int = 24,
        top_k: int = 120,
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits
        neighbor_doc_indices = self._collect_story_and_stage_neighbors(
            hits,
            max_seed_docs=max_seed_docs,
            story_window=story_window,
            activity_story_sort_window=activity_story_sort_window,
            same_story_sweep=same_story_sweep,
            same_story_max_seed_docs=same_story_max_seed_docs,
            same_story_max_docs_per_story=same_story_max_docs_per_story,
        )
        merged = [dict(item) for item in hits]
        seen = {int(item["doc_index"]) for item in merged}
        neighbor_rank = 0
        for doc_index in neighbor_doc_indices:
            if doc_index in seen or not (0 <= doc_index < len(self.documents)):
                continue
            seen.add(doc_index)
            neighbor_rank += 1
            merged.append(
                {
                    "doc_index": doc_index,
                    "document": self.documents[doc_index],
                    "dense_score": None,
                    "sparse_score": None,
                    "minirag_score": None,
                    "fusion_score": 0.0,
                    "supplemental_source": "neighbor",
                    "supplemental_rank": neighbor_rank,
                }
            )
            if len(merged) >= top_k:
                break
        return merged[:top_k]

    def _collect_story_and_stage_neighbors(
        self,
        seed_hits: list[dict[str, Any]],
        *,
        max_seed_docs: int = 6,
        story_window: int = 2,
        activity_story_sort_window: int = 1,
        same_story_sweep: bool = False,
        same_story_max_seed_docs: int = 8,
        same_story_max_docs_per_story: int = 24,
    ) -> list[int]:
        candidate_indices: list[int] = []
        for seed_rank, hit in enumerate(seed_hits[:max_seed_docs]):
            doc = hit["document"]
            doc_index = int(hit["doc_index"])
            if doc_index not in candidate_indices:
                candidate_indices.append(doc_index)

            story_id = str(doc.get("story_id") or "").strip()
            if story_id:
                story_indices = self.story_doc_indices.get(story_id, [])
                try:
                    current_pos = story_indices.index(doc_index)
                except ValueError:
                    current_pos = -1
                if current_pos >= 0:
                    start = max(0, current_pos - story_window)
                    end = min(len(story_indices), current_pos + story_window + 1)
                    for neighbor in story_indices[start:end]:
                        if neighbor not in candidate_indices:
                            candidate_indices.append(neighbor)
                    if same_story_sweep and seed_rank < same_story_max_seed_docs:
                        for neighbor in self._story_sweep_indices(
                            story_indices,
                            current_pos,
                            max_docs=max(1, same_story_max_docs_per_story),
                        ):
                            if neighbor not in candidate_indices:
                                candidate_indices.append(neighbor)
                elif same_story_sweep and seed_rank < same_story_max_seed_docs:
                    for neighbor in story_indices[: max(1, same_story_max_docs_per_story)]:
                        if neighbor not in candidate_indices:
                            candidate_indices.append(neighbor)

            activity_id = str(doc.get("activity_id") or "").strip()
            stage_code = str(doc.get("stage_code") or "").strip()
            if activity_id and stage_code:
                for neighbor in self.stage_doc_indices.get((activity_id, stage_code), []):
                    if neighbor not in candidate_indices:
                        candidate_indices.append(neighbor)
            story_sort = doc.get("story_sort")
            if activity_id and isinstance(story_sort, int):
                for offset in range(-activity_story_sort_window, activity_story_sort_window + 1):
                    neighbor_sort = story_sort + offset
                    for neighbor in self.activity_story_sort_doc_indices.get(activity_id, {}).get(neighbor_sort, []):
                        if neighbor not in candidate_indices:
                            candidate_indices.append(neighbor)
        return candidate_indices

    @staticmethod
    def _story_sweep_indices(story_indices: list[int], center_pos: int, *, max_docs: int) -> list[int]:
        if center_pos < 0 or not story_indices or max_docs <= 0:
            return []
        ordered: list[int] = []
        for distance in range(len(story_indices)):
            positions = [center_pos] if distance == 0 else [center_pos - distance, center_pos + distance]
            for pos in positions:
                if 0 <= pos < len(story_indices):
                    doc_index = story_indices[pos]
                    if doc_index not in ordered:
                        ordered.append(doc_index)
                        if len(ordered) >= max_docs:
                            return ordered
        return ordered
