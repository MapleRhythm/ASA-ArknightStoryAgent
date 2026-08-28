from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_building import (
    HybridEvidenceChainBuildingMixin,
)


def test_internal_chain_member_markers_do_not_look_like_public_citations() -> None:
    mixin = HybridEvidenceChainBuildingMixin()
    mixin._document_stage_number = lambda document: None
    chain = {
        "members": [
            {
                "item": {
                    "doc_index": 1,
                    "document": {
                        "id": "a",
                        "clean_text": "甲",
                        "activity_name": "活动",
                        "story_sort": 1,
                    },
                }
            },
            {
                "item": {
                    "doc_index": 2,
                    "document": {
                        "id": "b",
                        "clean_text": "乙",
                        "activity_name": "活动",
                        "story_sort": 2,
                    },
                }
            },
        ]
    }

    rendered = mixin._render_chain_text(chain)

    assert "[CHAIN_MEMBER_1]" in rendered
    assert "[CHAIN_MEMBER_2]" in rendered
    assert "[E1]" not in rendered
    assert "[E2]" not in rendered
