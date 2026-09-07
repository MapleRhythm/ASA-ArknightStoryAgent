import json
from pathlib import Path

import pytest

from asa_arknight_story_agent.runtime_config import load_runtime_config


def test_missing_runtime_config_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Runtime config does not exist"):
        load_runtime_config(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "config_name",
    [
        "runtime_gpu_reranker_qwen35_4b.json",
        "runtime_cpu_qwen35_4b_no_reranker.json",
        "runtime_cpu_api_no_reranker.json",
        "runtime_gpu_reranker_qwen35_4b_evidence_id.json",
        "runtime_gpu_reranker_qwen35_4b_evidence_id_coverage_ablation.json",
    ],
)
def test_recommended_runtime_configs_keep_minirag_disabled(config_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "configs" / config_name).read_text(encoding="utf-8")
    )
    retrieval = payload["retrieval"]

    assert retrieval["enable_minirag"] is False
    assert retrieval["minirag_chapter_isolation"] is False
    assert retrieval["minirag_auto_second_retrieval"] is False
