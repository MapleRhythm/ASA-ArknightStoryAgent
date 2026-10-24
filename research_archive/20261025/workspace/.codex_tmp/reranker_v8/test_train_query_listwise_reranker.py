from __future__ import annotations

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path


_PROJECT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPTS_DIR = (
    _PROJECT_SCRIPTS_DIR
    if (_PROJECT_SCRIPTS_DIR / "train_query_listwise_reranker.py").exists()
    else Path(__file__).resolve().parent
)
sys.path.insert(0, str(SCRIPTS_DIR))

# Candidate selection itself has no torch/transformers dependency.  Lightweight
# stubs keep this unit test runnable on data-generation hosts where the training
# environment is intentionally absent.
torch_stub = types.ModuleType("torch")
torch_stub.nn = types.SimpleNamespace()
torch_stub.nn.functional = types.ModuleType("torch.nn.functional")
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.nn.functional", torch_stub.nn.functional)
torch_utils = types.ModuleType("torch.utils")
torch_utils_data = types.ModuleType("torch.utils.data")
class DatasetStub:
    @classmethod
    def __class_getitem__(cls, _item: object) -> type["DatasetStub"]:
        return cls


torch_utils_data.Dataset = DatasetStub
sys.modules.setdefault("torch.utils", torch_utils)
sys.modules.setdefault("torch.utils.data", torch_utils_data)
transformers_stub = types.ModuleType("transformers")
for name in ("AutoModelForSequenceClassification", "AutoTokenizer", "Trainer", "TrainingArguments"):
    setattr(transformers_stub, name, type(name, (), {}))
transformers_stub.set_seed = lambda _: None
sys.modules.setdefault("transformers", transformers_stub)

module_path = SCRIPTS_DIR / "train_query_listwise_reranker.py"
spec = importlib.util.spec_from_file_location("train_query_listwise_reranker", module_path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
TEACHER_GAIN = module.TEACHER_GAIN
select_group_candidates = module.select_group_candidates


def candidate(label: str, rank: int) -> dict[str, object]:
    return {
        "candidate_id": f"C{rank:03d}",
        "text": f"{label}-{rank}",
        "label": label,
        "teacher_rank": rank,
        "relevance": TEACHER_GAIN[label],
    }


def test_group16_prefers_graded_positives_and_hard_context() -> None:
    candidates = [
        *(candidate("direct_answer", rank) for rank in range(1, 5)),
        *(candidate("partial_answer", rank) for rank in range(5, 9)),
        *(candidate("supporting_context", rank) for rank in range(9, 13)),
        *(candidate("hard_context", rank) for rank in range(13, 33)),
        *(candidate("irrelevant", rank) for rank in range(33, 121)),
    ]

    selected = select_group_candidates(
        "测试问题", candidates, max_group_size=16, seed=20260822
    )
    counts = Counter(str(item["label"]) for item in selected)

    assert len(selected) == 16
    assert sum(counts[label] for label in ("direct_answer", "partial_answer", "supporting_context")) == 6
    assert all(counts[label] >= 1 for label in ("direct_answer", "partial_answer", "supporting_context"))
    assert counts["hard_context"] == 8
    assert counts["irrelevant"] == 2
    assert [
        int(item["teacher_rank"])
        for item in selected
        if item["label"] == "hard_context"
    ] == list(range(13, 21))


def test_missing_positive_tier_backfills_without_losing_hard_negatives() -> None:
    candidates = [
        candidate("direct_answer", 1),
        candidate("partial_answer", 2),
        *(candidate("hard_context", rank) for rank in range(3, 13)),
        *(candidate("irrelevant", rank) for rank in range(13, 31)),
    ]

    selected = select_group_candidates("测试问题", candidates, max_group_size=16, seed=7)
    counts = Counter(str(item["label"]) for item in selected)

    assert len(selected) == 16
    assert counts["direct_answer"] == 1
    assert counts["partial_answer"] == 1
    assert counts["hard_context"] == 10
    assert counts["irrelevant"] == 4


def test_selection_is_deterministic() -> None:
    candidates = [
        *(candidate("direct_answer", rank) for rank in range(1, 4)),
        *(candidate("hard_context", rank) for rank in range(4, 20)),
        *(candidate("irrelevant", rank) for rank in range(20, 60)),
    ]
    first = select_group_candidates("同一问题", candidates, max_group_size=16, seed=42)
    second = select_group_candidates("同一问题", candidates, max_group_size=16, seed=42)
    assert [item["candidate_id"] for item in first] == [
        item["candidate_id"] for item in second
    ]
