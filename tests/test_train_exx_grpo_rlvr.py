import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_exx_grpo_rlvr.py"
SPEC = importlib.util.spec_from_file_location("train_exx_grpo_rlvr", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def completion(payload):
    return [{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}]


def test_strict_schema_and_rewards_distinguish_unknown_evidence() -> None:
    good = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "阿米娅采取了行动。", "evidence_ids": ["E2"]}],
    }
    bad = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "阿米娅采取了行动。", "evidence_ids": ["E9"]}],
    }

    assert MODULE.validate_payload(good, {"E1", "E2"}) == []
    assert "fact_1_unknown_id" in MODULE.validate_payload(bad, {"E1", "E2"})
    assert MODULE.schema_reward([completion(good), completion(bad)], [["E1", "E2"], ["E1", "E2"]]) == [
        1.0,
        0.0,
    ]


def test_claim_citation_reward_binds_each_fact_to_its_hidden_teacher_ids() -> None:
    exact = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1", "E2"]}],
    }
    partial = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1", "E3"]}],
    }

    rewards = MODULE.claim_citation_reward(
        [completion(exact), completion(partial)],
        ["answer_directly", "answer_directly"],
        [
            [{"fact": "事实", "evidence_ids": ["E1", "E2"]}],
            [{"fact": "事实", "evidence_ids": ["E1", "E2"]}],
        ],
    )

    assert rewards == [1.0, 1 / 3]


def test_claim_matching_uses_exact_maximum_not_greedy_pairs() -> None:
    # Greedy takes 9 at (0, 1), leaving 6, while the optimum is 9 + 8.
    assert MODULE.maximum_bipartite_score([[9, 9], [6, 8]]) == 17


def test_reference_fact_reward_is_continuous_and_reference_based() -> None:
    exact = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "阿米娅带领罗德岛撤离。", "evidence_ids": ["E1"]}],
    }
    partial = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "阿米娅带领队伍前进。", "evidence_ids": ["E1"]}],
    }
    unrelated = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "天气晴朗。", "evidence_ids": ["E1"]}],
    }
    rewards = MODULE.reference_fact_reward(
        [completion(exact), completion(partial), completion(unrelated)],
        ["answer_directly"] * 3,
        [[{"fact": "阿米娅带领罗德岛撤离。", "evidence_ids": ["E1"]}]] * 3,
    )

    assert rewards[0] == 1.0
    assert rewards[0] > rewards[1] > rewards[2]


def test_duplicate_fact_penalty_rejects_reward_padding() -> None:
    unique = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "事实甲", "evidence_ids": ["E1"]},
            {"fact": "事实乙", "evidence_ids": ["E2"]},
        ],
    }
    padded = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "事实甲", "evidence_ids": ["E1"]},
            {"fact": "事实甲。", "evidence_ids": ["E1"]},
        ],
    }

    assert MODULE.duplicate_fact_penalty(
        [completion(unique), completion(padded)], ["answer_directly", "answer_directly"]
    ) == [0.0, -0.5]
    assert "fact_2_duplicate" in MODULE.validate_payload(padded, {"E1"})


def test_premature_answer_is_penalized_but_correct_retrieve_is_not() -> None:
    answer = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "猜测", "evidence_ids": ["E1"]}],
    }
    retrieve = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {
            "question": "还缺什么？",
            "query_type": "fact",
            "entities": [],
            "keywords": [],
            "expected_answer_type": "原因",
        },
    }

    assert MODULE.premature_answer_penalty(
        [completion(answer), completion(retrieve)], ["retrieve_more", "retrieve_more"]
    ) == [-1.0, 0.0]
    assert MODULE.action_reward(
        [completion(answer), completion(retrieve)], ["retrieve_more", "retrieve_more"]
    ) == [0.0, 1.0]


def test_semantic_gated_profile_masks_positive_credit_for_invalid_payloads() -> None:
    valid = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1"]}],
    }
    invalid = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E99"]}],
    }
    gated_action = MODULE.protocol_gated(MODULE.action_reward)

    assert MODULE.protocol_penalty(
        [completion(valid), completion(invalid)], [["E1"], ["E1"]]
    ) == [0.0, -1.0]
    assert gated_action(
        [completion(valid), completion(invalid)],
        [["E1"], ["E1"]],
        gold_action=["answer_directly", "answer_directly"],
    ) == [1.0, 0.0]


def test_reward_profiles_preserve_legacy_and_name_gated_components() -> None:
    legacy_funcs, legacy_weights = MODULE.build_rule_reward_stack("legacy")
    gated_funcs, gated_weights = MODULE.build_rule_reward_stack("semantic-gated")

    assert [func.__name__ for func in legacy_funcs] == [
        "json_reward",
        "schema_reward",
        "action_reward",
        "claim_citation_reward",
        "reference_fact_reward",
        "duplicate_fact_penalty",
        "premature_answer_penalty",
    ]
    assert legacy_weights == [1.0, 1.0, 1.5, 1.5, 1.0, 0.5, 0.5]
    assert [func.__name__ for func in gated_funcs] == [
        "protocol_penalty",
        "gated_action_reward",
        "gated_claim_citation_reward",
        "gated_reference_fact_reward",
        "duplicate_fact_penalty",
        "premature_answer_penalty",
    ]
    assert gated_weights == [1.0, 0.75, 1.0, 1.0, 0.75, 1.0]


def test_training_diagnostic_reports_pre_clip_bound_and_coefficient() -> None:
    clipped = MODULE.build_training_diagnostic(
        {"grad_norm": 4.0, "reward": 2.5, "epoch": 0.1}, max_grad_norm=2.0
    )
    untouched = MODULE.build_training_diagnostic({"grad_norm": 0.5}, max_grad_norm=2.0)

    assert clipped["grad_norm_pre_clip"] == 4.0
    assert clipped["grad_norm_post_clip_bound"] == 2.0
    assert clipped["grad_clip_coefficient"] == 0.5
    assert clipped["grad_was_clipped"] is True
    assert untouched["grad_norm_post_clip_bound"] == 0.5
    assert untouched["grad_clip_coefficient"] == 1.0
    assert untouched["grad_was_clipped"] is False


class BatchEncodingLike:
    input_ids = [[1, 2, 3, 4]]


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return BatchEncodingLike()


def test_tokenized_length_does_not_count_batch_encoding_keys() -> None:
    assert MODULE.tokenized_length(FakeTokenizer(), [{"role": "user", "content": "x"}]) == 4


class MappingBatchEncoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class MappingTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return MappingBatchEncoding(input_ids=[1, 2, 3, 4, 5], attention_mask=[1] * 5)


def test_tokenized_length_handles_mapping_batch_encoding() -> None:
    assert MODULE.tokenized_length(MappingTokenizer(), [{"role": "user", "content": "x"}]) == 5


def test_stratified_shortest_smoke_covers_all_actions() -> None:
    examples = []
    for action, lengths in {
        "answer_directly": [30, 10],
        "retrieve_more": [31, 11],
        "abstain": [32, 12],
    }.items():
        for index, length in enumerate(lengths):
            examples.append(
                MODULE.TrainingExample(
                    row_id=f"{action}-{index}",
                    prompt=[],
                    prompt_tokens=length,
                    visible_ids=["E1"],
                    gold_action=action,
                    gold_fact_bindings=[],
                )
            )

    selected = MODULE.select_examples(
        examples, max_rows=3, selection_order="stratified-shortest", seed=1
    )

    assert {item.gold_action for item in selected} == set(MODULE.ACTIONS)
    assert [item.prompt_tokens for item in selected] == [10, 11, 12]


def test_stratified_longest_smoke_selects_boundary_rows() -> None:
    examples = []
    for action, lengths in {
        "answer_directly": [30, 10],
        "retrieve_more": [31, 11],
        "abstain": [32, 12],
    }.items():
        for index, length in enumerate(lengths):
            examples.append(
                MODULE.TrainingExample(
                    row_id=f"{action}-{index}",
                    prompt=[],
                    prompt_tokens=length,
                    visible_ids=["E1"],
                    gold_action=action,
                    gold_fact_bindings=[],
                )
            )

    selected = MODULE.select_examples(
        examples, max_rows=3, selection_order="stratified-longest", seed=1
    )

    assert [item.prompt_tokens for item in selected] == [30, 31, 32]


def test_glm_semantic_reward_defaults_to_evidence_only_coding_plan() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--train-file",
            "train.json",
            "--base-model",
            "base",
            "--sft-adapter",
            "adapter",
            "--output-dir",
            "out",
            "--glm-semantic-reward",
        ]
    )

    assert args.glm_endpoint == "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    assert args.glm_model == "glm-5.3"
    assert args.reward_profile == "legacy"
    assert args.max_grad_norm == 1.0
    assert args.warmup_ratio == 0.0
    assert args.lr_scheduler_type == "linear"
    assert args.glm_reasoning_effort == "medium"
    assert args.glm_timeout == 180.0
    assert args.glm_max_tokens == 4096
    assert args.glm_max_attempts == 1
    assert args.glm_reward_weight == 1.0
