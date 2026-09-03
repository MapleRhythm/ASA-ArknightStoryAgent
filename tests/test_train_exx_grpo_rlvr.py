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


def test_protocol_violation_penalty_is_graded() -> None:
    valid = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1"]}],
    }
    unknown_id = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E9"]}],
    }
    malformed = '{"next_action":"answer_directly","supported_facts":['
    values = MODULE.protocol_violation_penalty(
        [completion(valid), completion(unknown_id), malformed],
        [["E1"], ["E1"], ["E1"]],
    )
    assert values[0] == 0.0
    assert values[1] < 0.0
    assert values[2] == -1.5


def test_concise_fact_penalty_only_applies_to_long_valid_answers() -> None:
    short = {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "事实", "evidence_ids": ["E1"]}],
    }
    long = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": f"事实{i}", "evidence_ids": ["E1"]} for i in range(1, 7)
        ],
    }
    retrieve = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {
            "question": "还需要什么？",
            "query_type": "fact",
            "entities": [],
            "keywords": [],
            "expected_answer_type": "原因",
        },
    }
    values = MODULE.concise_fact_penalty(
        [completion(short), completion(long), completion(retrieve)]
    )
    assert values == [0.0, -0.5, 0.0]


def test_near_duplicate_penalty_catches_reworded_reward_padding() -> None:
    unique = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "阿米娅批准了申请。", "evidence_ids": ["E1"]},
            {"fact": "杜宾安排了训练。", "evidence_ids": ["E2"]},
        ],
    }
    padded = {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "拜松在货船上发现糖果，并质疑为何用船运糖果。", "evidence_ids": ["E1"]},
            {"fact": "拜松在货船上发现糖果，并质疑为什么用船运糖果。", "evidence_ids": ["E1"]},
        ],
    }

    penalties = MODULE.near_duplicate_fact_penalty(
        [completion(unique), completion(padded)]
    )

    assert penalties[0] == 0.0
    assert penalties[1] < 0.0


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
    renamed_funcs, renamed_weights = MODULE.build_rule_reward_stack("protocol-gated-rules")
    glm_funcs, glm_weights = MODULE.build_rule_reward_stack("glm-semantic-gated")
    precision_funcs, precision_weights = MODULE.build_rule_reward_stack(
        "glm-precision-gated"
    )

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
    assert [func.__name__ for func in renamed_funcs] == [
        func.__name__ for func in gated_funcs
    ]
    assert renamed_weights == gated_weights
    assert [func.__name__ for func in glm_funcs] == [
        "protocol_penalty",
        "gated_near_duplicate_fact_penalty",
    ]
    assert glm_weights == [2.0, 1.5]
    assert [func.__name__ for func in precision_funcs] == [
        func.__name__ for func in glm_funcs
    ]
    assert precision_weights == glm_weights
    assert "action_reward" not in {func.__name__ for func in glm_funcs}
    assert "reference_fact_reward" not in {func.__name__ for func in glm_funcs}
    structural_funcs, structural_weights = MODULE.build_rule_reward_stack(
        "glm-precision-structural"
    )
    assert [func.__name__ for func in structural_funcs] == [
        "protocol_penalty",
        "protocol_violation_penalty",
        "gated_near_duplicate_fact_penalty",
        "gated_concise_fact_penalty",
    ]
    assert structural_weights == [1.5, 1.5, 1.5, 0.75]


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


def test_action_length_stratification_covers_actions_and_context_boundaries() -> None:
    examples = []
    for action in MODULE.ACTIONS:
        for index, length in enumerate([10, 20, 30, 40]):
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
        examples, max_rows=6, selection_order="stratified-action-length", seed=1
    )

    assert [item.gold_action for item in selected] == list(MODULE.ACTIONS) * 2
    for action in MODULE.ACTIONS:
        assert [item.prompt_tokens for item in selected if item.gold_action == action] == [10, 40]


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
    assert args.glm_reward_weight is None


def test_glm_semantic_profile_requires_judge_and_dominant_weight(monkeypatch) -> None:
    parser = MODULE.build_parser()
    base = [
        "--train-file",
        "train.json",
        "--base-model",
        "base",
        "--sft-adapter",
        "adapter",
        "--output-dir",
        "out",
        "--reward-profile",
        "glm-semantic-gated",
    ]
    without_judge = parser.parse_args(base)
    try:
        MODULE.validate_runtime_args(without_judge)
    except ValueError as exc:
        assert "requires --glm-semantic-reward" in str(exc)
    else:
        raise AssertionError("semantic profile was allowed without GLM")

    monkeypatch.setenv("BIGMODEL_API_KEY", "secret")
    valid = parser.parse_args([*base, "--glm-semantic-reward"])
    MODULE.validate_runtime_args(valid)
    assert valid.glm_reward_weight == MODULE.GLM_SEMANTIC_DEFAULT_WEIGHT

    weak = parser.parse_args(
        [*base, "--glm-semantic-reward", "--glm-reward-weight", "1.5"]
    )
    try:
        MODULE.validate_runtime_args(weak)
    except ValueError as exc:
        assert "requires --glm-reward-weight" in str(exc)
    else:
        raise AssertionError("semantic profile accepted a non-dominant GLM weight")
