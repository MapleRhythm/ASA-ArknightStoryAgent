"""CPU-only check of actual generation entry point, arguments and pair controls."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import run_canonical_reader_v2 as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


class DryGenerator:
    def __init__(self, config, item):
        self.max_tokens = config["max_tokens"]
        self.repeat_penalty = config["repeat_penalty"]
        self.item = item
        self.calls = []

    def generate(self, prompt, **kwargs):
        assert prompt == self.item["prompt"]
        assert kwargs == {
            "max_tokens": 2048, "temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.05,
        }
        self.calls.append(kwargs)
        return '{"next_action":"abstain","final_answer":"现有证据不足以确认。"}'


def main():
    pre = shared.load(HERE / "preflight.json")
    probe.verify(pre)
    helpers, frozen = probe.runtime_imports(), probe.sources()
    checks = []
    for item in pre["conditions"]:
        source = frozen[(item["id"], item["condition"])]
        generator = DryGenerator(pre["generator"], item)
        pipeline = probe.pipeline_for(pre, helpers, generator)
        assert pipeline.retriever is None and not pipeline.web_context_config.enabled
        result, trace = probe.invoke(pipeline, helpers, item, source)
        assert len(generator.calls) == 1
        assert trace[-1]["evidence_packing"] == item["packing_trace"]
        rendered = probe.render(pipeline, helpers, item["hypothesis"], {**source, "question": item["question"]})
        for key in rendered:
            assert rendered[key] == item[key]
        assert result["generation_diagnostics"]["raw_output"] == '{"next_action":"abstain","final_answer":"现有证据不足以确认。"}'
        checks.append({"id": item["id"], "condition": item["condition"], "status": "actual_first_round_path_matches"})
    by_key = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    controls = []
    for case in ("fixed58-017", "fixed58-023"):
        a, b = [by_key[(case, kind)] for kind in ("control", "candidate")]
        assert a["hypothesis"] == b["hypothesis"] and a["prompt"] == b["prompt"]
        assert a["visible_evidence"] == b["visible_evidence"]
        assert frozen[(case, "control")]["evidence"] == frozen[(case, "candidate")]["evidence"]
        controls.append(case)
    v40pre = shared.load(probe.V40 / "preflight.json")
    for case in v40pre["rows"]:
        a, b = [by_key[(case["id"], kind)] for kind in ("control", "candidate")]
        assert a["hypothesis"] == case["control"] and b["hypothesis"] == case["candidate"]
        for key in a["hypothesis"].keys() - {"entities", "keywords"}:
            assert a["hypothesis"][key] == b["hypothesis"][key]
    assert len(pre["pairs"]) == 12
    assert all({*pair["condition_order"]} == {"control", "candidate"} for pair in pre["pairs"])
    shared.save(HERE / "protocol_check_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "canonical_reader_first_round_cpu_protocol_check",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "check_script_sha256": shared.sha(Path(__file__)),
        "checks": checks, "identical_input_controls": controls,
        "fake_abstain_used_only_for_CPU_protocol_check": True,
        "model_calls": 0, "retrieval_calls": 0, "production_changed": False,
        "training_ready": False,
    })
    print({"actual_method_dry_checks": len(checks), "identical_input_controls": controls, "model_calls": 0})


if __name__ == "__main__":
    main()
