"""Read-only audit of canonical fallback trace completeness.

Replays all14 completed v45 second-round states, then two explicitly synthetic
failure-path checks. No new generation, source patch, action change or labels.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V45 = HERE.parent / "staged_round2_reader_20260910.v45"
sys.path.insert(0, str(V45))
import run_second_round_reader as probe
shared = probe.shared


def string_paths(value, needle, path="$"):
    found = []
    if isinstance(value, str) and needle in value:
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(string_paths(child, needle, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(string_paths(child, needle, f"{path}[{index}]"))
    return found


def prepare():
    assert not (HERE / "preflight.json").exists()
    pre45 = shared.load(V45 / "preflight.json")
    probe.reader.verify(pre45)
    close45 = shared.load(V45 / "checkpoint_v45.json")
    assert close45["round2_canonical_replays"] == 14 and close45["actual_new_llm_calls"] == 23
    paths = [
        Path(__file__), V45 / "run_second_round_reader.py", V45 / "preflight.json",
        V45 / "generations.json", V45 / "postflight_v1.json",
        V45 / "primary_fallback_audit_v1.json", V45 / "checkpoint_v45.json",
        probe.V44 / "preflight.json", probe.V44 / "retrieval.json",
        *sorted(probe.reader.SRC.rglob("*.py")),
    ]
    shared.save(HERE / "preflight.json", {
        "version": "canonical_final_fallback_trace_completeness_v47",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {str(p): shared.sha(p) for p in paths},
        "protocol": [
            "CPU replay all14 actual second-round methods using saved v45 raws/kwargs and exact cached states.",
            "Compare returned action/answer/conclusion/trace/state digest byte-for-byte to v45.",
            "Inspect the canonical FINAL round trace for the complete primary raw and individual warning strings.",
            "Do not confuse v45's external request ledger/CPU observation with ordinary canonical trace fields.",
            "A warning may survive in final output coincidentally; only exact raw and structured primary record prove attempt retention.",
            "Two explicit synthetic failure fixtures: rejected primary+failing fallback, and failing primary+failing fallback.",
            "Synthetic exceptions are CPU protocol checks only, not actual model failures or replacement v45 outcomes.",
            "No code patch, custom model retry, validator change, new model/retrieval/GPU call, training, label change or production promotion.",
        ],
        "production_changed": False, "training_ready": False, "gpu_used": False,
    })
    print({"actual_state_replays_planned": 14, "synthetic_failure_protocol_cases": 2,
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


class FailingFallback(probe.ReplayGenerator):
    def __init__(self, config, item, saved, fail_primary):
        super().__init__(config, item, saved)
        self.fail_primary = fail_primary
        self.requests = []

    def generate(self, prompt, **kwargs):
        index = self.calls
        planned = self.item["planned_requests"][index]
        assert prompt == planned["prompt"] and kwargs == planned["kwargs"]
        self.requests.append({"kind": planned["kind"], "prompt_sha256": shared.digest(prompt), "kwargs": kwargs})
        if index == 1:
            self.calls += 1
            raise ValueError("synthetic_final_direct_fallback_error_v47")
        if self.fail_primary:
            self.calls += 1
            raise ValueError("synthetic_primary_generation_error_v47")
        return super().generate(prompt, **kwargs)


def run():
    pre = shared.load(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    pre45 = shared.load(V45 / "preflight.json")
    gen45 = shared.load(V45 / "generations.json")
    primary = {
        (r["pair_id"], r["condition"]): r
        for r in shared.load(V45 / "primary_fallback_audit_v1.json")["rows"]
    }
    helpers, frozen = probe.staged.imports(), probe.sources()
    rows, summary = [], Counter()
    for item, saved in zip(pre45["records"], gen45["records"], strict=True):
        if item["terminal_after_round1"]:
            continue
        key = item["pair_id"], item["condition"]
        generator = probe.ReplayGenerator(pre45["generator"], item, saved)
        replay = probe.invoke(pre45, helpers, item, frozen[key], generator)
        assert all(saved[k] == value for k, value in replay.items())
        assert generator.calls == len(saved["requests"])
        final_step = saved["retrieval_trace"][-1]
        before = primary[key]["primary_conclusion"]
        assert before is not None
        raw = saved["requests"][0]["sanitized_raw_output"]
        paths = string_paths(final_step, raw)
        warning_paths = {w: string_paths(final_step, w) for w in before["grounding_warnings"]}
        fallback = len(saved["requests"]) == 2
        if fallback:
            fallback_raw = saved["requests"][1]["sanitized_raw_output"]
            assert raw != fallback_raw
            after_paths = string_paths(final_step, fallback_raw)
            assert after_paths
            summary["fallback_states"] += 1
            summary["fallback_states_missing_primary_raw"] += not bool(paths)
        else:
            after_paths = []
            assert paths
            summary["no_fallback_primary_raw_retained"] += 1
        rows.append({
            "pair_id": key[0], "condition": key[1], "canonical_replay_exact": True,
            "canonical_fallback_called": fallback,
            "primary_action": before["next_action"], "final_action": saved["conclusion"]["next_action"],
            "primary_raw_sha256": shared.digest(raw), "primary_raw_paths_in_final_round_trace": paths,
            "fallback_raw_paths_in_final_round_trace": after_paths,
            "primary_warning_paths_in_final_round_trace": warning_paths,
            "primary_warnings_not_present_in_final_round_trace": [w for w, found in warning_paths.items() if not found],
            "v45_out_of_band_requests_preserved": len(saved["requests"]),
        })
    assert len(rows) == 14
    assert summary == {
        "fallback_states": 9, "fallback_states_missing_primary_raw": 9, "no_fallback_primary_raw_retained": 5,
    }
    key = "fixed58-010:seed0", "control"
    item = next(r for r in pre45["records"] if (r["pair_id"], r["condition"]) == key)
    saved = next(r for r in gen45["records"] if (r["pair_id"], r["condition"]) == key)
    synthetic = []
    for fail_primary in (False, True):
        generator = FailingFallback(pre45["generator"], item, saved, fail_primary)
        captured = io.StringIO()
        original_source = deepcopy(frozen[key])
        with redirect_stdout(captured):
            replay = probe.invoke(pre45, helpers, item, frozen[key], generator)
        assert original_source == frozen[key] and generator.calls == 2
        log = captured.getvalue()
        assert "synthetic_final_direct_fallback_error_v47" in log
        step = replay["retrieval_trace"][-1]
        assert not string_paths(step, "synthetic_final_direct_fallback_error_v47")
        primary_error_paths = string_paths(step, "synthetic_primary_generation_error_v47")
        assert bool(primary_error_paths) == fail_primary
        if not fail_primary:
            assert replay["conclusion"] == primary[key]["primary_conclusion"]
        synthetic.append({
            "fixture": "both_calls_raise" if fail_primary else "rejected_primary_fallback_raises",
            "synthetic_only": True, "actual_model_calls": 0,
            "requests": generator.requests, "captured_stdout": log,
            "fallback_exception_retained_in_trace": False,
            "primary_exception_paths_in_trace": primary_error_paths,
            "final_action": replay["conclusion"]["next_action"],
            "final_answer": replay["final_answer"],
            "trace_sha256": probe.staged.object_digest(replay["retrieval_trace"]),
        })
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    shared.save(HERE / "audit_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "actual_state_replays": 14, "rows": rows, "summary_not_accuracy": summary,
        "synthetic_failure_protocol_cases": synthetic,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "production_changed": False, "training_ready": False,
        "diagnosis": "Final-direct replacement removes primary validation-rejected conclusion/raw from ordinary trace; synthetic fallback exception remains stdout-only.",
        "not_accuracy_improvement": True,
    })
    print({"state_replays": 14, "summary_not_accuracy": summary,
           "synthetic_protocol_cases": len(synthetic), "production_changed": False}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_fallback_trace_audit.py prepare|run")
    operations[sys.argv[1]]()
