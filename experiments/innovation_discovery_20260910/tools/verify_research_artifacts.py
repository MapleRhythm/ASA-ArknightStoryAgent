#!/usr/bin/env python3
"""Recompute isolated probes, verify saved hashes, and run both test suites.

Does not download literature, invoke a model, change production, or regenerate
the data under verification. Writes only its own verification report.
"""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import probe_project_mechanisms as p

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main():
    checks = []

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    manifest = p.load(ROOT / "RUN_MANIFEST.json")
    source_tree = REPO / "src/asa_arknight_story_agent"
    check("probe_script_hash", manifest["script_sha256"] == p.file_sha(Path(p.__file__)))
    for key, expected in manifest["source_hashes"].items():
        check(f"source_hash:{key}", p.file_sha(source_tree / key) == expected)
    current_tree = {str(path.relative_to(source_tree)): p.file_sha(path)
                    for path in sorted(source_tree.rglob("*.py"))}
    check("full_production_python_tree_snapshot", current_tree == manifest["source_tree_hashes"])
    for key, expected in manifest["artifact_hashes"].items():
        check(f"artifact_hash:{key}", p.file_sha(ROOT / key) == expected)
    rows = p.synthetic_scope_rows() + p.natural_scope_rows()
    check("scope_data_recomputed", rows == p.load(ROOT / "data/attribution_scope_probes.json"))
    check("lexical_audit_recomputed",
          p.lexical_probe(rows) == p.load(ROOT / "audits/lexical_scope_probe.json"))
    check("full_query_replay_recomputed",
          p.replays() == p.load(ROOT / "audits/full_query_pipeline_replay.json"))
    check("no_probe_used_for_training", all(row["exclude_from_training"] for row in rows))
    check("no_new_model_retrieval_or_training",
          manifest["model_calls"] == manifest["retrieval_calls"] == 0 and not manifest["training"])

    literature_counts = {"retrieved_records": 0, "unverified_records": 0,
                         "complete_pdf_text_pairs": 0}
    for path in sorted((ROOT / "sources").rglob("manifest*.json")):
        for record in p.load(path):
            key = f"{path.parent.name}/{record['key']}"
            status = record["status"]
            check(f"literature_status:{key}", status in {"retrieved", "unverified"})
            literature_counts[f"{status}_records"] += 1
            if record.get("file"):
                local_file = ROOT / record["file"]
                check(f"literature_body_hash:{key}",
                      local_file.is_file() and p.file_sha(local_file) == record["sha256"])
            if status == "retrieved":
                check(f"literature_transport_success:{key}",
                      record["http_status"] == "200" and record["curl_exit_code"] == 0)
                if record.get("file", "").endswith(".pdf"):
                    check(f"literature_pdf_text_extracted:{key}",
                          record.get("pdftotext_exit_code") == 0 and bool(record.get("text_file")))
                    if record.get("text_file"):
                        local_text = ROOT / record["text_file"]
                        check(f"literature_text_hash:{key}",
                              local_text.is_file() and p.file_sha(local_text) == record["text_sha256"])
                        literature_counts["complete_pdf_text_pairs"] += 1

    owned_markdown = [ROOT / filename for filename in
                      ("README.md", "RESEARCH_REPORT.md", "PILOT_PROTOCOL.md",
                       "sources/READING_NOTES_20260910.md")]
    for path in owned_markdown:
        text = path.read_text()
        for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target_file = path.parent / target.split("#", 1)[0]
            # This report is written after checks complete.
            if target_file.resolve() == (ROOT / "audits/verification.json").resolve():
                continue
            check(f"markdown_target:{path.name}:{target}", target_file.exists())
        check(f"no_trailing_whitespace:{path.name}",
              all(line == line.rstrip() for line in text.splitlines()))

    suites = []
    for directory in (ROOT / "tests",
                      p.PREVIOUS / "phase2_deep_audit/tests"):
        command = [sys.executable, "-m", "unittest", "discover", "-s",
                   str(directory.relative_to(REPO)), "-v"]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=55)
        output = result.stdout + result.stderr
        match = re.search(r"Ran (\d+) tests?", output)
        count = int(match.group(1)) if match else None
        suites.append({"command": command, "returncode": result.returncode,
                       "tests_run": count, "output": output})
        check(f"test_suite:{directory.relative_to(REPO)}",
              result.returncode == 0 and count is not None)

    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO,
                          capture_output=True, text=True, timeout=15)
    check("git_diff_check", diff.returncode == 0, diff.stdout + diff.stderr)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=15)
    after_tree = {str(path.relative_to(source_tree)): p.file_sha(path)
                  for path in sorted(source_tree.rglob("*.py"))}
    check("verification_did_not_change_production_python", after_tree == current_tree)
    passed = all(item["passed"] for item in checks)
    report = {
        "verified_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "passed" if passed else "failed",
        "python_version": sys.version,
        "git_head": revision.stdout.strip(),
        "scope": "artifact integrity and deterministic regression, not semantic independent review or QA accuracy",
        "run_manifest_sha256": p.file_sha(ROOT / "RUN_MANIFEST.json"),
        "report_and_tool_hashes": {
            str(path.relative_to(ROOT)): p.file_sha(path)
            for path in sorted([*owned_markdown, *ROOT.glob("tools/*.py"), *ROOT.glob("tests/*.py")])
        },
        "literature_counts": literature_counts,
        "total_tests_run": sum(suite["tests_run"] or 0 for suite in suites),
        "checks": checks,
        "test_suites": suites,
    }
    p.write(ROOT / "audits/verification.json", report)
    print(json.dumps({key: report[key] for key in
                      ("status", "total_tests_run", "literature_counts")}, ensure_ascii=False, indent=2))
    for item in checks:
        if not item["passed"]:
            print("FAILED:", item["name"], item["detail"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
