"""Offline fact-level verifier application and counterfactual accounting.

No model is called here.  Three subcommands:

apply           Apply per-fact verifier verdicts to generated answers (one JSONL
                line per question, field ``raw_text``) under an explicit
                deletion policy, validating each verdict's evidence anchor
                against the task passages.  Writes filtered answers and an
                aggregate report that contains counts only.
counterfactual  Score annotated cases under an oracle verifier and under a
                grid of imperfect verifiers (false-deletion rate q on correct
                facts, catch rate r on wrong facts).  Annotations are
                assistant judgments; outputs are upper bounds / expectations,
                never model accuracy.
validate        Check a verdict file against the answers without applying it.

See SPEC.md for the verdict contract and the policy rationale.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

VERDICTS = {"supported", "unsupported", "binding_error", "uncertain"}
DELETABLE = {"unsupported", "binding_error"}
ROLES = {
    "speaker", "addressee", "participant", "possessor", "identity",
    "phase", "scene", "polarity", "relation", "other",
}
STATUS = ("unchanged", "pruned", "clause_repaired", "regenerate")

_FACT_OBJ = re.compile(r'\{[^{}]*?"fact"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*?\}', re.S)
_EIDS = re.compile(r'"evidence_ids"\s*:\s*\[([^\]]*)\]')
_ACTION = re.compile(r'"next_action"\s*:\s*"([a-z_]+)"')
_LABEL = re.compile(r"^([^：:\s]{1,16})[：:]")
_NOISE = re.compile(r"[\s\u3000，。、“”‘’！？…—（）()《》「」：:；;,.!?\"'\-]")


class ContractError(ValueError):
    """Input violates the verdict/answer contract; never silently skipped."""


# ---------------------------------------------------------------- answers

def parse_answer(raw: str) -> dict[str, Any]:
    """Extract next_action and ordered facts from a possibly invalid JSON answer."""
    try:
        obj = json.loads(raw)
        facts = [
            {"text": f["fact"], "evidence_ids": list(f.get("evidence_ids", []))}
            for f in obj.get("supported_facts", [])
        ]
        return {"next_action": obj.get("next_action"), "facts": facts, "valid_json": True}
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
        pass
    facts = []
    for m in _FACT_OBJ.finditer(raw):
        block = m.group(0)
        e = _EIDS.search(block)
        eids = re.findall(r"E\d+", e.group(1)) if e else []
        facts.append({"text": json.loads('"' + m.group(1) + '"'), "evidence_ids": eids})
    a = _ACTION.search(raw)
    return {"next_action": a.group(1) if a else None, "facts": facts, "valid_json": False}


def render_answer(next_action: str | None, facts: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "next_action": next_action,
            "supported_facts": [
                {"fact": f["text"], "evidence_ids": f["evidence_ids"]} for f in facts
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------- anchors

def _norm(s: str) -> str:
    return _NOISE.sub("", unicodedata.normalize("NFKC", s))


def _alias_equal(a: str, b: str, aliases: dict[str, set[str]]) -> bool:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return b in aliases.get(a, set()) or a in aliases.get(b, set())


def load_aliases(path: str | None) -> dict[str, set[str]]:
    """Alias file: JSON list of groups, e.g. [["name-a", "name-b"], ...]."""
    if not path:
        return {}
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for g in groups:
        for n in g:
            out.setdefault(n, set()).update(x for x in g if x != n)
    return out


def quote_line_label(doc_text: str, quote: str) -> tuple[bool, str | None, str | None]:
    """Return (found, explicit label of the quote's line, inherited label).

    The explicit label is None when the line has no ``name：`` prefix.  More
    than half of the passage lines in this corpus are unlabeled: narration,
    continuation of the previous speaker, or an interlocutor whose name was
    not rendered.  The inherited label is the nearest explicit label above the
    quote in the same passage; it is the continuation reading, which is right
    for most unlabeled lines but wrong when an unnamed interlocutor answers.
    """
    q = _norm(quote)
    if not q:
        return False, None, None
    inherited = None
    for line in doc_text.splitlines():
        m = _LABEL.match(line)
        if q in _norm(line):
            return True, (m.group(1) if m else None), (m.group(1) if m else inherited)
        if m:
            inherited = m.group(1)
    if q in _norm(doc_text):  # quote spans lines
        return True, None, None
    return False, None, None


def validate_anchor(verdict: dict[str, Any], docs: dict[str, str],
                    aliases: dict[str, set[str]], unlabeled: str = "inherit") -> tuple[bool, str]:
    """Deterministically check that a deletion verdict is grounded.

    Accepted when the quote exists verbatim (modulo punctuation/whitespace) in
    the named passage, and, for binding errors, the two role values differ.
    For speaker errors the evidence-side speaker must match the quote line's
    explicit label; on an unlabeled line the policy decides:
    ``strict`` rejects, ``inherit`` requires the inherited label to match,
    ``permissive`` accepts.
    """
    anchor = verdict.get("anchor") or {}
    eid, quote = anchor.get("evidence_id"), anchor.get("quote", "")
    if eid not in docs:
        return False, "anchor_evidence_missing"
    found, label, inherited = quote_line_label(docs[eid], quote)
    if not found:
        return False, "anchor_quote_not_in_evidence"
    if verdict["verdict"] != "binding_error":
        return True, "ok"
    ev, cl = anchor.get("evidence_role_value"), anchor.get("claim_role_value")
    if not ev or not cl:
        return False, "binding_roles_missing"
    if _alias_equal(ev, cl, aliases):
        return False, "binding_roles_identical"
    if verdict.get("binding_role") != "speaker":
        return True, "ok"
    if label is not None:
        return (True, "ok") if _alias_equal(label, ev, aliases) \
            else (False, "label_contradicts_evidence_role")
    if unlabeled == "permissive":
        return True, "ok_unlabeled_permissive"
    if unlabeled == "inherit" and inherited is not None and _alias_equal(inherited, ev, aliases):
        return True, "ok_unlabeled_inherited"
    return False, "unlabeled_speaker_unverified"


# ---------------------------------------------------------------- verdicts

def load_verdicts(path: str) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        v = json.loads(line)
        for k in ("position", "fact_index", "verdict"):
            if k not in v:
                raise ContractError(f"verdict line {n}: missing {k}")
        if v["verdict"] not in VERDICTS:
            raise ContractError(f"verdict line {n}: unknown verdict {v['verdict']!r}")
        if v.get("binding_role") is not None and v["binding_role"] not in ROLES:
            raise ContractError(f"verdict line {n}: unknown binding_role {v['binding_role']!r}")
        c = v.get("confidence")
        if c is not None and not 0.0 <= float(c) <= 1.0:
            raise ContractError(f"verdict line {n}: confidence outside [0,1]")
        span = v.get("error_span")
        if span is not None and (len(span) != 2 or not 0 <= span[0] < span[1]):
            raise ContractError(f"verdict line {n}: malformed error_span")
        key = (int(v["position"]), int(v["fact_index"]))
        if key in out:
            raise ContractError(f"verdict line {n}: duplicate verdict for {key}")
        out[key] = v
    return out


def _excise(text: str, span: list[int], min_chars: int, replacement: str = "") -> str | None:
    s, e = span
    if e > len(text):
        raise ContractError("error_span beyond fact text")
    rest = (text[:s] + replacement + text[e:]).strip()
    rest = re.sub(r"[，、；,;]{2,}", "，", rest)
    rest = re.sub(r"^[，、；,;。]+", "", rest)
    return rest if len(_norm(rest)) >= min_chars else None


def apply_policy(answer: dict[str, Any], verdicts: dict[int, dict[str, Any]],
                 docs: dict[str, str], *, mode: str = "fact", threshold: float = 0.9,
                 require_anchor: bool = True, min_clause_chars: int = 8,
                 aliases: dict[str, set[str]] | None = None,
                 unlabeled: str = "inherit") -> dict[str, Any]:
    """Apply verdicts (keyed by 1-based fact index) to one parsed answer."""
    aliases = aliases or {}
    n = len(answer["facts"])
    for idx in verdicts:
        if not 1 <= idx <= n:
            raise ContractError(f"verdict for fact {idx} but answer has {n} facts")
    kept, deleted, repaired, rejected, duplicates = [], [], [], [], []
    first_seen: dict[str, int] = {}
    for i, fact in enumerate(answer["facts"], 1):
        key = _norm(fact["text"])
        if key in first_seen:
            # An exact repeat carries no information; drop it regardless of
            # its verdict so a verifier only needs to judge unique facts.
            duplicates.append({"fact_index": i, "dup_of": first_seen[key]})
            continue
        first_seen[key] = i
        v = verdicts.get(i)
        if v is None or v["verdict"] not in DELETABLE:
            kept.append(fact)
            continue
        conf = float(v.get("confidence", 1.0))
        if conf < threshold:
            rejected.append({"fact_index": i, "reason": "below_threshold"})
            kept.append(fact)
            continue
        if require_anchor:
            ok, why = validate_anchor(v, docs, aliases, unlabeled)
            if not ok:
                rejected.append({"fact_index": i, "reason": why})
                kept.append(fact)
                continue
        span = v.get("error_span")
        if v.get("error_text"):
            et = v["error_text"]
            if fact["text"].count(et) != 1:
                raise ContractError(f"fact {i}: error_text must occur exactly once in the fact")
            span = [fact["text"].index(et), fact["text"].index(et) + len(et)]
        if mode == "clause" and span:
            # Repair is either excision of the error clause or a pure role
            # substitution: inside the error text, the claimed role value is
            # replaced by the anchored evidence-side value.  No free rewrite.
            repl: str | None = ""
            if v.get("repair") == "substitute":
                anchor = v.get("anchor") or {}
                cl, ev = anchor.get("claim_role_value"), anchor.get("evidence_role_value")
                seg = fact["text"][span[0]:span[1]]
                if cl and ev and cl in seg:
                    repl = seg.replace(cl, ev)
                else:
                    rejected.append({"fact_index": i, "reason": "substitution_role_not_in_error_text"})
                    repl = None
            rest = None if repl is None else _excise(fact["text"], span, min_clause_chars, repl)
            if rest is not None:
                kept.append({**fact, "text": rest})
                repaired.append({"fact_index": i, "verdict": v["verdict"],
                                 "binding_role": v.get("binding_role"),
                                 "repair": v.get("repair", "excise")})
                continue
        deleted.append({"fact_index": i, "verdict": v["verdict"],
                        "binding_role": v.get("binding_role"),
                        "addresses_question": v.get("addresses_question")})
    # The verifier marks facts that carry the answer to the question.  If every
    # such fact was deleted (none survived intact or as a repaired clause),
    # pruning would ship an answer without its core: route to regeneration.
    deleted_idx = {d["fact_index"] for d in deleted}
    core_idx = {i for i, v in verdicts.items() if v.get("addresses_question") is True}
    core_lost = bool(core_idx) and core_idx <= deleted_idx
    if not deleted and not repaired and not duplicates:
        status = "unchanged"
    elif not kept or core_lost:
        status = "regenerate"
    elif repaired:
        status = "clause_repaired"
    else:
        status = "pruned"
    return {"status": status, "facts": kept, "deleted": deleted,
            "repaired": repaired, "rejected": rejected, "duplicates": duplicates}


# ---------------------------------------------------------------- counterfactual

def _case_probability(case: dict[str, Any], q: float, r: float, mode: str) -> float:
    """P(case is complete) with independent per-fact verifier errors.

    Wrong facts must all be removed (fact mode) or removed/repaired (clause
    mode); each is handled with probability r.  Correct facts are falsely
    deleted with probability q.  Every required point needs a surviving
    coverer: a correct fact listing it in ``covers`` or, in clause mode, a
    repaired wrong fact listing it in ``covers_after_repair``.
    """
    # Exact duplicates are removed deterministically before verification, so
    # they neither need catching nor can be falsely deleted independently.
    facts = [f for f in case["facts"] if f.get("dup_of") is None]
    required = set(case.get("required_points", []))
    wrong = [f for f in facts if f["label"] != "supported"]
    correct = [f for f in facts if f["label"] == "supported"]
    p_wrong = r ** len(wrong)
    if p_wrong == 0.0:
        return 0.0
    repaired_cover: set[str] = set()
    if mode == "clause":
        for f in wrong:
            repaired_cover |= set(f.get("covers_after_repair", []))
    total = 0.0
    for alive in itertools.product((True, False), repeat=len(correct)):
        p = 1.0
        cover = set(repaired_cover)
        for f, a in zip(correct, alive):
            p *= (1 - q) if a else q
            if a:
                cover |= set(f.get("covers", []))
        if p and required <= cover:
            total += p
    return p_wrong * total


def _is_boundary(case: dict[str, Any], mode: str) -> bool:
    """``boundary`` is true/false or a list of modes where success is a judgment call."""
    b = case.get("boundary", False)
    return mode in b if isinstance(b, list) else bool(b)


def counterfactual(annotations: dict[str, Any], grid_q: list[float], grid_r: list[float],
                   k_unannotated: list[int]) -> dict[str, Any]:
    cases = annotations["cases"]
    for c in cases:
        for f in c["facts"]:
            if f["label"] not in {"supported", "unsupported", "binding_error"}:
                raise ContractError(f"case {c['position']}: bad label {f['label']!r}")
    unannotated = int(annotations.get("unannotated_complete", 0))
    baseline = int(annotations.get("baseline_complete", 0))
    out: dict[str, Any] = {
        "version": annotations.get("version"),
        "status": "assistant_judgment_upper_bound_not_model_accuracy",
        "baseline_complete": baseline,
        "n_questions": annotations.get("n_questions"),
        "annotated_cases": len(cases),
        "unannotated_complete": unannotated,
        "oracle": {},
        "grid": [],
    }
    for mode in ("fact", "clause"):
        sure, boundary, failed = [], [], []
        for c in cases:
            ok = _case_probability(c, 0.0, 1.0, mode) == 1.0
            bucket = boundary if ok and _is_boundary(c, mode) else sure if ok else failed
            bucket.append(c["position"])
        hist = {c["position"] for c in cases if c.get("historical") == "complete"}

        def total(successes: list[int]) -> int:
            # baseline already counts historically complete annotated cases
            return baseline - len(hist - set(successes)) + len(set(successes) - hist)

        out["oracle"][mode] = {
            "complete_positions": sorted(sure),
            "boundary_positions": sorted(boundary),
            "not_complete_positions": sorted(failed),
            "historically_complete_lost": sorted(hist & set(failed)),
            "total_complete_low": total(sure),
            "total_complete_high": total(sure + boundary),
        }
    for q in grid_q:
        for r in grid_r:
            for k in k_unannotated:
                row = {"q": q, "r": r, "k_necessary_unannotated": k}
                for mode in ("fact", "clause"):
                    exp_annot = sum(_case_probability(c, q, r, mode) for c in cases
                                    if not _is_boundary(c, mode))
                    exp_annot_b = exp_annot + sum(_case_probability(c, q, r, mode)
                                                  for c in cases if _is_boundary(c, mode))
                    exp_un = unannotated * (1 - q) ** k
                    row[mode] = {"expected_complete_low": round(exp_annot + exp_un, 3),
                                 "expected_complete_high": round(exp_annot_b + exp_un, 3)}
                out["grid"].append(row)
    return out


# ---------------------------------------------------------------- cli

def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _docs_by_position(tasks_path: str) -> dict[int, dict[str, str]]:
    out = {}
    for i, t in enumerate(_load_jsonl(tasks_path), 1):
        pos = int(t.get("position", i))
        out[pos] = {d["evidence_id"]: d["text"] for d in t["documents"]}
    return out


def _group(verdicts: dict[tuple[int, int], dict[str, Any]]) -> dict[int, dict[int, dict[str, Any]]]:
    g: dict[int, dict[int, dict[str, Any]]] = {}
    for (p, i), v in verdicts.items():
        g.setdefault(p, {})[i] = v
    return g


def cmd_apply(a: argparse.Namespace) -> int:
    gens = _load_jsonl(a.generated)
    docs = _docs_by_position(a.tasks)
    verdicts = _group(load_verdicts(a.verdicts))
    unknown = set(verdicts) - set(range(1, len(gens) + 1))
    if unknown:
        raise ContractError(f"verdicts reference unknown positions {sorted(unknown)}")
    aliases = load_aliases(a.aliases)
    report = {"mode": a.mode, "threshold": a.threshold, "require_anchor": not a.no_anchor,
              "questions": len(gens), "status": dict.fromkeys(STATUS, 0),
              "deleted_by_verdict": {}, "deleted_by_role": {}, "rejected_by_reason": {},
              "facts_before": 0, "facts_after": 0, "positions_by_status": {s: [] for s in STATUS}}
    rows = []
    for pos, g in enumerate(gens, 1):
        ans = parse_answer(g["raw_text"])
        res = apply_policy(ans, verdicts.get(pos, {}), docs.get(pos, {}), mode=a.mode,
                           threshold=a.threshold, require_anchor=not a.no_anchor,
                           min_clause_chars=a.min_clause_chars, aliases=aliases,
                           unlabeled=a.unlabeled)
        raw = g["raw_text"] if res["status"] in ("unchanged", "regenerate") \
            else render_answer(ans["next_action"], res["facts"])
        rows.append({"position": pos, "id": g.get("id"), "status": res["status"],
                     "raw_text": raw, "deleted": res["deleted"], "repaired": res["repaired"],
                     "rejected": res["rejected"], "duplicates": res["duplicates"]})
        report["duplicates_removed"] = report.get("duplicates_removed", 0) + len(res["duplicates"])
        report["status"][res["status"]] += 1
        report["positions_by_status"][res["status"]].append(pos)
        report["facts_before"] += len(ans["facts"])
        report["facts_after"] += len(ans["facts"]) if res["status"] == "regenerate" else len(res["facts"])
        for d in res["deleted"] + res["repaired"]:
            report["deleted_by_verdict"][d["verdict"]] = report["deleted_by_verdict"].get(d["verdict"], 0) + 1
            role = d.get("binding_role") or "none"
            report["deleted_by_role"][role] = report["deleted_by_role"].get(role, 0) + 1
        for r in res["rejected"]:
            report["rejected_by_reason"][r["reason"]] = report["rejected_by_reason"].get(r["reason"], 0) + 1
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["status"]))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    gens = _load_jsonl(a.generated)
    docs = _docs_by_position(a.tasks)
    verdicts = load_verdicts(a.verdicts)
    aliases = load_aliases(a.aliases)
    problems: dict[str, int] = {}
    for (pos, idx), v in verdicts.items():
        if not 1 <= pos <= len(gens):
            raise ContractError(f"unknown position {pos}")
        n = len(parse_answer(gens[pos - 1]["raw_text"])["facts"])
        if not 1 <= idx <= n:
            raise ContractError(f"position {pos}: fact {idx} of {n}")
        if v["verdict"] in DELETABLE:
            ok, why = validate_anchor(v, docs.get(pos, {}), aliases, a.unlabeled)
            if not ok:
                problems[why] = problems.get(why, 0) + 1
    print(json.dumps({"verdicts": len(verdicts), "anchor_problems": problems}))
    return 0


def cmd_counterfactual(a: argparse.Namespace) -> int:
    ann = json.loads(Path(a.annotations).read_text(encoding="utf-8"))
    res = counterfactual(ann, a.q, a.r, a.k)
    text = json.dumps(res, indent=2) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({m: {k: v for k, v in res["oracle"][m].items() if k.startswith("total")}
                      for m in res["oracle"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("apply")
    vp = sub.add_parser("validate")
    for s in (ap, vp):
        s.add_argument("--generated", required=True, help="generated_native46.jsonl (field raw_text)")
        s.add_argument("--tasks", required=True, help="tasks.jsonl with documents[].evidence_id/text")
        s.add_argument("--verdicts", required=True, help="per-fact verdict JSONL (SPEC.md)")
        s.add_argument("--aliases", help="JSON list of alias groups")
        s.add_argument("--unlabeled", choices=("strict", "inherit", "permissive"), default="inherit",
                       help="speaker anchors on lines without an explicit label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--mode", choices=("fact", "clause"), default="fact")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--no-anchor", action="store_true", help="skip anchor validation (ablation only)")
    ap.add_argument("--min-clause-chars", type=int, default=8)
    ap.set_defaults(fn=cmd_apply)
    vp.set_defaults(fn=cmd_validate)
    cp = sub.add_parser("counterfactual")
    cp.add_argument("--annotations", required=True)
    cp.add_argument("--out")
    cp.add_argument("--q", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.109])
    cp.add_argument("--r", type=float, nargs="+", default=[1.0, 0.9, 0.818])
    cp.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    cp.set_defaults(fn=cmd_counterfactual)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
