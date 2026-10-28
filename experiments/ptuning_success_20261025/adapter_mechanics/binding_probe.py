"""Stage 0 binding probe (data_mechanisms/BINDING_TARGET_SPEC.md): teacher-forced
log-probability of a correct clause vs its single-role wrong twin.

Inference only, no generation, no training. Default CLI validates the probe item
file and the model/adapter prerequisites and exits nonzero on anything missing.
--execute scores every item with the SAME no-cache forward the training loss used
(run_ptuning_v2_hybrid.loss_for): prompt with all 20 passages, adapter prefix,
target text teacher-forced. Item files contain story text: keep them, and the
per-item output, in private task evidence. The aggregate holds numbers only.

Item JSONL (one record per pair x condition):
  pair_id, panel ("training_side" | "dev46_natural"), condition ("cued" | "uncued"),
  prefix_fact_count (0 for cued, >=1 for uncued), prompt, documents (20, each text
  inside prompt), truncated=false, target_prefix, correct_clause, wrong_clause,
  target_suffix (optional, default ""). The three target pieces are raw target
  text exactly as it would appear in the native JSON target (already JSON-escaped);
  e.g. target_prefix ends with '"fact":"' and target_suffix starts with '"}'.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_parity as parity  # noqa: E402  (stdlib-only at import time)

PANELS = ("training_side", "dev46_natural")
CONDITIONS = ("cued", "uncued")
REQUIRED = ("pair_id", "panel", "condition", "prefix_fact_count", "prompt", "documents",
            "truncated", "target_prefix", "correct_clause", "wrong_clause")


def validate_items(items):
    """Mechanical checks only; semantic truth of either clause is the reviewers' call."""
    errors, seen = [], set()
    for n, item in enumerate(items):
        where = f"item {n} ({item.get('pair_id')})"
        missing = [k for k in REQUIRED if k not in item]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        key = (item["pair_id"], item["condition"])
        if key in seen:
            errors.append(f"{where}: duplicate pair_id x condition")
        seen.add(key)
        if item["panel"] not in PANELS:
            errors.append(f"{where}: panel must be one of {PANELS}")
        if item["condition"] not in CONDITIONS:
            errors.append(f"{where}: condition must be one of {CONDITIONS}")
        k = item["prefix_fact_count"]
        if not isinstance(k, int) or (item["condition"] == "cued") != (k == 0) or k < 0:
            errors.append(f"{where}: cued needs prefix_fact_count 0, uncued >= 1")
        if item["correct_clause"] == item["wrong_clause"] or not item["correct_clause"] \
                or not item["wrong_clause"]:
            errors.append(f"{where}: clauses must be nonempty and differ")
        docs = item["documents"]
        if item["truncated"] is not False or len(docs) != 20 \
                or not all(d.get("text") and d["text"] in item["prompt"] for d in docs):
            errors.append(f"{where}: require 20 untruncated passages, each inside the prompt")
    return errors


def clause_span(offsets, start, end):
    """Token indices overlapping chars [start, end); clean if no token straddles an edge."""
    span = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not span:
        raise ValueError("Clause produced no tokens")
    clean = offsets[span[0]][0] >= start and offsets[span[-1]][1] <= end
    return span, clean


def differing_span(a, b):
    """Common token prefix/suffix of two target sequences; returns per-sequence diff ranges."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return (p, len(a) - s), (p, len(b) - s)


def encode_target(tokenizer, prefix, clause, suffix):
    text = prefix + clause + suffix
    encoding = tokenizer.encode(text, add_special_tokens=False)
    span, clean = clause_span(encoding.offsets, len(prefix), len(prefix) + len(clause))
    return dict(ids=list(encoding.ids), clause=span, clean_boundaries=clean)


def summarize(logps, indices):
    values = [logps[i] for i in range(*indices)] if isinstance(indices, tuple) \
        else [logps[i] for i in indices]
    total = sum(values)
    return dict(tokens=len(values), sum=total, mean=total / len(values) if values else None)


def score_pair(token_logps_c, token_logps_w, enc_c, enc_w):
    """Clause-level and differing-span statistics from per-target-token log-probs."""
    diff_c, diff_w = differing_span(enc_c["ids"], enc_w["ids"])
    c = dict(clause=summarize(token_logps_c, enc_c["clause"]),
             differing=summarize(token_logps_c, diff_c),
             shared_suffix=summarize(token_logps_c, (diff_c[1], len(enc_c["ids"]))))
    w = dict(clause=summarize(token_logps_w, enc_w["clause"]),
             differing=summarize(token_logps_w, diff_w),
             shared_suffix=summarize(token_logps_w, (diff_w[1], len(enc_w["ids"]))))
    margin = c["differing"]["sum"] - w["differing"]["sum"]
    return dict(correct=c, wrong=w, differing_margin=margin, prefers_correct=margin > 0,
                clause_sum_margin=c["clause"]["sum"] - w["clause"]["sum"],
                clause_mean_margin=c["clause"]["mean"] - w["clause"]["mean"],
                clean_boundaries=enc_c["clean_boundaries"] and enc_w["clean_boundaries"],
                differing_tokens=[diff_c[1] - diff_c[0], diff_w[1] - diff_w[0]])


def aggregate(results):
    """Per (panel, condition); panels are never pooled."""
    groups = defaultdict(list)
    for r in results:
        groups[f"{r['panel']}/{r['condition']}"].append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        n = len(rows)
        out[key] = dict(
            pairs=n,
            prefers_correct_fraction=sum(r["prefers_correct"] for r in rows) / n,
            mean_differing_margin=sum(r["differing_margin"] for r in rows) / n,
            mean_clause_sum_margin=sum(r["clause_sum_margin"] for r in rows) / n,
            mean_clause_mean_margin=sum(r["clause_mean_margin"] for r in rows) / n,
            unclean_boundaries=sum(not r["clean_boundaries"] for r in rows))
    return out


def target_logprobs(model, adapter, control, prompt_ids, target_ids):
    """Exactly run_ptuning_v2_hybrid.loss_for, returning per-token log-probs."""
    import torch
    ids = torch.tensor([prompt_ids + target_ids], device="cuda")
    start = adapter.length + len(prompt_ids) - 1
    positions = torch.arange(start, start + len(target_ids), device="cuda")
    with torch.no_grad(), control.activate():
        logits = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                       use_cache=False, logits_to_keep=positions).logits[0].float()
    picked = logits.log_softmax(-1).gather(-1, torch.tensor(target_ids, device="cuda")[:, None])
    return picked[:, 0].tolist()


def prerequisite_args(args):
    """Reuse decode_parity's identity table; the item file replaces prompts/continuations."""
    files = {k: v for k, v in parity.COMMON.items() if k != "prompts"}
    files.update({k: v for k, v in parity.ARMS[args.arm]["files"].items() if k != "continuations"})
    return files


def check_prerequisites(args, check_environment=True):
    report = parity.check_prerequisites(args, check_environment, files=prerequisite_args(args))
    item_errors = []
    if not args.items.is_file():
        item_errors.append(f"missing item file {args.items}")
    else:
        items = parity.load_rows(args.items)
        item_errors = validate_items(items)
        report["items"] = dict(path=str(args.items), sha256=parity.sha(args.items),
                               count=len(items))
    if args.parity_summary is not None:
        if args.parity_summary.is_file():
            summary = json.loads(args.parity_summary.read_text())
            report["decode_parity_verdicts"] = {k: v["verdict"] for k, v in summary["results"].items()}
        else:
            item_errors.append(f"missing parity summary {args.parity_summary}")
    report["item_errors"] = item_errors
    report["prerequisites_present"] = report["prerequisites_present"] and not item_errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(parity.ARMS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True, help="PRIVATE probe item JSONL")
    parser.add_argument("--parity-summary", type=Path,
                        help="optional decode_parity summary_<arm>.json, recorded not gated")
    parser.add_argument("--out", type=Path, required=True, help="new PRIVATE evidence directory")
    parser.add_argument("--execute", action="store_true", help="explicit GPU execution")
    args = parser.parse_args()
    report = check_prerequisites(args)
    print(json.dumps(report, indent=2))
    if not report["prerequisites_present"]:
        raise SystemExit("Prerequisites missing or invalid (no fallback); see JSON above")
    if not args.execute:
        return
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(args.source.resolve()))
    from backend import PromptEncoder, load_backend
    from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture
    tokenizer = Tokenizer.from_file(str(args.source / "official/tokenizer.json"))
    model, manifest = load_backend(args.artifact, dtype=torch.bfloat16)
    model.eval()
    if args.arm == "v2":
        adapter = DeepPrefix.load(args.adapter, architecture(model.config), manifest["source_sha256"])
        control = PrefixController(model, adapter)
    else:
        encoder = PromptEncoder.load(args.adapter)
        encoder.eval()
        adapter, control = parity.InputPrompt(encoder), parity.NullController()
    results = []
    try:
        with (args.out / f"items_{args.arm}.jsonl").open("x") as stream:
            for item in parity.load_rows(args.items):
                prompt_ids = tokenizer.encode(item["prompt"], add_special_tokens=False).ids
                suffix = item.get("target_suffix", "")
                enc_c = encode_target(tokenizer, item["target_prefix"], item["correct_clause"], suffix)
                enc_w = encode_target(tokenizer, item["target_prefix"], item["wrong_clause"], suffix)
                if adapter.length + len(prompt_ids) + max(len(enc_c["ids"]), len(enc_w["ids"])) > 32768:
                    raise ValueError(f"{item['pair_id']}: context exceeded; refusing to truncate")
                lc = target_logprobs(model, adapter, control, prompt_ids, enc_c["ids"])
                lw = target_logprobs(model, adapter, control, prompt_ids, enc_w["ids"])
                result = dict(pair_id=item["pair_id"], panel=item["panel"],
                              condition=item["condition"],
                              prefix_fact_count=item["prefix_fact_count"],
                              prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                              prompt_tokens=len(prompt_ids), **score_pair(lc, lw, enc_c, enc_w))
                results.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
    finally:
        control.close()
        with (args.out / f"aggregate_{args.arm}.json").open("x") as stream:
            json.dump(dict(kind="teacher_forced_binding_preference_not_QA_accuracy",
                           arm=args.arm, items_sha256=report["items"]["sha256"],
                           complete=len(results) == report["items"]["count"],
                           groups=aggregate(results)), stream, indent=1)


if __name__ == "__main__":
    main()
