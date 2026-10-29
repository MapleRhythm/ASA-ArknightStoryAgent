"""Training-coverage check: does the 1117-row training set contain the hard cases
that the dev set fails on?

Two questions:
1. How were training questions created, and what non-answer actions do they teach?
2. Do any non-answer training rows have a premise that is absent from the evidence?

The premise detector needs no hand-written term list. For each question it finds
content character n-grams (n=2..6) that never occur in the 20 passages after
removing a small fixed set of question-form words. The score is the length of
the longest question span with no evidence support. It is calibrated on dev46
(teacher action labels only) before it is applied to training rows.

Output is aggregate counts only; no story text or model answers.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# Question-form words that carry no story content. Fixed before looking at scores.
FORM_WORDS = [
    "为什么", "为何", "怎么", "怎样", "如何", "什么", "哪些", "哪个", "哪里", "哪儿",
    "是否", "有没有", "能否", "可以", "是谁", "谁", "多少", "几", "吗", "呢", "了",
    "的", "是", "在", "和", "与", "对", "把", "被", "让", "给", "向", "从", "到",
    "之间", "时候", "之后", "之前", "当时", "这个", "那个", "这些", "那些", "他们",
    "她们", "它们", "他", "她", "它", "其", "所", "以", "而", "并", "又", "也", "都",
    "要", "会", "能", "想", "做", "有", "没有", "一", "个", "中", "上", "下", "里",
    "分别", "最终", "最后", "后来", "原因", "关系", "身份", "目的", "态度", "看法",
]
CJK = re.compile(r"[\u4e00-\u9fff]+")
SPEAKER_TAG = re.compile(r"^([^\s\uff1a:]{1,12})[\uff1a:]", re.M)
BINDING_KINDS = ("speaker_direction", "speaker_identity")


def labelled_speakers_named(question, passages):
    """Count line-label speakers (>=2 chars) from the passages that the question names.

    Approximation: aliases, titles and unlabelled speakers are not counted.
    """
    speakers = {m.group(1) for text in passages for m in SPEAKER_TAG.finditer(text)}
    return sum(1 for s in speakers if len(s) >= 2 and s in question)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action(target):
    try:
        return json.loads(target).get("next_action")
    except (ValueError, AttributeError):
        return "unparsable"


def content_runs(question):
    text = question
    for word in sorted(FORM_WORDS, key=len, reverse=True):
        text = text.replace(word, " ")
    return [run for run in CJK.findall(text) if len(run) >= 2]


def longest_unsupported(question, passages):
    """Length of the longest content run (n>=2) with no occurrence in any passage."""
    joined = "\n".join(passages)
    best = 0
    for run in content_runs(question):
        for size in range(len(run), 1, -1):
            if size <= best:
                break
            if any(run[i:i + size] not in joined for i in range(len(run) - size + 1)):
                # some window of this size is absent; check whether every shorter
                # window inside it is also absent, i.e. the span is truly unsupported
                for i in range(len(run) - size + 1):
                    span = run[i:i + size]
                    if all(span[j:j + 2] not in joined for j in range(len(span) - 1)):
                        best = max(best, size)
                        break
    return best


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--expect-train-sha", default="659586261f7ae4dfafcae352c75f68c623cea5b393557630b9ff324f03828cb5")
    ap.add_argument("--expect-val-sha", default="046adcee638d5a7c36c93836eb12e96ec29e883337153bee0d687849069a3d8d")
    ap.add_argument("--private-out", help="optional per-row scores with question ids (private)")
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    for path, expect in ((args.train, args.expect_train_sha), (args.val, args.expect_val_sha)):
        if sha(path) != expect:
            raise SystemExit(f"sha mismatch for {path}")

    train, val = load(args.train), load(args.val)
    passages = lambda row: [d["text"] for d in row["documents"]]

    # 1. provenance and actions
    groups = Counter()
    for r in train:
        groups[(r.get("targeted_kind") or "base_none",
                "generated_from_evidence" if r.get("question_generation_response_id") else "no_generation_id",
                action(r["target"]))] += 1

    # 2. detector calibrated on dev46 with teacher labels
    val_scores = [(i + 1, longest_unsupported(r["question"], passages(r)), action(r["target"]))
                  for i, r in enumerate(val)]
    pos = [s for _, s, a in val_scores if a != "answer_directly"]
    neg = [s for _, s, a in val_scores if a == "answer_directly"]
    thresholds = {}
    for t in range(1, 9):
        thresholds[t] = dict(
            flagged_nonanswer=sum(s >= t for s in pos), nonanswer=len(pos),
            flagged_answer=sum(s >= t for s in neg), answer=len(neg))
    train_scores = [(r["id"], r.get("targeted_kind") or "base_none", action(r["target"]),
                     longest_unsupported(r["question"], passages(r))) for r in train]

    def dist(rows):
        return dict(sorted(Counter(min(s, 8) for *_, s in rows).items()))

    binding = {}
    for kind in BINDING_KINDS:
        named = Counter(min(labelled_speakers_named(r["question"], passages(r)), 2)
                        for r in train if r.get("targeted_kind") == kind)
        binding[kind] = {"named_0": named[0], "named_1": named[1], "named_2_or_more": named[2]}

    nonanswer_train = [x for x in train_scores if x[2] != "answer_directly"]
    answer_train = [x for x in train_scores if x[2] == "answer_directly"]
    summary = dict(
        semantic_judgments="none; all counts are deterministic string statistics",
        train_rows=len(train), val_rows=len(val),
        provenance_action_counts=[dict(kind=k, origin=o, action=a, n=n) for (k, o, a), n in sorted(groups.items())],
        detector="longest content run (form words removed) whose every bigram is absent from all 20 passages; scores capped at 8",
        dev46_teacher_actions=dict(Counter(a for *_, a in val_scores)),
        dev46_score_by_position=[dict(position=p, score=s, teacher_action=a) for p, s, a in val_scores],
        dev46_auc_nonanswer_vs_answer=auc(pos, neg),
        dev46_thresholds=thresholds,
        train_score_distribution_nonanswer=dist(nonanswer_train),
        train_score_distribution_answer=dist(answer_train),
        train_auc_nonanswer_vs_answer=auc([x[3] for x in nonanswer_train], [x[3] for x in answer_train]),
        binding_rows_labelled_speakers_named_in_question=binding,
        binding_note="counts only speakers that appear as line labels in the 20 passages; aliases and titles are missed",
    )
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.private_out:
        Path(args.private_out).write_text(json.dumps(dict(
            val=[dict(position=p, score=s, action=a, question=val[p - 1]["question"]) for p, s, a in val_scores],
            train=[dict(id=i, kind=k, action=a, score=s) for i, k, a, s in train_scores]),
            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(dev46_auc=summary["dev46_auc_nonanswer_vs_answer"],
                          train_auc=summary["train_auc_nonanswer_vs_answer"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
