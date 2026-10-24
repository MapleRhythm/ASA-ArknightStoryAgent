"""Build a frozen FairytaleQA test protocol with oracle and BM25 evidence."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query, documents):
    import math
    docs = [tokens(x) for x in documents]
    q = tokens(query)
    n = len(docs)
    avgdl = sum(map(len, docs)) / max(n, 1)
    dfs = Counter(term for doc in docs for term in set(doc))
    scores = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in q:
            freq = tf[term]
            if not freq:
                continue
            idf = math.log(1.0 + (n - dfs[term] + 0.5) / (dfs[term] + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (1 - 0.75 + 0.75 * len(doc) / avgdl))
        scores.append(score)
    return scores


def main():
    source, output, train = map(Path, sys.argv[1:])
    assert not output.exists()
    repo = source / "repo"
    source_manifest = json.loads((source / "source_manifest.json").read_text())
    for rel, expected in source_manifest["files"].items():
        assert sha(repo / rel) == expected, rel
    questions = repo / "data-by-train-split/questions/test"
    stories = repo / "data-by-train-split/section-stories/test"
    rows = []
    retrieval_hits = Counter()
    retrieval_coverage = Counter()
    retrieval_all = Counter()
    prompt_texts = []
    for qpath in sorted(questions.glob("*-questions.csv")):
        name = qpath.name.removesuffix("-questions.csv")
        spath = stories / f"{name}-story.csv"
        sections = {}
        with spath.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = int(row["section"])
                assert sid not in sections and row["text"].strip()
                sections[sid] = row["text"].strip()
        with qpath.open(encoding="utf-8-sig", newline="") as f:
            for q in csv.DictReader(f):
                gold_ids = [int(x) for x in re.findall(r"\d+", q["cor_section"])]
                assert gold_ids and set(gold_ids) <= set(sections)
                scores = bm25_scores(q["question"], [sections[i] for i in sections])
                ranked = sorted(sections, key=lambda sid: (-scores[list(sections).index(sid)], sid))
                retrieved_ids = ranked[:3]
                for k in (1, 3):
                    retrieval_hits[k] += int(bool(set(gold_ids) & set(ranked[:k])))
                    retrieval_coverage[k] += len(set(gold_ids) & set(ranked[:k])) / len(set(gold_ids))
                    retrieval_all[k] += int(set(gold_ids) <= set(ranked[:k]))
                oracle = "\n\n".join(f"[Section {i}] {sections[i]}" for i in gold_ids)
                retrieved = "\n\n".join(f"[Section {i}] {sections[i]}" for i in retrieved_ids)
                base = {
                    "id": f"{name}:{q['question_id']}", "story": name,
                    "question_id": int(q["question_id"]), "question": q["question"].strip(),
                    "answers": [q["answer1"].strip(), q["answer4"].strip()],
                    "attribute": q["attribute1"].strip(), "scope": q["local-or-sum"].strip(),
                    "explicitness": [q["ex-or-im1"].strip(), q["ex-or-im2"].strip()],
                    "gold_section_ids": gold_ids, "retrieved_section_ids": retrieved_ids,
                    "retrieval_scores": [scores[list(sections).index(i)] for i in retrieved_ids],
                }
                assert all(base["answers"])
                for condition, evidence in (("oracle", oracle), ("bm25_top3", retrieved)):
                    system = (
                        "You answer questions about a children's story. Use only the evidence below. "
                        "Answer in English with one concise phrase or sentence. Do not explain your reasoning. "
                        "If the evidence is insufficient, answer exactly: insufficient evidence."
                    )
                    user = f"Evidence:\n{evidence}\n\nQuestion: {base['question']}\nAnswer:"
                    prompt_texts.extend([system, user])
                    rows.append({**base, "condition": condition, "system": system, "user": user})
    assert len(rows) == 2014 and len({r["id"] for r in rows}) == 1007
    train_blob = train.read_text(encoding="utf-8")
    exact_overlap_questions = sorted({r["question"] for r in rows if r["question"] in train_blob})
    exact_overlap_answers = sorted({a for r in rows for a in r["answers"]
                                    if len(a) >= 20 and a in train_blob})
    output.mkdir()
    save(output / "rows.json", rows)
    save(output / "protocol.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "FairytaleQA", "official_repository_commit": source_manifest["commit"],
        "source_manifest_sha256": sha(source / "source_manifest.json"),
        "test_questions": 1007, "test_stories": 23, "conditions": ["oracle", "bm25_top3"],
        "rows": len(rows),
        "retrieval": {
            "method": "BM25 over human-defined sections within the question's story",
            "tokenization": "lowercase ASCII alphanumeric words", "top_k": 3,
            "answerable_hit_at_1": retrieval_hits[1] / 1007,
            "answerable_hit_at_3": retrieval_hits[3] / 1007,
            "mean_gold_section_recall_at_1": retrieval_coverage[1] / 1007,
            "mean_gold_section_recall_at_3": retrieval_coverage[3] / 1007,
            "all_gold_sections_at_3": retrieval_all[3] / 1007,
            "definition": "at least one annotated evidence section retrieved",
        },
        "generation": {"temperature": 0.0, "max_tokens": 96, "seed": 20260911},
        "scoring": {
            "references": "maximum over answer1 and answer4",
            "metrics": ["normalized_exact_match", "token_f1", "rouge_l_f1"],
            "normalization": "SQuAD-style lowercase, punctuation/article/whitespace removal",
        },
        "training_overlap_audit": {
            "training_file": str(train), "training_sha256": sha(train),
            "exact_question_substrings": exact_overlap_questions,
            "exact_answer_substrings_len_ge_20": exact_overlap_answers,
        },
        "bound_files": {
            "prepare_fairytaleqa.py": sha(Path(__file__)),
            "rows.json": sha(output / "rows.json"),
        },
        "production_changed": False, "test_data_used_for_training": False,
        "note": "BM25 is an isolated public-benchmark baseline, not ASA hybrid retrieval. "
                "Overlap audit covers only the 910 records for this run; base/parent pretraining is unknown. "
                "Metrics are local reproducible lexical metrics, not an official leaderboard submission.",
    })
    print(json.dumps(json.loads((output / "protocol.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
