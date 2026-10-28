#!/usr/bin/env python3
"""Verify a PRIVATE manual audit of the input-layer model's 22 non-complete answers
against immutable sources, and emit a plot-free aggregate summary.

Semantic categories come from a manual assistant audit (not human gold, not an
automatic grader). This script only (a) proves the audit points at unchanged
sources and exact answer/evidence offsets, (b) runs one deterministic model-free
diagnostic: whether pre-specified question premise terms occur anywhere in the
20 supplied passages, and (c) aggregates. It never writes source files and never
copies story text or answers into its output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    text = path.read_text()
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)['cases']


def bigrams(text):
    text = ''.join(ch for ch in text if ch.isalnum())
    return {text[i:i + 2] for i in range(len(text) - 1)}


def overlap_ranks(question, documents):
    """Rank passages by share of question character bigrams they contain (1 = highest)."""
    q = bigrams(question)
    score = {d['evidence_id']: len(q & bigrams(d['text'])) / len(q) for d in documents}
    order = sorted(score, key=lambda e: (-score[e], int(e[1:])))
    return {e: i for i, e in enumerate(order, 1)}


def boundary_summary(source_root, sample_path):
    sample = json.loads(sample_path.read_text())
    path = source_root / Path(sample['source_path']).name
    if sha(path.read_bytes()) != sample['source_sha256']:
        raise ValueError('changed 486 source')
    ids = {json.loads(line)['id']: json.loads(line) for line in path.read_text().splitlines() if line.strip()}
    for row in sample['training_sample']:
        rec = ids[row['id']]
        target = rec['target'] if isinstance(rec['target'], str) else json.dumps(rec['target'])
        if rec.get('targeted_kind') != 'evidence_boundary' or '"answer_directly"' in target:
            raise ValueError(f'sample row is not a non-answer evidence_boundary row: {row["id"]}')
    return {
        'sampling': 'deterministic: non-answer evidence_boundary rows sorted by sha256(id), first 14; manual reading',
        'training_sample_n': len(sample['training_sample']),
        'training_sample_class_counts': dict(Counter(r['class'] for r in sample['training_sample'])),
        'dev_action_failure_class_counts': dict(Counter(r['class'] for r in sample['dev_action_failures'])),
        'dev_action_failure_positions': [r['position'] for r in sample['dev_action_failures']],
        'all486_questions_generated_from_evidence': sum(1 for r in ids.values() if r.get('question_generation_response_id')),
        'all486_rows': len(ids),
    }


def analyze(source_root, audit_path, terms_path, sample_path=None):
    audit = json.loads(audit_path.read_text())
    src = {}
    for key, rel in audit['source_paths'].items():
        path = source_root / rel
        if sha(path.read_bytes()) != audit['source_file_sha256'][key]:
            raise ValueError(f'changed source: {key}')
        src[key] = load(path)
    tasks, outputs, review = src['task'], src['input'], src['input_review']

    noncomplete = sorted(c['position'] for c in review if c['category'] != 'complete')
    audited = sorted(c['position'] for c in audit['cases'])
    if audited != noncomplete:
        raise ValueError('audit does not cover exactly the historical non-complete set')

    rows = []
    for case in audit['cases']:
        pos = case['position']
        task, out = tasks[pos - 1], outputs[pos - 1]
        if not (task['id'] == out['id'] == case['caseID']):
            raise ValueError(f'id mismatch at {pos}')
        s = case['sources']
        if task['bonsai_prompt_sha256'] != s['task_prompt_sha256']:
            raise ValueError(f'prompt hash mismatch at {pos}')
        if sha(out['raw_text'].encode()) != s['input_raw_sha256'] or s['input_read_char_range'] != [0, len(out['raw_text'])]:
            raise ValueError(f'answer hash/range mismatch at {pos}')
        docs = {d['evidence_id']: d for d in task['documents']}
        for read in s['evidence_read']:
            doc = docs[read['evidence_id']]
            if sha(doc['text'].encode()) != read['sha256'] or doc['sha256'] != read['sha256'] or read['char_range'] != [0, len(doc['text'])]:
                raise ValueError(f'evidence mismatch at {pos} {read["evidence_id"]}')
        if s['all20_read'] and len(s['evidence_read']) != len(docs):
            raise ValueError(f'all20 flag inconsistent at {pos}')
        issue = case['first_material_issue']
        if issue:
            a, b = issue['raw_char_range_0based_halfopen']
            if out['raw_text'][a:b] != issue['text']:
                raise ValueError(f'answer offset mismatch at {pos}')
        for req in case['required_evidence_not_used']:
            a, b = req['doc_char_range_0based_halfopen']
            if docs[req['evidence_id']]['text'][a:b] != req['text']:
                raise ValueError(f'evidence offset mismatch at {pos}')
        ranks = overlap_ranks(task['question'], task['documents'])
        cited = sorted({e for e in docs if f'"{e}"' in out['raw_text']}, key=ranks.get)
        needed = sorted({r['evidence_id'] for r in case['required_evidence_not_used']}, key=ranks.get)
        rows.append({
            'position': pos, 'round': task['round'],
            'question_overlap_rank_cited': [ranks[e] for e in cited],
            'question_overlap_rank_required_unused': [ranks[e] for e in needed], 'historical_input_category': review[pos - 1]['category'],
            'primary_category': case['primary_category'], 'secondary_categories': case['secondary_categories'],
            'mechanism': case['mechanism'], 'fix_class': case['fix_class'],
            'offline_conversion_likelihood': case['offline_conversion_likelihood'],
            'first_issue_fact_index': issue['fact_index_1based'] if issue else None,
            'input_fact_count': case['input_fact_count'],
            'teacher_label_action': case['teacher_label_action'], 'input_action': case['input_action'],
            'evidence_passages_read': len(s['evidence_read']), 'all20_read': s['all20_read'],
        })

    # Deterministic premise-term coverage over all 46 (model-free, substring match).
    terms = json.loads(terms_path.read_text())
    premise = []
    for task in tasks:
        pos = int(task['position'])
        text = ''.join(d['text'] for d in task['documents'])
        missing = [t for t in terms[str(pos)] if t not in text]
        premise.append({'position': pos, 'historical_input_category': review[pos - 1]['category'],
                        'n_terms': len(terms[str(pos)]), 'n_missing': len(missing)})
    flagged = [p for p in premise if p['n_missing']]

    by = lambda key, sel=rows: dict(sorted(Counter(r[key] for r in sel).items(), key=lambda kv: str(kv[0])))
    high = [r['position'] for r in rows if r['offline_conversion_likelihood'] == 'high']
    return {
        'source_integrity_and_offsets_match': True,
        'semantic_judgments': 'manual assistant audit after reading listed passages; not independent human gold',
        'denominator': '22 historical non-complete input-layer answers (13 partial, 9 wrong) of the repeatedly used 46-question dev set',
        'new_generation_performed': False, 'new_model_accuracy_measured': False,
        'primary_category_counts': by('primary_category'),
        'primary_by_historical': {h: by('primary_category', [r for r in rows if r['historical_input_category'] == h]) for h in ('partial', 'wrong')},
        'any_binding_error_count': sum(1 for r in rows if r['primary_category'] == 2 or 2 in r['secondary_categories']),
        'fix_class_counts': by('fix_class'),
        'offline_conversion_likelihood_counts': by('offline_conversion_likelihood'),
        'high_likelihood_positions': high,
        'counterfactual_note': 'high = assistant judges the historical rubric would accept the answer after the named minimal change; '
                               'an offline counterfactual upper bound, never a model accuracy',
        'action_calibration': {
            'input_answer_directly_on_46': sum(1 for o in outputs if '"answer_directly"' in o['raw_text']),
            'non_complete_where_teacher_label_is_not_answer': sorted(r['position'] for r in rows if r['teacher_label_action'] != 'answer_directly'),
        },
        'premise_term_coverage': {
            'rule': 'pre-specified key terms from question text; flag if any term occurs in none of the 20 passages',
            'flagged_positions': [p['position'] for p in flagged],
            'flagged_by_historical_category': dict(Counter(p['historical_input_category'] for p in flagged)),
            'unflagged_by_historical_category': dict(Counter(p['historical_input_category'] for p in premise if not p['n_missing'])),
        },
        'binding_cue': binding_cue(audit['cases']),
        'boundary_training_vs_dev': boundary_summary(source_root, sample_path) if sample_path else None,
        'cases': rows,
    }


def binding_cue(cases):
    inst = [b for c in cases for b in c.get('binding_instances', [])]
    return {
        'definition': 'cued = question names both the correct and the wrongly bound party AND the misbound proposition sits under an '
                      'explicit speaker label naming the correct agent; otherwise uncued (pronoun, second person, vocative, alias, '
                      'reported speech, unlabeled line, scene)',
        'instances': len(inst), 'cases_with_binding': sum(1 for c in cases if c.get('binding_instances')),
        'cued_instances': sum(b['cued'] for b in inst),
        'question_cue_counts': dict(Counter(b['question_cue'] for b in inst)),
        'evidence_locus_counts': dict(Counter(b['evidence_locus'] for b in inst)),
        'primary_binding_cases_first_instance_cued': sum(c['binding_instances'][0]['cued'] for c in cases if c['primary_category'] == 2),
        'primary_binding_cases': sum(1 for c in cases if c['primary_category'] == 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--terms', type=Path, required=True)
    ap.add_argument('--boundary-sample', type=Path, help='private manual audit of 486 evidence_boundary rows')
    ap.add_argument('--summary-out', type=Path)
    args = ap.parse_args()
    out = json.dumps(analyze(args.source_root, args.audit, args.terms, args.boundary_sample), ensure_ascii=False, indent=2) + '\n'
    if args.summary_out:
        args.summary_out.write_text(out)
    else:
        print(out, end='')


if __name__ == '__main__':
    main()
