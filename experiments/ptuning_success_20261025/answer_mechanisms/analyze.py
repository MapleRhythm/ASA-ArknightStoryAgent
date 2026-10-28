#!/usr/bin/env python3
"""Check a PRIVATE manual semantic audit against immutable source artifacts.

This is a provenance/location diagnostic, NOT an automatic semantic grader.
Never writes source files or copies story/output text into its summary.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    # Explicit check: `assert` would vanish under python -O and pass a tampered audit.
    if not condition:
        raise ValueError(message)


def analyze(source_root, annotation_path):
    audit = json.loads(annotation_path.read_text())
    source = {}
    for key, relative in audit['source_paths'].items():
        path = source_root / relative
        require(sha_bytes(path.read_bytes()) == audit['source_file_sha256'][key], f'Changed source: {key}')
        source[key] = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                       if relative.endswith('.jsonl') else json.loads(path.read_text())['cases'])
    compact = []
    for case in audit['cases']:
        pos = case['position']
        task, before, after = (source[key][pos - 1] for key in ('task', 'input', 'v2'))
        require(task['id'] == before['id'] == after['id'] == case['caseID'], f'Case ID mismatch: {pos}')
        require(task['bonsai_prompt_sha256'] == case['sources']['task_prompt_sha256'], f'Prompt hash mismatch: {pos}')
        for name, row in [('input', before), ('v2', after)]:
            require(sha_bytes(row['raw_text'].encode()) == case['sources'][name + '_raw_sha256'], f'Output hash mismatch: {pos} {name}')
            require(case['sources'][name + '_read_char_range'] == [0, len(row['raw_text'])], f'Output not fully read: {pos} {name}')
        documents = {d['evidence_id']: d for d in task['documents']}
        for read in case['sources']['evidence_read']:
            doc = documents[read['evidence_id']]
            require(sha_bytes(doc['text'].encode()) == read['sha256'] == doc['sha256'], f'Passage hash mismatch: {pos}')
            require(read['char_range'] == [0, len(doc['text'])], f'Passage not fully read: {pos}')
            require(read['line_range'] == [1, len(doc['text'].splitlines())], f'Passage not fully read: {pos}')
        error = case['first_material_v2_error']
        if error:
            start, end = error['raw_char_range_0based_halfopen']
            require(after['raw_text'][start:end] == error['text'], f'Error offset mismatch: {pos}')
        compact.append({
            'position': pos, 'transition': case['transition'],
            'mechanism': case['mechanism'], 'scope_status': case['scope_status'],
            'first_material_error_fact': error['fact_index_1based'] if error else None,
            'adequate_clean_prefix_end_fact': case['minimal_adequate_clean_prefix_end_fact'],
            'evidence_passages_read': len(case['sources']['evidence_read']),
        })
    lost = [c for c in compact if c['transition'] == 'lost_complete']
    return {
        'source_integrity_and_offsets_match': True,
        'semantic_judgments': 'manual assistant audit; not independent human gold',
        'denominator': 'selected 13 historical losses and 4 gains; not all 46',
        'lost_scope_status_counts': dict(Counter(c['scope_status'] for c in lost)),
        'new_generation_performed': False,
        'new_model_accuracy_measured': False,
        'counterfactual_accuracy_claimed': False,
        'cases': compact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path)
    args = parser.parse_args()
    summary = json.dumps(analyze(args.source_root, args.annotations), ensure_ascii=False, indent=2) + '\n'
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary)
    else:
        print(summary, end='')


if __name__ == '__main__':
    main()
