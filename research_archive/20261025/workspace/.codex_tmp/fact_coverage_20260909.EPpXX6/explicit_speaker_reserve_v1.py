"""Research-only soft reserve for literal question speakers; no learned labels."""


def explicit_speakers(question, vocabulary):
    """Use unchanged corpus speaker strings, longest overlapping matches only."""
    if not isinstance(question, str):
        raise TypeError("question must be text")
    matches = []
    for name in vocabulary:
        if not isinstance(name, str) or not 2 <= len(name) <= 32:
            continue
        start = question.find(name)
        while start >= 0:
            matches.append((start, start + len(name), name))
            start = question.find(name, start + 1)
    matches.sort(key=lambda m: (-(m[1] - m[0]), m[0], m[2]))
    accepted = []
    for match in matches:
        if any(match[0] < other[1] and other[0] < match[1] for other in accepted):
            continue
        accepted.append(match)
    return list(dict.fromkeys(m[2] for m in sorted(accepted)))


def reserve_ranked(ranked, anchors, *, top_k, reserve, id_of, text_of):
    """Reserve <=r full-anchor docs; keep every candidate and its original text.

    This is not an entailment gate. Name cooccurrence does not imply that a
    document answers the question. Retain all nonselected candidates too.
    """
    if type(top_k) is not int or type(reserve) is not int or not 0 <= reserve <= top_k:
        raise ValueError("invalid top_k/reserve")
    ids = [id_of(row) for row in ranked]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document id")
    if not all(isinstance(a, str) and a for a in anchors) or len(anchors) != len(set(anchors)):
        raise ValueError("invalid anchors")
    if reserve == 0 or len(anchors) < 2 or len(ranked) <= top_k:
        return list(ranked)
    required = [id_of(row) for row in ranked if all(a in text_of(row) for a in anchors)][:reserve]
    chosen = set(required)
    for row in ranked:
        if len(chosen) >= top_k:
            break
        chosen.add(id_of(row))
    return [row for row in ranked if id_of(row) in chosen] + [
        row for row in ranked if id_of(row) not in chosen]
