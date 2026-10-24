"""Express catalog links only among names already in the frozen hypothesis."""
import json

SUFFIX = "<|im_end|>\n<|im_start|>assistant\n{"
PREFIX = (
    "\nentity_alias_groups_from_retrieval_catalog: "
    "以下各组内的名字按检索别名表指向同一实体；只补充名称等价关系，"
    "不提供亲属关系、事件或问题答案："
)


def groups_for(catalog, hypothesis):
    entities = hypothesis["entities"]
    present = set(entities) | set(hypothesis["keywords"])
    groups, seen = [], set()
    for entity in entities:
        aliases = catalog.get(entity, [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("invalid catalog entry")
        names = list(dict.fromkeys([entity, *(a for a in aliases if a in present)]))
        if len(names) < 2:
            continue
        for name in names:
            if not name or len(name) > 80 or any(x in name for x in ("\r", "\n", "<|", "|>")):
                raise ValueError("unsafe name in catalog group")
        identity = frozenset(names)
        if identity not in seen:
            groups.append(names)
            seen.add(identity)
    return groups


def inject(prompt, groups):
    if not prompt.endswith(SUFFIX):
        raise ValueError("unexpected legacy reader wrapper")
    if not groups:
        return prompt
    return prompt[:-len(SUFFIX)] + PREFIX + json.dumps(groups, ensure_ascii=False) + SUFFIX


def verify(prompt, candidate, groups):
    assert candidate == inject(prompt, groups)
    if groups:
        added = PREFIX + json.dumps(groups, ensure_ascii=False)
        position = len(prompt) - len(SUFFIX)
        assert candidate[:position] + candidate[position + len(added):] == prompt
    else:
        assert candidate == prompt
