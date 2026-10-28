# Preregistered slot/binding/scope A/B

`preregistered_scope_ab.json` (experiment `full20_slot_binding_scope_ab_v2`)
compares, on a fixed adapter and decoder, the unchanged native prompt (A) with
the same prompt plus the instruction in `../answer_mechanisms/prompt_arm.json`
(B). All 20 passages, the action schema and decoding stay identical. Two panels
are registered and analyzed separately: v2 (where the scope/binding mechanism
was diagnosed) and input-layer (the current best adapter, the only panel that can
move the 28/46 objective). Nothing has been generated.

Whole-answer correctness is defined by `SLOT_RUBRIC.md`: necessary slots fixed
from the question before generation, the role/identity/phase/modality
qualifiers each slot carries, which supported facts are merely unnecessary
(no penalty), and which errors are material anywhere in the answer.

## What is frozen, and how it is enforced

`frozen_inputs` in the preregistration records SHA-256 of `prompt_arm.json`,
`SLOT_RUBRIC.md` and the private per-task slot file (46 tasks, 62 slots, assistant
draft from question text, not yet evidence-checked). `paired_protocol.py
prepare` refuses to plan when any hash differs, when arm B's text differs from
`prompt_arm.json`, when slot rows were written under another rubric hash, or when
a real (non-synthetic) config lacks `frozen_inputs`. `analyze` refuses judgments
whose slots differ from the frozen ones. A change to any frozen file is a new
preregistration version.

Version 1 of this preregistration had a different arm-B wording than
`prompt_arm.json`; both were untested. Version 2 uses `prompt_arm.json` as the
single source and records the v1 text hash under `supersedes`. Plans prepared
under v1 are void.

## Run (after the decode-parity gate and with the real runtime)

```bash
X=ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/protocol_experiments
PYTHONDONTWRITEBYTECODE=1 python3 $X/paired_protocol.py prepare \
  --tasks <dev46 tasks.jsonl> --config $X/preregistered_scope_ab.json \
  --rubric <private slot file> [--panel input_layer] --out <private run dir>
# generate exactly one native completion per request, outside this tool
python3 $X/paired_protocol.py blind --run <run> --results <results.jsonl> --out <review dir>
python3 $X/paired_protocol.py analyze --run <run> --review <review dir> --judgments <judgments.jsonl>
```

All run, review and judgment files contain story text and stay in private task
evidence. `historical_audit.py` is the read-only source/decoder audit of the two
historical runs.

## Tests

```bash
cd ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/protocol_experiments
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v
```

Synthetic fixtures; they also assert that the committed preregistration's hashes
match the committed `prompt_arm.json` and `SLOT_RUBRIC.md`.
