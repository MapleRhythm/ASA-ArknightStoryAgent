# Reviewed-subset deep P-tuning training and evaluation

User requested training followed by evaluation on2026-10-19.

Remote root:
`${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012`

Launched at01:53:35 Asia/Shanghai, PID3664749, GPU2
(`GPU-90144416-2626-05e7-0b89-a4e45559ddf9`, RTX4090).
Launch identity is not proof of continuing liveness.

Run: `ptuning_v2_deep_curated_20261019_attempt01`

Data: `curated_reader_data_20261019_v1`

- Train1117 = original audited750 +367 selected original expansion labels.
- Selection: assistant pass or new GLM pass, excluding all known fail,
  needs_review and GLM-flagged IDs. Conflicts lose to the negative judgment.
- Also exclude original positions200,202–206 due to unresolved completeness
  or review read-scope concerns.
- Unreviewed labels and assistant-repaired candidates are not used.
- Source questions, evidence, prompts and retained labels are unchanged.
- Historical46-example validation file is byte-for-byte unchanged.
- No claim that all486 expansion labels were reviewed or certified.
- No claim that assistant/GLM reviews are independent human gold.
- New manifest is explicitly experimental training authorization, not an
  overwrite of any previous candidate readiness flag.

Audited train SHA256:
`659586261f7ae4dfafcae352c75f68c623cea5b393557630b9ff324f03828cb5`

Fixed46 val SHA256:
`046adcee638d5a7c36c93836eb12e96ec29e883337153bee0d687849069a3d8d`

Training reuses the existing frozen Bonsai2 ternary27B differentiable backend,
16-token hybrid per-layer P-tuning v2,4,456,448 trainable parameters,
AdamW lr2e-5,1epoch, seed20261012, CPU activation offload and per-step
transactional adapter/optimizer/RNG checkpoints. It initializes from the
original initial shallow adapter via layer calibration, not a trained final
adapter. Real27B gate precedes training; full20 evidence, no input truncation.
No other jobs are stopped and production is unchanged.

The historical runner is not modified. `train_curated_deep_20261019.py`
uses a separate data audit and includes itself in checkpoint code identity,
then reuses the tested gate/train/evaluate implementation.
Two new selection tests plus six existing CPU prefix/checkpoint tests passed
remotely before launch. Full dataset provenance, protocol, family/document/
question split isolation and val-byte identity checks passed.

After training, the runner automatically generates all46 fixed-evidence
historical development questions with the native protocol, up to1024 output
tokens, no answer validation, saving raw text and token IDs in
`generated_native46.jsonl` and `generation_receipt.json`.
Semantic correctness must still be reviewed separately after generation.
This is not a blind benchmark or retrieval end-to-end score. Never substitute
training loss or JSON validity for answer correctness.

Monitor run `workflow.json`, `progress.jsonl`, `gate.json`, `run.json`,
`latest_checkpoint.json` and the adjacent `.log`.

Confirmed gate completion at01:55:46 Asia/Shanghai:
initialization logits max-abs0.0, short backward/cache checks passed,
longest full20-evidence backward passed, peak allocated GPU memory
16,405,768,704 bytes. Workflow advanced to `training`.
This is a runtime health result, not an accuracy result.

Resume only after checking the old process is absent and the gate completed:

```sh
CUDA_VISIBLE_DEVICES=GPU-90144416-2626-05e7-0b89-a4e45559ddf9 \
  ${REMOTE_PYTHON} \
  train_curated_deep_20261019.py workflow --resume \
  --data ${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012/curated_reader_data_20261019_v1 \
  --out ${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012/ptuning_v2_deep_curated_20261019_attempt01
```

The existing runner does not support resuming after `training_complete=true`
to redo generation. If interrupted during generation, inspect and implement
an isolated evaluation-only recovery rather than rerunning training or
overwriting partial generation output.
