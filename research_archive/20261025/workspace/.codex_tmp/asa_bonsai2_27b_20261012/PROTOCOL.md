# Bonsai 2 27B isolated trial — 2026-10-12

Production release and existing GPU jobs must remain unchanged.

## Locked artifacts

- Model: prism-ml/Ternary-Bonsai-2-27B-gguf
- HF revision: b072e1d3b35a0a630cece372c2127528e0994386
- File: Ternary-Bonsai-2-27B-PTQ1_0.gguf (5946648928 bytes)
- SHA256: 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
- Runtime: PrismML-Eng/llama.cpp prism-b10772-e8fd209, CUDA 12.8 x64.
- Runtime archive SHA256: 94bb4a8cf97dddb9ff87f02d654dfdf2744ee5f05a5d9df0463efa8d7ac47759
- Remote stage: ${REMOTE_ASA_ROOT}/bonsai2_27b_20261012

The HF files and GitHub release are downloaded through mirrors because direct
connections failed; checksums are checked against the upstream API metadata
served through those transports. This does not independently authenticate mirrors.
Official model cards identify Qwen3.8-27B as the base, not Qwen3.5.

## Reader smoke comparison

Preselect Q1,Q3,Q21,Q23,Q26,Q30,Q31,Q39,Q40,Q50 before generation.
Use the final valid 9B run (eval50_fix2), not discarded eval50.
Replay its conclusion prompts with exactly its 20 complete evidence documents.
These are development questions, not a blind or unseen benchmark.
Do not send old answers, gold labels, or review notes to the model.

First run: thinking disabled, same conclusion token/sampling limits as 9B.
Second run, if needed: all ten questions again in medium thinking mode, official
thinking sampling, 16384 output tokens; report separately, never pick the best
answer per question. Neither run is an end-to-end retrieval accuracy estimate.

No runtime answer/citation/factual validation. Only parsing and input completeness
checks. If context/output limits interrupt a call, count it and disclose it.
Keep the full input and native API response. Grade strict/partial/wrong/refusal
against the previously frozen protocol and original story after generation.

## Optional live pipeline

Reuse production retrieval and prompts in an isolated process, replace generation
only. Full 20 evidence, no content validation; monitor any schema fallbacks.
No formal publication or production pointer changes.

## P-tuning gate

GGUF serving alone is not differentiable P-tuning. Before claiming training,
demonstrate a real soft-prompt forward/backward with nonzero finite prompt
gradients, frozen backbone, preserved Hadamard transforms, and a saved/restored
adapter. Do not rename hard-prompt tuning or LoRA as P-tuning.
The inspected Bonsai 2 model card/backend docs describe inference, not a supported
PEFT P-tuning interface. This is currently unverified, not proof of impossibility.
Do not silently substitute the previous-generation Bonsai-27B-unpacked model.
