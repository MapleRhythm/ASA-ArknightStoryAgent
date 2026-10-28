# Adapter mechanics: decoder parity and binding probe

The two historical 46-question evaluations used different decoders. The
input-layer run (24/46 whole-answer correct) called HF `model.generate`; the
optimized v2 run (15/46) called the custom cached argmax loop
`ptuning_v2_hybrid.greedy_generate`. The question this directory answers: does
the decoder difference contribute anything to the gap, or are the two decoders
the same function of (adapter, prompt)?

## Static audit result: 未发现不变量违例

Read against the sealed sources (`backend.py` 397cbfa8…, `ptuning_v2_hybrid.py`
4b1f0028…, `run_ptuning_v2_hybrid.py`, `train_curated_shallow_20261019.py`,
`train_optimized_v2_20261019.py`) and the Transformers 5.6.2 snapshot used on
the remote run (`modeling_qwen3_5.py` 3e2b6239…, `generation/utils.py`,
`masking_utils.py`, `cache_utils.py`). No invariant violation found, so this
directory contains no patch.

| invariant | where it holds | why |
|---|---|---|
| prefix inserted exactly once, never re-accumulated in KV or GDN state | `greedy_generate` calls `adapter.prepare` only in prefill; decode passes one `input_ids` token. `PrefixController._replace` returns its input unchanged unless `mode == "prefill"`; `_gdn_hook` returns `None` in decode. | Decode cannot re-inject: if the mode were wrongly left at prefill, `_replace` raises on a 1-token input (`value.shape[1] < length`) and `_gdn_hook` raises on a populated GDN cache. Both failures would be loud. |
| RoPE position = cache length | Neither decoder passes custom positions to the model per step. `Qwen3_5TextModel.forward` uses `arange(T) + cache.get_seq_length()`; HF builds `cumsum(mask) - 1` on an all-ones mask, then appends `last + 1`. Same values. | Prefix K is replaced *before* RoPE (hook on `k_norm` output), so prefix keys sit at positions 0–15 in training (`loss_for`, no cache) and in both decoders. |
| attention mask | Custom passes none. HF passes an all-ones 2-D mask, which `_ignore_causal_mask_sdpa` drops (SDPA `is_causal` on prefill, no mask at q_len 1) and `_update_linear_attn_mask` drops for GDN. | Both decoders end up calling the same kernels. |
| cache structure | HF `_prepare_cache_for_generation` passes `config` for linear-attention models, and the model's own fallback does too. In both decoders this gives `DynamicCache(config=…)` with `LinearAttentionLayer` for the 48 GDN layers. | A `DynamicCache` is never auto-compiled. |
| EOS / stop | Both stop at the first of `[248046, 248044]` and keep it; budget 1024; `pad = 248046`. HF takes argmax over an fp32 copy of the BF16 logits, which converts exactly, and `torch.argmax` breaks ties by first index on both paths. | All 92 historical outputs end in 248046 and contain no earlier EOS. Prompt token counts re-encode exactly (46/46). Decoding `token_ids` reproduces `raw_text` (92/92). |
| prefill vs decode on one path | Both decoders run the same model forward: prefill with `inputs_embeds` + `logits_to_keep=1`, then one-token decode steps on the returned cache. | The historical 27B gate measured cached-vs-recompute differences of 0.078–0.125 max-abs logit, KL ≤ 6e-4, with matching top-1 in all 6 checks. |

One real hazard, which the historical runs did not hit: if HF `generate` is
called on a `DeepPrefix` model without something setting `control.mode`, the
mode stays `None` and the deep prefix is **silently absent**. The v2 evaluation
never did this (it used `greedy_generate`). `decode_parity.py` drives HF through
an explicit phase bridge and checks the controller mode on every forward.

**Prediction for `--execute`.** On the same adapter and prompt, both decoders
should produce bitwise-identical per-step logits and identical tokens. Each
historical run should also reproduce its own `token_ids`, provided the GPU,
kernels and package versions are unchanged. If that holds, the decoder is
removed as an explanation for 24 vs 15, and the gap belongs to the adapter,
optimizer, training dynamics or scope behavior.

## `decode_parity.py`

By default it only checks prerequisites. It imports neither torch nor a model
and exits with status 1, listing every missing or mismatched file with its
expected SHA-256 plus any environment mismatch (Transformers 5.6.2 and the hashes
of four of its source files; torch, triton, tokenizers and safetensors present).
`--execute` loads the 27B backbone once and runs each requested task through:

1. HF `generate`, called exactly as the historical input-layer eval called it
   (`inputs_embeds`, all-ones mask, greedy, EOS list, pad, 1024);
2. the sealed `greedy_generate`, unchanged, on the same adapter. The input-layer
   `PromptEncoder` enters it through a two-line shim (`InputPrompt`,
   `NullController`). The v2 `DeepPrefix` enters HF through the phase bridge;
3. a cache-vs-full-recompute replay along the historical continuation, at
   steps 0, 1, 2, 7, 31, 127 and at the divergence step, if there is one.

Each step records the top-16 logits and a SHA-256 of the raw BF16 logit vector.
The run also checks the forward trace: one forward per token, prefill only at
step 0 with `prefix + prompt` tokens, cache length `P+L+k-1` at decode step k,
effective RoPE range equal to the cache range, a mask (if any) that is all ones
and covers exactly the keys, and the controller mode. It also checks that every
emitted token is a maximal logit of its own step, and that generation stops at
the first EOS or exactly at the budget.

```bash
SRC=.codex_tmp/asa_bonsai2_ptuning_20261012          # sealed; read only
EV=.memory/tasks/2026-10-25-ptuning-success/evidence/decode-parity
# input-layer arm (historical decoder: HF)
python3 decode_parity.py --arm input --source $SRC --artifact $SRC/artifact \
  --adapter $SRC/ptuning_input_curated_20261019_attempt01/trained/final_adapter \
  --prompts $SRC/glm_reviewed_full20_data_v1_20261014/val.jsonl \
  --continuations $SRC/ptuning_input_curated_20261019_attempt01/generated_native46.jsonl \
  --out $EV/run_input --id <task id> [--id ...] [--execute]
# v2 arm (historical decoder: explicit)
python3 decode_parity.py --arm v2 --source $SRC --artifact $SRC/artifact \
  --adapter <v2 run>/final_adapter \
  --prompts $SRC/glm_reviewed_full20_data_v1_20261014/val.jsonl \
  --continuations $SRC/v2_optimized_snapshot_20261022/generated_native46.jsonl \
  --out $EV/run_v2 --id <task id> [--id ...] [--execute]
```

Required files, which are absent on this machine until 241 is reachable: the
packed backbone (`artifact/manifest.json` 77b2c71c…, `signs.safetensors`, and
every tensor file the manifest lists); the input adapter
`prompt_encoder.safetensors` 157456cc… with its `adapter_config.json`; the v2
`deep_prefix.safetensors` with its `adapter_config.json`, whose `weight_sha256`
`DeepPrefix.load` enforces (the step-1117 checkpoint recorded c392ebc7…). Present
locally, with matching SHA-256: the tokenizer (0997f410…), dev46 `val.jsonl`
(046adcee…, equal to both runs' `data_sha256.val`), both `run.json` files
(fa634c43…, 0b221b33…) and both historical outputs (1cc1c65e…, 291ebcd0…).

Cost: one model load (≈16 GB peak in the historical runs), then per task one
generation per decoder plus about seven full recomputes. Historical generation
averaged 77 s (input) and 118 s (v2) per task. Start with 2–3 tasks per arm.
If those are bitwise equal at every step, `--all` adds little. Output goes to a
new directory (`summary_<arm>.json` and one JSON per task). The output contains
answer token IDs, so it stays in private evidence.

### Verdicts (`classify`)

| verdict | meaning | implementation bug? |
|---|---|---|
| `equivalent_bitwise` | Same tokens; every step's logits are byte-identical. | No. This is the predicted outcome. |
| `equivalent_tokens_numeric_drift` | Same tokens; some logits differ in bits (e.g. nondeterministic kernels). | No. Check `max_topk_logit_difference` against the 0.078–0.125 gate scale. |
| `numeric_near_tie_divergence` | First token difference at step d, where both top-1 margins are ≤ 0.25 (2 BF16 ulps at \|logit\| 16–32) and no shared top-16 logit differed by > 0.5 up to d. | No. It is a numeric tie. Later text then legitimately differs, so do not compare content past d. |
| `IMPLEMENTATION_BUG` | Any trace or stop violation; the same logit bytes produced different tokens; a token emitted that is not a maximal logit; different stop with identical tokens; or a confident divergence (margin > 0.25 or drift > 0.5). | Yes. `reasons` names the step and the check that failed. |

Further readings:
- `anomalies` flags prefill logits that differ in bits. Both decoders should run
  identical kernels on identical inputs, so any prefill difference points at
  mask, position or prefix handling even when the tokens agree.
- `historical_reproduced=false` with an `equivalent_*` verdict means the current
  environment differs from the historical one: GPU, kernels, weights or package
  versions. It is not a decoder difference. Check identity before any content
  comparison.
- In the cache replay, a top-1 disagreement with `full_margin` > 0.25, or KL
  > 1e-2, means the cache does not reproduce recomputation along the historical
  path. That is a cache bug, even when the two decoders agree with each other,
  because they share the cache path.

Even if every check passes, this tool says nothing about answer correctness.
It removes one confound from the input-vs-v2 comparison.

## `binding_probe.py` (BINDING_TARGET_SPEC Stage 0)

This script scores teacher-forced log-probabilities of correct vs single-role
wrong clauses under the unchanged native prompt with all 20 passages. The
forward pass is literally the training loss path (`run_ptuning_v2_hybrid.loss_for`:
no cache, `control.activate()`, `logits_to_keep` = target positions). It uses no
generation, so decoder parity is not a precondition; it shares only the
environment and identity checks above. `--parity-summary` records the decoder
verdicts alongside the probe if you have them.

Items (private JSONL, one per pair × condition) are described in the script
docstring: `pair_id, panel, condition, prefix_fact_count, prompt, documents,
truncated, target_prefix, correct_clause, wrong_clause[, target_suffix]`. The
target pieces are raw target text as it appears in the native JSON. The script
tokenizes the full target string `prefix + clause + suffix` in one call,
matching how training encoded targets, and locates the clause by character
offsets. A clause whose edge token merges with surrounding text counts as
`clean_boundaries=false` and is reported, not hidden.

For each item it outputs the correct and the wrong clause's log-prob sum and
mean over the clause tokens, and the same over the token span where the two
target sequences differ (the common token prefix and suffix are aligned). It
also outputs the log-prob of the shared suffix, which measures downstream
consistency, plus `differing_margin` (the spec's primary metric) and
`prefers_correct`. The aggregate is kept per `panel/condition` and never pooled.
Example:

```bash
python3 binding_probe.py --arm input --source $SRC --artifact $SRC/artifact \
  --adapter <input final_adapter> --items <private items.jsonl> \
  --out $EV/probe_input [--parity-summary $EV/run_input/summary_input.json] [--execute]
```

The default mode also validates the item file mechanically: required fields,
panel/condition values, `prefix_fact_count` 0 for cued and ≥ 1 for uncued,
distinct non-empty clauses, 20 untruncated passages each inside the prompt, and
no duplicate pair × condition. Whether either clause is true is the reviewers'
judgment.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 test_decode_contract.py --source $SRC
```

The tests use the standard library only. They extract the sealed
`greedy_generate` by AST and run it against recording fakes: prefill once, no
prefix in decode, EOS kept, budget, no truncation, and the input-layer shim. They
also cover trace, emission and stop checks with every violation class, every
verdict branch, prerequisite reporting (missing files with expected SHA, wrong
SHA, pilot `run.json`), and the probe's span, margin, aggregate and validator
logic. They do not run torch, so they do not test neural numerics.
