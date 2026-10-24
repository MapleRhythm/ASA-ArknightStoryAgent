# COMPLETE: native 4B / current LoRA thinking comparison

Current status 2026-09-15: ALL WORK COMPLETE. No active inference/training task.

The original 40-generation run finished 2026-09-14 at15:56:37 Asia/Shanghai.
This continuation downloaded final artifacts, ran canonical CPU replay once
(36 successful protocol outputs reproduced exactly), cross-file audited all40
inputs/LoRA IDs/full bodies, and completed all40 source-based semantic reviews.
8 unit tests passed (repeated runs are not additional tests).

Final main results (n=10 per condition):

| condition | raw final content correct | after original gate correct | mean generation seconds |
|---|---:|---:|---:|
| base_off | 2 | 1 | 13.398 |
| base_on | 4 | 1 | 262.830 |
| lora_off | 2 | 1 | 11.521 |
| lora_on | 3 | 2 | 375.824 |

All40 stop, no token-budget exhaustion or input truncation; 4 protocol errors
(base_on Q8/Q12, lora_on Q15/Q20). These remain denominator10.
Current LoRA's raw extra correct Q20 is a compressed-answer boundary and protocol
failure; requiring explicit hug/burn prose makes raw2 instead of3. After-gate
extra success Q8 is quote improvement, NOT corrected semantics. Base_on Q7 is
partial under frozen detailed rubric; allowing implied capacity/protection/new
life gives raw5 and gate2. See REPORT.md for all boundaries and honest limits.

Reports/artifacts:
- REPORT.md, summary.json, semantic_review.json (complete=true)
- attempt1/{results.jsonl,events.jsonl,manifest.json,completion.json,postflight.json,idle_samples.json}
- artifact_audit.json, review_bindings.json, preflight.json, inputs.json, run.log
- summarize_results.py, bind_reviews.py, audit_artifacts.py reproduce analysis.
All outputs use exclusive-create: don't rerun into their existing files.
results SHA8c25b21ee753914c32ac9955a288b40b4b683f7a04ceb46b9d9c6996bbcf1b87.

PID627032/PID627769 absent. At2026-09-15 09:20 all GPUs~15MiB, no compute apps.
Other GPU1/2 jobs were never stopped by this experiment.
No production edits, training, paid API, model promotion, service restart,
new test run, commit/push or subagents. Do not start more without user request.

This is fixed first-round reader diagnosis, not full-chain performance, and not
the previous production sampler/template. Only four arms internally paired.
Conclusion: limited gains, latency cost large, no evidence it reaches60%.

---

## Historical progress notes (superseded; do not resume or relaunch)

Updated 2026-09-14 around 14:17 Asia/Shanghai. NOT COMPLETE.

User authorized the 10-question native4B/currentLoRA x thinking on/off experiment.
Exactly 40 generations, fixed original first-round top20 complete evidence.
No training, no paid API, no production changes, no agents.

- Local directory: `${ASA_ROOT}/.codex_tmp/asa_thinking4b_20260914.GBfYMN`
- Remote directory: `${REMOTE_USER}@${REMOTE_HOST}:/tmp/asa-think4b.goQBCv`
- Remote canonical: `${REMOTE_HOME}/ASA-ArknightStoryAgent`
- Remote parent PID627032, vLLM engine PID627769. GPU0 only.
- Other GPU1/2 jobs PID608774/608975 preserved.
- SSH master socket `/tmp/asa-think4b-20260914.sock`, password entered hidden.
  Do not save or repeat credentials.
- `run_factorial.py run` is already running: DO NOT relaunch or overwrite.
- `run.log`, `attempt1/events.jsonl`, `attempt1/results.jsonl` are live.
- Completion requires `attempt1/completion.json`, then run script `check` once.
- Remote tail tool session6981 watches run.log; other SSH-start session60678 remains
  open while the background shell exists and may finish when the job exits.
- Runtime Python `${REMOTE_PYTHON}`
- PYTHONPATH `${REMOTE_HOME}/ASA-ArknightStoryAgent/.python_packages/train:${REMOTE_HOME}/ASA-ArknightStoryAgent/.vendor/train_override:${REMOTE_HOME}/ASA-ArknightStoryAgent/src`
- CUDA_VISIBLE_DEVICES=0, CUDA_DEVICE_ORDER=PCI_BUS_ID.
- Preflight SHA `f1515f082e308fdcb71e8f894073313deb34f4cc538a906cb33ea232cf38aa8a`.
- No new goals were created. No relevant skills or AGENTS apply.

All protocol and scripts in local directory. Preflight is downloaded.
8 unit tests passed, base/adapter tokenizer encodings and templates equal.
Same sampled seed per question, common native template and model-card thinking
sampling for all arms; NOT the old production low-temp/brace-prefill baseline.
Output budget32768 for all, context40960, max7452 input; no truncation.
One engine with enable_lora, requests alternate None vs explicit LoRARequest.
It is reader-only, not new full-chain accuracy.

At last observed progress14/40 (around14:49):
- Q1 base_off: raw partial; omits fourth swing and explicit changed assessment.
  Wrong reference IDs/quotes => grounded retrieve_more.
- Q1 base_on: 7596 tokens/345.234s, raw partial; misses explicit third full/fourth
  unavailable detail; altered concatenated quotes => grounded retrieve_more.
- Q1 lora_off: raw partial; misses third/fourth and later assessment, structured
  fact says not fatal contrary to final answer; quote invalid => retrieve_more.
- Q1 lora_on: 8378 tokens/610.181s; raw AND delivered partial. Still misses
  explicit third-full/fourth-unavailable detail. Original gate passes despite
  both refs using bare "7" when quotes come from E5 and E7. Do not silently fix.
- Q4 base_on: 5039 tokens/230.587s; final_answer correct. Quote added one extra
  对 character => grounded retrieve_more. Extra inferred emotional "保持联系"
  not treated as actual communication; note preserved in semantic_review.json.
- Q4 lora_off: 77tokens/6.386s, retrieve_more, no answer though evidence sufficient.
- Q4 lora_on partial/partial (best friend correct, falsely claims preservation
  reason missing despite E3/E5); 4355tokens/314.243s.
- Q4 base_off wrong/abstain: wrongly living together currently, flips speaker
  and interprets unable-to-speak as providing real responses. Group membership
  itself is explicitly in source and NOT the error.
- Q6 lora_off correct/correct, brief "bought by Kal'tsit and Ansel" accepted by
  original rubric (do not suddenly require logging camp/date).
- Q6 lora_on correct/correct, 4531tokens/327.853s, adds logging camp but same
  strict correct status as9s nonthink.
- Q6 base_off correct/correct,286tokens/13.235s. Pronoun she reasonably Kal'tsit.
- Q6 base_on correct/correct,3781tokens/172.384s.
- Q7 lora_on partial/partial,3567tokens/256.885s; misses finite blood.
  Its malformed JSON is recovered by ORIGINAL parser; recovered_from_truncated_json
  is a parser label, actual finish=stop, NOT token-budget or input truncation.
- Q7 base_off partial/abstain; protection and Columbia right, finite blood omitted,
  added unsupported "挚爱"/bloodline-sacrifice gloss and reversed announcer in fact.
  Keep partial; strict fails regardless. Quotes altered.
- Q7 base_on request15 in flight.

Need finish all40, download immutable final artifacts, run canonical replay check,
individually score completed final answers (NEVER reasoning) using frozen original
rubric and original story text, write semantic_review.json and run
summarize_results.py, then REPORT.md. Summary distinguishes raw semantic strict vs
strict correct after original quote validator. Errors/length remain denominator10.
Don't silently alter settings after seeing outputs, cherry-pick, or claim online60%.

Selected old QIDs [1,4,6,7,8,12,14,15,19,20].
Original rubric/references:
`../asa_e2e20_20260914.9KSGhY/PROTOCOL.md` and `reference_sources/`.
Current original system/user + actual20 chunks are in `inputs.json`.
First-round original pipeline hypotheses preserved; no gold sent to model.

Evidence checks reviewed:
- Q1 E5=two swings stop + battle fatal; E8=third full/fourth unavailable,
  initially simple habit then more complicated/apology/rest.
- Q4 E1/E2=best friend before group and explicitly group member; E5=never forgot,
  E3=doll important. Do not misjudge group membership.
- Q6 E1=Ansel/Kal'tsit encountered logging camp, next day Kal'tsit bought her.
- Q7 E1/2/4 key limited blood/Grani assumes risk; E6 is secondary Moegirl
  description of extra treasure enabling Bob/team move to Columbia.
  Direct end-letter self-sufficiency chunk is NOT among these first20.
  Do not add evidence midrun; all four conditions share limitation.
- Q8 E2=neuro-agent origin / obey Kal'tsit orders.
- Q12 E4/E5 initial rejection/psychological shadow present; final change reasoning
  not in20. No full credit for incomplete answer even if cautious.
- Q14 E4 same source: no specific injector name or injection date.
- Q15 E1=message about previously mentioned test, she goes medical; specific test
  name or result not given. Accept precise boundary answer, not wholesale refusal.
- Q19 E1=embezzlement shortfall / falsely report infected destroyed equipment /
  preserve expensive equipment; theft excuse was rejected, factory not his.
- Q20 E10=vines restrain attackers, E12=vines breach + embraces Susie despite
  burns, E18=comfort. Need both rescue and calming, not just one.

User updates should be Chinese and frequent, concrete; thinking runs are slow
(native first thinking~5m45), so keep monitoring. No new test run is needed.
