# ASA continuation: v35–v58

CURRENT 2026-09-12: v58 DeepSeek-flash pilot interrupted before any response.
Only one request_started marker, zero request_finished/HTTP responses, zero answers.
Cause: credential echoed by SSH PTY; process stopped immediately and credential will
not be reused. No billing conclusion. 28-request plan was not completed.
Remote248 dir `experiments/asa_v58_deepseek_pilot_20260912`; canonical241 dir same
under rank_v8_coverage. Interruption archive SHA
0cd80e5e50c964b42ae747a64edaf1cc47f2c55c2931932c13b821869a5c656a.
Local report `.codex_tmp/e2e_v58/DEEPSEEK_PILOT_INTERRUPTED_20260912.md`.
New credential required before another DeepSeek request; use non-TTY pipe and no
credential persistence. Local safe input tests 7 passed. No new labels, training,
production/model/retrieval/service changes.

v58 attempt05 (2026-09-12) retried the same frozen14-question DeepSeek pilot using
non-TTY stdin; first request started but timed out after about270s, with zero HTTP
responses/answers and 27 requests unstarted. No retry after timeout, no billing
conclusion, no production/training/retrieval change. Remote directory
`experiments/asa_v58_deepseek_pilot_20260912_attempt05`; local report
`.codex_tmp/e2e_v58/DEEPSEEK_ATTEMPT05_TIMEOUT_20260912.md`. Do not rerun closed
attempts without a new credential and an explicit new attempt.

CURRENT 2026-09-11: v55 training COMPLETE, v56 public4B evaluation COMPLETE,
v57 GLM14-question pilot attempt02 COMPLETE. No production promotion.
All v55-running/v56-paused paragraphs below are HISTORICAL.
v55 final adapter SHA0adb413f5ba8b40423341cfd0f1e17f2624684c43e9ff8c7035f2ba115426dd1
returned241 `experiments/asa_v55_a03_20260911/incoming_final_v56/verified/model`;
43files verified, originalcomplete unchanged. Training114steps,trainloss.609019,valloss.617578.
v56 `experiments/asa_v56_eval_20260911` on248: FairytaleQA1007questions,
oracle/BM25top3 x historicalparent/new4B=4028generationrequests,allstop;
completed15:03:19+08. ParentF1oracle.597707/BM25.500868;
candidateF1oracle.594857/BM25.502665. Not currentproductionSODAcomparison.
v56fullgenerationSHA c7796f4f43a8165e60efb954d90ae6136c5b64f50fe06c1ba7d79454b032c3a7.
v56finalarchive completion_v57.tgz returned241/completed, hashesverified.
v57 user requested GLMreplace4B reader; unchangedretrieval/messages/scorer.
14questions,2per7attributes selectedSHAwithoutpredictions,28APIcalls.
Attempt01 URLError,0answers; PythondefaultCApointsnonexistentoldenv.
Attempt02 explicitSSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; all28HTTP200/stop.
UsedprojectdocumentedBIGMODEL_API_KEY configuration in248processonly,
standardmodelAPI,NOTcodingendpoint; nonewcredentialcopied/logged.
GLM5.3requiresenabledthinking;requestedlow/4096tokens vs4Bnothink/96,
so matchedinputs but NOTmatchedcompute. No wholebenchmarkGLMrun.
Same14questionF1:candidateoracle.698810/BM25.581952;
GLMoracle.574285/BM25.457958. Lexicaldroppartlylongeranswers;
examplescheckedshowcorrectextraanswertextpenalized; notsemanticoverallrank.
GLMreportedtokens15427;28requestseconds52.30;noactualbillingclaim.
Completed248dir `experiments/asa_v57_glm_pilot_20260911_attempt02`;
PID3950574exited. DoNOTrerun/retry/overwriteclosedattempts.
Localreport `.codex_tmp/e2e_v57/GLM_COMPARISON_20260911_v57.md`.
GLMcompletearchiveSHA0e5796e3679720d21856f63a22b3ea8c0b8b72c69effb8cbbb602463ca643177.
No newtraining,labels,productionmodel/config/retrieval/servicechanges,commits,pushes,subagents.

CURRENT v55 attempt03 RUNNING on248 (supersedes all attempt01/02 status below).
User approved historical910train/95val and A100sharing; no promotion authority.
Run `${REMOTE_STORE}/exx_grounding_v1/runs/asa_v55_a03_20260911`;
PID3735215, tmux `asa_sft_v55_a03`. Do NOT duplicate/relaunch.
Verified7/114 steps, loss0.49566242099 at startup snapshot; NOT complete.
Read actualsteps/model/trainer_log.jsonl/failure.json/complete.json before claims.
Trainer began04:23:15UTC, firststep04:23:52UTC on2026-09-11.
Onlymemorychange: existingLiger0.8.2 fusedlinearCE, no RMSNorm/SwiGLU/rope patch.
Keeps85% allocatorlimit, dataset/labels/fulltokens/seed/parent/optimizerunchanged.
Shortmktemp directory fixes attempt02 AF_UNIX path overflow (0trainingsteps).
Attempt01 stoppedOOMafter5steps; no checkpoint. BothfailedPIDs exited.
Residentother-user450914remains; no otherprocesses stopped or altered.
Kernelactualtinymodel float32+bf16 loss/14gradienttensor comparisons pass.
All1005rows retainexactuntruncatedtokenencodings acrossall3versions.
Fresh241canonicaldir under `.../rank_v8_coverage/experiments/asa_v55_a03_20260911`.
Step7archive returned241/incoming_from_248, SHA verified; not finalweights.
SnapshotSHA6ec14d35d96b37a59738837f5a72c38da063b7356847acb703c99afa8df3b2ca;
preflightSHA00844551658f2d70f0238c291559ea78f315995026f66e7ae3f548996d5af030.
Local `.codex_tmp/e2e_v55/attempt03/TRAINING_STATUS_20260911_v55.md`
and checkpoint_v55_a03_start.json are the currenthand-off.
Originalstartupreports preserved but are historical; never implyattempt01stillruns.
Benchmark/retrieval evaluation stillpaused, no scores; no automaticposttrain
evalinstalled. Model remainsindependentcandidate, not currentonlineSODAcontinuation.

Latest v55 UPDATE: attempt01 STOPPED after5 optimizer steps with CUDA OOM
at04:04:46UTC. Its startup reports below are HISTORICAL, not current-running.
PID3725104 exited; resident other-user450914 preserved. No checkpoint yet.
Preparing fresh attempt02 using existing Liger0.8.2 fused linear crossentropy
ONLY, same full data/seed/parent/85% allocator guard. Kernel comparison required.
Do not relaunch old attempt or claim completion. Current new attempt not launched.

Earlier v55 startup (2026-09-11): TRAINING ACTUALLY RAN on248, NOT complete.
User explicitly authorized training historical910train/95val and sharing
248 A100 with resident other-user PID450914. This supersedes older no-training
and exclusive-GPU restrictions for THIS run only. Do not kill others.
Run `${REMOTE_STORE}/exx_grounding_v1/runs/asa_historical_sft_20260911_v55`;
PID3725104, tmux `asa_sft_v55`. Do not duplicate/restart/overwrite.
Read `steps.jsonl`, `model/trainer_log.jsonl`, `failure.json` or `complete.json`
for current state; startup snapshot had2/114 optimizer steps, loss0.6771358848.
Trainer began04:01:52UTC and first optimizer step completed04:02:27UTC.
BaseQwen3.5-4B + historicalSFTparent `exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827`;
NOT the currently deployed SODAadapter. Oneepoch,LR3e-6,rank16,batch1/accum8,bf16.
Actual910/95 records fit12000 tokens without truncation; max9830/8855.
Directory name559 is NOT rowcount. Historical source245GLM-compressed+
757copied+3fallback-copied records; not all newGLM labels,not humangold.
v54 remains0newlabels; no new annotations or dataset expansion this run.
Canonical241 experiment:
`${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/asa_historical_sft_20260911_v55`.
SnapshotSHA24eace21c559c530826fd8326ef8c6b98046121c1cb339116f5749dd15e7abbf;
preflightSHA51f96c7b303b9bd49e37ce42399bf07756424e405996e11c3e9c7530caf83d1c.
Initial startup archive alreadyreturned241/incoming_from_248 andverified,
NOT finalmodelreturn. Local `.codex_tmp/e2e_v55/TRAINING_STARTED_20260911_v55.md`
and checkpoint_v55_start.json have links/receipts. No production/modelpromotion.
Benchmark+retrieval eval paused at user's request to prioritize training;
no benchmark scores exist from this turn. No automatic follow-on eval installed.
Sharingmemoryallocatorlimit85%,not control of otherprocesses or allCUDAmemory.
Failure retained in failure.json; no automaticretry/truncation/lowerbudget.
Task auth allows this historical run; do not infer newlabel/defaultswitch/service/
commit/push authority. Previous v54 blocked status below is historical.

Latest v54 (2026-09-11 Asia/Shanghai): user selected option3, GLM
claim/evidence annotation. LOCAL pilot stopped after first HTTP401;
0 successful labels,8 remaining requests NOT sent,15 unit tests passed.
User-supplied credential is not copied into this handoff or artifacts.
Need provider/API-address confirmation or a valid official BigModel credential.
No retries,training,data expansion,production changes or GPU work.
Canonical241 SSH still times out; NO v54 remote directory or sync exists.
Local `.codex_tmp/e2e_v54/GLM_ANNOTATION_PILOT_STATUS_20260911_v54.md`
and `checkpoint_v54.json` contain the sanitized receipt and frozen SHA bindings.
Preserve this failed attempt; do not rerun prepare/run or overwrite its files.
No active jobs remain. Earlier v35–v53 status paragraphs below are historical.

Training-count correction from cached v52 audit: train190=110positive+80negative;
val18=9positive+9negative; total208=119positive+89negative.
Do not call119/71 the train split. Parent training history remains incomplete;
208 referenced rows do not establish the model's entire training-data volume.
September10 naming instructions below apply to historical v35–v53, not new v54.

Latest v53: model experiment COMPLETE, candidate NOT PROMOTED. User now
authorizes all GPUs; fresh check found all idle, so original GPU2 was used.
25 actual calls completed after >=61s lock-held idle check. GPU2 released.
Current version `span_semantic_shadow_20260910.v53_attempt02` is CLOSED:
do NOT rerun prepare/run/check/close or overwrite any preflight/log/output.
9 known erroneous/unsupported frozen checks all raw-supported;5 pass schema,
4 rejected only extra fields. Six new drafts all fail combined gate.
See `CANONICAL_SPAN_SEMANTIC_REJECTION_20260910_v53.md` and checkpoint_v53.json.
No production/source/config/model/training changes in v53.

Latest: v49–v52 are fully closed and cached. The v49 production trace-only
fix remains installed and passed326 release tests. No later production edits.
v51 generic ownership prompt was rejected after24 actual first-round calls;
GPU2 released. v52 current referenced-data contract audit completed208 rows.
No active GPU/test/installation/close job remains from this continuation.

Canonical repository remains `${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`.
Local files are cache/prototypes only. MiniRAG permanently disabled; no subagents,
training, default-model switch, service restart, commit or push.
Every new GPU run must recheck GPU2 continuously for >=60 seconds under the existing lock.
GPU UUID: `GPU-90144416-2626-05e7-0b89-a4e45559ddf9`.
Use numeric `CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID` after verifying
the index/UUID/PCI mapping. Installed vLLM rejects UUID-form visibility.
Never read or print SSH askpass content. Do not pass sandbox_permissions.

All dates in experiment names continue the Beijing-time September 10 series.
Machine timestamps are UTC September 9; this is an explicit timezone distinction.

## Completed and closed

- v35 alias prompt: 32 completed calls, both conditions 10 direct/6 abstain, NOT accuracy.
  Do not promote. No-intervention pairs: raw equal 5/8, display 6/8, action 7/8.
  Lappland seed0 candidate adds reversed chronology, verified against E1.
  17 source / 18 output / 9 warning bindings; checkpoint_v35.json.
  Shutdown observation explicitly labeled assistant-recorded, not original persisted log.
  Canonical `docs/CANONICAL_ALIAS_GROUP_REJECTION_20260910_v35.md`.
- v36 read-only entity diagnosis: 8 cases plus synthetic normalization witness.
  The regex splits continuous Chinese into 2–8-char windows before function-word splitting.
  Fragments are appended as entities; this logic predates recent cleaning changes.
  Old fixed58 JSONL has 58 questions/92 trace steps, only normalized hypotheses,
  not raw planner payloads. Do not infer raw by subtracting heuristic entities.
  checkpoint_v36.json and `docs/CANONICAL_ENTITY_FRAGMENT_DIAGNOSIS_20260910_v36.md`.
- v37 first attempt failed before generation (0 actual calls) due UUID CUDA env.
  Kept original preflight/log/idle receipt/startup_failure_v1.json.
  Successful new directory: `outputs/experiments/planner_entity_provenance_20260910.v37_attempt02`.
  Same 58 prompts/seeds/models/params, with numeric GPU mapping and fresh idle guard.
  58 stop/ok generations, 54 valid payloads, 4 invalid-intent full fallbacks.
  52 valid payloads receive 152 code-added entity occurrences; NOT 152 error labels.
  All 152 additions remain in keywords under model-only final-entity projection.
  Actual D.D.D. case: model writes `DD.D.`, heuristic recovers `D.D.D.`; projection
  drops that correct spelling from entities while leaving it in keywords. Do not promote.
  58 actual canonical planner-method replays match control exactly.
  10 protocol tests passed locally and on 241 (not additive 20).
  Successful run log contains EngineCore shutdown ERROR and process-group warning;
  exit0, all outputs saved and checks pass. GPU2 subsequently released.
  preflight SHA: `3b32cec2b73f5785811502d78441a1e4111e26b5f30833980f4d91073fa05fbe`.
  checkpoint_v37.json and `docs/CANONICAL_PLANNER_ENTITY_PROVENANCE_20260910_v37.md`.
- v38 CPU quote fallthrough audit:
  Replay all 32 v35 outputs using empty evidence objects vs original retrieved
  objects vs visible-only objects, holding raw/hypothesis/visible text identical.
  All groups: 20 direct/12 abstain, 4 legacy-token entries; zero action/display changes.
  A fully synthetic 甲乙/丙丁 fixture proves quote/grounded/strict clean-quote
  success can fall through to a legacy identity-anchor rejection, while the
  reconstruction branch returns early. Evidence_id already returns early.
  The unsupported extra final-answer sentence in the fixture is REMOVED, not displayed.
  Do not claim actual-case regression or remove validation gates.
  checkpoint_v38.json and `docs/CANONICAL_QUOTE_LEGACY_FALLTHROUGH_20260910_v38.md`.
- v39 CPU field-local intent recovery, all 58 v37 payloads:
  54 valid payloads unchanged; 4 known query_type-in-intent enum failures recover
  while preserving the original error in a sidecar. Arbitrary enums/bad schemas still reject.
  Basic repair changes query_type/expected_answer_type and is NOT selected.
  Routing-frozen variant keeps full-fallback routing/question/context and ALL old
  cues; merges validated recovered cues, refusing any overflow instead of truncating.
  Restored question-spelling names: 007 骸骨荒原; 010 梁洵/宗师; 046 槐天裴; 049 none.
  17 protocol tests passed locally and remotely (not additive 34).
  No production change; only routing-frozen variant can enter isolated retrieval.
  checkpoint_v39.json and `docs/CANONICAL_PLANNER_FIELD_FALLBACK_20260910_v39.md`.

Production source changes in the broader resumed task remain the prior v33
layout-only fix. This continuation v35–v39 added experiments/docs, not release source.
Do not forget v33 was a real source change; do not claim the entire work had none.

## v40 completed and closed

Directory: `outputs/experiments/routing_frozen_retrieval_20260910.v40`.
Local prototype: `.codex_tmp/e2e_v40/run_retrieval_probe.py`.
Preflight prepared successfully, SHA:
`102cd50265306bfccbf384db0e22a5d0d0bb358938931896fd5246b5a9f39721`.

Six cases × control/candidate = 12 first-round retrieval conditions:
007,010,017,023,046,049. 017 and 023 are byte-identical hypothesis controls.
Uses ONLY v39 routing-frozen candidate. Same indexes/embedding/reranker/current
canonical source and frozen v27 retrieval configuration. No LLM text generation,
next round, web context, MiniRAG, training or production update.

Initial query counts:
007 16/16; 010 16/16; 017 13/13; 023 16/16; 046 16/16; 049 15/16.
Thus the final pair has an additional query, not identical request budgets.
Do not attribute outcomes solely to names or call retrieval changes accuracy.

Captures actual dense/sparse searches, full evidence objects, trace, prepared
objects and named standardized minimal first-round preview (not every production prompt path).

Run 31801 and check 44280 both exited 0; do not poll/repeat them. GPU2 released.
All 12 retrieval conditions succeeded, LLM generation 0. Postflight copied locally.
017/023 full objects and visible evidence are byte-identical. 049 visible evidence
is byte-identical despite changed objects and hypotheses; 007/010/046 differ.
All six visible pairs reviewed (identical packets read once plus equality check).
35 exact visible-source/condition bindings passed canonical close; checkpoint_v40.json.
Canonical report: docs/CANONICAL_ROUTING_FROZEN_RETRIEVAL_20260910_v40.md.

007 both already have Nastie core; candidate adds reunion but retains distractors.
010 candidate preview E4 adds master's Ling-elder-brother dialogue absent in control preview.
046 both include Mo Yi's motive; candidate E3 adds instruction link via [uc]info
story-summary document, NOT dialogue. Candidate E1 also contains another sword
woman's escape motive; do not attribute it to Mo Yi.
049 both include personality simulation and Trevor Friston name.
017 contains train-accident dialogue plus Juanna/Qanipalaat family distractors.
023 both explicitly contain continuing instruction because Blaze made an effort.
No correctness score or production promotion. v40 preview's default ~5000-char
budget differs from actual frozen runtime 12000-char budget.

## v41 completed and closed

New canonical directory: outputs/experiments/routing_frozen_canonical_reader_20260910.v41.
Local successful script: .codex_tmp/e2e_v41/run_canonical_reader_v2.py.
No new retrieval or planner generation. Use all six v40 frozen pairs, two paired
seeds, actual canonical generate_conclusion at round1/2, actual 12000-char packing,
real prepared evidence objects for all validators, unchanged SODA LoRA/legacy
template. 24 attempts; no second round/direct final fallback/custom repair.
CPU prepare 59317 failed before generation because v27 did not hash training_args.bin;
the failed original script and assistant-transcribed failure receipt are preserved.
v2 explicitly hashes those metadata files without loading/unpickling them, still
verifying every previous model/tokenizer hash. Prepare 10725 exit0, preflight:
f8baad37130e308c69dc317eef3c7df7fc042925c1799529359fc9f2e170c02a.
Longest input 8274 tokens, all fit 12288 context + requested 2048 output.
Run 5757 exit0, check 22352 exit0, close 14285 exit0. All 24 generations stop,
24 canonical method replays identical. 12 additional CPU dry-path checks pass
(fake abstain only for parameter/packing check, not model calls).
GPU2 released; only unrelated GPU0 PID2127909 remained in postflight process list.
Run log has shutdown complete + process-group warning; NO ERROR line this time.
35/39/40/41 production LoRA remains unchanged; no native template substituted.

Control 4 direct/8 retrieve_more, candidate 6 direct/6 retrieve_more: NOT accuracy.
All 24 raw/actions/displays reviewed; 22 exact source bindings + 24 output bindings.
010 candidate two seeds: narrow supported Ling-elder-brother answer with E6; both
controls retrieve_more. Two-round endpoints NOT tested.
007 control seed1 reverses parent relationship ("她母亲娜斯提"), displayed; other
three identify the core sought person, but citation scope for extra identity/age
claims is not fully certified.
017 unchanged-input control: seed0 both retrieve; seed1 both wrongly display
Qanipalaat rescue as Pramanix's family story. Existing semantic check miss.
023 all4 have effort core but quote E2 instead of actual E1, rejected to retrieve.
046 control raw wrongly Yun Qingping; candidate raw correctly Mo Yi plus spectacle
motive, but both candidate outputs still rejected on literal term 动机. Don't relax.
049 seed0 rejected on 制造; seed1 identical accepted output claims Friston made the
machine, but quote only says it is a manufactured tool/not original Friston. Creator
relation not certified. Actual visible E2 also explicitly calls it a copy with memories.
4 unchanged-input pairs: raw equal2, display/action equal4; not deterministic by seed.

evidence_trace_v1.json is post-hoc witness diagnosis, never model input:
010 control source was retrieved rank2/prepared rank4 but 3489-character chain
budget-skipped; candidate visible E3/E6. Thus NOT "failed to retrieve source".
046 both full evidence sets have Mo Yi instruction SUMMARY; control not prepared,
candidate E5. 049 actual visible packets identical despite different hypotheses.

Canonical report: docs/CANONICAL_ROUTING_FROZEN_READER_20260910_v41.md.
checkpoint_v41.json exists on241; fetch to local if not yet cached.
No GPU runs active; do not rerun completed stages or overwrite receipts.
Next investigation: CPU-only subject/reference presence diagnostic on existing
outputs; no new production validator, no auto E-ID repair, no weakened gates.

## v42 and v43 completed and closed

- v42 `outputs/experiments/named_subject_reference_diagnosis_20260910.v42`:
  unchanged-source, CPU-only named-subject presence diagnostic, not a validator.
  56 historical outputs (24 v41 +32 v35) parsed. Only four outputs flagged,
  all the same Pramanix question, two facts each. Fifteen protocol tests pass
  locally and on241 (not 30 tests). Does not detect relation reversals/creator
  attribution. Longer unrelated name mentions can mask absence; cross-block
  pronouns can cause false alarms. checkpoint_v42.json cached.
- v43 `outputs/experiments/broad_subject_reference_diagnosis_20260910.v43`:
  unchanged v42 diagnostic over all58 v27 final outputs and SHA-bound cached
  reconstructed visible evidence. Only9 outputs eligible,16 fact-subject checks,
  only017 flagged (two facts). Never say remaining57 passed semantics.
  No LLM/GPU/retrieval, no new actions, no production gate.
  close session55845 exit0; checkpoint_v43.json fetched locally.
  preflight SHA348cc7544749dfdc3496bcea5b706df86e6a2f0b9c88c222d1f4c40e7adb0b56.
  Canonical docs/CANONICAL_BROAD_SUBJECT_DIAGNOSIS_20260910_v43.md.

## v44 completed and closed

Canonical outputs/experiments/staged_round2_retrieval_20260910.v44.
Script run_second_round_retrieval.py compiled locally, uploaded and frozen.
prepare session77703 exit0:24 exact canonical first-round state replays;
10 terminal answers unchanged,14 unresolved states queued for actual round2.
preflight SHA c6389a800db29516b9c792770784fa5907056dbab7ae2e80e2c0cfe2c63b0775.
Run session3279 exit0 with fresh GPU2 lock/idle window; run_v1.log persisted.
Check12249 and close26465 exit0. Fourteen retrievals all successful, ten terminals
unchanged,24 first-round replays checked. checkpoint_v44.json cached, GPU2 released.
No ERROR/Traceback in run log. Canonical report uploaded:
docs/CANONICAL_STAGED_ROUND2_RETRIEVAL_20260910_v44.md.
No new LLM calls in v44. No first-round resampling or dropping wrong terminals.
Do not rerun/overwrite preflight/run receipts.

## v45 completed and closed

Canonical outputs/experiments/staged_round2_reader_20260910.v45.
run_second_round_reader.py frozen and uploaded. prepare36797 exit0;
14 CPU synthetic-abstain protocol checks verified both actual canonical primary
and final-direct-fallback paths (NOT real model calls). Both prompts frozen,
including second evidence preparation for fallback.
Longest planned input8460 tokens; no overflow at12288 context+2048 output.
preflight SHA271a430b5ccfd0bf36fb9048f167b9a02b3a6ed05785f71a9bfb8ef1672305f7.
Run34576 exit0 with NEW GPU2 lock/60second idle window. 24 endpoints complete:
10 v41 terminal answers unchanged,14 round2 states, up to28 new model calls.
No additional retrieval/planner, same LoRA/legacy template/budgets/validators.
Save primary AND fallback raws; do not infer final quality from action counts.
Actual23 calls:14 primary+9 final-direct, allstop,14 method replays identical.
Check30687, primary-audit13317, close49620 exit0. GPU2 released, no ERROR/Traceback,
process-group shutdown warning preserved. checkpoint_v45.json cached.
28 source bindings+24 output bindings passed. Report prepared locally:
docs/CANONICAL_STAGED_ROUND2_READER_20260910_v45.md uploaded.
Final control7direct/5abstain;candidate9direct/3abstain, NOT accuracy.
010 control both see correctE3 now but bindE7/E8, stillabstain;candidate narrow
round1 Ling-elder-brother answers unchanged.
017 seed0 identicalinput:control quote drops意外 thenfallbackabstains;candidate
adds Juanna's father/hunting/medicine/station story as Pramanix's and displays it.
Seed1 old Qanipalaat misattributions preserved.
023 allfour round2nowE1correct,effortcoredisplayed;extra厨艺进步interpretive.
046 controls stillwrongYunQingping,candidateprimaryMoYi with summaryE4instruction;
allabstain. Candidate seed1 also quoteE2wrong+制造warning;do not relax动机.
049 controlseed0 soleabstain->directfallback stillunprovedTrevorcreatorclaim.
Candidatefallback malformed/unparsed,missing_supported_facts,abstain.
Keep visible copy/memory and answers-to-Kristen support;energy/blood imagery
does not certify literal blood donation. Bothseed1 oldcreatorclaims unchanged.

## v46 completed and closed

Canonical outputs/experiments/staged_subject_reference_diagnosis_20260910.v46.
Sameunchangedv42diagnostic, all10carried+14primary+9fallbackraws=33/24endpoints.
33diagnosticparser/functionreplays;not33strict-validrawJSONs.
Only4raws/7fact-subjectchecks eligible.2flags onlyold017seed1 outputs.
New017candidatefalseclaimmissed becauseE5hasalias恩雅 next to Juanna's speech.
Close additionally proves SAME exactCHAIN_MEMBER_6contains bothnames;even
singlemembernamepresence misses. CPU-onlyposthocprojection,notmodelinput.
prepare84436,run73502,close17014 all exit0. checkpoint_v46.json cached.
preflight143e44dbfa24e2a97add41e8306d9f1791683331fc535d2a632b0a004a2e7278.
Report docs/CANONICAL_STAGED_SUBJECT_DIAGNOSIS_20260910_v46.md uploaded.
No GPU/LLM/retrieval/production changes.

## v47 completed and closed

Investigate final-direct fallback losing the primary validation-rejected
conclusion/raw in ordinary retrieval_trace. v45 separate request ledger and
CPU primary capture preserve it, but ordinary canonical trace may not.
Canonical outputs/experiments/fallback_trace_completeness_20260910.v47.
prepare71655/run3279(new reused tool ID, NOT the old v44 GPU run)/close27075 exit0.
14 actual-state replays exact;9 fallback states all omit primary fullraw in
ordinary final-roundtrace;5 nofallback retainprimary. v45externalledger intact.
Two CPU syntheticfailurecases show fallbackexceptionstdoutonly;primary
generationexception isretained in generation_failures whenbothfail.
These are notactualmodelerrors:all23actualv45callsstop/ok.
preflight22ba1e3d27127cd23fd1a23771b348d30b596fa88b99d9b4c71b5c15f711ec2d.
checkpoint_v47.json cached. Report prepared locally
docs/CANONICAL_FALLBACK_TRACE_COMPLETENESS_20260910_v47.md uploaded.
Original v9 report is retrieval-diagnostic TERM propagation, not this issue.

## v48 completed and closed

Canonical outputs/experiments/fallback_trace_retention_candidate_20260910.v48.
Production source remains unchanged throughout v35–v48.
Original answer_conclusion_generation.py SHA:
43b3611bad56687a112258d0bbf2504126207f63fe5592e3136e87b4be63259a.
Candidate imports unchanged helpers from exact original source copy and only
adds append-only final_direct_fallback_attempts field to terminal-roundtrace.
Includes primary conclusions/generationfailures and fallbacksuccess/error.
prepare45123 exit0,preflight
ad886e7c16404a56e95d8b346550668e1d3ea5b2a27d54a7ed8b9603a5545bec.
Unit75829 exit0:20passed. run11768 reported14 baseline/candidate pairs identical
after removingONLYnewsidecar,10terminalcarryunchanged,9primaryrecordsretained,
2syntheticfailurepairs identicallegacybehavior/stdout and newerrorrecordretained.
run11768 polled exit0.

Broader baseline release tests first attempt24285 exit2 atCOLLECTION (0testsrun):
KeyError ASA_NO_MINIRAG_RETRIEVERS. Preserve canonical_release_baseline_tests_v1.xml
and .log;do notoverwrite or claimtestfailure. Read full no-MiniRAG test and v33
verify_installed.py;they require ASA_NO_MINIRAG_CONFIG/CONFIGS/RETRIEVERS/RUNNER,
ASA_ROOT_SRC,ASA_RUN_CPU_INFERENCE_SCRIPT. New localrun_release_baseline_v2.py
sets these exactly and CUDA_VISIBLE_DEVICES empty, hashesallconfigs/tests/runner,
preserves firstfailure, writesattempt02 receipts. Uploaded;61543 exit0,
306 baseline release tests pass. These arebaseline,not306candidatetests.
close93071 exit0;checkpoint_v48.json cached.
CANONICAL_FALLBACK_TRACE_CANDIDATE_20260910_v48.md uploaded to canonical docs.

## v49 production trace-only integration completed and closed

Canonical outputs/experiments/fallback_trace_retention_integration_20260910.v49.
One existing production source updated:
release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/generation/answer_conclusion_generation.py.
One new test:release/ASA-ArknightStoryAgent/tests/test_final_direct_trace_retention.py.
IMPORTANT:the broaderresumedtask now hasv49 realproductionedit aswellaspriorv33.
No answer/validator/prompt/E-ID/model/config/service change. Only append-only
final_direct_fallback_attempts capturesprimary and fallbackconclusion/error.
ExactAST matchesv48 testedcandidate;all otherfunctions exactoldAST.
Originalpreimage canonicalV48/answer_conclusion_generation_original.py,
SHA43b3611bad56687a112258d0bbf2504126207f63fe5592e3136e87b4be63259a.
Localdrafts editedwithapply_patch,uploadedtoexperimentdir,prepare8311 exit0.
Guardandtwoexplicitcopytargets95443 exit0. preapply_guard_v1.json/application_v1.log.
Source/newtestareALREADYinstalled;do not repeatapplication/overwriteguard.
Postapplyfreshprocesscheck24994 exit0:326/326release tests passed,0fail/error/skip.
14 actualcanonicalcachedreplaysafterstripONLYnewsidecar areidentical;10terminalcarry
unchanged;9fallbackstatesretainprimary;5nonfallbackstatesaddnofield.
2syntheticfailurefixtures preserveoldresult/stdout andrecordfallbackerror.
Allotheroldsource/model/config/testhashesguarded. No new LLM/retrieval/GPUcalls.
InstalledsourceSHA a69d1a78258cf4f8c2e2cdc10f424752707481099a7ac23264a9c3bebefd550d.
InstalledtestSHA 02fb292bf7448685a4a6c9feb33b3db4f41df5118bfa6bbbeec4f0413faa8d43.
PreflightSHA3098d76e80da7547f1b6007ba01d01b22e26693337fb597169cba1c05d8d35f3.
Resumedturn:localclosecompiled;reportupload9309/scriptupload90269 both exit0.
Canonicalclose52031 exit0;checkpoint_v49.json generated. Cachetransfer61074 exit0,
checkpoint+close_v1.log cached. Do not repeat installation, check, or close.
Canonicalreport:docs/CANONICAL_FALLBACK_TRACE_RETENTION_INSTALLED_20260910_v49.md.
This is diagnostic completeness, not semantic accuracy improvement.
Afterv49allhistoricalv45/v48verify-current-sourcechecks intentionallyfail for
the ONEpatchedsource;nevereditoldmanifests. Usev49explicitold/newhash verification.
No GPU task active. No services restarted, no training/defaultswitch/commit/push.

## v50 source-bound relation sensitivity audit completed and closed

Canonical outputs/experiments/relation_claim_sensitivity_20260910.v50.
Script audit_relation_claims.py;prepare87326/run46257 both exit0.
Three deliberately selected endpoints, four cached calls:
007controlseed1(round1),017candidateseed0(round2),049controlseed0(primary+fallback).
Actual canonical replays preserve all old fields;onlyv49sidecar stripped for comparison.
Transparent validator capture proves observed bad facts are already in modelraw,
parsedfacts,validatedfacts,finaldisplay;notintroducedbyv33layout/v49trace.
Six assistant-authored one-fact projections pair direction/owner/creator claims;
ALLsix accepted/displayed,ALLsix havelexicalwarnings,nonehardquoteissues.
Six separateinvalidEID/absentquotecontrolsallrejected,claimsnotdisplayed.
These areCPUfixtures,NOTnewmodeloutputs/humangold/accuracy. Juannacorrectfactdoes
notanswerthePramanixquestion;unsupportedcreatorclaimnotproveduniversallyfalse.
SixsourceexcerptsSHA-boundtoactualEblocks,CHAIN_MEMBER,Unicodecodepointoffsets.
Cachetransfer85279 exit0;preflight/audit/logsarelocal. Reportupload56741 exit0.
close_relation_claims.pycompiled;upload26212 exit0;close34002 exit0.
Freshprocessreplaymatchesallactual/syntheticfields. checkpoint_v50.json cached
withtransfer58134exit0. Canonicalreport:
docs/CANONICAL_RELATION_CLAIM_SENSITIVITY_20260910_v50.md.
No production/model/prompt/label/retrieval/GPU changes.

## v51 isolated ownership-rule prompt comparison completed and closed

Local .codex_tmp/e2e_v51/run_ownership_prompt.py+README.mdcompiled.
Canonical outputs/experiments/ownership_rule_prompt_20260910.v51;
mkdir13848/upload23050 exit0;prepare41613 exit0.
24plannedfirst-roundcalls:6v41CONTROLhypotheses/retrievalinputs x2newseeds x
baseline/ownership_rules. DoNOTusev39routingcandidate. Onegenericsystemaddition;
evidence/user/schema/legacybraceprefix/budgets/model/validatorsunchanged.
Usesunmodifiedv41generationrecordingwithtemporaryprocesscallbacksandone
outgoingpromptbridge;allcallbacksrestored;NOPRODUCTIONSOURCEPATCH.
Prepared12CPUsyntheticprotocolchecks;longestinput8363fits12288with2048output.
PreflightSHAc4d3bf2c16ad3ba4683527c56ac5f9df50412d2575db51e384a3ca18b7bb65d3.
Run38356 exit0:24actualcalls,allstop;freshGPU2lock/60secondidlewindowverified.
Cachetransfer90112exit0. Check85325exit0:24exactcanonicalreplays.
Fullgeneration/log/idlecache56949exit0. Local snapshots01(first8) and02(first19)
werecomparedagainstfinalgenerations,exact;all24raws/actions/displaysread.
ExistingmodelandnumericGPUmappingonly. Must reviewall24raws/displays/actualvisibleevidence,
notactioncounts. ThisisdevelopmentnotE2E/independentvalidation.
Reportupload87167/scriptupload48903exit0;sourcebind+close62405exit0.
15sourceexcerpts x2conditions=30bindings;24outputbindings.
Canonicaldocs/CANONICAL_OWNERSHIP_RULE_PROMPT_20260910_v51.md.
checkpoint/postflight/review/logcache21644exit0,allcached.
Baseline3direct/9retrieve_more;ownership1direct/11retrieve_more,NOTaccuracy.
007baselinebothgiveavailableNastiecore;ownershipbothwithholdit.
010allretrieve_more.017allretrieve_more,nooldmisattributiondisplayedbyEITHER.
023baselineE2/candidateE3wrong,sourceE1,allrejected.046baselinewrongYunQingping
rejectedbyexistingrules;candidatebothretrieve_more,notcorrectMoYianswers.
049baseline0unsupportedcreatoraccepted;baseline1manufacturetermrejected;
candidate0stillwrongcreatorbutquotechangescurlydoubletosinglequotes=>rejected;
candidate1unsupportedcreatorACCEPTED. Copy/memorysupported,creatornotcertified.
Rejectpromptpromotion. No productionchange,training,defaultswitch,restart,push.
Logshutdowncomplete+processgroupwarning;NOERROR/Traceback;GPU2releasedatclose.
Importantnegativefinding:017/046candidatefourrawcontextscontainbadexplanations,
butALLFOURactualnormalizedfollow_up_hypothesis.dialogue_contextareEMPTY.
DoNOTclaimrawcontextpoisonednextstate;oldnormalizeralreadypreservescontext.

## v52 read-only current LoRA referenced-data contract audit completed and closed

Canonicaloutputs/experiments/referenced_training_contract_20260910.v52.
Local .codex_tmp/e2e_v52/audit_referenced_data.py+README.md compiled.
mkdir5304/upload42440exit0;combinedprepare/run79271exit0.
No training, label edits, model/retrieval/GPUcalls or training_args.binunpickling.
Current adapter README names dataset:
soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_train.
Modelcard autogenerated;notbyitselfhistoricalcontent-SHAproof.
Found historical docs/current_full_flow_and_datasets_for_review.md, dated2026-06-03.
It describes oldMiniRAG/weakgrounding;DO NOTadoptthoseoutdatedruntimeconfigs.
Relevantdatafolderscurrent:
data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged
and..._gpu3_clean_v1,plus_gpu3_gpu0/_gpu3_gpu1/_gpu3_gpu2 shards.
ExactmodelcarddatasetnameRESOLVEDto..._gpu3_merged/dataset_info.json:
train.json190records,val.json18records,total208,build_summary69verifierstates,
50teacherfullchainrecords;119True/89FalseKTOtotal,100hypothesis/108conclusion.
Matchingconfigsrc/config/llama_factory_soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_config.yaml
namesexactcurrentoutputadapter,registryandmergeddatadir;templateqwen3_nothink,
cutoff_len5632,kto,3epochs. READONLY;didNOTexecutedo_trainconfiguration.
Parentadapterdeclared:model/lora/soda_blackbox_deepseek_v1_550_parallel_qwen35_4b_lr2e6_beta001_epoch1.
DoNOTclaim208rowsareentiretraininghistory;currentreferencechainisnothistoricalbyteproof.
training_args.bin may be hashed but MUST NOT load/unpickle.
Rowsaresystem/tools/conversations(human,gpt)/kto_tag/task_type/meta.
meta.preference_rolecanstillteacher_positiveevenwhenkto_tagFALSE;usekto_tag.
All208recordsinspected,63train+6valpositiveconclusionsallconclusion_v2/plainanswer,
allstrictJSONobjects,ALLwithout supported_facts/inferred_facts/evidence_refs/final_answer.
All69positiveconclusionpromptshavenumbered1.docchunk evidence,no[E#]heading.
Trainpositiveconclusion47direct/16retrieve;val3direct/3retrieve,NOTaccuracy.
OneinvalidJSON(falsehypothesis),onewrongtaskschema(falseconclusion),NOTpositive
traininglabelerrors. TwelvefalseKTOrowshavestale teacher_positive metadata.
Train49distinctquestions/val8,shared7questions,14of18valrowshavetrainquestion.
EXACTpromptenvelopeoverlapZERO;doNOTcallsamequestiondifferentstateexactpromptleakage.
Onevalquestionabsentfromtrain:为什么卓娅决定不顾危险要去彼得海姆中学？
Sixv51questionshaveZEROexactquestiontextoverlapwithregistered208rows.
ThisdoesNOTproveabsenceofparaphrases,parenttrainingexposure,orblinddevelopment.
Preflight8f571c15d5fd0f4067551a3144d929a28c614805f2c1540692034b865a4cbcab.
Cache55818exit0;report66202/script12697uploads exit0;close91105exit0.
Closefreshprocesscomputesall208again,exactallfields. checkpoint+closelogcache32313exit0.
Canonicaldocs/CANONICAL_REFERENCED_TRAINING_CONTRACT_20260910_v52.md.
No causal/accuracy/humangoldclaims,nohistoricaltraining-byteproof;nohistoryrewrite.
Currentv52scriptcomparesall208recordsandactualquestiontext/promptSHAoverlap,
plusthesixv51runtimegrounded_action_v1requests;neverjoinonnumericID suffix.

All completed checkpoint/report copies are local under `.codex_tmp/e2e_v35` ... `e2e_v52`;
v37 successful artifacts are under `.codex_tmp/e2e_v37/attempt02`. No queued run remains.
Remaining semantic limits: relation direction/first-person owner/creator attribution
are not solved by lexical quote checks or v49 diagnostics. Never claimoverallASAquality
completed,orimplicitlyauthorize training/defaultswitch/restart/automaticEIDrebinding.

## v53 new exact-span + semantic shadow prototype; CPU done, GPU pending

User explicitly approved implementing the previously recommended offline
fact-verifier + deterministic citation prototype ("做"). Still no production
integration, training, model switch, MiniRAG, services, agents, commits/push.

First canonical dir: outputs/experiments/span_semantic_shadow_20260910.v53.
Frozen preflight SHA4bcda775ef2b4065d0e0dd335ae4a4e14b361119990875aca619c8d479c920b2.
28 base protocol tests passed locally/remotely (same28, not56).
run_v1.log: original lock's FIRST gpu_idle check failed, no model engine/calls,
no idle_samples or generation journal. Other user's GPU2 Python jobs present.
CPU integration v1 then failed because long source spans exceed existing
quote<=80/max2 refs per fact. All initial source/preflight/logs remain intact.

CURRENT second canonical dir:
outputs/experiments/span_semantic_shadow_20260910.v53_attempt02.
Local: .codex_tmp/e2e_v53/attempt02.
Wrapper run_shadow_v2.py reuses frozen first-runner explicitly; protocol
span_protocol_v2.py partitions allsource into <=80codepoint T IDs, <=2refs/fact.
Old version uses S IDs; cannot reuse IDs across protocols. E/source text all
retained, full SHA/blockSHA/offsets stored; no heuristic speaker metadata.
Separate checker same4B SODA LoRA, NOT independent judge. Fullcontext available;
freshdraft passes only existing canonical gate AND eachcheck supports+answers
AND checker supportspans already selected by draft. No rebind/retry/dropfacts.

Second preflight SHA:
54abd4c81cafd3b39865d75cc9e60082daafc33b1b805b69cf50a5a1ff1d2ee0.
15frozenclaimchecks (sixv50contrasts x2seeds +threehistoricalbadfacts),
6freshdrafts(threequestions x2seeds), <=12draftfactchecks: max33newcalls.
Longestfixedprompt8895+2048 fits12288; dynamicoverflow preserved,nottruncated.
Second10additionaltests passedlocally/remotely (same10).
check_cpu_v2.py succeeded:489spanbindings(162/164/163),3syntheticcanonicalfixtures,
6invalidEID/absentquotecontrolsrejected,12syntheticgatechecks,21tasksrebuiltexact.
These are NOT semantic performance/model answers/humangold.

GPU2 repeatedly busy; latest observed other-user lightgts PID2357598,
837MiB total /26%usage; PIDs change as tasks cycle. Do not rely on old PID.
GPU1 last observed18MiB/0%,but switchingGPU requires explicit user authority.
No v53attempt02 GPU job started, no background waiter. Needfresh continuous>=60s
idlecheckINSIDE originalGPU2lock before run; use numericCUDA2+PCI_BUS_ID.
Do not rerunprepare/overwrite failures. AfterGPUrun performcheck+allraw/source
review and recordfalseaccepts+supported-coreloss,notactionaccuracy.

Report CANONICAL_SPAN_SEMANTIC_SHADOW_20260910_v53.md and CPU-only
checkpoint_cpu_v53.json document incomplete GPU stage. Empty human holdout
review template lives in firstdir; it is NOT a ready/evaluated dataset.

CPU checkpoint session50623 exited0; recomputed all CPU integration fields in
a fresh process. Latest persisted GPU2 PID2362877,814MiB process allocation.
checkpoint_cpu_v53.json and checkpoint_cpu_v1.log fetched with74642 exit0.
All11 checkpoint basis artifact SHA bindings verified locally (exec93372c).
No active tool jobs remain. GPU stage still not started, no success claimed.

## v53 GPU continuation COMPLETE (supersedes earlier pending status)

User: "所有GPU都可用" authorizes all availableGPUs. Runtimecheck all3 idle;
keptoriginalGPU2/numericCUDA2, no device/config/prompt/model changes.
run51003 exit0; fresh13samples~61seconds insideoriginallock.25actualcalls,
24stop+1length;15protocolok/10errors.6drafts only2protocolvalid =>
4childchecks, so25actual NOT33plannedmaximum.
check26735 exit0:15successes exactcanonical replays, schema/syntax errors
reproduced, all25journalrows preserved. Noretries/E-IDrepair/extra-keyignoring.
Snapshot01=first10rows, snapshot02=first16rows, both equal finalprefix.

review_and_close_v53.py uploaded; review68276 exit0;all25raws read,25prompt/token
IDs/LoRA identityrecomputed,50sourcebindings(includingrejectedraws),17unique
packet/span bindings reviewed withsourcecontext. Source review isASSISTANT,
not humangold. All15frozenraw verdicts supported/answers. 9negativev50-based
checks (6badcontrast+3historical) allrawsupported;5actualpass,4extrafieldsrejected.
6source-consistent contrasts allpass but Juannafact2 incorrectlyanswersPramanix;
motherpositive rationale misidentifies whomNastie thanks;notcertifiedentailment.
No accuracy claim/generaltheorydisproof.

6freshdrafts NONE combinedpass. Parentbothmissingunresolved+3refs/firstfact,
rawNastiecore plusmotherreversal. Pramanixseed0repeatwrongstationfact to2048;
seed1invalidtrailingcomma;bothtrainaccidentcore+Juannastorywrongowner.
SelectedT25/T26donotindependentlyestablishPramanixidentity;T24speakerEnciodes.
Creatorbothdraftprotocolvalid with6exactcopiedrefs;bothrejectedbyEXISTING
canonicalmissing制造. Seed0T5personalitysimulationcorechildpasses,butwhole
answerwithheld. Seed1creatorchildrawwrong supported;schemaok butT48/T56
notdrafterT47/T137 =>citation-subsetreject,NOTsemanticcorrection.
Seed1helper/mastermindfactdropsperspectivalqualifier;notcertifiedpositive.
Allfailed/successfulraws,prompts,preparedsourceobjects,oldoutputs retained.

Callsecondsincludingpostcallvalidation total383.77199,median9.61227,
range7.75523–143.91499;excludesengine/idle;no pairedlatencybaseline.
RunlogShutdowncomplete+processgroupwarning;noERROR/Traceback,otherwarningskept.
NewreportCANONICAL_SPAN_SEMANTIC_REJECTION_20260910_v53.md uploaded.
Close65267exit0;checkpoint_v53.json:reviewrecomputedexact,50bindings,
GPU2postshutdownidle15MiB/0%/noPIDs at2026-09-10T12:21:26Z.
HistoricalCPUcheckpointandearlierfailedattemptremainunchanged,notrerun.
No independenthumanholdout;rejectconcretecandidateatdevelopmentstage.
No production/prompt/model/training/MiniRAG/service/commit/push/agentschange.
Checkpoint+closelog cache45247exit0;all16basisSHA localbindingsverified97f5c0.
No active run/check/review/transferjobs remain. Do not repeat completed stages.

## v58 DeepSeek local pilot COMPLETE (2026-09-12)

Previous timeout investigation and repair:
248 resolved api.deepseek.com but IPv4:443 timed out; 241 had stale
127.0.0.1:8888 proxy configuration. Local direct access returned immediate401,
so the failure was network path/proxy, not request schema. Do not repeat the
failed pseudo-TTY/empty-stdin/path/hash/remote-timeout attempts (v58 initial,
attempt03, attempt04, attempt05); their directories remain preserved.

Patched .codex_tmp/e2e_v58/deepseek_compare.py with an 8s no-key network
preflight, DEEPSEEK_THINKING=disabled, DEEPSEEK_PILOT_SOURCE, and no automatic
retry or credential persistence. Added launch_deepseek_local.sh. Key was
provided through hidden input only and was not written to files/logs.

local_run5 completed 28/28 HTTP requests successfully (14 fixed questions x
oracle evidence and BM25 Top-3 evidence), all finish_reason=stop. 2026-09-12
results:
- New trained4B: oracle tokenF1 .698810, ROUGE-LF1 .6738, EM .3571;
  BM25 tokenF1 .581952, ROUGE-LF1 .5454, EM .3571.
- DeepSeek-flash: oracle tokenF1 .739109, ROUGE-LF1 .7155, EM .4286;
  BM25 tokenF1 .632237, ROUGE-LF1 .6016, EM .4286.
- GLM-5.3: oracle tokenF1 .5743, ROUGE-LF1 .5404, EM .2857;
  BM25 tokenF1 .4580, ROUGE-LF1 .4274, EM .0714.
DeepSeek vs new4B token-F1 win/tie/loss: oracle 4/6/4, BM25 4/7/3.
This is a 14-question lexical pilot, not a 1007-question benchmark, not an
end-to-end production evaluation, and not evidence that GLM annotation quality
is the cause. Do not replace production SODA or alter retrieval/training.

Local artifacts: .codex_tmp/e2e_v58/local_run5/ and
.codex_tmp/e2e_v58/DEEPSEEK_COMPARISON_20260912.md. Report uploaded to 241:
${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/
asa_v58_deepseek_pilot_20260912_local_run5/DEEPSEEK_COMPARISON_20260912.md
SHA-256 591e8062d9a71d977c31cbdaa851e7c7dea13f2150255141fce2483dcb595c.
API totals: 14,517 prompt tokens, 272 completion tokens, 14,789 total;
22.58s request time. No active jobs remain.
Local working-copy hash rechecked after continuation is
591e8062d9a71d977c31cbdaa851dd081acf6f0df95016b8bcc6da2630d61281; retain
the previously recorded remote-upload hash above for the transferred artifact.

## 2026-09-13 follow-up pilot attempt

User requested another small 4B validation. No fresh inference was run:
local workspace has no 4B weights/runtime, and passwordless SSH to both
${REMOTE_USER}@${REMOTE_HOST} and ${REMOTE_USER}@${REMOTE_HOST} returned
`Permission denied (publickey,password)`. Do not present prior 14-question
results or cached scope-control outputs as a new pilot. Resume only after
remote credentials or a local model/runtime is restored.

## 2026-09-13 4B six-question follow-up

SSH password authentication was restored for both 241 and 248. On 241, the
current trained 4B runtime was found at
${REMOTE_HOME}/ASA-ArknightStoryAgent/model/qwen3.5-4b with LoRA
teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2.
Default /bin Python failed because its Transformers did not recognize qwen3_5;
the reasoning environment had Transformers 5.6.2 and vLLM 0.26.0.

First reasoning-env launch failed during FlashInfer JIT because the system CUDA
installation lacks curand_precalc.h. Retried with VLLM_USE_FLASHINFER_SAMPLER=0;
then vLLM required more KV cache than the default 0.62 utilization allowed
(max_num_seqs 256 versus 180 Mamba blocks). Retried without repository changes
using --gpu-memory-utilization 0.80. Six questions then completed successfully
with the trained 4B and normal retrieval, no web context, on GPU0. Run output
was /tmp/asa_small_20260913_reasoning3.jsonl on 241; no active inference
process remained after shutdown. Temporary input/output files were not copied
into the repository.

Manual semantic review against source evidence:
- Q1 古米/坚雷 training habit: correct.
- Q2 凛冬's leadership: leader identity correct, but the claimed route to
  leadership (sweeping clubs/announcing withdrawal) is unsupported/wrong;
  partial only.
- Q3 真理 identity/schools: wrong name and wrong school relations.
- Q4 薇卡 relation/retained item: abstained despite evidence; wrong/incomplete.
- Q5 梅拿到无人机: essentially correct (slip/fall injury and 真理 cleaning,
  spraying medicine, advising medical follow-up).
- Q6 泡普卡 arrival: correct; she was found at a lumber mill by 凯尔希 and
  安赛尔, then bought by 凯尔希.

Strict full-answer correctness is therefore 3/6 = 50%; counting Q2 as partial
and allowing Q5 paraphrase gives roughly 3.5–4/6 (58%–67%). This is a fresh
six-question smoke check, not a benchmark or a promotion decision. Runtime
workarounds were temporary; production configuration/model was not changed.

## 2026-09-13 DeepSeek API Chinese six-question follow-up

Using the same six questions and the same top-10 retrieved evidence objects
from the fresh 4B run, DeepSeek-flash API was called with temperature=0,
thinking disabled, max_tokens=1024. All 6/6 requests returned HTTP 200 and
finish_reason=stop; aggregate request time was 6.814s. Credential was entered
through hidden stdin and did not appear in the script or output.

Manual semantic review:
- Q1 古米/坚雷 training habit: correct and complete.
- Q2 凛冬 leadership: leader identity is right, but the claimed cause
  (sweeping surrounding clubs) is unsupported/wrong; partial only.
- Q3 真理 name/school relations: wrong/incomplete (wrong full name and
  incorrect school assignment).
- Q4 薇卡 relation/retained item: correct.
- Q5 梅/unmanned aircraft: correct.
- Q6 泡普卡 arrival: core answer correct, though it omits the lumber-mill
  context.

Strict semantic score is approximately 4/6 = 66.7% (Q6 accepted as correct);
with a completeness penalty for Q6, 3.5–4/6. The same six-question 4B run was
manually scored 3/6 strict, so API was better by about one question in this
tiny paired check. This is not a benchmark and does not establish production
replacement. Local output:
.codex_tmp/deepseek_chinese_pilot_20260913.jsonl
SHA-256 7fb8e03f75a59c1358d377c8a8a2ee57076bc257942111bd0f8dfc7a27fe54d1.
