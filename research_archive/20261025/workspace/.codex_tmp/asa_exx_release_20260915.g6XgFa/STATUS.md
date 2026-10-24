# COMPLETE — 2026-09-15 Exx test + formal default

Final update (supersedes historical progress below): second run20/20 completed,
76/76stop,56x20=1120 full evidence checks passed. Correct1/20, partial2/20,
abstain17/20; last raw content correct4/20 (Q6/8/13/18), three lost to lexical
false rejection. REPORT.md, review.json, summary.json and all artifacts saved.
Publication executed at12:27:36 Beijing time; root CLI verify-only passes and
current-release points to releases/exx-20260915-v1. publication.json records backups.
No public push, service restart, further generation/training needed.
Only final entry/cleanup verification and report transfer may still be in progress.

## Historical work log (not current status)

User explicitly asked to retest previous questions on new chain and make new chain formal.
No subagents authorized. No public git/HF push authorized or performed.

Remote: ${REMOTE_USER}@${REMOTE_HOST}, ControlMaster socket /tmp/asa-exx-release-20260915.sock.
Canonical root ${REMOTE_HOME}/ASA-ArknightStoryAgent. Original legacy root is goldenglow;
new Exx package previously only under release/ASA-ArknightStoryAgent.

## Completed

- New v55 adapter verified SHA 0adb413f5ba8b40423341cfd0f1e17f2624684c43e9ff8c7035f2ba115426dd1.
- First poorly configured run /tmp/asa_newexx_e2e20_20260915 stopped (SIGINT own PID1303329),
  15 completed answers retained under local failed_attempt1; engine1306518 gone.
  Root cause: package constructor clamps 0 char budgets to120 and ignores full-doc config.
  Not a meaningful model evaluation; report as failed integration attempt, not suppress it.
- Downloaded remote package into THIS local directory; patched ONLY this staging copy:
  zero sentinel/full-doc plumbing, direct top20 chunk selection, all full-prompt guard,
  Exx planner prefix removal, no hardcoded512-token conclusion cap (1536 configured).
  107 relevant local tests passed, plus 3 dispatch tests (110 distinct total).
  Remote reasoning Python lacks pytest; no dependencies installed.
- Persistent frozen release built at
  ${REMOTE_HOME}/ASA-ArknightStoryAgent/releases/exx-20260915-v1.
  Model copied (not relying on temp checkpoint); base/reranker/embedding/index symlinks.
  RELEASE_LOCK hashes source/config/all base & adapter weights/tokenizers.
  Official entry scripts/run_exx_official.py verifies lock; verified successfully.
- Second real end-to-end run started on GPU1 PID1335003 (engine1335654):
  remote output ${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/exx_release_e2e20_20260915_v1
  remote stage /tmp/asa-exx-release-stage.iuTmWQ
  log stage/attempt2.log. At last check12/20 completed, 36x20 full evidence checks pass.
  Same questions and scoring as prior20; real new retrieval, no gold injection/API.
  New package implementation differs from prior monolith; not weight-only A/B.
  Baseline config EXACT previous20 config except explicit Exx/model/generation changes.

## Publication prepared but NOT executed yet

Local publish_release.py and files uploaded to remote stage.
It backs up current root CLI/config/launcher/README under outputs/release_backups/exx-20260915-v1.
Replaces default root CLI (dispatch only; old implementation retained), root default configs,
old-name run_latest_kto_interactive.sh + new run_exx_official.{py,sh}, docs/README;
creates current-release -> releases/exx-20260915-v1 last.
Explicit other historical configs still use old implementation. New default fails closed if missing.
Rollback script restores exact preimages and refuses overwriting subsequent modifications.
No actual publication until execute:
python /tmp/asa-exx-release-stage.iuTmWQ/publish_release.py
  --canonical ${REMOTE_HOME}/ASA-ArknightStoryAgent
  --staging /tmp/asa-exx-release-stage.iuTmWQ

Need complete20, download artifacts, audit/review all final answers, summarize/write report;
publish default (user authorized regardless of quality but DO NOT claim60% if fails);
verify canonical default config aliases and shell entry --verify-only;
save publication receipt and report both local and remote. No extra inference necessary.

Existing workspace dirty repos preserved; only this staging copy and new remote version created.
Local first15 run answers all abstain due missing evidence. In valid second run, first12:
Q1 partial; Q2/3/4/5 rejected; Q6 correct; Q7/8/9/10/11/12 rejected.
Q8 has semantically correct raw and E5 is the correct source, but lexical guard
rejects phrases “她想知道/前提条件/卡蒂时附加”; still count final rejected.
Do not invent optimistic results. Quality gate is separate from version pinning.
