# COMPLETE — 2026-09-15 Exx v4 grounding fix

Final status supersedes historical notes below:
V4 final2 completed at14:02:05 Beijing, Q13correct, Q19abstain/failed claims
not exposed.5calls5stop;60evidence items;max input5224/output100;0runtime errors.
369 tests passed (366source +3promotion/rollback). All artifacts and review saved.
Published14:18:00 Beijing v1→v4; four default entry --verify-only checks pass;
default configs identical. Rollback script installed and hashed. All6 own
worker/engine PIDs exited, GPU1 18MiB/0%; unrelated GPU0 activity untouched.
REPORT.md,review.json,publication.json,final2_audit.json,entry_verification.json
complete locally. Remote acceptance sync/SSH close are final housekeeping.
No further generation, score tuning, training or code change required.

## Historical progress — not current status

User asks to fix Exx false rejection / useful-answer loss.
No subagents, training, paid APIs, public push or service restart.
Canonical local git tree dirty; all edits in independent staging; no original
file overwritten. Remote default still v1.

Remote socket /tmp/asa-exx-fix-20260915.sock, authentication already established.
Do not read/print password files.
Stage /tmp/asa-exx-grounding-v4.f6PTQq.
Release ${REMOTE_HOME}/ASA-ArknightStoryAgent/releases/exx-20260915-v4.
Final TWO fresh questions Q13/Q19, no further whole batches:
out ${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/exx_grounding_fix_final2_20260915_v4
log /tmp/asa-exx-grounding-v4.f6PTQq/smoke8.log (name retained, only2Q).
Launch via launch_smoke.py requested; inspect launch.json for PID.

All366 local source tests passed. Replay original20 first/last outputs40:
first Q8/13/18 all accepted, bad Q11/19 still rejected;
last accepted Q1/6/13/18 (lastQ8 different wording still rejected).
This does NOT change v1 score1/20 or mean v4 has measured20Q accuracy.

Trials preserved:

- v2 ../asa_exx_grounding_fix_20260915.jKfyIG/smoke8:
  1correct/8, Q10partial, other6reject.30calls28stop2length;515.74sec.
- v3 ../asa_exx_grounding_v3_20260915.YqBgQd/smoke8:
  3correct/8 Q6/8/18, Q13partial, Q10/11/16reject, Q19unreliable partial.
  26calls25stop1length;377.94sec. Neither v2 nor v3 promoted.

V4 changes:

- narrow discourse normalization for explicit reporting speaker, epistemic
  phrase, question temporal scope, question rule topic+literal full rule body;
- explicit plural antecedent SAME speaker turn, one group, literal "all" and
  matching predicate. Fix Q13 "兄弟姐妹都" vs "他们都死了".
- keep partial facts DIAGNOSTIC ONLY (v3 exposed semantically unreliable causal
  partial; lexical pass not sufficient). Never surface a partial failed claim.
- automatic same-evidence rewrite OFF in official default. Trials showed no
  robust improvement, repeats and output length exhaustion.
- old Exx final same-prompt fallback also disabled.
- preserve original candidates/partial provenance/rebinding/revalidation.

Remaining:
1 finish final2; download/audit/review Q13 correct, Q19failed claims not exposed.
2 create review.json with exact final2 result SHA, qids[13,19],
  replay_target_first_candidates_accepted[8,13,18], q13_strict_correct true,
  q19_failed_claims_not_exposed true, new_major_errors0.
3 report honest all trials; no combined invented4/8 or new20 score.
4 promote_fix.py (--review path --apply) CAS v1→v4 with doc backups. Already
  updated for v4 but needs local release-tool test adaptation if desired.
  Install rollback_exx_20260915_v4.py only if destination absent.
5 verify all default entry aliases/config hashes; save publication/lock and
  acceptance report on remote, symlink local relative raw references if needed.
6 check own processes exited/GPU free; close SSH; final Chinese concise
  "fix published,3 known cases recovered via targeted tests/replay,
  not60%, old20 not rerun; report link".
