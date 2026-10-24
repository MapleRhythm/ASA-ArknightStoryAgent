#!/usr/bin/env bash
set -euo pipefail

ROOT=${ASA_ROOT}
ASKPASS=$ROOT/.codex_tmp/ssh_askpass_asa.sh
LOG=$ROOT/.codex_tmp/asa_watch_and_resume_241.log
SSH=(ssh -o ConnectTimeout=8 -o ConnectionAttempts=1 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${REMOTE_USER}@${REMOTE_HOST})
SCP=(scp -o ConnectTimeout=8 -o ConnectionAttempts=1 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

export SSH_ASKPASS=$ASKPASS
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] watcher started"

for attempt in $(seq 1 2160); do
  if timeout 12 setsid -w "${SSH[@]}" true; then
    echo "[$(date '+%F %T')] 241 reachable on attempt $attempt"
    setsid -w "${SCP[@]}" \
      "$ROOT/.codex_tmp/build_dense_sidecar_index.py" \
      "$ROOT/.codex_tmp/audit_query_answerability_single_split.py" \
      "$ROOT/.coverage_v8_remote/query_answerability_v3/scripts/audit_query_answerability_labels.py" \
      "$ROOT/.codex_tmp/asa_blind67_label_retry_controller_241.sh" \
      "$ROOT/.codex_tmp/asa_blind67_finalize_241.sh" \
      "$ROOT/.codex_tmp/asa_embedding_4b_chain_241.sh" \
      "$ROOT/.codex_tmp/asa_resume_241.sh" \
      ${REMOTE_USER}@${REMOTE_HOST}:/tmp/
    setsid -w "${SSH[@]}" 'chmod +x /tmp/asa_*_241.sh /tmp/build_dense_sidecar_index.py /tmp/audit_query_answerability_single_split.py && bash /tmp/asa_resume_241.sh'
    echo "[$(date '+%F %T')] bootstrap complete"
    exit 0
  fi
  if (( attempt % 15 == 0 )); then
    echo "[$(date '+%F %T')] still waiting, attempt=$attempt"
  fi
  sleep 20
done

echo "[$(date '+%F %T')] watcher timed out"
exit 1
