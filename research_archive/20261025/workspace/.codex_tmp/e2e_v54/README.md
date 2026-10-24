# v54 GLM claim–evidence annotation pilot

Date: 2026-09-11 Asia/Shanghai; receipts explicitly use UTC.
User chose option3 (claim–evidence entailment) and supplied an API credential.
Do not include it in files/logs/arguments/headers dumps or version control.

Scope: nine unique v50/v53 development claims over three complete visible
source packets: six assistant-authored contrasts and three historical model
claims. Full SHA-bound cached inputs from closed canonical v53 are used;
all source, speaker text and neighboring context are retained.
GLM receives question, claim and source only, not previous labels/reasons.
Composite claims require support for all clauses; atomic decomposition and
training-data expansion are separate future work, not implied by this pilot.

Official standard API endpoint:
`https://open.bigmodel.cn/api/paas/v4/chat/completions`.
Model: `glm-5.3`, thinking enabled, high effort, temperature0, max8192 tokens.
No coding-plan endpoint, third-party proxy or credential discovery.
Endpoint/model/thinking parameter documentation was retrieved from official
BigModel docs on 2026-09-11. Pricing is not assumed or estimated here.

Budget bound: <=9 requests, one at a time, no automatic retries. Stop on first
network/auth/format failure. First real annotation doubles as connectivity
check; do not send whole dataset before checking this pilot.

Labels: supported/contradicted/insufficient; relevance independently
answers/context/unrelated/unclear. Exact span resolution is deterministic.
JSON and source identity are strict; no bracket/fence/schema/citation repair.
Raw responses, model usage and failures are preserved, but credentials and
HTTP error body/message are never persisted. Redirects are refused.
Use getpass on a non-echoing terminal. No old labels/config/model modifications.

The canonical241 SSH connection currently times out. This directory remains
local cache/preparation or API staging, NOT the canonical completed dataset.
All outputs require later hash-checked sync to a fresh canonical experiment
directory. Do not claim successful sync or training readiness while offline.
GLM labels are machine candidates, not human gold, and these known development
questions are not independent holdout. No training/GPU/service/commit/push.

Run `python3 -m unittest discover -s .codex_tmp/e2e_v54`, then
`python3 .codex_tmp/e2e_v54/glm_annotation.py prepare`.
Use `run` in a TTY for credential entry; existing journals/preflight are never
overwritten. If unsuccessful, preserve this attempt and diagnose before a new
versioned retry. Never repeat a charge-bearing successful request silently.
