# Guarded trace-only integration

Only the canonical answer_conclusion_generation.py diagnostic bookkeeping and
a new regression test are in scope. Preserve every existing dirty change.
Expected pre-image: 43b3611bad56687a112258d0bbf2504126207f63fe5592e3136e87b4be63259a.

Require exact AST equivalence of all unchanged functions, and equivalence of
the changed function/helper to the tested v48 isolated candidate. Keep a
canonical pre-image, use explicit source-hash guards, and verify all unrelated
source/config/runner/retriever/test hashes before and after integration.

After installation, run the full release suite with its existing no-MiniRAG
environment and replay all fourteen actual v45 states using installed code.
Strip only final_direct_fallback_attempts for strict legacy-state comparison.
Keep ten first-round terminals unchanged and both synthetic error fixtures.
No service restart, model/planner/retrieval run, training, commit, push, semantic
rule change, automatic E-ID rebinding or claim of increased answer correctness.
