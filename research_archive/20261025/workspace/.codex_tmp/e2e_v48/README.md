# Isolated final-direct trace retention candidate

Local staging only. No production source change or generation is authorized by
this file. Test preserving canonical primary/fallback attempt diagnostics in a
new terminal-step trace field without changing prompts, outputs, validators,
fallback eligibility, calls, sampling, E-ID bindings, or actions.

Before any source promotion, compare all fourteen frozen v45 canonical
second-round replays after removing only the new trace sidecar, and exercise
synthetic success/failure/no-fallback/empty-trace paths. Preserve v47's original
diagnosis and all pre-patch source hashes. Trace completeness is not accuracy.
