# v55 attempt02: fused-loss retry after preserved OOM

Attempt01 completed5 optimizer steps, then exceeded the85% allocator limit
at a long-example backward pass requesting8.61GiB. Its original snapshot,
logs, startup report and failure are retained. There was no saved checkpoint
yet; attempt02 restarts from the SAME original parent, not an invented resume.

Only memory-computation change: use the already-present Liger0.8.2
Qwen3.5 fused linear cross entropy; do NOT change RMSNorm/SwiGLU/rope.
Data, order seed, full input context, supervised labels, optimizer and epoch
settings remain identical. Existing package files are read-only; explicit
PYTHONPATH makes this dependency visible in this run only. GPU loss/gradient
comparison on tiny actual Qwen3.5 conditional models must pass before training.
No guarantee of bitwise identical training trajectories is made.
The85% allocation guard remains; do not raise it to crowd the resident task.

Date: 2026-09-11 Asia/Shanghai. User explicitly authorized training on the
historical 910 train / 95 validation records and sharing the 248 A100 without
stopping its resident process. This is not the failed v54 GLM annotation pilot.

241 is canonical. Freeze original configuration, unchanged historical dataset,
scripts, model/adapter identities and run-specific configuration there first;
copy the immutable snapshot to a fresh 248 run directory and verify hashes.
No edits to old data, old checkpoints, installed packages or production settings.
No API calls, new labels, benchmark test-set training or service changes.

Initialization follows the existing dataset's SFT configuration:
Qwen3.5-4B plus `exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827`.
It is NOT continuation of the currently deployed SODA adapter.
LoRA rank16, learning rate3e-6, one epoch, batch1/accumulation8, bf16,
gradient checkpointing, qwen3_nothink, cutoff12000, full original records.
Verify full encoded samples fit before launching; do not silently truncate.
The historical machine labels have not become independently human-verified
gold. Validation loss is not narrative QA accuracy or retrieval quality.

Run-specific changes: fresh dataset snapshot, output/run/cache paths, explicit
seed20260911, first-step logging, retain this run's checkpoints, and no reporting
to external services. PyTorch allocator limit85% of A100 memory; this is not a
guarantee covering all non-PyTorch allocations or other users' future demand.
Startup requires free GPU memory >=36000MiB and >=6GiB output disk space.
Only the already approved resident PID450914 may preexist at initial launch.
Do not kill or modify that process. Use a run lock and fresh start receipt.

CPU preflight compares unlimited encoding against actual cutoff encoding for
all1005 records, checks nonempty supervised tokens and exact dataset SHA.
Training runs in a detached tmux session; process and first optimizer-step
receipts are authoritative evidence of actual startup. Failures remain saved.
Completion does not promote the candidate; benchmark/read/retrieval evaluation
is still separate work. Return generated artifacts to a fresh241 incoming
directory when complete, with hash checks, before any potential promotion.
