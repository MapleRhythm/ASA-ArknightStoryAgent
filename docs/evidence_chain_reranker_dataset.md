# Evidence Chain Reranker Dataset

This pipeline prepares Minimax annotation prompts and converts model outputs into reranker training files.

The current prompt is answerability-oriented: the teacher must mark `answer`,
`answer_evidence`, and hard negatives such as `background_only` and
`answer_adjacent`. This trains the reranker to prefer chains that can actually
answer the question over same-character or same-story background chains.

Runtime reranking should use the same sequence budget as training. The runtime
configs include `retrieval.reranker_max_length: 1024`; if you train with a
larger `MAX_LENGTH`, update runtime configs as well.

## 1. Generate Annotation Prompt

Generate one prompt from ordered story files:

```bash
python scripts/evidence_chain_dataset.py make-prompts \
  --story-files \
    data/ArknightsGameData/zh_CN/gamedata/story/activities/act33side/level_act33side_09_beg.txt \
    data/ArknightsGameData/zh_CN/gamedata/story/activities/act33side/level/act33side_09_a2.txt \
  --prompt-id act33side_bb9 \
  --output-dir outputs/evidence_chain_prompts/act33side_bb9
```

Generate one prompt from a directory glob:

```bash
python scripts/evidence_chain_dataset.py make-prompts \
  --story-dir data/ArknightsGameData/zh_CN/gamedata/story/activities/act33side \
  --glob '*.txt' \
  --prompt-id act33side_main \
  --output-dir outputs/evidence_chain_prompts/act33side_main
```

The script writes:

- `*.prompt.txt`: prompt text to send to Minimax.
- `prompts.jsonl`: structured prompt record.
- `manifest.json`: source file list and prompt metadata.

## 2. Call Minimax-Compatible API

The API key is read from the `MINIMAX_API_KEY` environment variable. Do not put the key in the repo.

```bash
export MINIMAX_API_KEY='your_api_key_here'
export MINIMAX_MODEL='your_minimax_fast_model_name'
```

Call the API:

```bash
python scripts/evidence_chain_dataset.py call-api \
  outputs/evidence_chain_prompts/act33side_bb9/act33side_bb9.prompt.txt \
  --api-base https://api.svips.org \
  --model "$MINIMAX_MODEL" \
  --output-json data/processed/evidence_chain_annotations/act33side_bb9.json \
  --response-format-json
```

The command writes:

- `act33side_bb9.json`: parsed JSON content returned by the model.
- `act33side_bb9.json.raw.json`: raw API response for debugging.

If the endpoint does not support OpenAI-compatible `response_format`, remove `--response-format-json`.

## 3. Save Minimax Output Manually

Save the model response as a JSON file, for example:

```text
data/processed/evidence_chain_annotations/act33side_bb9.json
```

Expected schema:

```json
{
  "all_evidence": [
    {
      "id": "E1",
      "text": "证据片段内容"
    }
  ],
  "rerank_dataset": [
    {
      "query": "用户问题",
      "query_type": "fact",
      "answer": "由 gold chain 支持的简短答案",
      "answer_evidence": ["E2"],
      "answer_focus": "E2 直接揭示答案，E1 提供必要上下文。",
      "candidate_chain_list": [
        {
          "label": "positive",
          "type": "gold",
          "chain": ["E1", "E2"],
          "score": 1.0,
          "score_reason": "证据链按原文时序完整回答问题。"
        }
      ]
    }
  ]
}
```

Allowed `query_type` values:

- `fact`
- `relation`
- `causality`
- `reasoning`

Recommended candidate types:

- `gold`: complete correct chain, score `1.0`.
- `shuffled_order`: same evidence as gold but wrong order, score `0.45-0.65`.
- `irrelevant_mixed`: partly related plus misleading or unrelated evidence, score `0.2-0.45`.
- `incomplete`: related but missing key links, score `0.3-0.55`.
- `background_only`: same character/story/event background but no answer-bearing evidence, score `0.05-0.30`.
- `answer_adjacent`: adjacent to the answer scene but missing the answer-bearing evidence, score `0.25-0.50`.
- `same_entity_distractor`: same entity but answers a different question, score `0.15-0.40`.
- `partial_answer`: contains part of the answer but misses a key conclusion, score `0.35-0.60`.
- `misleading_chain`: relevant-looking but would lead to a wrong answer, score `0.20-0.60`.

Recommended chain lengths:

- `fact` / `relation` gold chains: 2-4 evidence items.
- `causality` / `reasoning` / `reveal` / `mystery` / `answerability` gold chains: 3-8 evidence items.
- Complex cases may use up to 10 evidence items.
- `background_only` hard negatives should often be 4-8 evidence items, so the model learns that long context without answer-bearing evidence is still insufficient.
- Keep `answer_evidence` early in the chain, ideally before 70% of the token budget.

## 4. Validate And Export Training Files

```bash
python scripts/evidence_chain_dataset.py export \
  data/processed/evidence_chain_annotations/act33side_bb9.json \
  --output-dir data/processed/evidence_chain_reranker/act33side_bb9
```

The script writes:

- `annotations.cleaned.jsonl`: normalized annotation payloads with generated `chain_text`.
- `reranker_listwise.jsonl`: one record per query with scored candidates.
- `reranker_pairwise.jsonl`: one record per positive/negative pair.
- `flag_embedding_reranker.jsonl`: `{"query", "pos", "neg"}` format for common BGE/FlagEmbedding reranker training.
- `validation_issues.jsonl`: warnings and errors.
- `manifest.json`: export summary.

## 5. Batch Generation

Batch mode uses one story folder as one complete prompt. For example,
`data/ArknightsGameData/zh_CN/gamedata/story/activities/act42side`
is recursively merged into one `act42side.prompt.txt`; it is not split by
stage file or character budget unless you explicitly set `--max-source-chars`.
By default, folders with less than `--min-source-chars 10000` rendered source
characters are skipped to avoid very small gameplay/system fragments.

Default batch paths now write answerability data to:

- `outputs/evidence_chain_prompts/batch_v2_answerability/`
- `data/processed/evidence_chain_annotations/batch_v2_answerability/`
- `data/processed/evidence_chain_reranker/batch_v2_answerability/`

Run a dry run first:

```bash
python scripts/run_evidence_chain_batch.py \
  --dry-run \
  --target-samples 1000 \
  --max-prompts 90 \
  --sample-activities 120 \
  --min-questions 10 \
  --max-questions 10 \
  --seed 20260509
```

Generate about 1000 usable samples:

```bash
export MINIMAX_API_KEY='your_api_key_here'
export MINIMAX_MODEL='MiniMax-M2.7-highspeed'

python scripts/run_evidence_chain_batch.py \
  --target-samples 1000 \
  --max-prompts 90 \
  --sample-activities 120 \
  --min-questions 10 \
  --max-questions 10 \
  --seed 20260509 \
  --api-base https://api.svips.org \
  --model "$MINIMAX_MODEL" \
  --max-tokens 30000 \
  --min-source-chars 10000 \
  --chunk-source-chars 70000 \
  --priority-activities act10mini,act17side,act34side \
  --sleep-seconds 0.5
```

If you use the svips key variable from API mode, pass it directly:

```bash
export SVIPS_API_KEY='your_api_key_here'

python scripts/run_evidence_chain_batch.py \
  --target-samples 20 \
  --max-prompts 6 \
  --sample-activities 20 \
  --min-questions 10 \
  --max-questions 10 \
  --seed 20260510 \
  --api-base https://api.svips.org \
  --api-key-env SVIPS_API_KEY \
  --model MiniMax-M2.7-highspeed \
  --max-tokens 30000 \
  --min-source-chars 10000 \
  --chunk-source-chars 70000 \
  --priority-activities act10mini,act17side,act34side \
  --sleep-seconds 0.5
```

Resume an interrupted run:

```bash
python scripts/run_evidence_chain_batch.py \
  --target-samples 1000 \
  --max-prompts 90 \
  --sample-activities 120 \
  --min-questions 10 \
  --max-questions 10 \
  --seed 20260509 \
  --api-base https://api.svips.org \
  --model "$MINIMAX_MODEL" \
  --max-tokens 30000 \
  --resume
```

Batch outputs:

- `outputs/evidence_chain_prompts/batch_v2_answerability/`: generated prompt files and `batch_plan.jsonl`.
- `data/processed/evidence_chain_annotations/batch_v2_answerability/`: raw model annotation JSON files.
- `data/processed/evidence_chain_reranker/batch_v2_answerability/`: merged reranker training data.

## 6. Fine-Tune Reranker

Run a dry run first:

```bash
DRY_RUN=true bash scripts/run_reranker_sft.sh
```

Train the local reranker from `model/reranker/bge-reranker-v2-m3`:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
TRAIN_FILE=data/processed/evidence_chain_reranker/batch_v1/reranker_pairwise.jsonl \
OUTPUT_DIR=model/reranker/bge-reranker-v2-m3-evidence-chain \
OVERWRITE_OUTPUT_DIR=true \
bash scripts/run_reranker_sft.sh
```

For the answerability v2 dataset, use:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHON_BIN=/home/zhb/miniconda3/envs/train/bin/python \
TRAIN_FILE=data/processed/evidence_chain_reranker/batch_v2_answerability/reranker_pairwise.jsonl \
OUTPUT_DIR=model/reranker/bge-reranker-v2-m3-evidence-chain-answerability \
OVERWRITE_OUTPUT_DIR=true \
bash scripts/run_reranker_sft.sh
```

Useful overrides:

- `MAX_LENGTH=1024`: maximum query/evidence-chain sequence length.
- `TRAIN_FILE=data/processed/evidence_chain_reranker/batch_v2_answerability/reranker_pairwise.jsonl`: train on the answerability dataset.
- `EPOCHS=2`: training epochs.
- `LEARNING_RATE=2e-5`: reranker fine-tuning learning rate.
- `PER_DEVICE_TRAIN_BATCH_SIZE=2` and `GRADIENT_ACCUMULATION_STEPS=8`: effective batch size per process.
- `TRAIN_GPUS=0,1`: used when `CUDA_VISIBLE_DEVICES` is not set.
