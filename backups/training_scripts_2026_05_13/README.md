# 训练脚本备份 — 2026-05-13

本目录是新增 MiniRAG / CRAG / Self-RAG / DPO 四个方案的数据生成与训练逻辑改动**之前**的快照。

## 备份内容

| 文件 | 角色 |
|---|---|
| `train_evidence_chain_reranker.py` | reranker pairwise softplus 训练脚本（pre-DPO） |
| `run_reranker_sft.sh` | reranker 训练入口脚本 |
| `run_train.sh` | LLaMA Factory 4B SFT 入口脚本 |
| `evidence_chain_dataset.py` | reranker 训练数据生成（pre-MiniRAG 三元组） |
| `generate_sft_from_teacher.py` | 4B SFT 数据生成入口 |
| `sft_teacher.py` | 教师 prompt 与样本规范化（pre-Self-RAG token / pre-CRAG refinement） |
| `sft_teacher_generation.json` | 教师生成配置（pre-`max_evidence_docs_per_request=12`） |

## 还原方法

```bash
cp backups/training_scripts_2026_05_13/train_evidence_chain_reranker.py scripts/
cp backups/training_scripts_2026_05_13/run_reranker_sft.sh scripts/
cp backups/training_scripts_2026_05_13/run_train.sh scripts/llama_factory/
cp backups/training_scripts_2026_05_13/evidence_chain_dataset.py scripts/
cp backups/training_scripts_2026_05_13/generate_sft_from_teacher.py scripts/
cp backups/training_scripts_2026_05_13/sft_teacher.py src/<主代码包>/data/
cp backups/training_scripts_2026_05_13/sft_teacher_generation.json configs/
```

## 改动范围（备份之后做的事）

- **A. MiniRAG**：`evidence_chain_dataset.py` PROMPT_TEMPLATE 增加 `entity_relations` 三元组字段；新增 `scripts/build_minirag_index.py` 从 reranker 标注产物构图
- **B. CRAG-lite**：`sft_teacher.py` `build_conclusion_prompt_bundle` 注入 knowledge-refinement 双段 evidence 示例
- **C. Self-RAG**：`sft_teacher.py` conclusion / hypothesis 输出 schema 增加 `reflect_tokens` 字段
- **D. DPO reranker**：新增 `scripts/build_dpo_reranker_dataset.py` + `scripts/train_dpo_reranker.py`
</content>
</invoke>
