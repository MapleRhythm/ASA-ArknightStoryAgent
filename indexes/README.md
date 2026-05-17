# 索引目录

发布版已保留 MiniRAG 图索引：

```text
indexes/arknights_story_minirag/graph.json
```

该图包含 chunk/entity 映射、别名表、实体共现和 teacher relation，可直接被运行时配置使用。

主检索索引需要在本地根据游戏数据构建：

```bash
python scripts/build_retrieval_index.py --device cpu
```

期望输出：

```text
indexes/arknights_story/documents.jsonl
indexes/arknights_story/faiss.index
indexes/arknights_story/bm25_tokens.pkl
indexes/arknights_story/operator_aliases.json
```

如需基于本地 documents 重建 MiniRAG：

```bash
python scripts/build_minirag_index.py --progress
```
