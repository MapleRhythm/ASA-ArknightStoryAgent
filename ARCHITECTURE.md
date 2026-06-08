# 链路代码结构

本文档说明发布版中检索与推理链路的代码边界，便于维护时定位问题。

## 公共入口

- `scripts/build_retrieval_index.py`：构建故事语料索引与 MiniRAG 图。
- `scripts/query_retrieval.py`：直接调试检索结果。
- `scripts/run_cpu_inference.py`：本地推理入口。
- `api-mode/run_api_inference.py`：API 模式入口。
- `src/asa_arknight_story_agent/inference/cpu_pipeline.py`：推理管线装配入口。
- `src/asa_arknight_story_agent/retrieval/hybrid.py`：混合检索器装配入口。
- `src/asa_arknight_story_agent/retrieval/minirag.py`：MiniRAG 索引装配入口。

## 数据构建

- `data/story_parser.py`：构建最终语料文档，组合剧情正文、档案、语音。
- `data/story_text.py`：解析剧情文本并切分片段。
- `data/story_metadata.py`：加载游戏表中的关卡、活动、章节等元数据。
- `data/story_meta_resolver.py`：将剧情文件路径解析成文档元信息。
- `data/operator_aliases.py`：构建干员别名表。

## 检索链路

- `retrieval/hybrid.py`：只负责混合检索器加载与索引初始化。
- `retrieval/hybrid_components/`：dense/sparse 检索、RRF 融合、证据链构建、query 分析、打分调整。
- `retrieval/minirag.py`：只负责 MiniRAGIndex 加载与兼容导出。
- `retrieval/minirag_components/`：实体抽取、图构建、关系打分、PPR 传播、章节 scope。

## 推理链路

- `inference/pipeline/`：推理状态、轮次编排、结果生成。
- `inference/retrieval/`：推理时的检索轮次、MiniRAG 扩展、rerank 组合。
- `inference/generation/`：假设生成、结论生成、prompt 渲染。
- `inference/payload/`：模型 JSON/类 JSON 输出解析与规范化。
- `inference/evidence/`：证据准备、选择、渲染、CRAG 精炼。
- `inference/grounding/`：答案 grounding、引用检查、回退策略。
- `inference/planning/`：问题理解、实体抽取、追问/续检索规划。
- `inference/web_context/`：可选网页上下文检索。
- `inference/model_runtime/`：llama.cpp 与 vLLM 运行器。
- `inference/common/`：共享词典、正则和文本工具。

## 维护原则

- 公共入口保持稳定，内部 helper 优先放入对应子包。
- 单个模块只承载一种职责；若同时出现加载、打分、解析、编排，应继续拆分。
- 发布版不得包含训练、上传、内部路径或临时运行产物。
