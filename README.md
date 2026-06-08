# ASA-ArknightStoryAgent

这里是 ASA（ArknightStoryAgent）！这个项目旨在搭建一个可以纯本地离线部署的 ai agent。项目采用RAG检索+微调Qwen 3.5 4b模型，明日方舟游戏文本数据以及萌娘百科设定数据作为检索对象，旨在从千万级文本内容中对剧情细节进行挖掘以及对复杂情节进行推理，项目配套有交互式前端以及一键环境配置脚本！（温蒂赛高！）

发布版入口在 `release/ASA-ArknightStoryAgent/`。完整部署、配置和链路说明见 [release/ASA-ArknightStoryAgent/README.md](release/ASA-ArknightStoryAgent/README.md)。

## 快速开始

```bash
cd release/ASA-ArknightStoryAgent
bash scripts/setup_gpu_reranker_qwen35_4b.sh
source .venv-gpu/bin/activate
python scripts/build_retrieval_index.py --device cuda
bash scripts/run_gpu_reranker_qwen35_4b.sh "岁兽是什么？"
```

CPU 本地和 CPU API 流程见发布版 README。
