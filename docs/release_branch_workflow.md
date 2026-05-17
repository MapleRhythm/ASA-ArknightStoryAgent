# 双分支发布工作流

目标：

- 开发分支保留完整训练、数据生成、评测和推理代码。
- 发布分支只保留 `release/ASA-ArknightStoryAgent/` 内的推理发布版内容。
- 开发分支更新后，用一个指令同步到发布分支。

## 分支建议

推荐命名：

```text
master 或 dev-training      # 原始训练开发版
inference-release           # 推理发布版
```

如果你想把当前 `master` 改名为开发分支：

```bash
git branch -m dev-training
```

不改名也可以，脚本会把当前分支视为开发分支。

## 一键同步

在开发分支运行：

```bash
bash scripts/sync_inference_release_branch.sh
```

脚本会：

- 校验发布目录里的 Python、JSON、shell 脚本。
- 创建或复用 `inference-release` 分支。
- 创建或复用 worktree：`.worktrees/inference-release`。
- 把 `release/ASA-ArknightStoryAgent/` 同步为发布分支根目录内容。

默认只同步，不提交。查看结果：

```bash
git -C .worktrees/inference-release status
```

## 同步并提交

```bash
COMMIT=1 bash scripts/sync_inference_release_branch.sh
```

自定义提交信息：

```bash
COMMIT=1 COMMIT_MESSAGE="release: sync inference package" \
  bash scripts/sync_inference_release_branch.sh
```

## 同步、提交并推送

```bash
COMMIT=1 PUSH=1 bash scripts/sync_inference_release_branch.sh
```

首次推送后，GitHub 上会出现两个分支：

```text
dev-training 或 master
inference-release
```

可以把 `inference-release` 设置为给用户看的发布分支，或单独创建 GitHub Release。

## 常用变量

```bash
RELEASE_BRANCH=inference-release
WORKTREE_DIR=.worktrees/inference-release
RELEASE_SOURCE_DIR=release/ASA-ArknightStoryAgent
COMMIT=0
PUSH=0
SKIP_VALIDATE=0
FORCE=0
```

如果 release worktree 有未提交改动，脚本默认会停止，避免覆盖手工修改。确认要用开发分支发布目录覆盖它时：

```bash
FORCE=1 bash scripts/sync_inference_release_branch.sh
```

## 推荐规则

- 不要直接在 `inference-release` 分支长期手改功能代码；发布改动应先回到开发分支的 `release/ASA-ArknightStoryAgent/`。
- 发布分支只放可部署推理版，不放训练数据、teacher 生成脚本、wandb、LLaMA-Factory 配置和训练输出。
- MiniRAG 图可以保留在发布分支：`indexes/arknights_story_minirag/graph.json`。
- 模型权重、原始游戏数据、API key、日志不要提交到任一分支。
