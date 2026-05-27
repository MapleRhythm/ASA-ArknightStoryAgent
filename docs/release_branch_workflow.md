# 双分支发布工作流

目标：

- `main`：主分支，给用户看的推理发布版，只包含可部署内容。
- `dev-training`：副分支，保留完整训练、数据生成、评测和开发代码。
- 在 `dev-training` 更新发布目录后，用一个指令同步到 `main`。

## 推荐分支结构

```text
main           # 推理发布版，GitHub 默认分支
dev-training   # 原始训练开发版
```

当前仓库如果还在 `master`，建议先把它改成开发分支：

```bash
git branch -m dev-training
```

然后用同步脚本创建/更新 `main`。

## 一键同步到主分支

在开发分支运行：

```bash
bash scripts/sync_inference_release_branch.sh
```

脚本默认会：

- 校验 `release/ASA-ArknightStoryAgent/` 里的 Python、JSON、shell 脚本。
- 创建或复用 `main` 分支。
- 创建或复用 worktree：`.worktrees/main`。
- 把 `release/ASA-ArknightStoryAgent/` 同步为 `main` 分支根目录内容。

默认只同步，不提交。查看发布分支状态：

```bash
git -C .worktrees/main status
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

## 同步、提交并推送主分支

```bash
COMMIT=1 PUSH=1 bash scripts/sync_inference_release_branch.sh
```

推送开发分支：

```bash
git push -u origin dev-training
```

推送后到 GitHub 仓库设置里把默认分支设为：

```text
main
```

## 常用变量

```bash
RELEASE_BRANCH=main
WORKTREE_DIR=.worktrees/main
RELEASE_SOURCE_DIR=release/ASA-ArknightStoryAgent
COMMIT=0
PUSH=0
SKIP_VALIDATE=0
FORCE=0
```

如果 `main` worktree 有未提交改动，脚本默认会停止，避免覆盖手工修改。确认要用开发分支发布目录覆盖它时：

```bash
FORCE=1 bash scripts/sync_inference_release_branch.sh
```

## 推荐规则

- 不要在 `main` 长期手改功能代码；发布改动应先进入 `dev-training` 的 `release/ASA-ArknightStoryAgent/`。
- `main` 只放可部署推理版，不放训练数据、teacher 生成脚本、wandb、LLaMA-Factory 配置和训练输出。
- MiniRAG 图可以保留在 `main`：`indexes/arknights_story_minirag/graph.json`。
- 模型权重、原始游戏数据、API key、日志不要提交到任一分支。
