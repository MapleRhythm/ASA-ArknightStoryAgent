#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
RELEASE_SOURCE_DIR="${RELEASE_SOURCE_DIR:-$ROOT_DIR/release/ASA-ArknightStoryAgent}"
RELEASE_BRANCH="${RELEASE_BRANCH:-inference-release}"
WORKTREE_DIR="${WORKTREE_DIR:-$ROOT_DIR/.worktrees/$RELEASE_BRANCH}"
COMMIT="${COMMIT:-0}"
PUSH="${PUSH:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
FORCE="${FORCE:-0}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-sync inference release from development branch}"

if [[ ! -d "$RELEASE_SOURCE_DIR" ]]; then
  echo "Release source directory not found: $RELEASE_SOURCE_DIR" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not found." >&2
  exit 1
fi

if [[ "$SKIP_VALIDATE" != "1" ]]; then
  echo "[validate] python syntax"
  python -m py_compile $(
    find "$RELEASE_SOURCE_DIR/src" "$RELEASE_SOURCE_DIR/scripts" "$RELEASE_SOURCE_DIR/api-mode" \
      -name '*.py' -print
  )

  echo "[validate] json configs"
  for file in "$RELEASE_SOURCE_DIR"/configs/*.json "$RELEASE_SOURCE_DIR"/api-mode/runtime_api.json; do
    python -m json.tool "$file" >/dev/null
  done

  echo "[validate] shell scripts"
  for file in "$RELEASE_SOURCE_DIR"/scripts/*.sh; do
    bash -n "$file"
  done
fi

mkdir -p "$(dirname "$WORKTREE_DIR")"

CREATED_RELEASE_BRANCH=0
if git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
  if [[ ! -d "$WORKTREE_DIR/.git" && ! -f "$WORKTREE_DIR/.git" ]]; then
    echo "[worktree] add existing branch $RELEASE_BRANCH -> $WORKTREE_DIR"
    git worktree add "$WORKTREE_DIR" "$RELEASE_BRANCH"
  fi
else
  echo "[worktree] create orphan branch $RELEASE_BRANCH -> $WORKTREE_DIR"
  git worktree add --detach "$WORKTREE_DIR" HEAD
  git -C "$WORKTREE_DIR" switch --orphan "$RELEASE_BRANCH"
  git -C "$WORKTREE_DIR" rm -r --quiet --ignore-unmatch .
  CREATED_RELEASE_BRANCH=1
fi

if [[ "$CREATED_RELEASE_BRANCH" != "1" && -n "$(git -C "$WORKTREE_DIR" status --porcelain)" && "$FORCE" != "1" ]]; then
  cat >&2 <<MSG
Release worktree has uncommitted changes:
  $WORKTREE_DIR

Review or commit them first, or rerun with FORCE=1 to overwrite the worktree from:
  $RELEASE_SOURCE_DIR
MSG
  exit 1
fi

echo "[sync] $RELEASE_SOURCE_DIR/ -> $WORKTREE_DIR/"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-*' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$RELEASE_SOURCE_DIR"/ "$WORKTREE_DIR"/

echo "[status] release worktree"
git -C "$WORKTREE_DIR" status --short

if [[ "$COMMIT" == "1" ]]; then
  git -C "$WORKTREE_DIR" add -A
  if git -C "$WORKTREE_DIR" diff --cached --quiet; then
    echo "[commit] no changes to commit"
  else
    git -C "$WORKTREE_DIR" commit -m "$COMMIT_MESSAGE"
  fi
fi

if [[ "$PUSH" == "1" ]]; then
  git -C "$WORKTREE_DIR" push -u origin "$RELEASE_BRANCH"
fi

cat <<MSG

Done.

Development branch:
  $(git branch --show-current)

Release branch:
  $RELEASE_BRANCH

Release worktree:
  $WORKTREE_DIR

Useful next commands:
  git -C "$WORKTREE_DIR" status
  git -C "$WORKTREE_DIR" log --oneline -5
MSG
