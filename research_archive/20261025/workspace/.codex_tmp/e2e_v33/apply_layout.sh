#!/usr/bin/env bash
set -euo pipefail
cd ${REMOTE_HOME}/ASA-ArknightStoryAgent
layout_exp=outputs/experiments/grounded_fact_layout_20260910.v33
layout_src=release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py
layout_test=release/ASA-ArknightStoryAgent/tests/test_grounded_fact_layout_regression.py
test -f "$layout_exp/isolated_replay_v1.json"
test ! -e "$layout_exp/applied_source.sha256"
test ! -e "$layout_test"
test ! -e "$layout_src.v33-staged"
cmp -s "$layout_src" "$layout_exp/baseline_grounded_fact_answers.py"
git diff -- "$layout_src" > "$layout_exp/preexisting_source_diff.patch"
cp -p "$layout_exp/grounded_fact_answers.py" "$layout_src.v33-staged"
cp -p "$layout_exp/test_grounded_fact_layout_regression.py" "$layout_test"
mv -T "$layout_src.v33-staged" "$layout_src"
sha256sum "$layout_src" "$layout_test" > "$layout_exp/applied_source.sha256"
git diff --check -- "$layout_src" "$layout_test"
printf '%s\n' "Applied only the grounded-fact layout change and its new regression tests."
