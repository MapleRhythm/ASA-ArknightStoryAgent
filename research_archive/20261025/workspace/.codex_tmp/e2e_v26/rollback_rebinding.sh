#!/usr/bin/env bash
set -euo pipefail
cd ${REMOTE_HOME}/ASA-ArknightStoryAgent
release=release/ASA-ArknightStoryAgent
target="$release/src/asa_arknight_story_agent/inference/grounding"
prior=outputs/rollback_backups/20260909_quote_rebind_batch_exit_preapply
backup=outputs/rollback_backups/20260909_quote_rebind_withdrawn_v26
stage=outputs/experiments/quote_rebind_withdrawal_20260909.v26
test -d "$stage"
test ! -e "$backup"
for file in quote_checks.py quote_match_utils.py quote_validation.py; do
    test -f "$target/$file" && test ! -L "$target/$file"
    test -f "$prior/$file" && test ! -L "$prior/$file"
done
printf '%s\n' \
  "0f3826f79b73d9d77bae54093415e73a2d5efa3b01126bb8bc8a01ad6e11cfe2  $target/quote_checks.py" \
  "6abc53c3d26b2a0902147d6180ce3b9dc048093b55379735edb4465f25a3103d  $target/quote_match_utils.py" \
  "a4ad250f42f94b2884d3696d1cf46745f1e5c74d65d80d8aa3bb4306afd55e5b  $target/quote_validation.py" \
  "280c4525574447e7eb0b39e504be9120f4504b5fd2e38b9cd70ca6b64cb1b0d3  $release/tests/test_unique_quote_rebinding.py" \
  "2108ce590be98361c896e82a7e41f86fa007cef14f5521521be634f648ec07d2  $prior/quote_checks.py" \
  "d17cf94bc34089fd5fd49a1dde205e4779ade8476088f7ffa8366fb5507132a1  $prior/quote_match_utils.py" \
  "a1a8b719dc6a7892e6a060b87f8169291c576826831ba0404c03d61c4fcdbb7c  $prior/quote_validation.py" |
  sha256sum --check
mkdir "$backup"
for file in quote_checks.py quote_match_utils.py quote_validation.py; do
    cp -p "$target/$file" "$backup/$file"
done
cp -p "$release/tests/test_unique_quote_rebinding.py" "$backup/test_unique_quote_rebinding.py"
for file in quote_checks.py quote_match_utils.py quote_validation.py; do
    cp -p "$prior/$file" "$target/$file"
    cmp "$prior/$file" "$target/$file"
done
cp "$stage/test_unique_quote_rebinding.py" "$release/tests/test_unique_quote_rebinding.py"
sha256sum "$target/quote_checks.py" "$target/quote_match_utils.py" \
    "$target/quote_validation.py" "$release/tests/test_unique_quote_rebinding.py"
git diff --check
