#!/usr/bin/env bash
# Baseline build (no SE_* macros) + unit tests, to verify the always-compiled
# parts of the search-feature integration. Gentle -j to coexist with labeling.
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
cmake -S "$REPO" -B "$K/v-sbase" >/dev/null 2>&1
if cmake --build "$K/v-sbase" --target engine unit_tests -j4 > "$K/sbuild.log" 2>&1; then
  echo "BASELINE BUILD OK"
  "$K/v-sbase/unit_tests" 2>&1 | tail -3
else
  echo "BASELINE BUILD FAILED"
  grep -iE "error" "$K/sbuild.log" | head -8
fi
