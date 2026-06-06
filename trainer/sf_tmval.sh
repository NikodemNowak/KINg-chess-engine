#!/usr/bin/env bash
# Validate the time-management fix: build the fixed engine, run unit tests, then
# SPRT vs the current production engine at overhead=200 (bug zone) and overhead=30.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king

cmake -S $REPO -B $K/v-tmfix >/dev/null 2>&1
cmake --build $K/v-tmfix --target engine unit_tests -j 22 >/dev/null 2>&1 \
  && echo "built v-tmfix" || { echo "build FAILED"; exit 1; }

echo "=== unit_tests ==="
$K/v-tmfix/unit_tests 2>&1 | tail -3

echo "=== SPRT @ overhead=200 (bug zone): tmfix vs allsf ==="
CONC=11 bash $REPO/trainer/ccmatch_oh.sh $K/v-tmfix/engine $K/v-allsf/engine tmfix allsf 400 8+0.08 tmval200 200

echo "=== SPRT @ overhead=30 (no-regression): tmfix vs allsf ==="
CONC=11 bash $REPO/trainer/ccmatch_oh.sh $K/v-tmfix/engine $K/v-allsf/engine tmfix allsf 400 8+0.08 tmval30 30
echo "=== DONE ==="
