#!/usr/bin/env bash
# Re-test the fast-TC-neutral features at the competition's SLOW TC (singular
# proved fast-TC under-measures depth-dependent features). Each is tested ON TOP
# of the now-default singular extensions, vs the singular-only baseline.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
# singular-only baseline (singular is now default-on in source):
cmake -S "$REPO" -B "$K/v-sbaseS" >/dev/null 2>&1
cmake --build "$K/v-sbaseS" --target engine -j10 >/dev/null 2>&1 && echo "built singular baseline"
for M in SE_PROBCUT SE_CONTHIST2; do
  cmake -S "$REPO" -B "$K/v-s-$M" -DCMAKE_CXX_FLAGS="-D$M" >/dev/null 2>&1
  cmake --build "$K/v-s-$M" --target engine -j10 >/dev/null 2>&1 && echo "built $M"
  echo "=== SPRT $M+singular vs singular @ 30+0.3 (slow) ==="
  CONC=11 bash "$K/ccmatch.sh" "$K/v-s-$M/engine" "$K/v-sbaseS/engine" "$M" sing 400 30+0.3 "slow_$M"
done
echo "=== DONE ==="
