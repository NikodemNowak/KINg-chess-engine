#!/usr/bin/env bash
# Build the 77M net and test whether scaling 33M->77M (same lam1.0 recipe) helps.
# Also probe whether search overrides the suspicious static KQ-vs-K eval.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king

cmake -S $REPO -B $K/v-allsf -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
  -DNNUE_NET=$REPO/nets/king_allsf_hl512.bin -DNNUE_SAMPLES=$REPO/trainer/s_allsf.txt >/dev/null 2>&1
cmake --build $K/v-allsf --target engine -j 22 >/dev/null 2>&1 \
  && echo "built v-allsf" || { echo "build FAILED"; exit 1; }

echo "=== search probe on KQ-vs-K (should find a large + score / mate) ==="
printf "position fen 8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1\ngo depth 10\nquit\n" | $K/v-allsf/engine 2>/dev/null | grep -E "score (cp|mate)" | tail -2
echo "=== search probe on K-vs-KQ (white to move, losing; should be large - score) ==="
printf "position fen 4k3/4q3/8/8/8/8/8/4K3 w - - 0 1\ngo depth 10\nquit\n" | $K/v-allsf/engine 2>/dev/null | grep -E "score (cp|mate)" | tail -2

echo "=== SPRT: 77M(lam1.0) vs 33M(lam1.0) — does the data scale-up help? ==="
CONC=11 bash $K/ccmatch.sh $K/v-allsf/engine $K/v-sflam10/engine allsf sflam10 400 8+0.08 allsftest
echo "=== DONE ==="
