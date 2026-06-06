#!/usr/bin/env bash
# Decide the final-net lambda: SPRT lam1.0 (pure SF eval) vs lam0.9 (SF+WDL blend),
# both trained on the same 33M SF-labeled positions.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king

cmake -S $REPO -B $K/v-sflam10 -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
  -DNNUE_NET=$REPO/nets/king_sflam10_hl512.bin -DNNUE_SAMPLES=$REPO/trainer/s_sflam10.txt >/dev/null 2>&1
cmake --build $K/v-sflam10 --target engine -j 22 >/dev/null 2>&1 \
  && echo "built v-sflam10" || { echo "build FAILED"; exit 1; }

echo "=== SPRT lambda: lam1.0 vs lam0.9 (both 33M SF) ==="
CONC=11 bash $K/ccmatch.sh $K/v-sflam10/engine $K/v-extrasf/engine lam10 lam09 400 8+0.08 lamtest
echo "=== DONE ==="
