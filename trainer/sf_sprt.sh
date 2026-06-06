#!/usr/bin/env bash
# WSL side of the SF-relabel A/B: build both engines, then SPRT.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king

for tag in extrasf extraeng; do
  cmake -S $REPO -B $K/v-$tag -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
    -DNNUE_NET=$REPO/nets/king_${tag}_hl512.bin -DNNUE_SAMPLES=$REPO/trainer/s_${tag}.txt >/dev/null 2>&1
  cmake --build $K/v-$tag --target engine -j 22 >/dev/null 2>&1 \
    && echo "built v-$tag" || { echo "build $tag FAILED"; exit 1; }
done

echo "=== SPRT A: net_sf vs net_eng (label isolation, identical 33M positions) ==="
CONC=11 bash $K/ccmatch.sh $K/v-extrasf/engine $K/v-extraeng/engine extrasf extraeng 400 8+0.08 sfvseng

echo "=== SPRT B: net_sf vs net69m (vs current best) ==="
if [ -x $K/v-69m/engine ]; then
  CONC=11 bash $K/ccmatch.sh $K/v-extrasf/engine $K/v-69m/engine extrasf net69m 400 8+0.08 sfvs69m
else
  echo "v-69m/engine missing -- skipping SPRT B"
fi
echo "=== ALL DONE ==="
