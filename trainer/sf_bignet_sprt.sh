#!/usr/bin/env bash
# Validate the bigger net (HL=1024 + OB=8) vs the HL=512 production at a SLOW TC
# (fast TC penalizes the larger net; the competition is 10-min, so 30+0.3 is a
# fairer proxy). Both nets trained on the same 77M data, lam 1.0 — isolates arch.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king

cmake -S "$REPO" -B "$K/v-hl1024ob8" -DEVAL=NNUE -DNNUE_HL=1024 -DNNUE_OB=8 -DNNUE_SCRELU=ON \
  -DNNUE_NET="$REPO/nets/king_hl1024ob8_77m.bin" -DNNUE_SAMPLES="$REPO/trainer/s_hl1024ob8.txt" >/dev/null 2>&1
cmake --build "$K/v-hl1024ob8" --target engine -j10 >/dev/null 2>&1 \
  && echo "built v-hl1024ob8" || { echo "BUILD FAIL"; exit 1; }

echo "=== SPRT HL1024+OB8 vs HL512 production @ 30+0.3 (slow TC) ==="
CONC=11 bash "$K/ccmatch.sh" "$K/v-hl1024ob8/engine" "$K/v-allsf/engine" hl1024ob8 hl512 400 30+0.3 bignet
echo "=== DONE ==="
