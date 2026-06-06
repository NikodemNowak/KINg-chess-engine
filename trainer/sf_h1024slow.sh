#!/usr/bin/env bash
# Does HL=1024 (8% lower val-loss on 112M) beat HL=512 at the competition's slow
# TC? Bigger net = better eval but ~2x NPS; the slow TC is where it can pay off.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
cmake -S "$REPO" -B "$K/v-h1024" -DEVAL=NNUE -DNNUE_HL=1024 -DNNUE_SCRELU=ON \
  -DNNUE_NET="$REPO/nets/king_112m_hl1024.bin" -DNNUE_SAMPLES="$REPO/trainer/s_112m_hl1024.txt" >/dev/null 2>&1
cmake --build "$K/v-h1024" --target engine -j8 >/dev/null 2>&1 && echo "built v-h1024"
echo "=== SPRT HL1024-112M vs HL512-112M (production) @ 30+0.3 (slow) ==="
CONC=11 bash "$K/ccmatch.sh" "$K/v-h1024/engine" "$K/v-112m/engine" h1024 h512 400 30+0.3 h1024slow
echo "=== DONE ==="
