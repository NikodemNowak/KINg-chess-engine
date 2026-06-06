#!/usr/bin/env bash
# SPRT the 112M HL=512 net vs the 77M production net, SAME search (singular+probcut
# default-on). Net change is ~TC-independent (eval quality, same arch/NPS), so 8+0.08.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
cmake -S "$REPO" -B "$K/v-112m" -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
  -DNNUE_NET="$REPO/nets/king_112m_hl512.bin" -DNNUE_SAMPLES="$REPO/trainer/s_112m.txt" >/dev/null 2>&1
cmake --build "$K/v-112m" --target engine -j8 >/dev/null 2>&1 && echo "built v-112m"
echo "=== SPRT 112M net vs 77M production (same search) @ 8+0.08 ==="
CONC=11 bash "$K/ccmatch.sh" "$K/v-112m/engine" "$K/v-pcver/engine" net112m net77m 500 8+0.08 net112m
echo "=== DONE ==="
