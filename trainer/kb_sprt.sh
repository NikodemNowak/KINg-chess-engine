#!/usr/bin/env bash
# SPRT a KB candidate vs the production net (king_nnue.bin, KB=1). Builds a fresh
# v-prod baseline from the committed production net, then plays the candidate vs it
# at fast TC (eval A/B — not depth-dependent, so 8+0.08 is valid). CPU-HEAVY: run
# only when the relabel is NOT saturating the CPU. Args: [candDir] [tag] [games].
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
CAND=${1:-$K/v-kb4}
TAG=${2:-kb4vsprod}
G=${3:-1000}

# Build the production baseline (KB=1, committed net) if not present.
if [ ! -x $K/v-prod/engine ]; then
  echo "[build] configuring v-prod (production KB=1 net) ..."
  cmake -S $REPO -B $K/v-prod -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON -DNNUE_KB=1 \
    -DNNUE_NET=$REPO/nets/king_nnue.bin -DNNUE_SAMPLES=$REPO/trainer/nnue_samples.txt >/dev/null 2>&1 \
    || { echo "[build] v-prod CONFIG FAILED"; exit 1; }
  cmake --build $K/v-prod --target engine -j 22 >/dev/null 2>&1 \
    && echo "[build] v-prod OK" || { echo "[build] v-prod BUILD FAILED"; exit 1; }
fi

[ -x $CAND/engine ] || { echo "candidate engine $CAND/engine missing — run kb_build_test.sh first"; exit 1; }

echo "=== SPRT: $TAG  ($G games, 8+0.08, from candidate POV) ==="
CONC=${CONC:-20} bash $K/ccmatch.sh $CAND/engine $K/v-prod/engine "$TAG" prod "$G" 8+0.08 "$TAG"
echo "=== SPRT $TAG DONE ==="
