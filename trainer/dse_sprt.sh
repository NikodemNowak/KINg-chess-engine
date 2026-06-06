#!/usr/bin/env bash
# SPRT the DSE (double/negative singular extension) search change vs production.
# DSE is DEPTH-DEPENDENT (like singular: -12 fast TC -> +28.7 slow TC), so it MUST
# be validated at the competition's SLOW TC (30+0.3), NOT fast TC. CPU-HEAVY +
# slow — lowest priority in the CPU queue (run after the KB arch SPRTs). The
# v-dse engine must already be built with -DSE_DSE=1 (see the CXX_FLAGS build).
# Args: [games] (default 2000).
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
G=${1:-2000}

# (Re)build v-dse with DSE enabled if missing.
if [ ! -x $K/v-dse/engine ]; then
  echo "[build] configuring v-dse (SE_DSE=1) ..."
  cmake -S $REPO -B $K/v-dse -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DSE_DSE=1" >/dev/null 2>&1 \
    || { echo "[build] v-dse CONFIG FAILED"; exit 1; }
  cmake --build $K/v-dse --target engine -j 22 >/dev/null 2>&1 \
    && echo "[build] v-dse OK" || { echo "[build] v-dse BUILD FAILED"; exit 1; }
fi
# Production baseline (default search, SE_DSE off).
if [ ! -x $K/v-prod/engine ]; then
  echo "[build] configuring v-prod (production, DSE off) ..."
  cmake -S $REPO -B $K/v-prod -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON -DNNUE_KB=1 \
    -DNNUE_NET=$REPO/nets/king_nnue.bin -DNNUE_SAMPLES=$REPO/trainer/nnue_samples.txt >/dev/null 2>&1 \
    || { echo "[build] v-prod CONFIG FAILED"; exit 1; }
  cmake --build $K/v-prod --target engine -j 22 >/dev/null 2>&1 \
    && echo "[build] v-prod OK" || { echo "[build] v-prod BUILD FAILED"; exit 1; }
fi

echo "=== SPRT: DSE vs prod  ($G games, 30+0.3 SLOW TC, from DSE POV) ==="
CONC=${CONC:-20} bash $K/ccmatch.sh $K/v-dse/engine $K/v-prod/engine dse prod "$G" 30+0.3 dsevsprod
echo "=== DSE SPRT DONE ==="
