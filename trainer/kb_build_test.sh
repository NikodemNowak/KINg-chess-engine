#!/usr/bin/env bash
# Build the KB=4 candidate engine + unit tests and run the NNUE bit-exact /
# incremental-vs-scratch contract tests. This is the CORRECTNESS gate for the
# king-bucket inference (esp. the king-move refresh_perspective path) — must pass
# before any SPRT. Args: [net] [samples] [kb] [tag]  (defaults to kb4-112M).
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
NET=${1:-$REPO/nets/king_kb4_112m.bin}
SAMP=${2:-$REPO/trainer/s_kb4_112m.txt}
KB=${3:-4}
OB=${4:-1}
TAG=${5:-kb4}
DIR=$K/v-$TAG

echo "[build] configuring v-$TAG (KB=$KB OB=$OB net=$(basename $NET)) ..."
cmake -S $REPO -B $DIR -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
  -DNNUE_KB=$KB -DNNUE_OB=$OB -DNNUE_NET=$NET -DNNUE_SAMPLES=$SAMP >/dev/null 2>&1 \
  || { echo "[build] CONFIG FAILED"; exit 1; }
cmake --build $DIR --target engine -j 22 >/dev/null 2>&1 \
  && echo "[build] engine OK" || { echo "[build] ENGINE BUILD FAILED"; exit 1; }
cmake --build $DIR --target unit_tests -j 22 >/dev/null 2>&1 \
  && echo "[build] unit_tests OK" || { echo "[build] TEST BUILD FAILED"; exit 1; }

# Run the doctest binary directly with an NNUE filter (the ctest target bundles
# ALL cases as a single "unit" test, so `ctest -R NNUE` would match nothing and
# falsely "pass"). The *NNUE* cases include the bit-exact sample gate AND the
# incremental==scratch legal-move-tree test — the latter is the critical KB gate
# (king moves change the bucket → must trigger refresh_perspective correctly).
echo "[test] running NNUE contract tests (bit-exact + incremental==scratch) ..."
"$DIR/unit_tests" -tc=*NNUE* 2>&1 | tail -8
RC=${PIPESTATUS[0]}
if [ "$RC" = "0" ]; then
  echo "[done] v-$TAG: NNUE contract PASSED — engine at $DIR/engine"
else
  echo "[FAIL] v-$TAG: NNUE contract FAILED (rc=$RC) — do NOT trust this net/build"
  exit 1
fi
