#!/usr/bin/env bash
# Build a v-best-config engine with extra -D flags. Args:
#   $1 = build dir name (under /home/nikodem/king)
#   $2 = extra CXX defines (e.g. "-DAGGR_LMR=1 -DCUTNODE=1")
# Matches v-best: NNUE 512/SCReLU/KB=1, king_174m.bin, LTO on, generic -O3.
# Builds engine + unit_tests, runs the tests. Niced so it never starves a running SPRT.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
dir=$1; flags=$2; net=${3:-$REPO/nets/king_174m.bin}
cmake -S "$REPO" -B "$K/$dir" \
  -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON -DNNUE_KB=1 \
  -DNNUE_NET="$net" -DNNUE_SAMPLES="$REPO/trainer/s_174m.txt" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$flags" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON >/tmp/${dir}_cfg.log 2>&1 \
  || { echo "CFG FAIL"; tail -8 /tmp/${dir}_cfg.log; exit 1; }
nice -n 19 cmake --build "$K/$dir" --target engine unit_tests -j12 >/tmp/${dir}_build.log 2>&1 \
  || { echo "BUILD FAIL"; tail -12 /tmp/${dir}_build.log; exit 1; }
echo "BUILD OK -> $K/$dir/engine"
"$K/$dir/unit_tests" 2>&1 | tail -3
