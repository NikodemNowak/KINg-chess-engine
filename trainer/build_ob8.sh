#!/usr/bin/env bash
# Build an OB=8 (king_ob8_174m.bin) engine variant with extra CXX defines.
# Args: $1 = build dir name (under /home/nikodem/king), $2 = extra CXX flags.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
dir=$1; flags=$2
cmake -S "$REPO" -B "$K/$dir" \
  -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON -DNNUE_KB=1 -DNNUE_OB=8 \
  -DNNUE_NET="$REPO/nets/king_ob8_174m.bin" -DNNUE_SAMPLES="$REPO/trainer/s_ob8_174m.txt" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$flags" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON >/tmp/${dir}_cfg.log 2>&1 \
  || { echo "$dir CFG FAIL"; tail -8 /tmp/${dir}_cfg.log; exit 1; }
cmake --build "$K/$dir" --target engine unit_tests -j20 >/tmp/${dir}_build.log 2>&1 \
  || { echo "$dir BUILD FAIL"; tail -12 /tmp/${dir}_build.log; exit 1; }
echo "$dir BUILT -> $K/$dir/engine"
"$K/$dir/unit_tests" 2>&1 | tail -2
