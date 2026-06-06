#!/usr/bin/env bash
# Build the baseline (all SE_* off) and each feature variant, then SPRT each
# ON-vs-OFF at 8+0.08. Highest-EV features first. Same 77M net throughout, so
# each match isolates exactly one search feature.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
GAMES=${1:-500}

echo "=== building baseline (no SE_* macros) ==="
cmake -S "$REPO" -B "$K/v-sbase" >/dev/null 2>&1
cmake --build "$K/v-sbase" --target engine -j22 >/dev/null 2>&1 \
  && echo "built baseline" || { echo "BASELINE BUILD FAIL"; exit 1; }

for M in SE_SINGULAR SE_PROBCUT SE_QSCHECK SE_CONTHIST2 SE_NMPEVAL SE_ASPDELTA SE_FIFTYSCALE SE_BADCAPLMR; do
  echo "=== building $M ==="
  cmake -S "$REPO" -B "$K/v-$M" -DCMAKE_CXX_FLAGS="-D$M" >/dev/null 2>&1
  if cmake --build "$K/v-$M" --target engine -j22 >/dev/null 2>&1; then
    echo "=== SPRT $M (on) vs baseline (off) ==="
    CONC=11 bash "$K/ccmatch.sh" "$K/v-$M/engine" "$K/v-sbase/engine" "$M" base "$GAMES" 8+0.08 "sprt_$M"
  else
    echo "$M BUILD FAIL — skipping"
  fi
done
echo "=== ALL SEARCH SPRTS DONE ==="
