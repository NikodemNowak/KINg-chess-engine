#!/usr/bin/env bash
# Fast per-macro syntax check of search.cpp (no full build, ~1 core).
SRC=/mnt/c/Users/nikod/Documents/uni/chess/src
FLAGS="-std=c++20 -fsyntax-only -mavx2 -DNNUE_HL=512 -DNNUE_SCRELU -I $SRC"
for M in BASE SE_PROBCUT SE_SINGULAR SE_QSCHECK SE_CONTHIST2 SE_FIFTYSCALE SE_ASPDELTA SE_NMPEVAL SE_BADCAPLMR; do
  if [ "$M" = BASE ]; then F=""; else F="-D$M"; fi
  if g++ $FLAGS $F "$SRC/search.cpp" 2>"/tmp/sc_$M.log"; then
    echo "$M: OK"
  else
    echo "$M: FAIL"
    grep -iE "error:" "/tmp/sc_$M.log" | head -4
  fi
done
