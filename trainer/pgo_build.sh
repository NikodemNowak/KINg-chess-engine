#!/usr/bin/env bash
# PGO build of v-best (the cumulative-best engine): instrument -> profile -> optimize+LTO.
# MATCHES v-best config exactly (AGGR_LMR=1, NNUE 512/SCReLU/KB=1, king_174m.bin, generic -O3,
# LTO). PGO only reorders hot/cold code + sharpens inlining from MEASURED search traces -> no
# search-logic change -> cannot regress strength, only raise NPS. AVX2 stays runtime-dispatched
# (generic baseline) so no SIGILL risk. SPRT the result vs v-best to confirm NPS->Elo.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
K=/home/nikodem/king
PROF=$K/pgo_prof
NET=$REPO/nets/king_174m.bin
SAMP=$REPO/trainer/s_174m.txt
# v-best -D options (everything except LTO + the PGO flags, which differ per pass)
COMMON=(-DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON -DNNUE_KB=1
        -DNNUE_NET="$NET" -DNNUE_SAMPLES="$SAMP" -DCMAKE_BUILD_TYPE=Release)
rm -rf "$PROF" "$K/v-pgo-gen" "$K/v-pgo-use"; mkdir -p "$PROF"

echo "=== PGO pass 1: instrumented build (-fprofile-generate, LTO off) ==="
cmake -S "$REPO" -B "$K/v-pgo-gen" "${COMMON[@]}" \
  -DCMAKE_CXX_FLAGS="-DAGGR_LMR=1 -fprofile-generate=$PROF" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate=$PROF" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF >/tmp/pgo1_cfg.log 2>&1 \
  || { echo "cfg1 FAIL"; tail -8 /tmp/pgo1_cfg.log; exit 1; }
cmake --build "$K/v-pgo-gen" --target engine -j12 >/tmp/pgo1_build.log 2>&1 \
  && echo "instrumented OK" || { echo "build1 FAIL"; tail -10 /tmp/pgo1_build.log; exit 1; }

echo "=== profile workload: deep searches over varied positions (search/eval/NNUE/movegen) ==="
{
  echo "uci"; echo "isready"; echo "setoption name Hash value 128"
  for fen in \
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" \
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" \
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" \
    "4rrk1/pp1n1ppp/2p5/q2p4/3P4/2P1PN2/PP3PPP/R2Q1RK1 b - - 0 1" \
    "2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1" ; do
    echo "position fen $fen"; echo "go depth 18"
  done
  echo "quit"
} | "$K/v-pgo-gen/engine" >/dev/null 2>&1
echo "profile gathered: $(find "$PROF" -name '*.gcda' | wc -l) gcda files"

echo "=== PGO pass 2: optimized build (-fprofile-use + LTO) ==="
cmake -S "$REPO" -B "$K/v-pgo-use" "${COMMON[@]}" \
  -DCMAKE_CXX_FLAGS="-DAGGR_LMR=1 -fprofile-use=$PROF -fprofile-correction -Wno-missing-profile -Wno-coverage-mismatch" \
  -DCMAKE_EXE_LINKER_FLAGS="" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON >/tmp/pgo2_cfg.log 2>&1 \
  || { echo "cfg2 FAIL"; tail -8 /tmp/pgo2_cfg.log; exit 1; }
cmake --build "$K/v-pgo-use" --target engine unit_tests -j12 >/tmp/pgo2_build.log 2>&1 \
  && echo "PGO+LTO build OK" || { echo "build2 FAIL"; tail -12 /tmp/pgo2_build.log; exit 1; }
echo "=== unit tests (PGO engine) ==="
"$K/v-pgo-use/unit_tests" 2>&1 | tail -3

echo "=== NPS check: v-best vs v-pgo-use (depth-20 startpos, last info line) ==="
for e in "$K/v-best/engine" "$K/v-pgo-use/engine"; do
  echo "-- $e"
  printf "position startpos\ngo depth 20\nquit\n" | "$e" 2>/dev/null | grep -E "^info depth 20" | tail -1
done
echo "=== DONE. Next: SPRT v-pgo-use vs v-best at 40+0.4 (after corrhist match frees the CPU) ==="
