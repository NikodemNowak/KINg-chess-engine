#!/usr/bin/env bash
# Compare nodes-to-depth across several positions (deterministic; timing-independent).
# Fewer nodes/depth = tighter tree = deeper at fixed time = the intended LMR effect.
# Args: depth then engine dir names.
set -u
K=/home/nikodem/king
D=${1:-16}; shift
FENS=(
  "startpos"
  "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
  "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
  "fen 2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1"
)
for e in "$@"; do
  tot=0
  printf '%-12s' "$e"
  for f in "${FENS[@]}"; do
    out=$( { printf 'uci\nsetoption name Threads value 1\nsetoption name Hash value 128\nisready\nposition %s\ngo depth %s\n' "$f" "$D"; sleep 6; } | "$K/$e/engine" 2>/dev/null )
    n=$(printf '%s\n' "$out" | grep -E "^info depth $D " | tail -1 | grep -oE 'nodes [0-9]+' | grep -oE '[0-9]+')
    n=${n:-0}; tot=$((tot + n))
    printf ' %9s' "$n"
  done
  printf '   total=%s\n' "$tot"
done
