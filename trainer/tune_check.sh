#!/usr/bin/env bash
# Verify the -DTUNE engine lists the SPSA options and that setoption changes search.
set -u
E=/home/nikodem/king/v-tune/engine
FEN="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

echo "=== options listed by 'uci' (expect LmrBase/RfpMargin/SeeMargin/...) ==="
printf 'uci\nquit\n' | "$E" 2>/dev/null | grep -iE "option name (lmrbase|rfpmargin|seemargin|nmpdiv|futbase)"

run() {  # $1 = optional setoption line; prints nodes at depth 15
  local setline=$1
  local out
  out=$( { printf 'uci\nsetoption name Threads value 1\nsetoption name Hash value 64\n'
           [ -n "$setline" ] && printf '%s\n' "$setline"
           printf 'isready\nposition fen %s\ngo depth 15\n' "$FEN"
           sleep 5; } | nice -n 19 "$E" 2>/dev/null )
  printf '%s\n' "$out" | grep -E "^info depth 15 " | tail -1 | grep -oE 'nodes [0-9]+'
}

echo "=== behavior change at depth 15 (single-thread, deterministic) ==="
echo "default      : $(run '')"
echo "LmrBase=40   : $(run 'setoption name LmrBase value 40')   (less reduction -> more nodes)"
echo "LmrBase=150  : $(run 'setoption name LmrBase value 150')  (more reduction -> fewer nodes)"
echo "RfpMargin=30 : $(run 'setoption name RfpMargin value 30') (more RFP pruning -> fewer nodes)"
