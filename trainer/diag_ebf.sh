#!/usr/bin/env bash
# EBF diagnostic: depth reached at a FIXED node budget (deterministic, independent
# of CPU contention as long as the search completes). KINg vs Tucano, single-thread.
# If KINg reaches much shallower depth at equal nodes -> worse EBF (the real gap).
KING=/home/nikodem/king/v-best/engine
TUC=/home/nikodem/opp/tucano/src/tucano_avx2
N=${1:-2000000}
FENS=(
  "startpos"
  "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
  "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
  "fen 2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1"
)
run() {  # $1 engine, $2 cwd, $3 label
  cd "$2"
  echo "=== $3 @ $N nodes ==="
  for f in "${FENS[@]}"; do
    out=$( { printf 'uci\nsetoption name Threads value 1\nsetoption name Hash value 128\nisready\nposition %s\ngo nodes %d\n' "$f" "$N"; sleep 30; } | "$1" 2>/dev/null )
    last=$(printf '%s\n' "$out" | grep -E '^info .*depth ' | tail -1)
    d=$(printf '%s\n' "$last" | grep -oE 'depth [0-9]+' | head -1)
    sc=$(printf '%s\n' "$last" | grep -oE 'score (cp|mate) -?[0-9]+')
    echo "  ${f:0:30} -> $d  ($sc)"
  done
}
run "$KING" /home/nikodem/king "KINg(v-best)"
run "$TUC" /home/nikodem/opp/tucano/src "Tucano"
