#!/usr/bin/env bash
# Diagnose WHERE the KINg-vs-Tucano gap comes from: depth reached + NPS at a fixed
# movetime, on a few positions. If Tucano reaches much higher depth -> search/NPS
# gap. If similar depth but Tucano still wins games -> eval-quality gap.
KING=/home/nikodem/king/v-prod/engine
TUC=/home/nikodem/tucano/src/tucano_avx2
run() {
  (printf 'uci\nisready\nposition %s\ngo movetime 4000\n' "$2"; sleep 5) \
    | timeout 9 "$1" 2>&1 | grep -iE '^info .*depth' | tail -1
}
POSES=(
  "startpos"
  "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
  "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
)
for pos in "${POSES[@]}"; do
  echo "=== ${pos:0:40} ==="
  echo -n "  KINg  : "; run "$KING" "$pos"
  echo -n "  Tucano: "; run "$TUC" "$pos"
done
