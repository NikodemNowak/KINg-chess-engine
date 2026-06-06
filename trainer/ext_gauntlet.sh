#!/usr/bin/env bash
# EXTERNAL ANCHOR gauntlet: KINg vs Tucano 12.17 NNUE — the first NON-self-play,
# SLOW-TC measurement of KINg's real strength (audit fix #1: breaks the self-play
# bubble + gives an absolute-ish anchor). Audit-aligned conditions: slow TC
# (40+0.4 proxy, ~depth 22-26), looser resign 900/5 (700/4 under-credits swindle),
# balanced color-swap book. Tally from KINg's POV.
set -u
CC="$HOME/cc/cutechess-cli/cutechess-cli"
BOOK="$HOME/king/book.epd"
TALLY="/mnt/c/Users/nikod/Documents/uni/chess/tools/tally.py"
KING=${1:-/home/nikodem/king/v-prod/engine}
KNAME=${2:-KINg}
G=${3:-400}
TC=${4:-40+0.4}
TUC=/home/nikodem/tucano/src/tucano_avx2
TNET=/home/nikodem/tucano/src/tucano_nn03.bin
PGN="$HOME/king/m_ext_${KNAME}.pgn"
LOG="$HOME/king/m_ext_${KNAME}.log"
R=$(( (G + 1) / 2 ))
rm -f "$PGN"
echo "=== EXTERNAL gauntlet: $KNAME vs Tucano  ($G games, $TC, from $KNAME POV) ==="
"$CC" -engine cmd="$KING" name="$KNAME" proto=uci "option.Move Overhead=30" \
      -engine cmd="$TUC" name=Tucano proto=uci option.EvalFile="$TNET" \
      -each tc="$TC" option.Hash=128 option.Threads=1 \
      -openings file="$BOOK" format=epd order=random -repeat -games 2 -rounds "$R" \
      -draw movenumber=40 movecount=8 score=10 -resign movecount=5 score=900 \
      -concurrency "${CONC:-8}" -recover -ratinginterval 50 -pgnout "$PGN" \
      > "$LOG" 2>&1
echo "--- result (KINg POV) ---"
python3 "$TALLY" "$PGN" "$KNAME"
echo "=== external gauntlet DONE ==="
