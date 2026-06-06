#!/usr/bin/env bash
# Like ccmatch.sh but takes Move Overhead as $8 (default 30) to test time-management
# behaviour at different overhead settings.
set -u
CC="$HOME/cc/cutechess-cli/cutechess-cli"
BOOK="$HOME/king/book.epd"
TALLY="/mnt/c/Users/nikod/Documents/uni/chess/tools/tally.py"
CAND="$1"; OPP="$2"; NC="$3"; NO="$4"; G="$5"; TC="$6"; TAG="$7"; OH="${8:-30}"
PGN="$HOME/king/m_${TAG}.pgn"
R=$(( (G+1)/2 ))
rm -f "$PGN"
"$CC" -engine cmd="$CAND" name="$NC" proto=uci \
      -engine cmd="$OPP" name="$NO" proto=uci \
      -each tc="$TC" option.Hash=128 option.Threads=1 "option.Move Overhead=$OH" \
      -openings file="$BOOK" format=epd order=random -repeat -games 2 -rounds "$R" \
      -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=700 \
      -concurrency "${CONC:-10}" -recover -ratinginterval 100 -pgnout "$PGN" \
      > "$HOME/king/m_${TAG}.log" 2>&1
python3 "$TALLY" "$PGN" "$NC"
