#!/usr/bin/env bash
# ccmatch.sh <cand> <opp> <nameCand> <nameOpp> <games> <tc> <tag> [extra opp options...]
# Unique per-tag PGN so concurrent/sequential matches never clobber each other.
# Prints a tally (Elo +/- CI, LOS) from the candidate's POV. CONC env = concurrency.
set -u
CC="$HOME/cc/cutechess-cli/cutechess-cli"
BOOK="$HOME/king/book.epd"
TALLY="/mnt/c/Users/nikod/Documents/uni/chess/tools/tally.py"
CAND="$1"; OPP="$2"; NC="$3"; NO="$4"; G="$5"; TC="$6"; TAG="$7"; shift 7
PGN="$HOME/king/m_${TAG}.pgn"
R=$(( (G+1)/2 ))
rm -f "$PGN"
"$CC" -engine cmd="$CAND" name="$NC" proto=uci \
      -engine cmd="$OPP" name="$NO" proto=uci "$@" \
      -each tc="$TC" option.Hash=128 option.Threads=1 "option.Move Overhead=30" \
      -openings file="$BOOK" format=epd order=random -repeat -games 2 -rounds "$R" \
      -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=700 \
      -concurrency "${CONC:-10}" -recover -ratinginterval 100 -pgnout "$PGN" \
      > "$HOME/king/m_${TAG}.log" 2>&1
python3 "$TALLY" "$PGN" "$NC"
