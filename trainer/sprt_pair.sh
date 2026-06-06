#!/usr/bin/env bash
# Generic paired SPRT at SLOW TC (the validated methodology). Args:
#   $1 = engine1 build dir name (under /home/nikodem/king), $2 = name1
#   $3 = engine2 build dir name, $4 = name2
#   $5 = games (default 600), $6 = concurrency (default 10)
# Writes m_<name1>vs<name2>.pgn. Run via Bash run_in_background (survives the turn).
set -u
K=/home/nikodem/king
CC=/home/nikodem/cc/cutechess-cli/cutechess-cli
d1=$1; n1=$2; d2=$3; n2=$4; games=${5:-600}; conc=${6:-10}; threads=${7:-1}
out=$K/m_${n1}vs${n2}.pgn
rounds=$((games / 2))
"$CC" \
  -engine cmd="$K/$d1/engine" name="$n1" proto=uci \
  -engine cmd="$K/$d2/engine" name="$n2" proto=uci \
  -each tc=40+0.4 option.Hash=128 option.Threads=$threads "option.Move Overhead=30" \
  -openings file="$K/book.epd" format=epd order=random \
  -repeat -games 2 -rounds "$rounds" \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=700 \
  -concurrency "$conc" -recover -ratinginterval 50 -pgnout "$out"
echo "DONE: $out"
