#!/usr/bin/env bash
# Parallel Stockfish relabeling driver.
# Splits an input "FEN | score | result" dataset into W shards, runs one SF
# labeling worker per shard, then concatenates the relabeled shards.
#
# Usage: sf_label_run.sh [IN] [OUT] [NODES] [W]
SF=/home/nikodem/king/stockfish/stockfish-ubuntu-x86-64-avx2
IN=${1:-/home/nikodem/king/nnue_extra.txt}
OUT=${2:-/home/nikodem/king/nnue_extra_sf.txt}
NODES=${3:-5000}
W=${4:-22}
SCRIPT=/mnt/c/Users/nikod/Documents/uni/chess/trainer/sf_label.py
WORK=/home/nikodem/king/sf_shards

rm -rf "$WORK"; mkdir -p "$WORK"
echo "[run] splitting $IN into $W shards ..."
split -n l/$W -d -a 2 "$IN" "$WORK/sh_"
echo "[run] launching $W workers @ nodes $NODES ..."
pids=()
for f in "$WORK"/sh_[0-9]*; do
  [ -e "$f" ] || continue
  python3 "$SCRIPT" "$SF" "$f" "$f.sf" "$NODES" &
  pids+=($!)
done
echo "[run] waiting for ${#pids[@]} workers ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[run] workers done (failures=$fail). concatenating ..."
cat "$WORK"/sh_[0-9]*.sf > "$OUT"
echo -n "[run] output lines: "; wc -l < "$OUT"
echo "[run] DONE -> $OUT"
