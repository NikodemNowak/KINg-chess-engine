#!/usr/bin/env bash
# Scale up the SF-labeled corpus: relabel the remaining disjoint self-play
# batches (nnue_big + extra2 + extra3) and merge with the already-labeled
# nnue_extra_sf.txt into one ~77M-position SF-labeled training set.
set -u
K=/home/nikodem/king
DATA=/mnt/c/Users/nikod/Documents/uni/chess/data
RUN=/mnt/c/Users/nikod/Documents/uni/chess/trainer/sf_label_run.sh

echo "[scaleup] concatenating sources to label ..."
cat "$DATA/nnue_big.txt" "$K/nnue_extra2.txt" "$K/nnue_extra3.txt" > "$K/rest.txt"
echo -n "[scaleup] rest.txt lines: "; wc -l < "$K/rest.txt"

echo "[scaleup] SF-labeling rest.txt @ nodes 5000, 22 workers ..."
bash "$RUN" "$K/rest.txt" "$K/rest_sf.txt" 5000 22

echo "[scaleup] merging extra_sf + rest_sf -> all_sf.txt ..."
cat "$K/nnue_extra_sf.txt" "$K/rest_sf.txt" > "$K/all_sf.txt"
echo -n "[scaleup] all_sf.txt lines: "; wc -l < "$K/all_sf.txt"
echo "[scaleup] DONE -> $K/all_sf.txt"
