#!/usr/bin/env bash
# A/B test: does Stockfish-relabeling beat the engine's own labels?
# Trains two HL=512 SCReLU nets on the SAME self-play positions -- one with SF
# labels, one with the original engine labels -- then SPRTs them head to head.
set -u
REPO=/mnt/c/Users/nikod/Documents/uni/chess
PRE=$REPO/trainer/preprocess_nnue.py
TRN=$REPO/trainer/train_nnue.py
K=/home/nikodem/king
SF_TXT=$K/nnue_extra_sf.txt
ENG_TXT=$K/nnue_extra.txt

echo "=== [1/4] preprocess both label sets ==="
rm -f $K/extra_sf.bin $K/extra_eng.bin
python3 $PRE --data $SF_TXT  --out $K/extra_sf.bin  || exit 1
python3 $PRE --data $ENG_TXT --out $K/extra_eng.bin || exit 1

echo "=== [2/4] train net_sf and net_eng (same recipe) ==="
python3 $TRN --cache $K/extra_sf.bin  --out $REPO/nets/king_extrasf_hl512.bin \
        --samples $K/s_extrasf.txt  --hl 512 --activation screlu --lam 0.9 --epochs 45 || exit 1
python3 $TRN --cache $K/extra_eng.bin --out $REPO/nets/king_extraeng_hl512.bin \
        --samples $K/s_extraeng.txt --hl 512 --activation screlu --lam 0.9 --epochs 45 || exit 1

echo "=== [3/4] build both engines ==="
for tag in extrasf extraeng; do
  cmake -S $REPO -B $K/v-$tag -DEVAL=NNUE -DNNUE_HL=512 -DNNUE_SCRELU=ON \
    -DNNUE_NET=$REPO/nets/king_${tag}_hl512.bin -DNNUE_SAMPLES=$K/s_${tag}.txt >/dev/null 2>&1
  cmake --build $K/v-$tag --target engine -j 22 >/dev/null 2>&1 && echo "built v-$tag" || { echo "build $tag FAILED"; exit 1; }
done

echo "=== [4/5] SPRT A (label isolation): net_sf vs net_eng (identical 33M positions) ==="
CONC=11 bash $K/ccmatch.sh $K/v-extrasf/engine $K/v-extraeng/engine extrasf extraeng 400 8+0.08 sfvseng

echo "=== [5/5] SPRT B (vs current best): net_sf vs net69m ==="
if [ -x $K/v-69m/engine ]; then
  CONC=11 bash $K/ccmatch.sh $K/v-extrasf/engine $K/v-69m/engine extrasf net69m 400 8+0.08 sfvs69m
else
  echo "v-69m/engine missing -- skipping SPRT B"
fi
echo "=== DONE ==="
