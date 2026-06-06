#!/usr/bin/env bash
# Robust Syzygy 5-men WDL downloader: the mirror serves files but no listing,
# and the canonical "stronger side first" naming is fiddly — so we enumerate every
# 3-5-men material combo in BOTH orderings and fetch each; invalid names 404 and
# are skipped. Net result: all valid .rtbw files, ~387MB.
set -u
D=/home/nikodem/king/syzygy
URL="http://tablebase.sesse.net/syzygy/3-4-5"
mkdir -p "$D"; cd "$D" || exit 1
python3 - > names.txt <<'PY'
import itertools
order = 'QRBNP'
names = set()
for n in (1, 2, 3):                     # non-king pieces -> 3,4,5 men
    for wc in range(0, n + 1):
        bc = n - wc
        for w in itertools.combinations_with_replacement(order, wc):
            for b in itertools.combinations_with_replacement(order, bc):
                ws = ''.join(sorted(w, key=order.index))
                bs = ''.join(sorted(b, key=order.index))
                names.add('K' + ws + 'vK' + bs)
                names.add('K' + bs + 'vK' + ws)
for x in sorted(names):
    print(x + '.rtbw')
PY
echo "candidate names: $(wc -l < names.txt)"
n=0
while read -r f; do
  if wget -q -nc "$URL/$f" 2>/dev/null && [ -s "$f" ]; then n=$((n + 1)); fi
done < names.txt
echo "downloaded: $n valid .rtbw files, total: $(du -sh "$D" | cut -f1)"
