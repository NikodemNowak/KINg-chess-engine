#!/usr/bin/env bash
# Syzygy 5-men WDL download via curl (wget fails on this mirror). Enumerate every
# 3-5-men material combo in both orderings; curl -sf downloads valid names, 404s skip.
set -u
D=/home/nikodem/king/syzygy
URL="http://tablebase.sesse.net/syzygy/3-4-5"
mkdir -p "$D"; cd "$D" || exit 1
rm -f list.html *.html
python3 - > names.txt <<'PY'
import itertools
order = 'QRBNP'
names = set()
for n in (1, 2, 3):
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
echo "candidates: $(wc -l < names.txt)"
n=0
while read -r f; do
  if curl -sf -o "$f" "$URL/$f"; then n=$((n + 1)); else rm -f "$f"; fi
done < names.txt
echo "downloaded: $n valid .rtbw files, total: $(du -sh "$D" | cut -f1)"
