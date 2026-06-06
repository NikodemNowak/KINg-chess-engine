#!/usr/bin/env bash
# Download 5-men Syzygy WDL (.rtbw) tables by parsing the directory listing and
# fetching each file explicitly (the mirror blocks wget recursion via robots.txt).
set -u
D=/home/nikodem/king/syzygy
URL="http://tablebase.sesse.net/syzygy/3-4-5"
mkdir -p "$D"; cd "$D" || exit 1
wget -q -O list.html "$URL/"
echo "list.html bytes: $(wc -c < list.html)"
grep -oE '[A-Za-z0-9]+v[A-Za-z0-9]+\.rtbw' list.html | sort -u > files.txt
echo "rtbw files to fetch: $(wc -l < files.txt)"
n=0
while read -r f; do
  if wget -q -nc "$URL/$f"; then n=$((n+1)); fi
done < files.txt
echo "downloaded: $n files, total: $(du -sh "$D" | cut -f1)"
