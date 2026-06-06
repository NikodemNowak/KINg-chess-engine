#!/usr/bin/env bash
# Summarize every SPRT match PGN: engine names + W/L/D + score% + rough Elo.
cd /home/nikodem/king || exit 1
for p in m_*.pgn; do
  [ -f "$p" ] || continue
  names=$(grep -hoE '^\[White "[^"]+"\]|^\[Black "[^"]+"\]' "$p" | sed -E 's/^\[(White|Black) "//; s/"\]$//' | sort -u | tr '\n' '/')
  w=$(grep -c 'Result "1-0"' "$p")
  l=$(grep -c 'Result "0-1"' "$p")
  d=$(grep -c 'Result "1/2-1/2"' "$p")
  tot=$((w + l + d))
  [ "$tot" -eq 0 ] && continue
  # score from White POV is meaningless across colors; cutechess alternates, so
  # use first engine (the "test" engine is engine1 = -engine cmd=... name=NAME, usually White in round 1)
  printf '%-30s games=%-4d  1-0=%-4d 0-1=%-4d D=%-4d  [%s]\n' "$p" "$tot" "$w" "$l" "$d" "$names"
done
