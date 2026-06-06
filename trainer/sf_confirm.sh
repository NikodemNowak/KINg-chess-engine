#!/usr/bin/env bash
# Confirm the SE_ASPDELTA winner with more games, and test SE_SINGULAR at a slow
# TC (it is depth-dependent — fast TC may have under-measured it).
set -u
K=/home/nikodem/king
echo "=== confirm SE_ASPDELTA: +1000 games @ 8+0.08 ==="
CONC=11 bash "$K/ccmatch.sh" "$K/v-SE_ASPDELTA/engine" "$K/v-sbase/engine" aspdelta base 1000 8+0.08 aspconf
echo "=== SE_SINGULAR @ slow TC 30+0.3: 400 games (depth-dependent) ==="
CONC=11 bash "$K/ccmatch.sh" "$K/v-SE_SINGULAR/engine" "$K/v-sbase/engine" singular base 400 30+0.3 singslow
echo "=== DONE ==="
