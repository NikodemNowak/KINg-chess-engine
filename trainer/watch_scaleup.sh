#!/usr/bin/env bash
sleep 7200
echo "=== scaleup watchdog @ +2h ==="
echo "stockfish procs: $(pgrep -c stockfish)"
echo "labeled: $(cat /home/nikodem/king/sf_shards/sh_[0-9]*.sf 2>/dev/null | wc -l) / $(wc -l < /home/nikodem/king/rest.txt)"
