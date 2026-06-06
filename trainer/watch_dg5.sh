#!/usr/bin/env bash
sleep 21600
echo "=== datagen watchdog @ +6h ==="
echo "engine procs: $(pgrep -c engine)"
echo "extra5b: $(wc -l < /home/nikodem/king/nnue_extra5b.txt 2>/dev/null)"
echo "extra5:  $(wc -l < /home/nikodem/king/nnue_extra5.txt 2>/dev/null)"
