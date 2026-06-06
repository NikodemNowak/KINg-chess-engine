#!/usr/bin/env bash
sleep 14400
echo "=== datagen watchdog @ +4h ==="
echo "engine procs: $(pgrep -c engine)"
echo "extra4 positions: $(wc -l < /home/nikodem/king/nnue_extra4.txt)"
