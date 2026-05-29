king_v1 = c87871fe9897800168be842996225f876788ba39 (Plan 1 material-only baseline)

Built from: cmake -S /mnt/c/Users/nikod/Documents/uni/chess -B /tmp/kb2 -DCMAKE_BUILD_TYPE=Release
Binary:     tools/baseline/king_v1  (Linux x86-64, NOT committed -- rebuild from above SHA)

To rebuild:
  wsl -d Ubuntu-24.04 bash -c "git -C /mnt/c/Users/nikod/Documents/uni/chess checkout c87871fe9897800168be842996225f876788ba39 && cmake -S /mnt/c/Users/nikod/Documents/uni/chess -B /tmp/kb_v1 -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/kb_v1 --target engine -j"
