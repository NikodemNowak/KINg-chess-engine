#pragma once

namespace king {

// In-engine Texel tuner.
//
// Usage (from main):  engine tune <epd-file> [maxPositions] [maxSeconds] [psqt]
//
//   <epd-file>    Path to a quiet-labeled EPD: "<FEN> ... c9 \"1-0|0-1|1/2-1/2\";"
//   maxPositions  Cap on positions loaded into memory (default 150000).
//   maxSeconds    Wall-clock cap for the coordinate-descent phase (default 1500).
//   psqt          If the literal token "psqt" is present, also tune the PSQT
//                 (768x2 params) after the scalar/material/HCE pass, time permitting.
//
// Optimises the scaling constant K, then runs coordinate descent over g_eval,
// printing MSE before/after and a paste-ready dump of the tuned defaults.
// Returns a process exit code (0 = success).
int run_tuner(int argc, char** argv);

} // namespace king
