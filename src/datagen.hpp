#pragma once
#include <string>

namespace king {

// Self-play data generation for NNUE training.
//
// Usage (from main): engine datagen <outfile> <numGames> [depth=8] [threads=N]
//
// Plays multithreaded self-play games, each with 8 random opening plies for
// diversity, then fixed-depth search for the remainder.  For each quiet,
// non-check position visited, writes a labeled line to <outfile>:
//
//   <FEN> | <score_cp_white_pov> | <result>
//
// where result ∈ {1.0, 0.5, 0.0} is the game outcome from White's POV.
//
// Returns process exit code (0 = success).
int run_datagen(int argc, char** argv);

} // namespace king
