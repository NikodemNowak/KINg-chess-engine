#pragma once
#include "position.hpp"

namespace king {

// Tapered piece-square-table evaluation.
// Source: PeSTO tables by Ronald Friederich, as published on the Chess Programming Wiki:
//   https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
// Tables are indexed from White's perspective with a1=0, h8=63 (rank-major, little-endian).
// For Black pieces, the square index is mirrored vertically: sq ^ 56 (flips rank).

int evaluate(const Position& pos);

} // namespace king
