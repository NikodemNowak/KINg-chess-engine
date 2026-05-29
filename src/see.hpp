#pragma once
#include "types.hpp"
#include "position.hpp"

namespace king {

// Static Exchange Evaluation.
//
// Evaluates the material outcome of the capture sequence initiated by `m` on
// its destination square, assuming both sides always recapture with their
// least-valuable attacker. Returns centipawns from the *moving side's*
// perspective: positive means the mover comes out ahead in material.
//
// Piece values used internally (independent of the search's PSQT/eval):
//   P=100, N=320, B=330, R=500, Q=900, K=10000.
//
// Notes / edge cases:
//   * A quiet move (no victim) returns 0 if its destination is undefended,
//     otherwise a negative score (the moving piece is hanging).
//   * En passant is handled (victim is a pawn on the captured-pawn square).
//   * X-ray attackers revealed behind a sliding/pawn capture are included.
//   * Never crashes on positions with no further attackers or king captures.
int see(const Position& pos, Move m);

// Convenience: true iff see(pos, m) >= threshold. (threshold default 0 = the
// "is this capture non-losing?" test used by qsearch / ordering.)
inline bool see_ge(const Position& pos, Move m, int threshold = 0) {
    return see(pos, m) >= threshold;
}

} // namespace king
