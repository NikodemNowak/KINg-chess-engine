#pragma once
#include "bitboard.hpp"

namespace king {

// Leaper attack tables (indexed by [color][square] for pawns, [square] for others)
extern Bitboard pawn_attacks[2][64];
extern Bitboard knight_attacks[64];
extern Bitboard king_attacks[64];

namespace attacks {
  // Initialize all leaper attack tables (idempotent)
  void init_leapers();
}

} // namespace king
