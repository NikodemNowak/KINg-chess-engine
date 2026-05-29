#pragma once
#include "bitboard.hpp"

namespace king {

// Leaper attack tables (indexed by [color][square] for pawns, [square] for others)
extern Bitboard pawn_attacks[2][64];
extern Bitboard knight_attacks[64];
extern Bitboard king_attacks[64];

// Slider attacks via plain magic bitboards (attacks::init_magics() must run first)
Bitboard rook_attacks(Square s, Bitboard occ);
Bitboard bishop_attacks(Square s, Bitboard occ);
inline Bitboard queen_attacks(Square s, Bitboard occ) { return rook_attacks(s, occ) | bishop_attacks(s, occ); }

namespace attacks {
  // Initialize all leaper attack tables (idempotent)
  void init_leapers();

  // Initialize slider magic tables (idempotent; independent of init_leapers)
  void init_magics();
}

} // namespace king
