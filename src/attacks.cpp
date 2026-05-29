#include "attacks.hpp"
#include "bitboard.hpp"

namespace king {

// Global attack tables
Bitboard pawn_attacks[2][64];
Bitboard knight_attacks[64];
Bitboard king_attacks[64];

namespace attacks {

void init_leapers() {
  // Initialize pawn attacks for both colors
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);

    // White pawns attack diagonally upward (NORTH)
    pawn_attacks[WHITE][s] = shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b);

    // Black pawns attack diagonally downward (SOUTH)
    pawn_attacks[BLACK][s] = shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
  }

  // Initialize knight attacks (branchless, wrap-safe)
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);

    Bitboard l1 = (b >> 1) & ~FILE_H_BB;
    Bitboard l2 = (b >> 2) & ~(FILE_G_BB | FILE_H_BB);
    Bitboard r1 = (b << 1) & ~FILE_A_BB;
    Bitboard r2 = (b << 2) & ~(FILE_A_BB | FILE_B_BB);
    Bitboard h1 = l1 | r1;
    Bitboard h2 = l2 | r2;
    knight_attacks[s] = (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
  }

  // Initialize king attacks (all 8 directions)
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);
    king_attacks[s] = shift<NORTH>(b) | shift<SOUTH>(b) | shift<EAST>(b) | shift<WEST>(b) |
                      shift<NORTH_EAST>(b) | shift<NORTH_WEST>(b) | shift<SOUTH_EAST>(b) | shift<SOUTH_WEST>(b);
  }
}

} // namespace attacks

} // namespace king
