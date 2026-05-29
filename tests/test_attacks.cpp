#include "doctest/doctest.h"
#include "attacks.hpp"
#include "bitboard.hpp"
using namespace king;

TEST_CASE("leaper attacks") {
  attacks::init_leapers();
  CHECK(knight_attacks[B1] == (square_bb(A3)|square_bb(C3)|square_bb(D2)));
  CHECK(popcount(knight_attacks[E4]) == 8);
  CHECK(king_attacks[A1] == (square_bb(A2)|square_bb(B1)|square_bb(B2)));
  CHECK(popcount(king_attacks[E4]) == 8);
  CHECK(pawn_attacks[WHITE][E4] == (square_bb(D5)|square_bb(F5)));
  CHECK(pawn_attacks[BLACK][E4] == (square_bb(D3)|square_bb(F3)));
  CHECK(pawn_attacks[WHITE][A2] == square_bb(B3));   // edge, no wrap
  CHECK(pawn_attacks[BLACK][H7] == square_bb(G6));   // edge, no wrap
}
