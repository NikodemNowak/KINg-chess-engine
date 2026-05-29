#include "doctest/doctest.h"
#include "bitboard.hpp"
using namespace king;
TEST_CASE("primitives") {
  CHECK(popcount(0ULL) == 0);
  CHECK(popcount(0xFFULL) == 8);
  CHECK(square_bb(E4) == (1ULL << 28));
  CHECK(lsb(square_bb(E4)) == E4);
  Bitboard b = square_bb(A1) | square_bb(H8);
  CHECK(more_than_one(b));
  CHECK(pop_lsb(b) == A1);
  CHECK(b == square_bb(H8));
}
TEST_CASE("shift no wrap") {
  CHECK(shift<NORTH>(square_bb(E4)) == square_bb(E5));
  CHECK(shift<SOUTH>(square_bb(E4)) == square_bb(E3));
  CHECK(shift<EAST>(square_bb(H4)) == 0ULL);
  CHECK(shift<WEST>(square_bb(A4)) == 0ULL);
  CHECK(shift<EAST>(square_bb(E4)) == square_bb(F4));
}
TEST_CASE("between and line") {
  bitboard::init();
  CHECK(between_bb(A1, A8) == (square_bb(A2)|square_bb(A3)|square_bb(A4)|square_bb(A5)|square_bb(A6)|square_bb(A7)));
  CHECK(between_bb(A1, H8) == (square_bb(B2)|square_bb(C3)|square_bb(D4)|square_bb(E5)|square_bb(F6)|square_bb(G7)));
  CHECK(between_bb(A1, B3) == 0ULL);          // not aligned
  CHECK(between_bb(A1, B1) == 0ULL);          // adjacent: nothing between
  CHECK(line_bb(A1, A5) == FILE_A_BB);        // whole a-file
  CHECK((line_bb(A1, H1) & RANK_1_BB) == RANK_1_BB);
  CHECK(line_bb(A1, B3) == 0ULL);             // not aligned

  // Anti-diagonal regression (H1–A8 direction, delta == 7 / -7)
  CHECK(between_bb(H1, A8) == (square_bb(G2)|square_bb(F3)|square_bb(E4)|square_bb(D5)|square_bb(C6)|square_bb(B7)));
  CHECK(between_bb(H8, A1) == (square_bb(G7)|square_bb(F6)|square_bb(E5)|square_bb(D4)|square_bb(C3)|square_bb(B2)));
  CHECK(line_bb(H1, A8) == line_bb(B7, G2));            // same anti-diagonal, full span
  CHECK((line_bb(H1, A8) & square_bb(H1)) != 0ULL);
  CHECK((line_bb(H1, A8) & square_bb(A8)) != 0ULL);
  CHECK(between_bb(H1, A8) != 0ULL);
}
