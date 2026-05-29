#include "doctest/doctest.h"
#include "attacks.hpp"
#include "bitboard.hpp"
using namespace king;

static Bitboard ref_attacks(Square s, Bitboard occ, bool rook) {
  Bitboard att = 0;
  int r = rank_of(s), f = file_of(s);
  static const int R[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
  static const int B[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
  for (int d = 0; d < 4; ++d) {
    int dr = rook ? R[d][0] : B[d][0];
    int df = rook ? R[d][1] : B[d][1];
    int rr = r + dr, ff = f + df;
    while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
      Square t = make_square(File(ff), Rank(rr));
      att |= square_bb(t);
      if (occ & square_bb(t)) break;
      rr += dr; ff += df;
    }
  }
  return att;
}
static uint64_t rng_state = 0x123456789abcdef0ULL;
static uint64_t xrng() {
  uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

TEST_CASE("slider basics") {
  attacks::init_magics();
  CHECK(popcount(rook_attacks(A1, 0ULL)) == 14);
  CHECK(popcount(rook_attacks(D4, 0ULL)) == 14);
  CHECK(popcount(bishop_attacks(A1, 0ULL)) == 7);
  CHECK(popcount(bishop_attacks(D4, 0ULL)) == 13);
  Bitboard occR = square_bb(A4);
  CHECK((rook_attacks(A1, occR) & square_bb(A4)) != 0ULL);
  CHECK((rook_attacks(A1, occR) & square_bb(A5)) == 0ULL);
  Bitboard occB = square_bb(F6);
  CHECK((bishop_attacks(D4, occB) & square_bb(F6)) != 0ULL);
  CHECK((bishop_attacks(D4, occB) & square_bb(G7)) == 0ULL);
}
TEST_CASE("magic sliders match reference (randomized, exhaustive)") {
  attacks::init_magics();
  int mismatches = 0;
  for (int s = A1; s <= H8; ++s)
    for (int i = 0; i < 2000; ++i) {
      Bitboard occ = xrng() & xrng();
      if (rook_attacks(Square(s), occ)   != ref_attacks(Square(s), occ, true))  ++mismatches;
      if (bishop_attacks(Square(s), occ) != ref_attacks(Square(s), occ, false)) ++mismatches;
      if (queen_attacks(Square(s), occ)  != (rook_attacks(Square(s),occ) | bishop_attacks(Square(s),occ))) ++mismatches;
    }
  CHECK(mismatches == 0);
}
