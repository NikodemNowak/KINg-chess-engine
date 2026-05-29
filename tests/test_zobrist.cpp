#include "doctest/doctest.h"
#include "zobrist.hpp"
#include "types.hpp"

using namespace king;

TEST_CASE("zobrist keys") {
  zobrist::init();
  uint64_t a = zobrist::psq[W_PAWN][E4];
  zobrist::init();  // idempotent + deterministic
  CHECK(zobrist::psq[W_PAWN][E4] == a);
  CHECK(zobrist::psq[W_PAWN][E4] != 0ULL);
  CHECK(zobrist::side != 0ULL);
  CHECK(zobrist::enpassant[FILE_E] != 0ULL);
  CHECK(zobrist::psq[W_PAWN][E4] != zobrist::psq[B_PAWN][E4]);   // distinct piece
  CHECK(zobrist::psq[W_PAWN][E4] != zobrist::psq[W_PAWN][E5]);   // distinct square
  CHECK(zobrist::castling[3] != zobrist::castling[12]);          // distinct masks
}
