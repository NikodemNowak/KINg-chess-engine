#include "doctest/doctest.h"
#include "types.hpp"
using namespace king;
TEST_CASE("square helpers") {
  CHECK(make_square(FILE_E, RANK_4) == E4);
  CHECK(file_of(E4) == FILE_E);
  CHECK(rank_of(E4) == RANK_4);
  CHECK(A1 == 0); CHECK(H8 == 63);
}
TEST_CASE("move normal") {
  Move m = make_move(E2, E4);
  CHECK(from_sq(m) == E2);
  CHECK(to_sq(m) == E4);
  CHECK(type_of(m) == NORMAL);
}
TEST_CASE("move promotion") {
  Move p = make_move(A7, A8, PROMO, QUEEN);
  CHECK(from_sq(p) == A7);
  CHECK(to_sq(p) == A8);
  CHECK(type_of(p) == PROMO);
  CHECK(promo_pt(p) == QUEEN);
}
TEST_CASE("move flags") {
  CHECK(type_of(make_move(E5, D6, EN_PASSANT)) == EN_PASSANT);
  CHECK(type_of(make_move(E1, G1, CASTLING)) == CASTLING);
}
TEST_CASE("piece helpers") {
  Piece wp = make_piece(WHITE, PAWN);
  Piece bk = make_piece(BLACK, KING);
  CHECK(color_of(wp) == WHITE);
  CHECK(piece_type(wp) == PAWN);
  CHECK(color_of(bk) == BLACK);
  CHECK(piece_type(bk) == KING);
}
