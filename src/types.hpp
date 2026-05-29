#pragma once
#include <cstdint>

namespace king {

using Bitboard = uint64_t;

enum Color { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

enum PieceType { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NO_PT = 6, PT_NB = 6 };

enum Piece {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 6, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE = 12, PIECE_NB = 12
};

enum File { FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };

enum Rank { RANK_1 = 0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

enum Square {
    A1 = 0,  B1,  C1,  D1,  E1,  F1,  G1,  H1,
    A2 = 8,  B2,  C2,  D2,  E2,  F2,  G2,  H2,
    A3 = 16, B3,  C3,  D3,  E3,  F3,  G3,  H3,
    A4 = 24, B4,  C4,  D4,  E4,  F4,  G4,  H4,
    A5 = 32, B5,  C5,  D5,  E5,  F5,  G5,  H5,
    A6 = 40, B6,  C6,  D6,  E6,  F6,  G6,  H6,
    A7 = 48, B7,  C7,  D7,  E7,  F7,  G7,  H7,
    A8 = 56, B8,  C8,  D8,  E8,  F8,  G8,  H8 = 63,
    NO_SQ = 64
};

enum MoveFlag { NORMAL = 0, PROMO = 1, EN_PASSANT = 2, CASTLING = 3 };

using Move = uint16_t;

// Square helpers
constexpr inline Square make_square(File f, Rank r) { return Square(r * 8 + f); }
constexpr inline File   file_of(Square s)           { return File(s & 7); }
constexpr inline Rank   rank_of(Square s)           { return Rank(s >> 3); }

// Piece helpers
constexpr inline Piece     make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr inline Color     color_of(Piece p)                 { return Color(p / 6); }
constexpr inline PieceType piece_type(Piece p)               { return PieceType(p % 6); }

// Move encoding: bits 0-5 = from, 6-11 = to, 12-13 = (promo-KNIGHT), 14-15 = flag
constexpr inline Move      make_move(Square from, Square to, MoveFlag flag = NORMAL, PieceType promo = KNIGHT) {
    return Move(from | (to << 6) | ((promo - KNIGHT) << 12) | (flag << 14));
}
constexpr inline Square    from_sq(Move m)   { return Square(m & 0x3F); }
constexpr inline Square    to_sq(Move m)     { return Square((m >> 6) & 0x3F); }
constexpr inline MoveFlag  type_of(Move m)   { return MoveFlag((m >> 14) & 3); }
constexpr inline PieceType promo_pt(Move m)  { return PieceType(KNIGHT + ((m >> 12) & 3)); }

} // namespace king
