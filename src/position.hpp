#pragma once
#include <cstdint>
#include <string>
#include "types.hpp"
#include "bitboard.hpp"

namespace king {

enum CastlingRight {
    NO_CASTLING = 0,
    WHITE_OO    = 1,
    WHITE_OOO   = 2,
    BLACK_OO    = 4,
    BLACK_OOO   = 8,
    ANY_CASTLING = 15
};

class Position {
public:
    Position();

    // FEN I/O
    void        set_fen(const std::string& fen);
    std::string fen() const;

    // Piece queries
    Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
    Bitboard pieces(PieceType pt)          const { return by_type_[pt]; }
    Bitboard pieces(Color c)               const { return by_color_[c]; }
    Bitboard occupied()                    const { return by_color_[WHITE] | by_color_[BLACK]; }
    Piece    piece_on(Square s)            const { return board_[s]; }
    Square   king_sq(Color c)              const { return lsb(pieces(c, KING)); }

    // Attack queries
    Bitboard attackers_to(Square s, Bitboard occ) const;
    bool     in_check(Color c) const;

    // State
    Color    side_to_move() const { return stm_; }
    uint64_t key()          const { return key_; }
    uint64_t compute_key()  const;

    // Incremental helpers — also used by T7 make/unmake
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

private:
    Bitboard by_type_[6]{};
    Bitboard by_color_[2]{};
    Piece    board_[64];
    Color    stm_;
    uint8_t  castling_;
    Square   ep_;
    int      halfmove_;
    int      fullmove_;
    uint64_t key_;
};

} // namespace king
