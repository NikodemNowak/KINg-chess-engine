#include "position.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"
#include <sstream>
#include <cassert>
#include <cctype>

namespace king {

// ── Constructor ───────────────────────────────────────────────────────────────

Position::Position() : stm_(WHITE), castling_(NO_CASTLING), ep_(NO_SQ),
                       halfmove_(0), fullmove_(1), key_(0)
{
    for (int i = 0; i < 64; ++i) board_[i] = NO_PIECE;
}

// ── Incremental helpers ───────────────────────────────────────────────────────

void Position::put_piece(Piece p, Square s) {
    board_[s]       = p;
    by_type_[piece_type(p)] |= square_bb(s);
    by_color_[color_of(p)]  |= square_bb(s);
    key_ ^= zobrist::psq[p][s];
}

void Position::remove_piece(Square s) {
    Piece p = board_[s];
    by_type_[piece_type(p)] &= ~square_bb(s);
    by_color_[color_of(p)]  &= ~square_bb(s);
    board_[s] = NO_PIECE;
    key_ ^= zobrist::psq[p][s];
}

void Position::move_piece(Square from, Square to) {
    Piece p = board_[from];
    remove_piece(from);
    put_piece(p, to);
}

// ── FEN parsing ───────────────────────────────────────────────────────────────

void Position::set_fen(const std::string& fen) {
    // Reset state
    for (int i = 0; i < 6; ++i) by_type_[i] = 0;
    by_color_[0] = by_color_[1] = 0;
    for (int i = 0; i < 64; ++i) board_[i] = NO_PIECE;
    stm_       = WHITE;
    castling_  = NO_CASTLING;
    ep_        = NO_SQ;
    halfmove_  = 0;
    fullmove_  = 1;
    key_       = 0;

    std::istringstream ss(fen);
    std::string token;

    // 1. Piece placement (rank 8 → rank 1)
    ss >> token;
    int rank = 7, file = 0;
    for (char ch : token) {
        if (ch == '/') {
            --rank;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += ch - '0';
        } else {
            static const std::string piece_chars = "PNBRQKpnbrqk";
            static const Piece pieces_from_char[] = {
                W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
            };
            size_t idx = piece_chars.find(ch);
            if (idx != std::string::npos) {
                Square s = make_square(File(file), Rank(rank));
                put_piece(pieces_from_char[idx], s);
                ++file;
            }
        }
    }

    // 2. Side to move
    ss >> token;
    stm_ = (token == "b") ? BLACK : WHITE;
    if (stm_ == BLACK) key_ ^= zobrist::side;

    // 3. Castling rights
    ss >> token;
    castling_ = NO_CASTLING;
    if (token != "-") {
        for (char ch : token) {
            switch (ch) {
                case 'K': castling_ |= WHITE_OO;   break;
                case 'Q': castling_ |= WHITE_OOO;  break;
                case 'k': castling_ |= BLACK_OO;   break;
                case 'q': castling_ |= BLACK_OOO;  break;
            }
        }
    }
    key_ ^= zobrist::castling[castling_];

    // 4. En passant square
    ss >> token;
    ep_ = NO_SQ;
    if (token != "-" && token.size() >= 2) {
        File f = File(token[0] - 'a');
        Rank r = Rank(token[1] - '1');
        ep_ = make_square(f, r);
        key_ ^= zobrist::enpassant[file_of(ep_)];
    }

    // 5. Halfmove clock
    ss >> halfmove_;

    // 6. Fullmove number
    ss >> fullmove_;
}

// ── FEN emission ──────────────────────────────────────────────────────────────

std::string Position::fen() const {
    static const char piece_to_char[] = "PNBRQKpnbrqk";

    std::string result;

    // 1. Piece placement (rank 8 → rank 1)
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Square s = make_square(File(f), Rank(r));
            Piece p  = board_[s];
            if (p == NO_PIECE) {
                ++empty;
            } else {
                if (empty) { result += char('0' + empty); empty = 0; }
                result += piece_to_char[p];
            }
        }
        if (empty) result += char('0' + empty);
        if (r > 0) result += '/';
    }

    // 2. Side to move
    result += (stm_ == WHITE) ? " w " : " b ";

    // 3. Castling rights
    if (castling_ == NO_CASTLING) {
        result += '-';
    } else {
        if (castling_ & WHITE_OO)  result += 'K';
        if (castling_ & WHITE_OOO) result += 'Q';
        if (castling_ & BLACK_OO)  result += 'k';
        if (castling_ & BLACK_OOO) result += 'q';
    }

    // 4. En passant
    result += ' ';
    if (ep_ == NO_SQ) {
        result += '-';
    } else {
        result += char('a' + file_of(ep_));
        result += char('1' + rank_of(ep_));
    }

    // 5. Halfmove clock and fullmove number
    result += ' ';
    result += std::to_string(halfmove_);
    result += ' ';
    result += std::to_string(fullmove_);

    return result;
}

// ── Key verification ──────────────────────────────────────────────────────────

uint64_t Position::compute_key() const {
    uint64_t k = 0;

    // All pieces
    for (int p = 0; p < 12; ++p) {
        Bitboard bb = by_color_[p / 6] & by_type_[p % 6];
        Bitboard tmp = bb;
        while (tmp) {
            Square s = pop_lsb(tmp);
            k ^= zobrist::psq[p][s];
        }
    }

    if (stm_ == BLACK) k ^= zobrist::side;
    k ^= zobrist::castling[castling_];
    if (ep_ != NO_SQ) k ^= zobrist::enpassant[file_of(ep_)];

    return k;
}

// ── Attack queries ────────────────────────────────────────────────────────────

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return
        (pawn_attacks[WHITE][s]  & pieces(BLACK, PAWN))
      | (pawn_attacks[BLACK][s]  & pieces(WHITE, PAWN))
      | (knight_attacks[s]       & pieces(KNIGHT))
      | (king_attacks[s]         & pieces(KING))
      | (bishop_attacks(s, occ)  & (pieces(BISHOP) | pieces(QUEEN)))
      | (rook_attacks(s, occ)    & (pieces(ROOK)   | pieces(QUEEN)));
}

bool Position::in_check(Color c) const {
    return (attackers_to(king_sq(c), occupied()) & by_color_[!c]) != 0;
}

} // namespace king
