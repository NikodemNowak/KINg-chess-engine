#include "position.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"
#include <sstream>
#include <cassert>
#include <cctype>
#include <cstdlib>

namespace king {

// ── Castling-rights update masks ──────────────────────────────────────────────
// AND-mask applied (per from/to square) to castling_ on every move. Default is
// ANY_CASTLING (15): no change. Squares that involve a king/rook strip the
// corresponding rights when touched (moved from, or captured on).
static uint8_t CASTLE_MASK[64];

static const bool castle_mask_inited = [] {
    for (int s = 0; s < 64; ++s) CASTLE_MASK[s] = ANY_CASTLING;
    CASTLE_MASK[E1] = uint8_t(~(WHITE_OO | WHITE_OOO) & ANY_CASTLING); // 12
    CASTLE_MASK[H1] = uint8_t(~WHITE_OO  & ANY_CASTLING);              // 14
    CASTLE_MASK[A1] = uint8_t(~WHITE_OOO & ANY_CASTLING);              // 13
    CASTLE_MASK[E8] = uint8_t(~(BLACK_OO | BLACK_OOO) & ANY_CASTLING); //  3
    CASTLE_MASK[H8] = uint8_t(~BLACK_OO  & ANY_CASTLING);              // 11
    CASTLE_MASK[A8] = uint8_t(~BLACK_OOO & ANY_CASTLING);              //  7
    return true;
}();

// ── Constructor ───────────────────────────────────────────────────────────────

Position::Position() : stm_(WHITE), castling_(NO_CASTLING), ep_(NO_SQ),
                       halfmove_(0), fullmove_(1), key_(0)
{
    for (int i = 0; i < 64; ++i) board_[i] = NO_PIECE;
#ifdef EVAL_NNUE
    acc_stack_.resize(ACC_SIZE); // copy-make ring (heap-backed)
#endif
}

// ── Incremental helpers ───────────────────────────────────────────────────────

// NOTE: put/remove/move_piece update ONLY board state (bitboards, mailbox, keys).
// The NNUE accumulator is maintained separately by do_move's copy-make update
// (and rebuilt from scratch in set_fen / copy_from), decoupled from the board so
// a move is one fused accumulator pass instead of per-piece add/sub calls.
void Position::put_piece(Piece p, Square s) {
    const PieceType pt = piece_type(p);
    const Color c = color_of(p);
    const Bitboard bb = square_bb(s);
    board_[s] = p;
    by_type_[pt] |= bb;
    by_color_[c] |= bb;
    key_ ^= zobrist::psq[p][s];
    if (pt == PAWN || pt == KING) pawn_key_ ^= zobrist::psq[p][s];
#if MULTICORR
    if (pt == KNIGHT || pt == BISHOP) minor_key_ ^= zobrist::psq[p][s];
    if (pt == ROOK   || pt == QUEEN)  major_key_ ^= zobrist::psq[p][s];
#endif
}

void Position::remove_piece(Square s) {
    const Piece p = board_[s];
    assert(p != NO_PIECE);
    const PieceType pt = piece_type(p);
    const Color c = color_of(p);
    const Bitboard bb = square_bb(s);
    by_type_[pt] &= ~bb;
    by_color_[c] &= ~bb;
    board_[s] = NO_PIECE;
    key_ ^= zobrist::psq[p][s];
    if (pt == PAWN || pt == KING) pawn_key_ ^= zobrist::psq[p][s];
#if MULTICORR
    if (pt == KNIGHT || pt == BISHOP) minor_key_ ^= zobrist::psq[p][s];
    if (pt == ROOK   || pt == QUEEN)  major_key_ ^= zobrist::psq[p][s];
#endif
}

void Position::move_piece(Square from, Square to) {
    const Piece p = board_[from];
    assert(p != NO_PIECE);
    const PieceType pt = piece_type(p);
    const Color c = color_of(p);
    const Bitboard fromTo = square_bb(from) | square_bb(to);

    board_[from] = NO_PIECE;
    board_[to] = p;
    by_type_[pt] ^= fromTo;
    by_color_[c] ^= fromTo;
    key_ ^= zobrist::psq[p][from] ^ zobrist::psq[p][to];
    if (pt == PAWN || pt == KING) pawn_key_ ^= zobrist::psq[p][from] ^ zobrist::psq[p][to];
#if MULTICORR
    if (pt == KNIGHT || pt == BISHOP) minor_key_ ^= zobrist::psq[p][from] ^ zobrist::psq[p][to];
    if (pt == ROOK   || pt == QUEEN)  major_key_ ^= zobrist::psq[p][from] ^ zobrist::psq[p][to];
#endif
}

// ── Controlled deep clone ─────────────────────────────────────────────────────

void Position::copy_from(const Position& o) {
    // 1. Copy all board bitboards and the mailbox.
    for (int i = 0; i < 6; ++i) by_type_[i] = o.by_type_[i];
    by_color_[0] = o.by_color_[0];
    by_color_[1] = o.by_color_[1];
    for (int i = 0; i < 64; ++i) board_[i] = o.board_[i];

    // 2. Copy scalar state.
    stm_      = o.stm_;
    castling_ = o.castling_;
    ep_       = o.ep_;
    halfmove_ = o.halfmove_;
    fullmove_ = o.fullmove_;
    key_      = o.key_;
    pawn_key_ = o.pawn_key_;
#if MULTICORR
    minor_key_ = o.minor_key_;
    major_key_ = o.major_key_;
#endif

    // 3. Copy the repetition history verbatim (so the clone detects draws by
    //    repetition exactly like the source).
    hist_ = o.hist_;

    // 4. Re-root the StateInfo chain. The clone owns a FRESH root with
    //    previous == nullptr, and st_ points at it. Search on the clone only
    //    ever undoes moves it itself made, so it never walks past this root;
    //    we still snapshot the current scalars into root_ for consistency.
    root_.previous      = nullptr;
    root_.captured      = NO_PIECE;
    root_.prev_ep       = o.ep_;
    root_.prev_castling = o.castling_;
    root_.prev_halfmove = o.halfmove_;
    root_.prev_fullmove = o.fullmove_;
    root_.prev_key      = o.key_;
    st_ = &root_;

#ifdef EVAL_NNUE
    // Rebuild the clone's accumulator from its (now-copied) board, resetting the
    // copy-make ring to ply 0. Track the king squares first.
    ksq_[WHITE] = king_sq(WHITE);
    ksq_[BLACK] = king_sq(BLACK);
    acc_ply_ = 0;
    nnue::refresh(acc_stack_[0], *this);
#endif
}

// ── Make / unmake ─────────────────────────────────────────────────────────────

void Position::do_move(Move m, StateInfo& st) {
    // 1. Snapshot current state into the new node and link it.
    st.previous      = st_;
    st.prev_ep       = ep_;
    st.prev_castling = castling_;
    st.prev_halfmove = halfmove_;
    st.prev_fullmove = fullmove_;
    st.prev_key      = key_;
    st_              = &st;
    // NNUE: the accumulator is updated by copy-make AFTER the board is settled (see
    // the EVAL_NNUE block near the end of do_move), not by put/remove_piece.

    // 2. Remove the old en-passant file from the key (if any).
    if (ep_ != NO_SQ) key_ ^= zobrist::enpassant[file_of(ep_)];

    // 3. Decode the move.
    Square   from = from_sq(m);
    Square   to   = to_sq(m);
    MoveFlag fl   = type_of(m);
    Piece    pc   = piece_on(from);
    Color    us   = stm_;
    Color    them = Color(!us);

    // 4. Determine captured piece.
    st.captured = (fl == EN_PASSANT) ? make_piece(them, PAWN) : piece_on(to);

    // 5. Halfmove clock: reset on pawn move or capture, else increment.
    if (piece_type(pc) == PAWN || st.captured != NO_PIECE) halfmove_ = 0;
    else                                                   halfmove_++;

    // 6. Remove captured piece.
    if (fl == EN_PASSANT)
        remove_piece(Square(us == WHITE ? to - 8 : to + 8));
    else if (st.captured != NO_PIECE)
        remove_piece(to);

    // 7. Move (or promote) the moving piece.
    if (fl == PROMO) {
        remove_piece(from);
        put_piece(make_piece(us, promo_pt(m)), to);
    } else {
        move_piece(from, to);
    }

    // 8. Castling: relocate the rook.
    if (fl == CASTLING) {
        Square rfrom, rto;
        switch (to) {
            case G1: rfrom = H1; rto = F1; break;
            case C1: rfrom = A1; rto = D1; break;
            case G8: rfrom = H8; rto = F8; break;
            case C8: rfrom = A8; rto = D8; break;
            default: rfrom = to; rto = to; break; // unreachable
        }
        move_piece(rfrom, rto);
    }

    // 9. New en-passant square (only on a double pawn push).
    ep_ = NO_SQ;
    if (piece_type(pc) == PAWN && std::abs(int(to) - int(from)) == 16)
        ep_ = Square((from + to) / 2);
    if (ep_ != NO_SQ) key_ ^= zobrist::enpassant[file_of(ep_)];

    // 10. Update castling rights (and key).
    key_ ^= zobrist::castling[castling_];
    castling_ &= CASTLE_MASK[from] & CASTLE_MASK[to];
    key_ ^= zobrist::castling[castling_];

    // 11. Flip side to move; bump fullmove after Black moves.
    stm_  = them;
    key_ ^= zobrist::side;
    if (us == BLACK) fullmove_++;

#ifdef EVAL_NNUE
    // Copy-make accumulator update: compute the child slot from the parent in ONE
    // fused pass per perspective (the parent slot is left intact → free unmake).
    // For KB>1 a king move (incl. castling) changes the moving side's whole bucket,
    // so rebuild the child from scratch (board is already in its post-move state).
    {
        nnue::Accumulator&       dst = acc_stack_[(acc_ply_ + 1) & ACC_MASK];
        const nnue::Accumulator& src = acc_stack_[acc_ply_ & ACC_MASK];
        if (nnue::KB > 1 && piece_type(pc) == KING) {
            ksq_[us] = to;
            nnue::refresh(dst, *this);
        } else {
            nnue::Feat adds[2], subs[2];
            int na = 0, ns = 0;
            if (fl == PROMO) {
                subs[ns++] = { us, PAWN, from };
                adds[na++] = { us, promo_pt(m), to };
                if (st.captured != NO_PIECE) subs[ns++] = { them, piece_type(st.captured), to };
            } else if (fl == EN_PASSANT) {
                subs[ns++] = { us, PAWN, from };
                adds[na++] = { us, PAWN, to };
                subs[ns++] = { them, PAWN, Square(us == WHITE ? to - 8 : to + 8) };
            } else if (fl == CASTLING) {
                Square rfrom, rto;
                switch (to) {
                    case G1: rfrom = H1; rto = F1; break;
                    case C1: rfrom = A1; rto = D1; break;
                    case G8: rfrom = H8; rto = F8; break;
                    case C8: rfrom = A8; rto = D8; break;
                    default: rfrom = to; rto = to; break; // unreachable
                }
                subs[ns++] = { us, KING, from };
                adds[na++] = { us, KING, to };
                subs[ns++] = { us, ROOK, rfrom };
                adds[na++] = { us, ROOK, rto };
            } else { // NORMAL (incl. a KB==1 king move)
                subs[ns++] = { us, piece_type(pc), from };
                adds[na++] = { us, piece_type(pc), to };
                if (st.captured != NO_PIECE) subs[ns++] = { them, piece_type(st.captured), to };
            }
            nnue::update_accumulator(dst, src, adds, na, subs, ns, ksq_[WHITE], ksq_[BLACK]);
            if (piece_type(pc) == KING) ksq_[us] = to; // keep tracked (KB>1 enemy perspective)
        }
        ++acc_ply_;
    }
#endif

    // 12. Record key in history for repetition detection.
    hist_.push_back(key_);
}

void Position::undo_move(Move m) {
    hist_.pop_back();
    StateInfo* st = st_;

    // 1. Flip side back: 'us' is the side that made the move being undone.
    stm_          = Color(!stm_);
    Color    us   = stm_;
    Color    them = Color(!us);
    Square   from = from_sq(m);
    Square   to   = to_sq(m);
    MoveFlag fl   = type_of(m);

    // 2. Undo the piece movement.
    if (fl == CASTLING) {
        move_piece(to, from);
        Square rfrom, rto;
        switch (to) {
            case G1: rfrom = H1; rto = F1; break;
            case C1: rfrom = A1; rto = D1; break;
            case G8: rfrom = H8; rto = F8; break;
            case C8: rfrom = A8; rto = D8; break;
            default: rfrom = to; rto = to; break; // unreachable
        }
        move_piece(rto, rfrom);
    } else if (fl == PROMO) {
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
    }

    // 3. Restore the captured piece.
    if (fl == EN_PASSANT)
        put_piece(make_piece(them, PAWN), Square(us == WHITE ? to - 8 : to + 8));
    else if (st->captured != NO_PIECE)
        put_piece(st->captured, to);

    // 4. Restore scalar state. The key is restored from the snapshot; the board
    //    is already physically correct via put/remove/move above.
    ep_       = st->prev_ep;
    castling_ = st->prev_castling;
    halfmove_ = st->prev_halfmove;
    fullmove_ = st->prev_fullmove;
    key_      = st->prev_key;
    // NNUE copy-make: unmake is FREE — just step the ring index back to the parent
    // slot (left intact by do_move). Restore the tracked king square for a king
    // move (only load-bearing for KB>1; the king is back on `from` by now).
#ifdef EVAL_NNUE
    if (piece_type(piece_on(from)) == KING) ksq_[us] = from;
    --acc_ply_;
#endif
    st_       = st->previous;
}

// ── Null-move make/unmake ─────────────────────────────────────────────────────

void Position::do_null_move(StateInfo& st) {
    // Snapshot current state (mirrors do_move preamble, captured = NO_PIECE).
    st.previous      = st_;
    st.prev_ep       = ep_;
    st.prev_castling = castling_;
    st.prev_halfmove = halfmove_;
    st.prev_fullmove = fullmove_;
    st.prev_key      = key_;
    st.captured      = NO_PIECE;
    st_              = &st;

    // Clear en-passant square (a null move passes without a pawn double-push).
    if (ep_ != NO_SQ) {
        key_ ^= zobrist::enpassant[file_of(ep_)];
        ep_   = NO_SQ;
    }

    // Flip side to move.
    stm_  = Color(!stm_);
    key_ ^= zobrist::side;

    // Advance halfmove clock (conservative; won't trigger draw checks on this).
    halfmove_++;

    // NOTE: hist_ is intentionally NOT modified — null moves are search-internal
    // and must not affect repetition detection.
}

void Position::undo_null_move() {
    StateInfo* st = st_;
    stm_      = Color(!stm_);
    ep_       = st->prev_ep;
    castling_ = st->prev_castling;
    halfmove_ = st->prev_halfmove;
    fullmove_ = st->prev_fullmove;
    key_      = st->prev_key;
    st_       = st->previous;
    // hist_ untouched (was not modified in do_null_move).
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
    pawn_key_  = 0;
#if MULTICORR
    minor_key_ = 0;
    major_key_ = 0;
#endif
    st_        = &root_;
    root_.previous = nullptr;

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
            if (idx != std::string::npos && file < 8 && rank >= 0) {
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
        int f = token[0] - 'a';
        int r = token[1] - '1';
        if (f >= 0 && f < 8 && r >= 0 && r < 8) {   // guard against a malformed FEN ep token
            ep_ = make_square(File(f), Rank(r));
            key_ ^= zobrist::enpassant[file_of(ep_)];
        }
    }

    // 5. Halfmove clock
    ss >> halfmove_;
    // Robustness: a FEN with a huge/negative halfmove field would overflow the
    // halfmove_++ in do_move during search (signed-int UB -> crash at -O3). Clamp to
    // the legal [0,100] range (100 = the 50-move draw threshold; beyond is already a
    // draw). A failed parse leaves halfmove_ at 0, which is in range. crash=loss.
    if (halfmove_ < 0)   halfmove_ = 0;
    if (halfmove_ > 100) halfmove_ = 100;

    // 6. Fullmove number
    ss >> fullmove_;
    if (fullmove_ < 1)    fullmove_ = 1;
    if (fullmove_ > 9999) fullmove_ = 9999;  // guard fullmove_++ overflow on absurd FENs

    // 7. Seed key history for repetition detection
    hist_.clear();
    hist_.push_back(key_);

#ifdef EVAL_NNUE
    // Track king squares, then rebuild the accumulator from scratch into ring slot
    // 0 (acc_ply_ reset to 0). put_piece no longer touches the accumulator, so this
    // from-scratch refresh is the single source of truth for the root board.
    ksq_[WHITE] = king_sq(WHITE);
    ksq_[BLACK] = king_sq(BLACK);
    acc_ply_ = 0;
    nnue::refresh(acc_stack_[0], *this);
#endif
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
    return (pawn_attacks[WHITE][s] & pieces(BLACK, PAWN))
      | (pawn_attacks[BLACK][s]    & pieces(WHITE, PAWN))
      | (knight_attacks[s]         & pieces(KNIGHT))
      | (king_attacks[s]           & pieces(KING))
      | (bishop_attacks(s, occ)    & (pieces(BISHOP) | pieces(QUEEN)))
      | (rook_attacks(s, occ)      & (pieces(ROOK)   | pieces(QUEEN)));
}

bool Position::attacked_by(Square s, Color by, Bitboard occ) const {
    const Bitboard pawns = (by == WHITE) ? pawn_attacks[BLACK][s] : pawn_attacks[WHITE][s];
    if (pawns & pieces(by, PAWN)) return true;
    if (knight_attacks[s] & pieces(by, KNIGHT)) return true;
    if (king_attacks[s] & pieces(by, KING)) return true;
    if (bishop_attacks(s, occ) & (pieces(by, BISHOP) | pieces(by, QUEEN))) return true;
    if (rook_attacks(s, occ) & (pieces(by, ROOK) | pieces(by, QUEEN))) return true;
    return false;
}

bool Position::in_check(Color c) const {
    Bitboard k = pieces(c, KING);
    if (!k) return false;                 // defensive: no king -> avoid OOB on lsb(0)=64
    return attacked_by(lsb(k), Color(!c), occupied());
}

// ── Draw detection ────────────────────────────────────────────────────────────

bool Position::is_repetition() const {
    int n     = (int)hist_.size();
    int limit = std::min(halfmove_, n - 1);
    // Positions with the same side to move are 2 plies apart; current is hist_[n-1].
    for (int i = n - 3; i >= n - 1 - limit; i -= 2)
        if (hist_[i] == key_) return true;
    return false;
}

bool Position::insufficient_material() const {
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    return popcount(pieces(KNIGHT) | pieces(BISHOP)) <= 1;
}

} // namespace king
