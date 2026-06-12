#include "movegen.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"

namespace king {

namespace {

// True iff square `s` is attacked by any piece of color `by` (uses current occ).
inline bool attacked(const Position& pos, Square s, Color by, Bitboard occ) {
    return pos.attacked_by(s, by, occ);
}

// Emit pawn moves to `to` from `from`: 4 promotion moves on the last rank,
// otherwise a single move with the given flag.
inline void add_pawn_move(MoveList& list, Square from, Square to, bool promo, MoveFlag flag) {
    if (promo) {
        list.add(make_move(from, to, PROMO, QUEEN));
        list.add(make_move(from, to, PROMO, ROOK));
        list.add(make_move(from, to, PROMO, BISHOP));
        list.add(make_move(from, to, PROMO, KNIGHT));
    } else {
        list.add(make_move(from, to, flag));
    }
}

} // namespace

void generate_pseudo(const Position& pos, MoveList& list) {
    const Color    us        = pos.side_to_move();
    const Color    them      = Color(!us);
    const Bitboard occ       = pos.occupied();
    const Bitboard empty     = ~occ;
    const Bitboard ownPieces = pos.pieces(us);
    const Bitboard enemy     = pos.pieces(them);

    // ── Pawns ───────────────────────────────────────────────────────────────
    const Square   ep        = pos.ep_square();
    const Rank     promoRank = (us == WHITE) ? RANK_8 : RANK_1;
    const int      forward   = (us == WHITE) ? 8 : -8;

    Bitboard pawns = pos.pieces(us, PAWN);
    
    // Single and double pushes
    Bitboard push1 = (us == WHITE) ? (pawns << 8) & empty : (pawns >> 8) & empty;
    Bitboard push2 = (us == WHITE) ? ((push1 & RANK_3_BB) << 8) & empty : ((push1 & RANK_6_BB) >> 8) & empty;

    Bitboard p1 = push1;
    while (p1) {
        Square to = pop_lsb(p1);
        Square from = Square(int(to) - forward);
        add_pawn_move(list, from, to, rank_of(to) == promoRank, NORMAL);
    }
    Bitboard p2 = push2;
    while (p2) {
        Square to = pop_lsb(p2);
        Square from = Square(int(to) - forward * 2);
        list.add(make_move(from, to, NORMAL));
    }

    // Captures and En Passant
    Bitboard p3 = pawns;
    while (p3) {
        Square from = pop_lsb(p3);
        Bitboard caps = pawn_attacks[us][from] & enemy;
        while (caps) {
            Square to = pop_lsb(caps);
            add_pawn_move(list, from, to, rank_of(to) == promoRank, NORMAL);
        }
        if (ep != NO_SQ && (pawn_attacks[us][from] & square_bb(ep)))
            list.add(make_move(from, ep, EN_PASSANT));
    }

    // ── Knights ─────────────────────────────────────────────────────────────
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square   from    = pop_lsb(knights);
        Bitboard targets = knight_attacks[from] & ~ownPieces;
        while (targets) list.add(make_move(from, pop_lsb(targets)));
    }

    // ── Bishops ─────────────────────────────────────────────────────────────
    Bitboard bishops = pos.pieces(us, BISHOP);
    while (bishops) {
        Square   from    = pop_lsb(bishops);
        Bitboard targets = bishop_attacks(from, occ) & ~ownPieces;
        while (targets) list.add(make_move(from, pop_lsb(targets)));
    }

    // ── Rooks ───────────────────────────────────────────────────────────────
    Bitboard rooks = pos.pieces(us, ROOK);
    while (rooks) {
        Square   from    = pop_lsb(rooks);
        Bitboard targets = rook_attacks(from, occ) & ~ownPieces;
        while (targets) list.add(make_move(from, pop_lsb(targets)));
    }

    // ── Queens ──────────────────────────────────────────────────────────────
    Bitboard queens = pos.pieces(us, QUEEN);
    while (queens) {
        Square   from    = pop_lsb(queens);
        Bitboard targets = queen_attacks(from, occ) & ~ownPieces;
        while (targets) list.add(make_move(from, pop_lsb(targets)));
    }

    // ── King ────────────────────────────────────────────────────────────────
    Square ksq = pos.king_sq(us);
    if (ksq != NO_SQ) {
        Bitboard ktargets = king_attacks[ksq] & ~ownPieces;
        while (ktargets) list.add(make_move(ksq, pop_lsb(ktargets)));

        // ── Castling (fully legality-checked) ────────────────────────────────
        const uint8_t cr = pos.castling_rights();
        if (us == WHITE) {
            // O-O: F1,G1 empty; E1,F1,G1 not attacked by BLACK.
            if ((cr & WHITE_OO)
                && (empty & square_bb(F1)) && (empty & square_bb(G1))
                && !attacked(pos, E1, BLACK, occ)
                && !attacked(pos, F1, BLACK, occ)
                && !attacked(pos, G1, BLACK, occ))
                list.add(make_move(E1, G1, CASTLING));

            // O-O-O: B1,C1,D1 empty; E1,D1,C1 not attacked by BLACK.
            if ((cr & WHITE_OOO)
                && (empty & square_bb(B1)) && (empty & square_bb(C1)) && (empty & square_bb(D1))
                && !attacked(pos, E1, BLACK, occ)
                && !attacked(pos, D1, BLACK, occ)
                && !attacked(pos, C1, BLACK, occ))
                list.add(make_move(E1, C1, CASTLING));
        } else {
            // O-O: F8,G8 empty; E8,F8,G8 not attacked by WHITE.
            if ((cr & BLACK_OO)
                && (empty & square_bb(F8)) && (empty & square_bb(G8))
                && !attacked(pos, E8, WHITE, occ)
                && !attacked(pos, F8, WHITE, occ)
                && !attacked(pos, G8, WHITE, occ))
                list.add(make_move(E8, G8, CASTLING));

            // O-O-O: B8,C8,D8 empty; E8,D8,C8 not attacked by WHITE.
            if ((cr & BLACK_OOO)
                && (empty & square_bb(B8)) && (empty & square_bb(C8)) && (empty & square_bb(D8))
                && !attacked(pos, E8, WHITE, occ)
                && !attacked(pos, D8, WHITE, occ)
                && !attacked(pos, C8, WHITE, occ))
                list.add(make_move(E8, C8, CASTLING));
        }
    }
}

void generate_captures(const Position& pos, MoveList& list) {
    const Color    us    = pos.side_to_move();
    const Color    them  = Color(!us);
    const Bitboard occ   = pos.occupied();
    const Bitboard empty = ~occ;
    const Bitboard enemy = pos.pieces(them);

    // ── Pawns: quiet promotions, captures (incl. capture-promos), en passant ──
    const Square ep        = pos.ep_square();
    const Rank   promoRank = (us == WHITE) ? RANK_8 : RANK_1;
    const int    forward   = (us == WHITE) ? 8 : -8;

    Bitboard pawns = pos.pieces(us, PAWN);
    
    // Quiet promotion push (noisy)
    Bitboard promos = (us == WHITE) ? (pawns << 8) & empty & RANK_8_BB : (pawns >> 8) & empty & RANK_1_BB;
    while (promos) {
        Square to = pop_lsb(promos);
        add_pawn_move(list, Square(int(to) - forward), to, true, NORMAL);
    }

    // Captures and En Passant
    while (pawns) {
        Square from = pop_lsb(pawns);
        Bitboard caps = pawn_attacks[us][from] & enemy;
        while (caps) {
            Square to = pop_lsb(caps);
            add_pawn_move(list, from, to, rank_of(to) == promoRank, NORMAL);
        }
        if (ep != NO_SQ && (pawn_attacks[us][from] & square_bb(ep)))
            list.add(make_move(from, ep, EN_PASSANT));
    }

    // ── Pieces: captures only (targets ∩ enemy), same square order as pseudo ──
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard t = knight_attacks[from] & enemy;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    Bitboard bishops = pos.pieces(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard t = bishop_attacks(from, occ) & enemy;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    Bitboard rooks = pos.pieces(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard t = rook_attacks(from, occ) & enemy;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    Bitboard queens = pos.pieces(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard t = queen_attacks(from, occ) & enemy;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    Square ksq = pos.king_sq(us);
    if (ksq != NO_SQ) {
        Bitboard t = king_attacks[ksq] & enemy;
        while (t) list.add(make_move(ksq, pop_lsb(t)));
    }
}

bool is_legal(Position& pos, Move m) {
    StateInfo st;
    pos.do_move(m, st);
    bool ok = !pos.in_check(Color(!pos.side_to_move()));
    pos.undo_move(m);
    return ok;
}

void generate_legal(Position& pos, MoveList& list) {
    MoveList ps;
    generate_pseudo(pos, ps);
    for (int i = 0; i < ps.size; ++i)
        if (is_legal(pos, ps.moves[i]))
            list.add(ps.moves[i]);
}

bool has_legal_moves(Position& pos) {
    MoveList ps;
    generate_pseudo(pos, ps);
    for (int i = 0; i < ps.size; ++i)
        if (is_legal(pos, ps.moves[i]))
            return true;
    return false;
}

} // namespace king
