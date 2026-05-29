#include "see.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include <algorithm>

namespace king {

namespace {

// SEE-internal piece values (deliberately distinct from the search's eval):
// the king is given a huge value so it is only ever used as the *last* attacker
// and capturing into it is never scored as beneficial.
constexpr int SEE_VALUE[7] = {
    100,   // PAWN
    320,   // KNIGHT
    330,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    10000, // KING
    0      // NO_PT
};

inline int see_value(PieceType pt) { return SEE_VALUE[pt]; }

inline Color opp(Color c) { return Color(c ^ 1); }

} // namespace

int see(const Position& pos, Move m) {
    const Square from = from_sq(m);
    const Square to   = to_sq(m);

    const Piece moverPiece = pos.piece_on(from);
    // Defensive: nothing to evaluate if there is no mover (shouldn't happen).
    if (moverPiece == NO_PIECE) return 0;

    PieceType nextVictim = piece_type(moverPiece);

    // ── Captured value (gain[0]) ────────────────────────────────────────────
    // Occupancy starts with the mover removed from `from`.
    Bitboard occ = pos.occupied() ^ square_bb(from);

    int gain[32];
    int d = 0;

    if (type_of(m) == EN_PASSANT) {
        gain[0] = see_value(PAWN);
        // The captured pawn sits on `to`'s file but on the mover's rank.
        const Square capSq = make_square(file_of(to), rank_of(from));
        occ ^= square_bb(capSq);
    } else {
        const Piece victim = pos.piece_on(to);
        gain[0] = (victim == NO_PIECE) ? 0 : see_value(piece_type(victim));
        // Promotions: keep it reasonable — treat the moving pawn as itself.
        // (Exact promotion SEE is a refinement; the value of the new queen is
        // not modelled, which only under-estimates winning promo-captures.)
    }

    // All pieces attacking `to` under the updated occupancy.  Intersect with
    // `occ` so attackers we later remove (and the mover already removed) drop
    // out, and so we never treat the piece sitting on `to` as its own attacker.
    Bitboard attackers = pos.attackers_to(to, occ) & occ;

    // Side to recapture is the opponent of the mover.
    Color stm = opp(color_of(moverPiece));

    // Sliders for X-ray reveal: bishops/queens on diagonals, rooks/queens on
    // ranks & files.
    const Bitboard bishopsQueens = pos.pieces(BISHOP) | pos.pieces(QUEEN);
    const Bitboard rooksQueens   = pos.pieces(ROOK)   | pos.pieces(QUEEN);

    while (true) {
        ++d;
        // Speculative material balance if `stm` recaptures with nextVictim and
        // the swap stops here.
        gain[d] = see_value(nextVictim) - gain[d - 1];

        // ── Find the least-valuable attacker for `stm` ──────────────────────
        const Bitboard stmAtt = attackers & pos.pieces(stm);
        PieceType lva = NO_PT;
        for (int pt = PAWN; pt <= KING; ++pt) {
            const Bitboard b = stmAtt & pos.pieces(stm, (PieceType)pt);
            if (b) {
                const Square s = lsb(b);
                occ ^= square_bb(s);          // this attacker now leaves the board
                nextVictim = (PieceType)pt;   // it becomes the next victim
                lva = (PieceType)pt;

                // ── Reveal X-ray attackers hidden behind `s` ───────────────
                // A pawn or bishop/queen capture can expose a diagonal slider;
                // a rook/queen capture can expose an orthogonal slider.
                if (pt == PAWN || pt == BISHOP || pt == QUEEN)
                    attackers |= bishop_attacks(to, occ) & bishopsQueens;
                if (pt == ROOK || pt == QUEEN)
                    attackers |= rook_attacks(to, occ) & rooksQueens;
                attackers &= occ;             // keep only still-present pieces
                break;
            }
        }

        if (lva == NO_PT) break;              // `stm` has no (more) attackers
        if (d + 1 >= 32) break;               // depth guard (cannot happen: ≤32 pieces)

        stm = opp(stm);
    }

    // ── Minimax the gain stack back to the root ─────────────────────────────
    // Each side will only continue the exchange if it improves on standing pat.
    while (--d > 0)
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);

    return gain[0];
}

} // namespace king
