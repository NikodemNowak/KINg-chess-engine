// syzygy.cpp – Fathom tablebase wrapper for KINg
// Compiles cleanly both as part of the engine and unit_tests.
// All probing is behind syzygy::enabled() guards; if no SyzygyPath is set
// the engine behaves exactly as without this file.

#include "syzygy.hpp"
#include "bitboard.hpp"  // popcount
#include "types.hpp"
#include "position.hpp"

// tbprobe.h is a C header; include as C.
extern "C" {
#include "fathom/tbprobe.h"
}

namespace king {
namespace syzygy {

// ── init / free ────────────────────────────────────────────────────────────────
void init(const std::string& path) {
    if (path.empty()) {
        tb_free();
        return;
    }
    tb_init(path.c_str());
    // TB_LARGEST is set by Fathom; 0 means no files found.
}

bool enabled() {
    return TB_LARGEST > 0;
}

unsigned largest() {
    return TB_LARGEST;
}

// ── Internal: build Fathom arguments from Position ───────────────────────────
// Fathom square encoding: bit i = square i (a1=0, h8=63) — same as ours.
// King's Square encoding: A1=0, H1=7, A2=8 … H8=63.  Matches.
//
// tb_probe_wdl signature:
//   (white, black, kings, queens, rooks, bishops, knights, pawns,
//    rule50, castling, ep, turn)
// For the inline tb_probe_wdl, castling != 0 returns FAILED immediately,
// and rule50 != 0 also returns FAILED.  We pass them in and let Fathom decide.
//
// En passant: Fathom wants 0 for no EP, otherwise the EP square index.
// Our Position::ep_square() returns NO_SQ (64) when there is no EP.

static unsigned fathom_ep(Square ep) {
    return (ep == NO_SQ) ? 0u : static_cast<unsigned>(ep);
}

// ── probe_wdl ─────────────────────────────────────────────────────────────────
unsigned probe_wdl(const Position& pos) {
    if (!enabled()) return TB_RESULT_FAILED;
    if (pos.castling_rights() != 0) return TB_RESULT_FAILED;
    if (static_cast<unsigned>(popcount(pos.occupied())) > TB_LARGEST)
        return TB_RESULT_FAILED;

    const uint64_t white   = pos.pieces(WHITE);
    const uint64_t black   = pos.pieces(BLACK);
    const uint64_t kings   = pos.pieces(KING);
    const uint64_t queens  = pos.pieces(QUEEN);
    const uint64_t rooks   = pos.pieces(ROOK);
    const uint64_t bishops = pos.pieces(BISHOP);
    const uint64_t knights = pos.pieces(KNIGHT);
    const uint64_t pawns   = pos.pieces(PAWN);

    const unsigned ep     = fathom_ep(pos.ep_square());
    const bool     turn   = (pos.side_to_move() == WHITE);

    // Probe with rule50=0: Fathom's tb_probe_wdl returns FAILED whenever rule50
    // != 0, which would suppress the TB signal for almost every mid-game
    // position. Passing 0 always yields the (50-move-unaware) WDL; the search
    // applies a (100 - halfmove_clock)/100 scale to decisive scores so a win
    // near the 50-move limit is valued cautiously.
    return tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights,
                        pawns, /*rule50=*/0u, /*castling=*/0, ep, turn);
}

// ── probe_root ────────────────────────────────────────────────────────────────
unsigned probe_root(const Position& pos) {
    if (!enabled()) return TB_RESULT_FAILED;
    if (pos.castling_rights() != 0) return TB_RESULT_FAILED;
    if (static_cast<unsigned>(popcount(pos.occupied())) > TB_LARGEST)
        return TB_RESULT_FAILED;

    const uint64_t white   = pos.pieces(WHITE);
    const uint64_t black   = pos.pieces(BLACK);
    const uint64_t kings   = pos.pieces(KING);
    const uint64_t queens  = pos.pieces(QUEEN);
    const uint64_t rooks   = pos.pieces(ROOK);
    const uint64_t bishops = pos.pieces(BISHOP);
    const uint64_t knights = pos.pieces(KNIGHT);
    const uint64_t pawns   = pos.pieces(PAWN);

    const unsigned rule50 = static_cast<unsigned>(pos.halfmove_clock());
    const unsigned ep     = fathom_ep(pos.ep_square());
    const bool     turn   = (pos.side_to_move() == WHITE);

    return tb_probe_root(white, black, kings, queens, rooks, bishops, knights,
                         pawns, rule50, /*castling=*/0, ep, turn,
                         /*results=*/nullptr);
}

} // namespace syzygy
} // namespace king
