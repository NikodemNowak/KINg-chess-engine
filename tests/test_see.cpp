#include "doctest/doctest.h"
#include "see.hpp"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
using namespace king;

static void see_init() {
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
}

// All values below are hand-computed. SEE is from the moving side's
// perspective: positive = the mover ends up ahead in material.
TEST_CASE("see values") {
    see_init();

    // ── Free pawn: exd5, nothing defends d5 ──────────────────────────────
    {
        Position p;
        p.set_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        CHECK(see(p, make_move(E4, D5)) == 100);
    }

    // ── Pawn for pawn: exd5 recaptured by c6 pawn → net 0 ────────────────
    {
        Position p;
        p.set_fen("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1");
        CHECK(see(p, make_move(E4, D5)) == 0);
    }

    // ── Rook grabs defended pawn: Rxd4 wins a pawn but loses the rook to
    //    the recapturing rook on e4 → 100 - 500 = -400 ───────────────────
    {
        Position p;
        p.set_fen("4k3/8/8/8/3pr3/8/3R4/4K3 w - - 0 1");
        CHECK(see(p, make_move(D2, D4)) == 100 - 500);
    }

    // ── Free pawn for the rook: Rxd5, d5 undefended → +100 ───────────────
    {
        Position p;
        p.set_fen("4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1");
        CHECK(see(p, make_move(D2, D5)) == 100);
    }
}

TEST_CASE("see x-ray: doubled rooks win a defended pawn") {
    see_init();
    // White Rd1, Rd2; Black Rd7, Pd4.  Rxd4 then RxR then RxR (back rook
    // revealed behind the front one) — white nets one pawn = +100.  This
    // requires X-ray attacker handling along the d-file.
    Position p;
    p.set_fen("4k3/3r4/8/8/3p4/8/3R4/3RK3 w - - 0 1");
    CHECK(see(p, make_move(D2, D4)) == 100);
}

TEST_CASE("see equal trade: rook takes rook, recaptured → 0") {
    see_init();
    // White Rd1, Kg1; Black Rd2, Rd7.  Rxd2 (+rook) then Rxd2 (-rook) and
    // white has no further attacker on d2 (king on g1 is out of reach) → 0.
    Position p;
    p.set_fen("4k3/3r4/8/8/8/8/3r4/3R2K1 w - - 0 1");
    CHECK(see(p, make_move(D1, D2)) == 0);
}

TEST_CASE("see quiet move onto attacked square is negative") {
    see_init();
    // White Rd1, Ke1; Black Qd8, Ke8.  Rd5 is a *quiet* move onto a square
    // attacked by the queen: the rook is simply lost → -500.
    Position p;
    p.set_fen("3qk3/8/8/8/8/8/8/3RK3 w - - 0 1");
    CHECK(see(p, make_move(D1, D5)) == -500);
}

TEST_CASE("see en passant capture is a pawn") {
    see_init();
    // White Pe5, Black Pd5 (just played d7-d5).  exd6 e.p. captures the
    // d5 pawn; nothing recaptures on d6 → +100.
    Position p;
    p.set_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    CHECK(see(p, make_move(E5, D6, EN_PASSANT)) == 100);
}

TEST_CASE("see least-valuable-attacker ordering") {
    see_init();
    // Black pawn d5 defended by a knight (c6 -> wait, knight defends d-? )
    // White pawn e4 captures d5; black recaptures with the *pawn* on c6
    // (cheapest), not the rook on d8.  Net 0 (pawn for pawn) rather than a
    // miscalculation that uses the rook.
    Position p;
    p.set_fen("3rk3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1");
    CHECK(see(p, make_move(E4, D5)) == 0);
}
