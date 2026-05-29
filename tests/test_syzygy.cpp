// test_syzygy.cpp – Syzygy tablebase probe tests for KINg.
// These tests require actual TB files in syzygy_test/.
// The test SKIPS gracefully when that directory is absent or empty.
//
// IMPORTANT: TBs are always disabled (syzygy::init("")) at the end of every
// test case so subsequent tests (search/uci etc.) are never affected.
#include "doctest/doctest.h"
#include "syzygy.hpp"
#include "position.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"
using namespace king;

// tbprobe.h WDL constants (redefined here so the test doesn't depend on Fathom's
// C header being directly visible in C++ with extern "C" gymnastics)
#ifndef TB_WIN
#define TB_WIN           4u
#define TB_CURSED_WIN    3u
#define TB_DRAW          2u
#define TB_BLESSED_LOSS  1u
#define TB_LOSS          0u
#define TB_RESULT_FAILED 0xFFFFFFFFu
#endif

// Path to the small TB subset used for testing.
// Can be overridden at compile time: -DTB_TEST_PATH="..."
#ifndef TB_TEST_PATH
#define TB_TEST_PATH "syzygy_test"
#endif

static void ensure_init() {
    static bool done = false;
    if (!done) {
        bitboard::init();
        attacks::init_leapers();
        attacks::init_magics();
        zobrist::init();
        done = true;
    }
}

// RAII guard: ensures TBs are always disabled when the scope exits, even on
// exceptions. This prevents TB state from leaking into subsequent test cases.
struct TBGuard {
    ~TBGuard() { syzygy::init(""); }
};

// ── tests that run only when TB files are present ─────────────────────────────
TEST_CASE("syzygy: disabled path — probe returns FAILED and engine is unchanged") {
    ensure_init();
    TBGuard g; // always disable at scope exit
    syzygy::init(""); // ensure disabled
    CHECK_FALSE(syzygy::enabled());

    // probe_wdl / probe_root on any position returns FAILED when disabled
    {
        Position p;
        p.set_fen("8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1");
        CHECK(syzygy::probe_wdl(p)  == TB_RESULT_FAILED);
        CHECK(syzygy::probe_root(p) == TB_RESULT_FAILED);
    }
    {
        Position p;
        p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        CHECK(syzygy::probe_wdl(p)  == TB_RESULT_FAILED);
        CHECK(syzygy::probe_root(p) == TB_RESULT_FAILED);
    }
}

TEST_CASE("syzygy: KQvK is WIN, KRvK is WIN, KQvK-black is LOSS") {
    ensure_init();
    TBGuard g; // always disable at scope exit

    // Try to init TBs. If they're not present, skip gracefully.
    syzygy::init(TB_TEST_PATH);
    if (!syzygy::enabled()) {
        WARN("Syzygy TB files not found at '" TB_TEST_PATH "' — skipping WDL probe tests");
        return;
    }

    // KQvK white to move → WIN
    {
        Position p;
        p.set_fen("8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1");
        unsigned wdl = syzygy::probe_wdl(p);
        CHECK(wdl == TB_WIN);
    }

    // KRvK white to move → WIN
    {
        Position p;
        p.set_fen("4k3/8/8/8/8/8/8/4KR2 w - - 0 1");
        unsigned wdl = syzygy::probe_wdl(p);
        CHECK(wdl == TB_WIN);
    }

    // KQvK black to move → LOSS (from black's POV)
    {
        Position p;
        p.set_fen("8/8/8/4k3/8/8/4Q3/4K3 b - - 0 1");
        unsigned wdl = syzygy::probe_wdl(p);
        CHECK(wdl == TB_LOSS);
    }

    // KvK lone kings → DRAW or FAILED (KvK may not be in every TB set)
    {
        Position p;
        p.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        unsigned wdl = syzygy::probe_wdl(p);
        if (wdl != TB_RESULT_FAILED) {
            CHECK(wdl == TB_DRAW);
        }
    }

    // probe_root for KQvK → valid result with WDL=WIN
    {
        Position p;
        p.set_fen("8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1");
        unsigned res = syzygy::probe_root(p);
        CHECK(res != TB_RESULT_FAILED);
        unsigned wdl = res & 0x0Fu;  // TB_GET_WDL
        CHECK(wdl == TB_WIN);
    }

    // KPvK → not FAILED (may be WIN or DRAW depending on position)
    {
        Position p;
        p.set_fen("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1");
        unsigned wdl = syzygy::probe_wdl(p);
        CHECK(wdl != TB_RESULT_FAILED);
    }

    // TBGuard destructor disables TBs
}
