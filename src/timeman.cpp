#include "timeman.hpp"
#include <algorithm>
#include <cstdint>

namespace king {

void TimeManager::init(const Limits& L, Color us, int overhead) {
    // fixed movetime
    if (L.movetime > 0) {
        // Use nearly all of the requested movetime. Only subtract a tiny fixed
        // guard (capped at 20ms) so the engine doesn't flag on slow I/O.
        // The full Move Overhead is NOT subtracted here — that cushion is for
        // clock mode where the engine manages its own time budget over the game.
        soft_ms = hard_ms = std::max<int64_t>(1, (int64_t)L.movetime - std::min(overhead, 20));
        return;
    }
    // no clock info (infinite / depth- or nodes-limited) -> effectively unlimited
    if (L.infinite || (L.time[us] == 0 && L.inc[us] == 0)) {
        soft_ms = hard_ms = INT64_MAX / 4;
        return;
    }
    int t   = L.time[us];
    int inc = L.inc[us];
    // panic: very low clock -> tiny fixed budget, move almost instantly, never flag
    if (t < 1000) {
        hard_ms = std::max<int64_t>(50, (int64_t)t / 10 - overhead);
        soft_ms = hard_ms;
        return;
    }
    int mtg      = (L.movestogo > 0) ? std::min(L.movestogo, 40) : 40;
    int64_t base = (int64_t)t / mtg + inc;
    int64_t soft = std::min<int64_t>(base, (int64_t)(t * 0.10));
    int64_t hard = std::min<int64_t>(soft * 4, (int64_t)(t * 0.40));
    // Reserve the communication overhead from the hard deadline only, and never
    // reserve more than half of it. Subtracting the full overhead from `soft`
    // (as before) made a low-but-not-panic clock (~1-9s, or even an 8s game
    // start when overhead is the 200ms default) collapse to ~1ms, so the engine
    // played instant, losing moves. Bounding `soft` by the overhead-adjusted
    // `hard` keeps sensible pacing while preserving flag-safety (hard <= 40% t).
    int64_t reserve = std::min<int64_t>(overhead, hard / 2);
    hard = std::max<int64_t>(1, hard - reserve);
    soft = std::min<int64_t>(soft, hard);
    soft_ms = std::max<int64_t>(1, soft);
    hard_ms = std::max<int64_t>(soft_ms, hard);  // hard always >= soft
}

} // namespace king
