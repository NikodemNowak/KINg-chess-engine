#include "timeman.hpp"
#include <algorithm>
#include <cstdint>

namespace king {

void TimeManager::init(const Limits& L, Color us, int overhead) {
    // fixed movetime
    if (L.movetime > 0) {
        // Reserve the move overhead, but never more than half of the allotted
        // time — otherwise a short movetime with a large overhead collapses the
        // search budget to ~0 and the engine plays at depth 1.
        int64_t ovh = std::min<int64_t>(overhead, (int64_t)L.movetime / 2);
        soft_ms = hard_ms = std::max<int64_t>(1, (int64_t)L.movetime - ovh);
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
    soft -= overhead;
    hard -= overhead;
    soft_ms = std::max<int64_t>(1, soft);
    hard_ms = std::max<int64_t>(soft_ms, hard);  // hard always >= soft
}

} // namespace king
