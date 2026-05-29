#pragma once
#include <cstdint>
#include "types.hpp"

namespace king {

struct Limits {
    int time[2]   = {0, 0};  // ms remaining, indexed by Color
    int inc[2]    = {0, 0};  // ms increment, indexed by Color
    int movestogo = 0;        // 0 = sudden death
    int movetime  = 0;        // 0 = not a fixed-movetime search
    int depth     = 0;        // 0 = no depth cap
    bool infinite = false;
};

struct TimeManager {
    int64_t soft_ms = 0;  // optimum: don't START a new ID iteration past this
    int64_t hard_ms = 0;  // maximum: ABORT search past this

    void init(const Limits& lim, Color us, int move_overhead);
};

} // namespace king
