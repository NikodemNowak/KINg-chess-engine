#pragma once
// ── Search tuning parameters (SPSA) ─────────────────────────────────────────
// In a normal (ship) build these are `constexpr`, so the compiler folds them and
// there is ZERO runtime cost vs the old hard-coded literals. In a `-DTUNE` build
// they become mutable globals settable via UCI `setoption name <Name> value <n>`,
// letting an external SPSA driver tune them against the slow-TC gauntlet. The
// winning values are then written back here as the new constexpr defaults.
//
// Defaults below EXACTLY reproduce the v-best engine (verified by identical
// fixed-depth node counts), so a -DTUNE build with no setoptions == v-best.

#ifdef TUNE
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#endif

namespace king { namespace sp {

#ifdef TUNE
#define SP_PARAM inline int
#else
#define SP_PARAM constexpr int
#endif

// Late move reduction table:  r = lmr_base/100 + ln(d)*ln(m) / (lmr_div/100)
SP_PARAM lmr_base     = 75;    // x100  → 0.75
SP_PARAM lmr_div      = 225;   // x100  → 2.25
SP_PARAM lmr_hist_div = 8192;  // quiet-history → reduction modulation divisor

// Null-move pruning reduction:  R = nmp_base + depth / nmp_div
SP_PARAM nmp_base = 3;
SP_PARAM nmp_div  = 3;

// Reverse futility pruning:  prune if staticEval - rfp_margin*depth >= beta
SP_PARAM rfp_margin = 75;

// Late move pruning move-count thresholds (improving / not improving)
SP_PARAM lmp_imp   = 4;
SP_PARAM lmp_noimp = 2;

// Frontier futility pruning:  evalForPruning + fut_base + fut_mult*depth <= alpha
SP_PARAM fut_base = 100;
SP_PARAM fut_mult = 80;

// Shallow SEE pruning margin:  see(move) < -see_margin * depth * depth
SP_PARAM see_margin = 20;

// Singular extension beta margin:  seBeta = ttScore - se_margin
SP_PARAM se_margin = 64;

// Aspiration window initial half-width
SP_PARAM asp_delta = 20;

// History pruning threshold:  skip late quiet if quietHist < -hp_mult*lmrDepth^2
SP_PARAM hp_mult = 512;

// History bonus:  min(hist_max, hist_quad*d*d + hist_lin*d - hist_lin)
SP_PARAM hist_quad = 4;
SP_PARAM hist_lin  = 120;
SP_PARAM hist_max  = 2048;

#undef SP_PARAM

#ifdef TUNE
// Registry for the UCI layer: list options (`uci`) and apply them (`setoption`)
// generically. {name, pointer, default, min, max}. Only present in -DTUNE builds.
struct SpEntry { const char* name; int* ptr; int def, lo, hi; };
inline std::vector<SpEntry>& registry() {
    static std::vector<SpEntry> r = {
        {"LmrBase",    &lmr_base,      75,  20, 200},
        {"LmrDiv",     &lmr_div,      225,  80, 500},
        {"LmrHistDiv", &lmr_hist_div, 8192,1024,32768},
        {"NmpBase",    &nmp_base,       3,   1,   6},
        {"NmpDiv",     &nmp_div,        3,   1,   8},
        {"RfpMargin",  &rfp_margin,    75,  30, 160},
        {"LmpImp",     &lmp_imp,        4,   1,  12},
        {"LmpNoimp",   &lmp_noimp,      2,   1,   8},
        {"FutBase",    &fut_base,     100,  20, 250},
        {"FutMult",    &fut_mult,      80,  30, 200},
        {"SeeMargin",  &see_margin,    20,   5,  60},
        {"SeMargin",   &se_margin,     64,  20, 160},
        {"AspDelta",   &asp_delta,     20,   6,  60},
        {"HpMult",     &hp_mult,      512, 128,2048},
        {"HistQuad",   &hist_quad,      4,   1,  12},
        {"HistLin",    &hist_lin,     120,  20, 300},
        {"HistMax",    &hist_max,     2048,512,8192},
    };
    return r;
}
inline bool ieq(const char* a, const std::string& b) {
    std::string s(a);
    if (s.size() != b.size()) return false;
    for (size_t i = 0; i < s.size(); ++i)
        if (std::tolower((unsigned char)s[i]) != std::tolower((unsigned char)b[i])) return false;
    return true;
}
inline bool set(const std::string& name, int val) {
    for (auto& e : registry())
        if (ieq(e.name, name)) { *e.ptr = std::max(e.lo, std::min(e.hi, val)); return true; }
    return false;
}
#endif // TUNE

}} // namespace king::sp
