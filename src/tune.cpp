// In-engine Texel tuner (coordinate descent over g_eval) — see tune.hpp.
//
// Standard Texel method:
//   * static eval only (no search), un-relativised to White's POV;
//   * fit a logistic scale K minimising MSE(result, sigmoid(K*eval/400));
//   * coordinate-descent each tunable weight by +/- step, keeping changes that
//     lower the MSE, over decreasing step sizes until a wall-clock cap is hit.
//
// The dataset is the public Zurichess "quiet-labeled.epd" (FEN + game result).

#include "tune.hpp"
#include "eval.hpp"
#include "position.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace king {

namespace {

using Clock = std::chrono::steady_clock;

// One labeled training position.  Positions are stored separately in a deque
// (Position is non-copyable/non-movable); this just records the label and a
// pointer to the stored Position.
struct Sample {
    const Position* pos;
    float           result; // White-POV game result: 1.0 / 0.5 / 0.0
};

std::deque<Position> g_positions;  // stable storage; addresses kept in samples
std::vector<Sample>  g_samples;

int g_threads = 1;

// ── EPD parsing ───────────────────────────────────────────────────────────────
// Returns true and fills (fen, result) if the line is a usable labeled position.
bool parse_epd_line(const std::string& line, std::string& fen, float& result) {
    if (line.empty() || line[0] == '#') return false;

    // Label lives between the first pair of double quotes.
    size_t q1 = line.find('"');
    if (q1 == std::string::npos) return false;
    size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    std::string label = line.substr(q1 + 1, q2 - q1 - 1);

    if      (label == "1-0")     result = 1.0f;
    else if (label == "0-1")     result = 0.0f;
    else if (label == "1/2-1/2") result = 0.5f;
    else if (label == "1/2")     result = 0.5f;
    else if (label == "0.5")     result = 0.5f;
    else if (label == "1.0")     result = 1.0f;
    else if (label == "0.0")     result = 0.0f;
    else return false;

    // FEN = first four whitespace-separated tokens (board stm castling ep).
    std::istringstream ss(line);
    std::string a, b, c, d;
    if (!(ss >> a >> b >> c >> d)) return false;
    fen = a + " " + b + " " + c + " " + d;
    return true;
}

// Load up to maxPositions samples from the EPD file. Returns count loaded.
size_t load_dataset(const std::string& path, size_t maxPositions) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "tune: cannot open dataset: " << path << "\n";
        return 0;
    }
    std::string line, fen;
    float result;
    size_t loaded = 0, skipped = 0;
    while (loaded < maxPositions && std::getline(in, line)) {
        if (!parse_epd_line(line, fen, result)) { ++skipped; continue; }
        g_positions.emplace_back();
        try {
            g_positions.back().set_fen(fen);
        } catch (...) {
            g_positions.pop_back();
            ++skipped;
            continue;
        }
        g_samples.push_back({ &g_positions.back(), result });
        ++loaded;
    }
    if (skipped)
        std::cerr << "tune: skipped " << skipped << " unparseable lines\n";
    return loaded;
}

// ── MSE (multi-threaded) ──────────────────────────────────────────────────────
// White-POV static eval for a sample.
inline int qeval_white(const Sample& s) {
    int e = evaluate(*s.pos); // side-to-move relative
    return (s.pos->side_to_move() == WHITE) ? e : -e;
}

inline double sigmoid(double k_over_400, int eval_cp) {
    return 1.0 / (1.0 + std::exp(-k_over_400 * eval_cp));
}

// Sum of squared errors over [lo, hi) given a precomputed K/400.
double sse_range(double k_over_400, size_t lo, size_t hi) {
    double sse = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        double s = sigmoid(k_over_400, qeval_white(g_samples[i]));
        double d = g_samples[i].result - s;
        sse += d * d;
    }
    return sse;
}

double mse(double K) {
    const size_t n = g_samples.size();
    if (n == 0) return 0.0;
    const double k_over_400 = K / 400.0;

    int T = g_threads;
    if (T <= 1 || n < 4096) {
        return sse_range(k_over_400, 0, n) / (double)n;
    }
    std::vector<std::thread> pool;
    std::vector<double> partial(T, 0.0);
    size_t chunk = (n + T - 1) / T;
    for (int t = 0; t < T; ++t) {
        size_t lo = (size_t)t * chunk;
        size_t hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&, t, lo, hi]() {
            partial[t] = sse_range(k_over_400, lo, hi);
        });
    }
    for (auto& th : pool) th.join();
    double sse = 0.0;
    for (double p : partial) sse += p;
    return sse / (double)n;
}

// ── K optimisation (ternary search) ───────────────────────────────────────────
double optimise_K() {
    double best = mse(0.5), bestK = 0.5;
    // Coarse scan over a wide range to bracket the minimum, then ternary refine.
    for (double K = 0.1; K <= 8.0; K += 0.1) {
        double e = mse(K);
        if (e < best) { best = e; bestK = K; }
    }
    double lo = std::max(0.0, bestK - 0.2);
    double hi = bestK + 0.2;
    for (int it = 0; it < 40; ++it) {
        double m1 = lo + (hi - lo) / 3.0;
        double m2 = hi - (hi - lo) / 3.0;
        if (mse(m1) < mse(m2)) hi = m2; else lo = m1;
    }
    return (lo + hi) / 2.0;
}

// ── Tunable-parameter table ───────────────────────────────────────────────────
// Pointers to every int in g_eval the descent is allowed to move.
struct Param { int* p; const char* name; };

void collect_scalar_params(std::vector<Param>& v) {
    EvalParams& g = g_eval;
    // Material (skip KING == index 5; it cancels and stays 0).
    for (int pt = 0; pt < 5; ++pt) v.push_back({ &g.mg_value[pt], "mg_value" });
    for (int pt = 0; pt < 5; ++pt) v.push_back({ &g.eg_value[pt], "eg_value" });

    v.push_back({ &g.bishop_pair_mg,   "bishop_pair_mg" });
    v.push_back({ &g.bishop_pair_eg,   "bishop_pair_eg" });
    v.push_back({ &g.rook_open_mg,     "rook_open_mg" });
    v.push_back({ &g.rook_open_eg,     "rook_open_eg" });
    v.push_back({ &g.rook_semiopen_mg, "rook_semiopen_mg" });
    v.push_back({ &g.rook_semiopen_eg, "rook_semiopen_eg" });
    v.push_back({ &g.pawn_isolated_mg, "pawn_isolated_mg" });
    v.push_back({ &g.pawn_isolated_eg, "pawn_isolated_eg" });
    v.push_back({ &g.pawn_doubled_mg,  "pawn_doubled_mg" });
    v.push_back({ &g.pawn_doubled_eg,  "pawn_doubled_eg" });

    // Passed pawn: indices 1..6 are meaningful (0 = own back rank, 7 = 8th rank).
    for (int r = 1; r <= 6; ++r) v.push_back({ &g.passed_mg[r], "passed_mg" });
    for (int r = 1; r <= 6; ++r) v.push_back({ &g.passed_eg[r], "passed_eg" });

    for (int i = 0; i < 4; ++i) v.push_back({ &g.mob_pivot[i], "mob_pivot" });
    for (int i = 0; i < 4; ++i) v.push_back({ &g.mob_mg[i],    "mob_mg" });
    for (int i = 0; i < 4; ++i) v.push_back({ &g.mob_eg[i],    "mob_eg" });

    // King attack units for N,B,R,Q (P,K stay 0).
    for (int pt = 1; pt <= 4; ++pt) v.push_back({ &g.king_attack_units[pt], "king_attack_units" });
    v.push_back({ &g.pawn_shield_mg,      "pawn_shield_mg" });
    v.push_back({ &g.king_safety_max,     "king_safety_max" });
    v.push_back({ &g.king_safety_divisor, "king_safety_divisor" });
}

void collect_psqt_params(std::vector<Param>& v) {
    EvalParams& g = g_eval;
    for (int pt = 0; pt < 6; ++pt)
        for (int sq = 0; sq < 64; ++sq)
            v.push_back({ &g.mg_psqt[pt][sq], "mg_psqt" });
    for (int pt = 0; pt < 6; ++pt)
        for (int sq = 0; sq < 64; ++sq)
            v.push_back({ &g.eg_psqt[pt][sq], "eg_psqt" });
}

// ── Coordinate descent ────────────────────────────────────────────────────────
// Returns the final MSE. Updates g_eval in place.
double coordinate_descent(double K, std::vector<Param>& params,
                          const std::vector<int>& steps,
                          Clock::time_point deadline, int& improvements) {
    double current = mse(K);
    improvements = 0;

    for (int step : steps) {
        if (Clock::now() >= deadline) break;
        bool any_pass_improved = true;
        int pass = 0;
        while (any_pass_improved) {
            if (Clock::now() >= deadline) break;
            any_pass_improved = false;
            ++pass;
            for (size_t i = 0; i < params.size(); ++i) {
                if ((i & 31) == 0 && Clock::now() >= deadline) break;
                int* p = params[i].p;
                int orig = *p;

                // Try +step.
                *p = orig + step;
                double up = mse(K);
                if (up < current - 1e-9) {
                    current = up;
                    any_pass_improved = true;
                    ++improvements;
                    continue;
                }
                // Try -step.
                *p = orig - step;
                double dn = mse(K);
                if (dn < current - 1e-9) {
                    current = dn;
                    any_pass_improved = true;
                    ++improvements;
                    continue;
                }
                // Neither helped — revert.
                *p = orig;
            }
            double secs_left =
                std::chrono::duration<double>(deadline - Clock::now()).count();
            std::fprintf(stderr,
                "  step=%d pass=%d  MSE=%.8f  improvements=%d  (%.0fs to cap)\n",
                step, pass, current, improvements, secs_left);
            std::fflush(stderr);
        }
    }
    return current;
}

// ── Paste-ready dump of g_eval ────────────────────────────────────────────────
void dump_psqt(std::ostream& os, const char* name, const int t[6][64]) {
    static const char* PT[6] = { "PAWN","KNIGHT","BISHOP","ROOK","QUEEN","KING" };
    os << "    // " << name << "\n    {\n";
    for (int pt = 0; pt < 6; ++pt) {
        os << "        // " << PT[pt] << "\n        {\n";
        for (int r = 0; r < 8; ++r) {
            os << "            ";
            for (int f = 0; f < 8; ++f) {
                char buf[16];
                std::snprintf(buf, sizeof buf, "%4d,", t[pt][r * 8 + f]);
                os << buf << (f == 7 ? "" : " ");
            }
            os << "\n";
        }
        os << "        },\n";
    }
    os << "    },\n";
}

void dump_params(std::ostream& os) {
    const EvalParams& g = g_eval;
    auto arr = [&](const char* nm, const int* a, int n) {
        os << "    // " << nm << "\n    { ";
        for (int i = 0; i < n; ++i) { os << a[i]; if (i + 1 < n) os << ", "; }
        os << " },\n";
    };
    os << "// ===== paste the following into DEFAULT_EVAL in eval.cpp =====\n";
    os << "static const EvalParams DEFAULT_EVAL = {\n";
    arr("mg_value[6]", g.mg_value, 6);
    arr("eg_value[6]", g.eg_value, 6);
    os << "    " << g.bishop_pair_mg << ", " << g.bishop_pair_eg << ", // bishop_pair mg,eg\n";
    os << "    " << g.rook_open_mg << ", " << g.rook_open_eg << ", // rook_open mg,eg\n";
    os << "    " << g.rook_semiopen_mg << ", " << g.rook_semiopen_eg << ", // rook_semiopen mg,eg\n";
    os << "    " << g.pawn_isolated_mg << ", " << g.pawn_isolated_eg << ", // pawn_isolated mg,eg\n";
    os << "    " << g.pawn_doubled_mg << ", " << g.pawn_doubled_eg << ", // pawn_doubled mg,eg\n";
    arr("passed_mg[8]", g.passed_mg, 8);
    arr("passed_eg[8]", g.passed_eg, 8);
    arr("mob_pivot[4]", g.mob_pivot, 4);
    arr("mob_mg[4]", g.mob_mg, 4);
    arr("mob_eg[4]", g.mob_eg, 4);
    arr("king_attack_units[6]", g.king_attack_units, 6);
    os << "    " << g.pawn_shield_mg << ", // pawn_shield_mg\n";
    os << "    " << g.king_safety_max << ", // king_safety_max\n";
    os << "    " << g.king_safety_divisor << ", // king_safety_divisor\n";
    dump_psqt(os, "mg_psqt[6][64]", g.mg_psqt);
    dump_psqt(os, "eg_psqt[6][64]", g.eg_psqt);
    os << "};\n";
    os << "// ===== end paste =====\n";
}

} // namespace

// ── Entry point ───────────────────────────────────────────────────────────────
int run_tuner(int argc, char** argv) {
    // argv: [0]=engine [1]=tune [2]=epd [3]=maxPositions [4]=maxSeconds [5]=psqt
    if (argc < 3) {
        std::cerr << "usage: engine tune <epd> [maxPositions] [maxSeconds] [psqt]\n";
        return 2;
    }
    std::string epd = argv[2];
    size_t maxPositions = (argc >= 4) ? std::strtoull(argv[3], nullptr, 10) : 150000;
    int    maxSeconds   = (argc >= 5) ? std::atoi(argv[4]) : 1500;
    bool   tune_psqt    = false;
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], "psqt") == 0) tune_psqt = true;

    unsigned hc = std::thread::hardware_concurrency();
    g_threads = (hc == 0) ? 1 : (int)hc;

    eval_set_defaults();

    std::cerr << "tune: loading up to " << maxPositions << " positions from "
              << epd << " ...\n";
    auto t_load0 = Clock::now();
    size_t n = load_dataset(epd, maxPositions);
    auto t_load1 = Clock::now();
    if (n == 0) { std::cerr << "tune: no positions loaded\n"; return 1; }
    std::cerr << "tune: loaded " << n << " positions in "
              << std::chrono::duration<double>(t_load1 - t_load0).count()
              << "s, threads=" << g_threads << "\n";

    // Optimise K on the untuned weights.
    std::cerr << "tune: optimising K ...\n";
    double K = optimise_K();
    double mse0 = mse(K);
    std::cerr << "tune: K=" << K << "  initial MSE=" << mse0 << "\n";

    auto deadline = Clock::now() + std::chrono::seconds(maxSeconds);

    // Phase 1: scalar / material / HCE weights (high value, fast).
    std::vector<Param> scalar;
    collect_scalar_params(scalar);
    std::cerr << "tune: phase 1 — " << scalar.size()
              << " scalar params, coordinate descent ...\n";
    int imp1 = 0;
    std::vector<int> steps = { 8, 4, 2, 1 };
    double mse1 = coordinate_descent(K, scalar, steps, deadline, imp1);
    std::cerr << "tune: phase 1 done  MSE=" << mse1
              << "  improvements=" << imp1 << "\n";

    // Phase 2: optional PSQT, if requested and time remains.
    double mse_final = mse1;
    if (tune_psqt && Clock::now() < deadline) {
        std::vector<Param> psqt;
        collect_psqt_params(psqt);
        std::cerr << "tune: phase 2 — " << psqt.size()
                  << " PSQT params, coordinate descent ...\n";
        int imp2 = 0;
        std::vector<int> psqt_steps = { 8, 4, 2, 1 };
        mse_final = coordinate_descent(K, psqt, psqt_steps, deadline, imp2);
        std::cerr << "tune: phase 2 done  MSE=" << mse_final
                  << "  improvements=" << imp2 << "\n";
    }

    std::cerr << "\ntune: SUMMARY\n";
    std::cerr << "  positions   = " << n << "\n";
    std::cerr << "  K           = " << K << "\n";
    std::cerr << "  MSE before  = " << mse0 << "\n";
    std::cerr << "  MSE after   = " << mse_final << "\n";
    std::cerr << "  improvement = " << (mse0 - mse_final) << " ("
              << (100.0 * (mse0 - mse_final) / mse0) << "%)\n\n";

    // Paste-ready dump to stdout (so it can be captured cleanly).
    dump_params(std::cout);
    std::cout.flush();
    return 0;
}

} // namespace king
