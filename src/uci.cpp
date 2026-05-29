#include "uci.hpp"
#include "position.hpp"
#include "movegen.hpp"
#include "search.hpp"
#include "timeman.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"
#include "tt.hpp"
#include "types.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <deque>
#include <algorithm>

#ifdef __linux__
#include <sched.h>
#include <fstream>
#include <cmath>
#endif

namespace king {

// ── available_cores() ─────────────────────────────────────────────────────────
// Returns the effective number of CPU cores available to this process.
// On Linux: MIN of sched_getaffinity count, cgroup CPU quota (v2 then v1),
//           and hardware_concurrency(), clamped to ≥1.
// On Windows / other: hardware_concurrency() clamped to ≥1.
int available_cores() {
#ifdef __linux__
    int hc = static_cast<int>(std::thread::hardware_concurrency());
    if (hc <= 0) hc = 1;

    // (a) sched_getaffinity: number of CPUs in the affinity mask
    int affinity_count = hc;
    {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
            affinity_count = static_cast<int>(CPU_COUNT(&cs));
        if (affinity_count <= 0) affinity_count = hc;
    }

    // (b) cgroup CPU quota
    int cgroup_count = hc;

    // Try cgroup v2 first: /sys/fs/cgroup/cpu.max  →  "<quota> <period>" or "max <period>"
    {
        std::ifstream f("/sys/fs/cgroup/cpu.max");
        if (f.is_open()) {
            std::string quota_str, period_str;
            if (f >> quota_str >> period_str && quota_str != "max") {
                try {
                    long long quota  = std::stoll(quota_str);
                    long long period = std::stoll(period_str);
                    if (quota > 0 && period > 0)
                        cgroup_count = static_cast<int>(
                            std::max(1LL, (long long)std::ceil((double)quota / (double)period)));
                } catch (...) {}
            }
        }
    }

    // Fall back to cgroup v1 if cgroup_count is still hc
    if (cgroup_count == hc) {
        std::ifstream fq("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        std::ifstream fp("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        if (fq.is_open() && fp.is_open()) {
            long long quota = -1, period = 0;
            fq >> quota;
            fp >> period;
            if (quota > 0 && period > 0)
                cgroup_count = static_cast<int>(
                    std::max(1LL, (long long)std::ceil((double)quota / (double)period)));
        }
    }

    // Effective = MIN of all three, clamped ≥1
    int result = std::min({hc, affinity_count, cgroup_count});
    return std::max(1, result);
#else
    unsigned hc = std::thread::hardware_concurrency();
    return std::max(1, hc == 0 ? 1 : static_cast<int>(hc));
#endif
}

namespace uci {

// ── Local helper: Move → UCI string ───────────────────────────────────────────
static std::string to_uci(Move m) {
    static const char* sq_name[64] = {
        "a1","b1","c1","d1","e1","f1","g1","h1",
        "a2","b2","c2","d2","e2","f2","g2","h2",
        "a3","b3","c3","d3","e3","f3","g3","h3",
        "a4","b4","c4","d4","e4","f4","g4","h4",
        "a5","b5","c5","d5","e5","f5","g5","h5",
        "a6","b6","c6","d6","e6","f6","g6","h6",
        "a7","b7","c7","d7","e7","f7","g7","h7",
        "a8","b8","c8","d8","e8","f8","g8","h8"
    };
    std::string s;
    s += sq_name[from_sq(m)];
    s += sq_name[to_sq(m)];
    if (type_of(m) == PROMO) {
        static const char promo_char[] = "nbrq"; // KNIGHT offset
        s += promo_char[promo_pt(m) - KNIGHT];
    }
    return s;
}

// ── Tokenizer ──────────────────────────────────────────────────────────────────
static std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> toks;
    std::istringstream ss(line);
    std::string t;
    while (ss >> t) toks.push_back(t);
    return toks;
}

// Helper: find token index (-1 if not found)
static int find_tok(const std::vector<std::string>& toks, const std::string& s, int from = 0) {
    for (int i = from; i < (int)toks.size(); ++i)
        if (toks[i] == s) return i;
    return -1;
}

// ── EngineState ────────────────────────────────────────────────────────────────
// Holds the Position along with the StateInfo chain so that StateInfo objects
// remain alive as long as the position state uses them.
// Uses std::deque (not std::vector) because push_back on a deque never
// invalidates existing element pointers — critical since pos.st_ points into it.
struct EngineState {
    Position pos;
    std::deque<StateInfo> history;
    std::mutex out_mtx;

    EngineState() {
        pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    void reset_to_startpos() {
        history.clear();
        pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    void reset_to_fen(const std::string& fen) {
        history.clear();
        pos.set_fen(fen);
    }

    void apply_move(Move m) {
        history.emplace_back();
        pos.do_move(m, history.back());
    }
};

// ── run ────────────────────────────────────────────────────────────────────────
void run(std::istream& in, std::ostream& out) {
    // Initialize all tables (idempotent)
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();

    EngineState es;

    // Size the transposition table to the default before any search; the
    // `Hash` option (setoption) can resize it later.
    tt.resize(64);

    // stop flag shared with the worker thread.
    // search::think checks this flag every 2047 nodes (nodes starts at 1 in
    // the searcher, so the first check fires at nodes=2048). This means a
    // stop/quit arriving immediately after 'go' does NOT abort the search
    // at node-0; the first 2047 node-evaluations are unconditional.
    std::atomic<bool> stop{false};
    std::thread worker;

    int hashMB       = 64;
    // Default Threads = available_cores() so that Docker --cpus / cpuset limits
    // are respected on Linux (cgroup v2/v1 + affinity), clamped to [1,256].
    int threads      = std::max(1, std::min(256, available_cores()));
    int moveOverhead = 200;

    auto join_worker = [&]() {
        if (worker.joinable()) worker.join();
    };

    auto stop_and_join = [&]() {
        stop.store(true);
        join_worker();
    };

    std::string line;
    while (std::getline(in, line)) {
        try {
            auto toks = tokenize(line);
            if (toks.empty()) continue;

            const std::string& cmd = toks[0];

            // ── uci ──────────────────────────────────────────────────────────
            if (cmd == "uci") {
                std::lock_guard<std::mutex> lk(es.out_mtx);
                out << "id name KINg\n";
                out << "id author KINg Team\n";
                out << "option name Hash type spin default 64 min 1 max 1024\n";
                out << "option name Threads type spin default " << threads
                    << " min 1 max 256\n";
                out << "option name Move Overhead type spin default 200 min 0 max 5000\n";
                out << "option name Ponder type check default false\n";
                out << "uciok\n";
                out.flush();
            }

            // ── isready ──────────────────────────────────────────────────────
            else if (cmd == "isready") {
                std::lock_guard<std::mutex> lk(es.out_mtx);
                out << "readyok\n";
                out.flush();
            }

            // ── ucinewgame ───────────────────────────────────────────────────
            else if (cmd == "ucinewgame") {
                stop_and_join();
                stop = false;
                es.reset_to_startpos();
                tt.clear(); // fresh table for a new game
            }

            // ── setoption ────────────────────────────────────────────────────
            else if (cmd == "setoption") {
                int name_idx  = find_tok(toks, "name",  1);
                int value_idx = find_tok(toks, "value", 1);
                if (name_idx == -1) continue;

                int name_end = (value_idx != -1) ? value_idx : (int)toks.size();
                std::string opt_name;
                for (int i = name_idx + 1; i < name_end; ++i) {
                    if (!opt_name.empty()) opt_name += ' ';
                    opt_name += toks[i];
                }

                std::string val_str;
                if (value_idx != -1 && value_idx + 1 < (int)toks.size())
                    val_str = toks[value_idx + 1];

                std::string opt_lower = opt_name;
                std::transform(opt_lower.begin(), opt_lower.end(), opt_lower.begin(), ::tolower);

                if (opt_lower == "hash") {
                    try {
                        hashMB = std::max(1, std::min(1024, std::stoi(val_str)));
                        tt.resize(hashMB); // (re)allocate + clear the TT
                    } catch (...) {}
                } else if (opt_lower == "threads") {
                    try { threads = std::max(1, std::min(256, std::stoi(val_str))); } catch (...) {}
                } else if (opt_lower == "move overhead") {
                    try { moveOverhead = std::max(0, std::min(5000, std::stoi(val_str))); } catch (...) {}
                }
                // Unknown options silently ignored
            }

            // ── position ─────────────────────────────────────────────────────
            else if (cmd == "position") {
                stop_and_join();
                stop = false;
                if ((int)toks.size() < 2) continue;

                int moves_kw = -1;

                if (toks[1] == "startpos") {
                    es.reset_to_startpos();
                    moves_kw = find_tok(toks, "moves", 2);
                } else if (toks[1] == "fen") {
                    // Collect up to 6 FEN fields starting at toks[2]
                    std::string fen_str;
                    int fields = 0;
                    int i = 2;
                    for (; i < (int)toks.size() && fields < 6; ++i) {
                        if (toks[i] == "moves") break;
                        if (!fen_str.empty()) fen_str += ' ';
                        fen_str += toks[i];
                        ++fields;
                    }
                    try { es.reset_to_fen(fen_str); } catch (...) {}
                    moves_kw = find_tok(toks, "moves", i);
                }

                if (moves_kw != -1) {
                    for (int i = moves_kw + 1; i < (int)toks.size(); ++i) {
                        const std::string& tok = toks[i];
                        MoveList ml;
                        generate_legal(es.pos, ml);
                        bool found = false;
                        for (int j = 0; j < ml.size; ++j) {
                            if (to_uci(ml.moves[j]) == tok) {
                                es.apply_move(ml.moves[j]);
                                found = true;
                                break;
                            }
                        }
                        if (!found) break;
                    }
                }
            }

            // ── go ───────────────────────────────────────────────────────────
            else if (cmd == "go") {
                Limits lim;

                for (int i = 1; i < (int)toks.size(); ++i) {
                    const std::string& k = toks[i];
                    if (k == "infinite") {
                        lim.infinite = true;
                    } else if (i + 1 < (int)toks.size()) {
                        const std::string& v = toks[i + 1];
                        try {
                            if      (k == "wtime")     { lim.time[WHITE]  = std::stoi(v); ++i; }
                            else if (k == "btime")     { lim.time[BLACK]  = std::stoi(v); ++i; }
                            else if (k == "winc")      { lim.inc[WHITE]   = std::stoi(v); ++i; }
                            else if (k == "binc")      { lim.inc[BLACK]   = std::stoi(v); ++i; }
                            else if (k == "movestogo") { lim.movestogo    = std::stoi(v); ++i; }
                            else if (k == "movetime")  { lim.movetime     = std::stoi(v); ++i; }
                            else if (k == "depth")     { lim.depth        = std::stoi(v); ++i; }
                        } catch (...) {}
                    }
                }

                // Stop any previous search and join its thread
                stop_and_join();
                stop = false;  // reset for the new search

                int ovh  = moveOverhead;
                int thrs = threads;

                // Worker operates on es.pos by reference. The main loop only
                // touches pos via position/ucinewgame commands, which require
                // stop_and_join() first, so there is no concurrent access.
                // All output (info lines and bestmove) is serialized via out_mtx.
                worker = std::thread([&es, &stop, &out, lim, ovh, thrs]() {
                    Move m = search::think(es.pos, lim, stop, ovh, thrs, out, &es.out_mtx);
                    std::string bm = (m != 0) ? to_uci(m) : "0000";
                    std::lock_guard<std::mutex> lk(es.out_mtx);
                    out << "bestmove " << bm << "\n";
                    out.flush();
                });
            }

            // ── stop ─────────────────────────────────────────────────────────
            else if (cmd == "stop") {
                stop = true;
                // Worker emits bestmove when think returns
            }

            // ── quit ─────────────────────────────────────────────────────────
            else if (cmd == "quit") {
                stop = true;
                join_worker();
                return;
            }

            // Unknown commands are silently ignored

        } catch (...) {
            // Defensive: never crash on any parse error
        }
    }

    // EOF reached: behave like quit
    stop = true;
    join_worker();
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
    run(std::cin, std::cout);
}

} // namespace uci
} // namespace king
