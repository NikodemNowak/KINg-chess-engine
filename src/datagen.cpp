// Self-play data generation for NNUE training — see datagen.hpp.
//
// Each worker thread plays whole games on its own Position + search state.
// Opening diversity is achieved via 8 random plies per game (splitmix64 PRNG,
// seed = baseSeed ^ (threadId * 0x9E3779B9 + gameIndex)).
//
// Recorded positions satisfy:
//   * side to move is NOT in check
//   * the chosen bestmove is NOT a capture / promotion (quiet only)
//
// Adjudication rules (for speed):
//   * |scoreWhitePOV| >= 1000 for 4 consecutive plies -> leading side wins
//   * game length > 200 plies -> 0.5 draw
//   * checkmate / stalemate / is_draw() -> normal result

#include "datagen.hpp"
#include "search.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "timeman.hpp"
#include "types.hpp"
#include "tt.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace king {

namespace {

using Clock = std::chrono::steady_clock;

// ── splitmix64 PRNG ───────────────────────────────────────────────────────────
struct Splitmix64 {
    uint64_t state;

    explicit Splitmix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }

    // Uniform in [0, n)
    int next_int(int n) {
        if (n <= 1) return 0;
        return (int)(next() % (uint64_t)n);
    }
};

// ── Helper: is move a capture or promotion? ───────────────────────────────────
static bool is_noisy(const Position& pos, Move m) {
    if (type_of(m) == PROMO)      return true;
    if (type_of(m) == EN_PASSANT) return true;
    return pos.piece_on(to_sq(m)) != NO_PIECE;
}

// ── Game result constants ─────────────────────────────────────────────────────
static const char* RESULT_STR[3] = { "0.0", "0.5", "1.0" };
// Index: 0 = Black wins, 1 = draw, 2 = White wins.

// ── Per-game buffer entry ─────────────────────────────────────────────────────
struct PosEntry {
    std::string fen;
    int         score_white_pov; // centipawns, White POV
};

// ── Play one self-play game ───────────────────────────────────────────────────
// threadId and gameIndex seed the PRNG for opening diversity.
// searchDepth: fixed depth for the non-opening moves.
// Returns a vector of (fen, scoreWhitePOV) pairs and fills resultIdx.
static std::vector<PosEntry> play_game(int threadId, int gameIndex,
                                       int searchDepth, int& resultIdx) {
    // Seed: mix thread id and game index so every game has a different sequence.
    const uint64_t baseSeed = UINT64_C(0xDEADBEEFCAFEBABE);
    Splitmix64 rng(baseSeed ^ ((uint64_t)threadId * UINT64_C(0x9E3779B9)
                               + (uint64_t)gameIndex));

    static const char* STARTPOS =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    Position pos;
    pos.set_fen(STARTPOS);

    // ── 8 random opening plies (heap-allocated StateInfo for safe move chain) ──
    static constexpr int OPENING_PLIES = 8;
    std::vector<StateInfo> states(256);
    int ply_count = 0;

    for (int i = 0; i < OPENING_PLIES; ++i) {
        MoveList ml;
        generate_legal(pos, ml);
        if (ml.size == 0 || pos.is_draw()) {
            resultIdx = 1;
            return {};
        }
        Move m = ml.moves[rng.next_int(ml.size)];
        pos.do_move(m, states[ply_count++]);
    }

    // ── Main game loop ────────────────────────────────────────────────────────
    std::vector<PosEntry> buffer;
    buffer.reserve(80);

    static constexpr int MAX_PLIES    = 200;
    static constexpr int ADJUDICATE_SCORE = 1000;
    static constexpr int ADJUDICATE_COUNT = 4;

    int adj_count = 0; // consecutive plies with |score| >= ADJUDICATE_SCORE
    int adj_winner = 0; // +1 = white ahead, -1 = black ahead (last seen)

    Limits lim;
    lim.depth = searchDepth;

    for (;;) {
        if (pos.is_draw()) {
            resultIdx = 1; // draw
            break;
        }

        MoveList ml;
        generate_legal(pos, ml);

        if (ml.size == 0) {
            // Checkmate or stalemate
            if (pos.in_check(pos.side_to_move())) {
                // Checkmate: the side to move has lost.
                resultIdx = (pos.side_to_move() == WHITE) ? 0 : 2;
            } else {
                resultIdx = 1; // stalemate
            }
            break;
        }

        if (ply_count - OPENING_PLIES >= MAX_PLIES) {
            resultIdx = 1; // 200-ply draw
            break;
        }

        // Search for best move + score.
        std::atomic<bool> stop_flag{false};
        search::SearchResult sr = search::think_result(pos, lim, stop_flag, 0, 1);

        if (sr.move == 0) {
            // No move returned (shouldn't happen; generate_legal found moves).
            resultIdx = 1;
            break;
        }

        // Score in White's POV.
        int score_white = (pos.side_to_move() == WHITE) ? sr.score : -sr.score;

        // Adjudication check.
        if (std::abs(score_white) >= ADJUDICATE_SCORE) {
            if (adj_count == 0) {
                adj_winner = (score_white > 0) ? 1 : -1;
                adj_count  = 1;
            } else if ((score_white > 0) == (adj_winner > 0)) {
                ++adj_count;
            } else {
                // Sign flipped (rare oscillation), reset.
                adj_winner = (score_white > 0) ? 1 : -1;
                adj_count  = 1;
            }
            if (adj_count >= ADJUDICATE_COUNT) {
                resultIdx = (adj_winner > 0) ? 2 : 0;
                break;
            }
        } else {
            adj_count = 0;
        }

        // Record the position if not in check and bestmove is quiet.
        bool in_check_stm = pos.in_check(pos.side_to_move());
        if (!in_check_stm && !is_noisy(pos, sr.move)) {
            buffer.push_back({ pos.fen(), score_white });
        }

        // Apply the move.
        if (ply_count >= (int)states.size()) states.resize(states.size() + 64);
        pos.do_move(sr.move, states[ply_count]);
        ++ply_count;
    }

    return buffer;
}

// ── Worker thread ─────────────────────────────────────────────────────────────
static void worker(int threadId, int numGames, int searchDepth,
                   std::ofstream& out_file, std::mutex& out_mtx,
                   std::atomic<int64_t>& games_done,
                   std::atomic<int64_t>& positions_written) {
    for (int g = 0; g < numGames; ++g) {
        int resultIdx = 1;
        std::vector<PosEntry> entries;

        // Retry if the random opening leads to immediate game end.
        for (int retry = 0; retry < 20; ++retry) {
            entries = play_game(threadId, g * 20 + retry, searchDepth, resultIdx);
            if (!entries.empty() || resultIdx != 1) break;
            // Empty and draw usually means game ended in opening — retry.
        }

        if (entries.empty()) {
            ++games_done;
            continue; // nothing to write for this game
        }

        // Build output lines for this game.
        const char* res = RESULT_STR[resultIdx];
        std::ostringstream oss;
        for (const auto& e : entries) {
            oss << e.fen << " | " << e.score_white_pov << " | " << res << "\n";
        }

        {
            std::lock_guard<std::mutex> lk(out_mtx);
            out_file << oss.str();
        }

        int64_t gd = ++games_done;
        positions_written += (int64_t)entries.size();
        (void)gd;
    }
}

} // anonymous namespace

// ── Entry point ───────────────────────────────────────────────────────────────
int run_datagen(int argc, char** argv) {
    // argv: [0]=engine [1]=datagen [2]=outfile [3]=numGames [4]=depth [5]=threads
    if (argc < 4) {
        std::cerr << "usage: engine datagen <outfile> <numGames> [depth=8] [threads=N]\n";
        return 2;
    }

    const std::string outfile  = argv[2];
    const int         numGames = std::atoi(argv[3]);
    const int         depth    = (argc >= 5) ? std::atoi(argv[4]) : 8;
    const int         hw       = (int)std::thread::hardware_concurrency();
    const int         numThreads = (argc >= 6)
                                   ? std::atoi(argv[5])
                                   : std::max(1, hw > 0 ? hw : 4);

    if (numGames <= 0) {
        std::cerr << "datagen: numGames must be > 0\n";
        return 2;
    }

    std::cerr << "datagen: outfile=" << outfile
              << "  games=" << numGames
              << "  depth=" << depth
              << "  threads=" << numThreads << "\n";

    // Make sure the TT is initialized once before spawning threads.
    if (tt.size() == 0) tt.resize(64);

    std::ofstream out(outfile, std::ios::app);
    if (!out) {
        std::cerr << "datagen: cannot open output file: " << outfile << "\n";
        return 1;
    }

    std::mutex              out_mtx;
    std::atomic<int64_t>    games_done{0};
    std::atomic<int64_t>    positions_written{0};

    // Distribute games across threads.
    const int gamesPerThread = numGames / numThreads;
    const int extra          = numGames % numThreads;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto t_start = Clock::now();

    for (int t = 0; t < numThreads; ++t) {
        int myGames = gamesPerThread + (t < extra ? 1 : 0);
        if (myGames <= 0) continue;
        threads.emplace_back(worker, t, myGames, depth,
                             std::ref(out), std::ref(out_mtx),
                             std::ref(games_done), std::ref(positions_written));
    }

    // Progress reporter on the calling thread (every ~3 seconds).
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        int64_t gd  = games_done.load();
        int64_t pos = positions_written.load();
        double  secs = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cerr << "datagen: " << gd << "/" << numGames << " games  "
                  << pos << " positions  "
                  << (secs > 0.0 ? (double)pos / secs : 0.0) << " pos/sec\n";
        if (gd >= (int64_t)numGames) break;
    }

    for (auto& th : threads)
        if (th.joinable()) th.join();

    out.flush();
    out.close();

    double secs = std::chrono::duration<double>(Clock::now() - t_start).count();
    int64_t pos = positions_written.load();
    std::cerr << "datagen: done  " << numGames << " games  "
              << pos << " positions  "
              << (secs > 0.0 ? (double)pos / secs : 0.0) << " pos/sec\n";
    return 0;
}

} // namespace king
