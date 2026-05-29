#include "search.hpp"
#include "movegen.hpp"
#include "bitboard.hpp"
#include "timeman.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace king {
namespace search {

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int INF     = 32000;
static constexpr int MATE    = 30000;
static constexpr int MAX_PLY = 64;

// ── Material evaluation ───────────────────────────────────────────────────────
// Returns a score relative to the side to move (positive = better for mover).
static int evaluate(const Position& p) {
    static const int val[6] = { 100, 320, 330, 500, 900, 0 }; // P N B R Q K
    int sc = 0;
    for (int pt = PAWN; pt <= QUEEN; ++pt)
        sc += val[pt] * (popcount(p.pieces(WHITE, (PieceType)pt))
                       - popcount(p.pieces(BLACK, (PieceType)pt)));
    return (p.side_to_move() == WHITE) ? sc : -sc;
}

// ── UCI move string ───────────────────────────────────────────────────────────
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
        static const char promo_char[] = "nbrq"; // KNIGHT=0..QUEEN=3 offset from KNIGHT
        s += promo_char[promo_pt(m) - KNIGHT];
    }
    return s;
}

// ── Searcher ──────────────────────────────────────────────────────────────────
struct Searcher {
    std::atomic<bool>* stop;
    int64_t            hard_ms;
    std::chrono::steady_clock::time_point start;
    uint64_t           nodes;
    bool               aborted;

    bool times_up() {
        if ((nodes & 2047) == 0) {
            if (stop->load()) return true;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
            if (ms >= hard_ms) return true;
        }
        return false;
    }

    int negamax(Position& pos, int depth, int alpha, int beta, int ply) {
        if (aborted || times_up()) { aborted = true; return 0; }
        ++nodes;

        if (depth <= 0) return evaluate(pos);

        MoveList ml;
        generate_pseudo(pos, ml);

        int best = -INF, legal = 0;
        for (int i = 0; i < ml.size; ++i) {
            StateInfo st;
            pos.do_move(ml.moves[i], st);
            // Skip illegal pseudo-moves: after do_move the mover is now the
            // non-side-to-move, so we check that color for king safety.
            if (pos.in_check(Color(!pos.side_to_move()))) {
                pos.undo_move(ml.moves[i]);
                continue;
            }
            ++legal;
            int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            pos.undo_move(ml.moves[i]);
            if (aborted) return 0;
            if (score > best) best = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }

        if (legal == 0)
            return pos.in_check(pos.side_to_move()) ? -(MATE - ply) : 0;

        return best;
    }
};

// ── think ─────────────────────────────────────────────────────────────────────
Move think(Position& pos, const Limits& L, std::atomic<bool>& stop,
           int overhead, int /*threads*/) {
    TimeManager tm;
    tm.init(L, pos.side_to_move(), overhead);

    MoveList root;
    generate_legal(pos, root);
    if (root.size == 0) return 0; // no legal move (mate/stalemate); UCI layer handles

    Move best = root.moves[0]; // safety default: always a legal move

    Searcher s;
    s.stop     = &stop;
    s.hard_ms  = tm.hard_ms;
    s.start    = std::chrono::steady_clock::now();
    s.nodes    = 0;
    s.aborted  = false;

    int maxDepth = (L.depth > 0) ? std::min(L.depth, MAX_PLY) : MAX_PLY;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        int alpha = -INF, beta = INF, bestScore = -INF;
        Move iterBest = best;

        for (int i = 0; i < root.size; ++i) {
            StateInfo st;
            pos.do_move(root.moves[i], st);
            int score = -s.negamax(pos, depth - 1, -beta, -alpha, 1);
            pos.undo_move(root.moves[i]);
            if (s.aborted) break;
            if (score > bestScore) {
                bestScore = score;
                iterBest  = root.moves[i];
            }
            if (score > alpha) alpha = score;
        }

        if (s.aborted) break;
        best = iterBest;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - s.start)
                      .count();
        std::cout << "info depth " << depth
                  << " score cp " << bestScore
                  << " nodes " << s.nodes
                  << " time " << ms
                  << " pv " << to_uci(best) << "\n";

        if (ms >= tm.soft_ms) break; // don't start an iteration we likely can't finish
    }

    return best;
}

} // namespace search
} // namespace king
