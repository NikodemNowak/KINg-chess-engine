#include "search.hpp"
#include "movegen.hpp"
#include "bitboard.hpp"
#include "timeman.hpp"
#include "tt.hpp"
#include "crash.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <ostream>
#include <mutex>
#include <algorithm>
#include <utility>

namespace king {
namespace search {

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int INF     = 32000;
static constexpr int MATE    = 30000;
static constexpr int MAX_PLY = 64;

// ── Mate-score TT adjustment ────────────────────────────────────────────────
// Mate scores are stored relative to the *root*, but searched relative to the
// node. A score of (MATE - ply) means "mate in `ply` half-moves from here".
// On store we fold `ply` out (toTT); on read we fold the current `ply` back in
// (fromTT). These are exact inverses.
static inline bool is_mate_score(int s) {
    return s >= MATE - MAX_PLY || s <= -(MATE - MAX_PLY);
}
static inline int16_t toTT(int score, int ply) {
    if (score >=  MATE - MAX_PLY) score += ply;
    else if (score <= -(MATE - MAX_PLY)) score -= ply;
    return (int16_t)score;
}
static inline int fromTT(int score, int ply) {
    if (score >=  MATE - MAX_PLY) score -= ply;
    else if (score <= -(MATE - MAX_PLY)) score += ply;
    return score;
}

// ── Material values ───────────────────────────────────────────────────────────
// Centipawn value of a piece type (indexed by PieceType: P N B R Q K).
static inline int value_of(PieceType pt) {
    static const int val[6] = { 100, 320, 330, 500, 900, 0 };
    return val[pt];
}

// ── Material evaluation ───────────────────────────────────────────────────────
// Returns a score relative to the side to move (positive = better for mover).
static int evaluate(const Position& p) {
    int sc = 0;
    for (int pt = PAWN; pt <= QUEEN; ++pt)
        sc += value_of((PieceType)pt) * (popcount(p.pieces(WHITE, (PieceType)pt))
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

    // ── Quiescence search ──────────────────────────────────────────────────
    // Extends the leaf search through "noisy" moves (captures/promotions, and
    // all evasions while in check) so the static eval is only taken in a quiet
    // position. This removes the horizon effect inside capture sequences.
    int qsearch(Position& pos, int alpha, int beta, int ply) {
        if (aborted || times_up()) { aborted = true; return 0; }
        ++nodes;

        if (ply >= MAX_PLY) return evaluate(pos);

        const bool inCheck = pos.in_check(pos.side_to_move());

        int standPat = 0;
        int best;
        if (inCheck) {
            // Cannot "stand pat" while in check: we must answer the check, so
            // every legal evasion is searched (not just captures).
            best = -INF;
        } else {
            standPat = evaluate(pos);
            if (standPat >= beta) return standPat; // fail-high: opponent won't allow this
            if (standPat > alpha) alpha = standPat;
            best = standPat;
        }

        MoveList ml;
        generate_pseudo(pos, ml);

        // Precompute MVV-LVA keys for ordering. For non-capture/non-promo moves
        // (only relevant when in check) the key is irrelevant; we keep them
        // last via a very small key.
        int keys[256];
        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            const bool isEp      = (type_of(m) == EN_PASSANT);
            const Piece capPiece = pos.piece_on(to_sq(m));
            const bool isCapture = isEp || (capPiece != NO_PIECE);
            const bool isPromo   = (type_of(m) == PROMO);
            if (isCapture) {
                PieceType victim   = isEp ? PAWN : piece_type(capPiece);
                PieceType attacker = piece_type(pos.piece_on(from_sq(m)));
                keys[i] = value_of(victim) * 16 - value_of(attacker);
                if (isPromo) keys[i] += value_of(QUEEN); // promo-captures sort high
            } else if (isPromo) {
                keys[i] = value_of(QUEEN); // quiet promotions: still noisy, order high
            } else {
                keys[i] = -1; // quiet (only used when in check); search after noisy
            }
        }

        // Selection sort by key descending (move count here is small).
        for (int i = 0; i < ml.size; ++i) {
            int bi = i;
            for (int j = i + 1; j < ml.size; ++j)
                if (keys[j] > keys[bi]) bi = j;
            if (bi != i) { std::swap(ml.moves[i], ml.moves[bi]); std::swap(keys[i], keys[bi]); }
        }

        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            const bool isEp      = (type_of(m) == EN_PASSANT);
            const Piece capPiece = pos.piece_on(to_sq(m));
            const bool isCapture = isEp || (capPiece != NO_PIECE);
            const bool isPromo   = (type_of(m) == PROMO);

            // When not in check, only consider captures and promotions.
            if (!inCheck && !isCapture && !isPromo) continue;

            // Delta pruning (only when not in check, plain captures): if even
            // winning the victim plus a safety margin cannot reach alpha, skip.
            if (!inCheck && isCapture && !isPromo) {
                int victim = value_of(isEp ? PAWN : piece_type(capPiece));
                if (standPat + victim + 200 <= alpha) continue;
            }

            StateInfo st;
            pos.do_move(m, st);
            // Legality: after do_move the mover is the non-side-to-move; legal
            // iff that mover's king is not in check.
            if (pos.in_check(Color(!pos.side_to_move()))) {
                pos.undo_move(m);
                continue;
            }
            int score = -qsearch(pos, -beta, -alpha, ply + 1);
            pos.undo_move(m);
            if (aborted) return 0;
            if (score > best)  best  = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break; // beta cutoff
        }

        // If in check and no legal move was found, it is checkmate.
        if (inCheck && best == -INF) return -(MATE - ply);

        return best;
    }

    int negamax(Position& pos, int depth, int alpha, int beta, int ply) {
        if (aborted || times_up()) { aborted = true; return 0; }
        ++nodes;

        const int  alphaOrig = alpha;
        const bool isPV       = (beta - alpha) > 1; // wide window ⇒ PV node

        // ── TT probe ───────────────────────────────────────────────────────
        Move ttMove = 0;
        TTEntry tte;
        bool ttHit = tt.probe(pos.key(), tte);
        if (ttHit) {
            ttMove = tte.move;
            // Cutoff only on non-PV nodes with a deep-enough entry whose bound
            // is compatible with the window. (Keeps the PV exact/intact.)
            if (!isPV && tte.depth >= depth) {
                int s = fromTT(tte.score, ply);
                Bound b = Bound(tte.genBound & 3);
                if (b == BOUND_EXACT) return s;
                if (b == BOUND_LOWER && s >= beta)  return s;
                if (b == BOUND_UPPER && s <= alpha) return s;
            }
        }

        if (depth <= 0) return qsearch(pos, alpha, beta, ply);

        MoveList ml;
        generate_pseudo(pos, ml);

        // ── Move ordering: try the TT move first if it is in this list ──────
        // (Verifying membership guards against key16 collisions.)
        if (ttMove != 0) {
            for (int i = 0; i < ml.size; ++i) {
                if (ml.moves[i] == ttMove) {
                    if (i != 0) std::swap(ml.moves[0], ml.moves[i]);
                    break;
                }
            }
        }

        int  best     = -INF;
        Move bestMove = 0;
        int  legal    = 0;
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
            if (score > best) {
                best     = score;
                bestMove = ml.moves[i];
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }

        if (legal == 0)
            return pos.in_check(pos.side_to_move()) ? -(MATE - ply) : 0;

        // ── TT store ─────────────────────────────────────────────────────────
        Bound bound = (best <= alphaOrig) ? BOUND_UPPER
                    : (best >= beta)       ? BOUND_LOWER
                    :                        BOUND_EXACT;
        tt.store(pos.key(), bestMove, toTT(best, ply), 0, (uint8_t)depth, bound);

        return best;
    }
};

// ── think ─────────────────────────────────────────────────────────────────────
Move think(Position& pos, const Limits& L, std::atomic<bool>& stop,
           int overhead, int /*threads*/,
           std::ostream& out, std::mutex* out_mtx) {
    TimeManager tm;
    tm.init(L, pos.side_to_move(), overhead);

    // Ensure the TT is sized before the first search (covers direct callers
    // that never went through the UCI `Hash` option). Default 64 MB.
    if (tt.size() == 0) tt.resize(64);
    tt.new_search();

    MoveList root;
    generate_legal(pos, root);
    if (root.size == 0) return 0; // no legal move (mate/stalemate); UCI layer handles

    Move best = root.moves[0]; // safety default: always a legal move
    crash::arm_fallback(to_uci(best).c_str());

    Searcher s;
    s.stop     = &stop;
    s.hard_ms  = tm.hard_ms;
    s.start    = std::chrono::steady_clock::now();
    s.nodes    = 1;  // start at 1 so the first stop/time check fires at nodes=2048
    s.aborted  = false;

    int maxDepth = (L.depth > 0) ? std::min(L.depth, MAX_PLY) : MAX_PLY;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        // Order the previous iteration's best move first (it is also the TT
        // move for the root). This makes the new window tighten fastest.
        for (int i = 0; i < root.size; ++i) {
            if (root.moves[i] == best) {
                if (i != 0) std::swap(root.moves[0], root.moves[i]);
                break;
            }
        }

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
        // Store the completed root result (EXACT — full window, full search).
        tt.store(pos.key(), best, toTT(bestScore, 0), 0, (uint8_t)depth, BOUND_EXACT);
        crash::arm_fallback(to_uci(best).c_str());

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - s.start)
                      .count();
        if (out_mtx) out_mtx->lock();
        out << "info depth " << depth
            << " score cp " << bestScore
            << " nodes " << s.nodes
            << " time " << ms
            << " pv " << to_uci(best) << std::endl;
        if (out_mtx) out_mtx->unlock();

        if (ms >= tm.soft_ms) break; // don't start an iteration we likely can't finish
    }

    return best;
}

} // namespace search
} // namespace king
