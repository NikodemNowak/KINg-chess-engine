#include "search.hpp"
#include "movegen.hpp"
#include "bitboard.hpp"
#include "timeman.hpp"
#include "tt.hpp"
#include "see.hpp"
#include "crash.hpp"
#include "eval.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <ostream>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <utility>
#include <cmath>

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

// ── Late Move Reduction table ─────────────────────────────────────────────────
// LMR[depth][moveCount] gives the reduction in plies for a late quiet move.
// Formula: max(0, floor(0.75 + ln(depth) * ln(moveCount) / 2.25))
// Precomputed once; LMR[0][*] = LMR[*][0] = 0 (boundary guard).
static int LMR[64][64];

static void init_lmr() {
    for (int d = 0; d < 64; ++d)
        for (int m = 0; m < 64; ++m) {
            if (d == 0 || m == 0)
                LMR[d][m] = 0;
            else
                LMR[d][m] = (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
        }
}

// One-time init guard (init_lmr() is called from think() before the first search).
static std::atomic<bool> lmr_ready{false};

// ── Material values ───────────────────────────────────────────────────────────
// Centipawn value of a piece type — kept for move-ordering / SEE helpers.
static inline int value_of(PieceType pt) {
    static const int val[6] = { 100, 320, 330, 500, 900, 0 };
    return val[pt];
}

// evaluate() is now provided by eval.cpp (tapered PeSTO PSQT).
// The declaration lives in eval.hpp; we pull it into this namespace via:
using king::evaluate;

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

// ── Move-ordering constants ───────────────────────────────────────────────────
static constexpr int SCORE_TT_MOVE     =  2'000'000;
static constexpr int SCORE_CAPTURE     =  1'000'000; // good/equal capture base; +MVV-LVA
static constexpr int SCORE_KILLER1     =    900'000;
static constexpr int SCORE_KILLER2     =    800'000;
static constexpr int HISTORY_MAX       = 1 << 20;    // ~1M; clamp to stay below killer scores
static constexpr int SCORE_BAD_CAPTURE = -1'000'000; // SEE<0 capture base; tried after quiets

// ── Searcher ──────────────────────────────────────────────────────────────────
struct Searcher {
    std::atomic<bool>* stop;
    int64_t            hard_ms;
    std::chrono::steady_clock::time_point start;
    uint64_t           nodes;
    bool               aborted;

    // ── Move-ordering tables ───────────────────────────────────────────────
    Move killers[MAX_PLY][2];          // 2 killer (quiet) moves per ply
    int  history[2][64][64];           // [side-to-move][from][to] for quiet moves

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

                // SEE pruning: don't search captures that lose material outright
                // (the static exchange comes out negative).  This is the main
                // qsearch sharpener — equal/winning captures and promotions are
                // always kept.
                if (see(pos, m) < 0) continue;
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

    // ── PVS negamax ───────────────────────────────────────────────────────────
    // Principal Variation Search: the first move is searched with the full
    // window; subsequent moves use a zero-window (scout) search. Only if the
    // scout exceeds alpha AND is inside beta is a costly re-search done.
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

        const bool inCheck = pos.in_check(pos.side_to_move());
        const Color stm    = pos.side_to_move();

        // ── Static evaluation (cached for pruning) ────────────────────────────
        // Not meaningful while in check (king is in danger, eval is unstable).
        const int staticEval = inCheck ? -INF : evaluate(pos);

        // ── Reverse Futility Pruning (static null move) ───────────────────────
        // If the static eval already beats beta by a large margin (depth*75), the
        // position is so good we can prune without searching further.
        if (!isPV && !inCheck && depth <= 8
            && beta < MATE - MAX_PLY && beta > -(MATE - MAX_PLY)
            && staticEval - 75 * depth >= beta)
            return staticEval;

        // ── Null-move pruning (NMP) ───────────────────────────────────────────
        // Skip our turn and see if the opponent can still fail high. If they
        // can't beat beta even with a free move, the position is so good that we
        // can prune without searching further.
        // Conditions: not PV, not in check, depth >= 3, side has non-pawn
        // material (avoids zugzwang in pure K+P endings), and beta is finite.
        bool hasNonPawn = (pos.pieces(stm, KNIGHT) | pos.pieces(stm, BISHOP)
                         | pos.pieces(stm, ROOK)   | pos.pieces(stm, QUEEN)) != 0;
        if (!isPV && !inCheck && depth >= 3 && hasNonPawn
            && beta < MATE - MAX_PLY && beta > -(MATE - MAX_PLY)) {
            int R = 3 + depth / 3;
            StateInfo nullSt;
            pos.do_null_move(nullSt);
            int nullScore = -negamax(pos, std::max(0, depth - 1 - R), -beta, -beta + 1, ply + 1);
            pos.undo_null_move();
            if (aborted) return 0;
            if (nullScore >= beta) return beta; // fail-high prune (return bound)
        }

        // ── Internal Iterative Reduction (IIR) ───────────────────────────────
        // If we have no TT move at this node (no prior search result to guide
        // ordering), do a shallower search by reducing depth by 1. This cheap
        // sacrifice pays off on the next visit when the TT move is available and
        // the full-depth search is better ordered.
        // Conditions: no TT move, depth >= 4, not in check.
        if (!ttMove && depth >= 4 && !inCheck) depth -= 1;

        MoveList ml;
        generate_pseudo(pos, ml);

        // ── Move scoring (for ordering) ───────────────────────────────────────
        // Assign a score to each pseudo-legal move; we iterate in descending
        // score order using selection sort (lazy, one pass per iteration).
        //
        // Score buckets (non-overlapping):
        //   TT move:        2,000,000
        //   Captures/promos: 1,000,000 + MVV-LVA bonus  (max ~14*16+900 ≈ 1,127,300)
        //   Killer 1:          900,000
        //   Killer 2:          800,000
        //   Quiet history:     history[stm][from][to]   (clamped to < 800,000)
        int scores[256];
        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            if (ttMove != 0 && m == ttMove) {
                scores[i] = SCORE_TT_MOVE;
            } else {
                const bool isEp      = (type_of(m) == EN_PASSANT);
                const Piece capPiece = pos.piece_on(to_sq(m));
                const bool isCapture = isEp || (capPiece != NO_PIECE);
                const bool isPromo   = (type_of(m) == PROMO);

                if (isCapture || isPromo) {
                    // MVV-LVA: most-valuable-victim / least-valuable-attacker
                    PieceType victim   = isEp ? PAWN
                                       : isCapture ? piece_type(capPiece)
                                       : PAWN; // quiet promo: no real victim
                    PieceType attacker = piece_type(pos.piece_on(from_sq(m)));
                    int mvvlva = value_of(victim) * 16 - value_of(attacker);
                    if (isPromo) mvvlva += value_of(QUEEN);
                    // Split captures by SEE: good/equal captures keep the high
                    // band; captures that lose material (SEE<0) drop below all
                    // quiets/history so they are tried last.  Promotions are
                    // always treated as good (kept high).
                    if (!isPromo && isCapture && see(pos, m) < 0)
                        scores[i] = SCORE_BAD_CAPTURE + mvvlva;
                    else
                        scores[i] = SCORE_CAPTURE + mvvlva;
                } else {
                    // Quiet move: check killers then history
                    if (ply < MAX_PLY && m == killers[ply][0]) {
                        scores[i] = SCORE_KILLER1;
                    } else if (ply < MAX_PLY && m == killers[ply][1]) {
                        scores[i] = SCORE_KILLER2;
                    } else {
                        scores[i] = history[stm][from_sq(m)][to_sq(m)];
                    }
                }
            }
        }

        int  best      = -INF;
        Move bestMove  = 0;
        int  legal     = 0;
        int  moveCount = 0;  // post-legality counter (for LMR/LMP thresholds)
        bool firstMove = true;

        for (int i = 0; i < ml.size; ++i) {
            // ── Selection sort: pick the highest-scored remaining move ─────
            int bi = i;
            for (int j = i + 1; j < ml.size; ++j)
                if (scores[j] > scores[bi]) bi = j;
            if (bi != i) {
                std::swap(ml.moves[i], ml.moves[bi]);
                std::swap(scores[i],   scores[bi]);
            }

            Move m = ml.moves[i];

            // ── Classify move BEFORE do_move (board is still unmoved) ─────
            // isQuiet: not a capture, not en passant, not a promotion.
            const bool isEpPre      = (type_of(m) == EN_PASSANT);
            const Piece capPiecePre = pos.piece_on(to_sq(m));
            const bool isQuiet      = !(isEpPre || (capPiecePre != NO_PIECE)
                                                 || (type_of(m) == PROMO));

            // ── Late Move Pruning (LMP) ───────────────────────────────────
            // Skip very late quiet moves at low depth when not in check and
            // we already have a non-losing best score.  (moveCount is the
            // post-legality count so early moves always pass the threshold.)
            // Only prune at depth >= 3 to avoid hiding tactics at shallow nodes.
            if (!isPV && !inCheck && isQuiet
                    && depth >= 3 && depth <= 6
                    && moveCount >= 4 + depth * depth
                    && best > -(MATE - MAX_PLY)) {
                continue;
            }

            // ── Futility Pruning (frontier) ───────────────────────────────
            // At very shallow depths, if the static eval plus a margin cannot
            // reach alpha, this quiet move almost certainly can't raise alpha.
            // Guard: only fire when the position isn't already losing (eval >=
            // -150) to avoid pruning in sharp positions where material eval
            // underestimates piece activity.
            if (!isPV && !inCheck && isQuiet && depth <= 6
                    && best > -(MATE - MAX_PLY)
                    && staticEval >= -150
                    && staticEval + 100 + 80 * depth <= alpha) {
                continue;
            }

            // ── Shallow SEE pruning ───────────────────────────────────────
            // At low depth in a non-PV node, prune moves whose static exchange
            // is clearly bad: losing captures, or quiets that walk onto a
            // square where the piece is lost.  The margin grows with depth² so
            // we prune more conservatively as depth rises.  Guarded by
            // best > mate-loss so the first move (best == -INF) is never pruned.
            if (!isPV && !inCheck && depth <= 6
                    && best > -(MATE - MAX_PLY)
                    && see(pos, m) < -20 * depth * depth) {
                continue;
            }

            StateInfo st;
            pos.do_move(m, st);
            // Skip illegal pseudo-moves: after do_move the mover is now the
            // non-side-to-move, so we check that color for king safety.
            if (pos.in_check(Color(!pos.side_to_move()))) {
                pos.undo_move(m);
                continue;
            }
            ++legal;
            ++moveCount;

            // ── Does this move give check? (opponent now to move) ─────────
            const bool givesCheck = pos.in_check(pos.side_to_move());

            // ── Check extension ───────────────────────────────────────────
            // Extend by 1 ply when this move gives check. Checks are forcing
            // moves that deserve deeper exploration; the MAX_PLY guard in the
            // recursion prevents depth runaway.
            const int extension = (givesCheck && ply < MAX_PLY - 1) ? 1 : 0;
            const int newDepth = depth - 1 + extension;
            int score;

            if (firstMove) {
                // First (best) move: full-window search, no reduction.
                score = -negamax(pos, newDepth, -beta, -alpha, ply + 1);
            } else {
                // ── Late Move Reduction (LMR) ─────────────────────────────
                int r = 0;
                if (depth >= 4 && moveCount >= 6 && isQuiet
                        && !inCheck && !givesCheck) {
                    r = LMR[std::min(depth, 63)][std::min(moveCount, 63)];
                    if (isPV && r > 0) r -= 1;          // reduce less on PV
                    if (r < 0) r = 0;
                    if (r > newDepth - 1) r = newDepth - 1; // never below 1 ply
                    if (r < 0) r = 0;
                }

                // Reduced zero-window scout.
                score = -negamax(pos, newDepth - r, -alpha - 1, -alpha, ply + 1);

                // Failed high while reduced → re-search at full depth (zero window).
                if (!aborted && score > alpha && r > 0)
                    score = -negamax(pos, newDepth, -alpha - 1, -alpha, ply + 1);

                // Still beating alpha but below beta → full-window re-search.
                if (!aborted && score > alpha && score < beta)
                    score = -negamax(pos, newDepth, -beta, -alpha, ply + 1);
            }

            pos.undo_move(m);
            if (aborted) return 0;
            firstMove = false;

            if (score > best) {
                best     = score;
                bestMove = m;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) {
                // ── Beta cutoff: update killers and history for quiet moves ──
                const bool isEpPost      = (type_of(m) == EN_PASSANT);
                const Piece capPiecePost = pos.piece_on(to_sq(m)); // piece already restored
                const bool isCapturePost = isEpPost || (capPiecePost != NO_PIECE);
                const bool isPromoPost   = (type_of(m) == PROMO);
                if (!isCapturePost && !isPromoPost) {
                    // Update killers (keep 2 distinct killers per ply)
                    if (ply < MAX_PLY && m != killers[ply][0]) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                    }
                    // Update history with depth² bonus (clamped to HISTORY_MAX)
                    int& h = history[stm][from_sq(m)][to_sq(m)];
                    h += depth * depth;
                    if (h > HISTORY_MAX) h = HISTORY_MAX;
                }
                break; // beta cutoff
            }
        }

        if (legal == 0)
            return inCheck ? -(MATE - ply) : 0;

        // ── TT store ─────────────────────────────────────────────────────────
        Bound bound = (best <= alphaOrig) ? BOUND_UPPER
                    : (best >= beta)       ? BOUND_LOWER
                    :                        BOUND_EXACT;
        tt.store(pos.key(), bestMove, toTT(best, ply), 0, (uint8_t)depth, bound);

        return best;
    }
};

// ── Per-thread search worker ────────────────────────────────────────────────
// Runs the full iterative-deepening / aspiration / root-PVS search on ONE
// thread, using the thread's OWN Searcher `s` (nodes/killers/history) and its
// OWN Position clone `pos`. The transposition table (`tt`) and the abort flag
// (`s.stop`) are SHARED across threads — that sharing is the entire point of
// Lazy SMP: helpers deepen the shared TT so the main thread searches better.
//
// Only the MAIN thread (isMain) drives time management (breaks on soft-time and
// then raises the shared stop flag) and emits `info` lines / arms the crash
// fallback. Helpers stay silent and simply loop until the shared stop flag (or
// the hard time limit, polled in times_up()) tells them to quit.
//
// The thread's best move + score + reached depth are written to `outBest`,
// `outScore`, `outDepth`.
static void smp_worker(Searcher& s, Position& pos, const TimeManager& tm,
                       int maxDepth, bool isMain,
                       Move startBest, std::atomic<bool>& stop,
                       std::ostream& out, std::mutex* out_mtx,
                       Move& outBest, int& outScore, int& outDepth,
                       bool arm_crash = true) {
    MoveList root;
    generate_legal(pos, root);
    if (root.size == 0) { outBest = 0; outScore = 0; outDepth = 0; return; }

    Move best      = (startBest != 0) ? startBest : root.moves[0];
    int  bestScore = 0;
    int  bestDepth = 0;
    int  prevScore = 0;

    Move iterBest  = best;
    int  iterScore = 0;

    // ── root_search ─────────────────────────────────────────────────────────
    // PVS over the root legal moves within [alpha, beta]; updates iterBest /
    // iterScore. On abort, iterBest/iterScore hold partial results from the
    // moves searched so far this depth.
    auto root_search = [&](int depth, int alpha, int beta) -> int {
        // Put iterBest first so the PVS first-move is the best candidate.
        for (int i = 0; i < root.size; ++i) {
            if (root.moves[i] == iterBest) {
                if (i != 0) std::swap(root.moves[0], root.moves[i]);
                break;
            }
        }

        int  rootBest  = -INF;
        Move localBest = iterBest;
        bool firstMove = true;

        for (int i = 0; i < root.size; ++i) {
            StateInfo st;
            pos.do_move(root.moves[i], st);
            int score;
            if (firstMove) {
                score = -s.negamax(pos, depth - 1, -beta, -alpha, 1);
            } else {
                score = -s.negamax(pos, depth - 1, -alpha - 1, -alpha, 1);
                if (!s.aborted && score > alpha && score < beta) {
                    score = -s.negamax(pos, depth - 1, -beta, -alpha, 1);
                }
            }
            pos.undo_move(root.moves[i]);

            if (s.aborted) break;
            firstMove = false;

            if (score > rootBest) {
                rootBest  = score;
                localBest = root.moves[i];
            }
            if (score > alpha) alpha = score;
            // No beta cutoff at root: always look for the best move.
        }

        if (rootBest > -INF) {
            iterBest  = localBest;
            iterScore = rootBest;
        }
        return rootBest;
    };

    for (int depth = 1; depth <= maxDepth; ++depth) {
        int delta = 20;
        int alpha, beta;
        if (depth >= 5) {
            alpha = prevScore - delta;
            beta  = prevScore + delta;
        } else {
            alpha = -INF;
            beta  =  INF;
        }

        int scoreThisDepth = 0;
        iterBest  = best;
        iterScore = 0;

        while (true) {
            scoreThisDepth = root_search(depth, alpha, beta);

            if (s.aborted) break;

            if (scoreThisDepth <= alpha) {
                beta  = (alpha + beta) / 2;
                alpha = std::max(-INF, scoreThisDepth - delta);
                delta += delta / 2;
            } else if (scoreThisDepth >= beta) {
                beta = std::min(INF, scoreThisDepth + delta);
                delta += delta / 2;
            } else {
                break; // in window: accept
            }

            if (delta > 2000) { alpha = -INF; beta = INF; }
        }

        if (s.aborted) break;

        // Commit the completed depth.
        prevScore = scoreThisDepth;
        best      = iterBest;
        bestScore = scoreThisDepth;
        bestDepth = depth;

        // Store the completed root result (EXACT — full window, full search).
        // Shared TT: lockless XOR scheme makes concurrent stores safe.
        tt.store(pos.key(), best, toTT(scoreThisDepth, 0), 0, (uint8_t)depth, BOUND_EXACT);

        if (isMain) {
            if (arm_crash) crash::arm_fallback(to_uci(best).c_str());

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - s.start)
                          .count();
            if (out_mtx) out_mtx->lock();
            out << "info depth " << depth
                << " score cp " << scoreThisDepth
                << " nodes " << s.nodes
                << " time " << ms
                << " pv " << to_uci(best) << std::endl;
            if (out_mtx) out_mtx->unlock();

            // Soft-time management is the MAIN thread's job only. When the main
            // thread decides to stop starting new iterations, it raises the
            // shared stop flag so every helper aborts its current search too.
            if (ms >= tm.soft_ms) { stop.store(true); break; }
        } else {
            // Helpers ignore soft time; they keep deepening (filling the TT)
            // until the shared stop flag / hard limit aborts them.
            if (stop.load()) break;
        }
    }

    // Make sure helpers also unblock the main thread's join promptly if THIS
    // thread is the one that hit max depth or ran out of legal work.
    if (isMain) stop.store(true);

    outBest  = best;
    outScore = bestScore;
    outDepth = bestDepth;
}

// ── think ─────────────────────────────────────────────────────────────────────
// Lazy SMP entry point. Spawns `threads` workers (clamped to [1,256]). Each
// worker owns a private Searcher + a private Position clone (copy_from) and runs
// the iterative-deepening search above, all sharing the global TT and the `stop`
// flag. Thread 0 (run inline on the calling thread) drives time management,
// prints info/bestmove, and reports the result. Helpers are silent and are
// ALWAYS joined before returning (every exit path) so there is never a dangling
// thread and never a data race on per-thread state.
//
// GUARANTEE: returns exactly one legal Move (or 0 only when there is no legal
// move at all), never crashes, and always joins all helper threads.
Move think(Position& pos, const Limits& L, std::atomic<bool>& stop,
           int overhead, int threads,
           std::ostream& out, std::mutex* out_mtx) {
    TimeManager tm;
    tm.init(L, pos.side_to_move(), overhead);

    // One-time LMR table initialisation (fast: just 64*64 = 4096 entries).
    if (!lmr_ready) { init_lmr(); lmr_ready = true; }

    // Ensure the TT is sized before the first search (covers direct callers
    // that never went through the UCI `Hash` option). Default 64 MB.
    if (tt.size() == 0) tt.resize(64);
    tt.new_search();

    // No legal move → mate/stalemate; the UCI layer handles emitting bestmove.
    MoveList root0;
    generate_legal(pos, root0);
    if (root0.size == 0) return 0;

    Move startBest = root0.moves[0];        // safety default: always legal
    crash::arm_fallback(to_uci(startBest).c_str());

    const int nThreads = std::max(1, std::min(256, threads));
    const int maxDepth = (L.depth > 0) ? std::min(L.depth, MAX_PLY) : MAX_PLY;
    const auto startTime = std::chrono::steady_clock::now();

    // ── Allocate per-thread state ───────────────────────────────────────────
    // Each helper gets its OWN Searcher (nodes/killers/history) and its OWN
    // Position clone. Thread 0 uses index 0. Position clones must outlive the
    // worker threads, hence the vectors live here on the stack of think().
    std::vector<Searcher> searchers(nThreads);
    std::vector<Position> positions(nThreads);
    std::vector<Move>     bestMoves(nThreads, startBest);
    std::vector<int>      bestScores(nThreads, 0);
    std::vector<int>      bestDepths(nThreads, 0);

    for (int t = 0; t < nThreads; ++t) {
        Searcher& s = searchers[t];
        s.stop    = &stop;
        s.hard_ms = tm.hard_ms;
        s.start   = startTime;
        s.nodes   = 1; // first stop/time check fires at nodes=2048
        s.aborted = false;
        std::memset(s.killers, 0, sizeof(s.killers));
        std::memset(s.history, 0, sizeof(s.history));
        positions[t].copy_from(pos); // private clone with full repetition history
    }

    // ── Spawn helpers (threads 1..N-1) ──────────────────────────────────────
    std::vector<std::thread> helpers;
    helpers.reserve(nThreads > 0 ? nThreads - 1 : 0);
    for (int t = 1; t < nThreads; ++t) {
        helpers.emplace_back([&, t]() {
            smp_worker(searchers[t], positions[t], tm, maxDepth,
                       /*isMain=*/false, startBest, stop,
                       out, out_mtx, bestMoves[t], bestScores[t], bestDepths[t]);
        });
    }

    // ── Run the main thread (index 0) inline ────────────────────────────────
    smp_worker(searchers[0], positions[0], tm, maxDepth,
               /*isMain=*/true, startBest, stop,
               out, out_mtx, bestMoves[0], bestScores[0], bestDepths[0]);

    // ── Join ALL helpers (every exit path) ──────────────────────────────────
    // The main worker raised `stop` on exit, so helpers see it and finish.
    for (auto& h : helpers)
        if (h.joinable()) h.join();

    // ── Report thread 0's result ────────────────────────────────────────────
    // Simple, correct Lazy SMP: the main thread reports. Its best move comes
    // from its own deepest completed iteration; helpers only enriched the TT.
    Move best = bestMoves[0];
    if (best == 0) best = startBest; // ultra-defensive: never return null here
    return best;
}

// ── NullStreambuf / NullStream ────────────────────────────────────────────────
// A streambuf that discards all output. Used by think_result to suppress the
// UCI info lines that think() emits, without hitting a null-pointer in the
// default ostream constructor (which crashes on write attempts on some platforms).
struct NullStreambuf : std::streambuf {
    int overflow(int c) override { return c; }       // accept and discard
    std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
};

// ── think_result ──────────────────────────────────────────────────────────────
// Identical to think() but returns both the best move and the root score.
// Uses a null-sink ostream so no info lines are emitted — suitable for datagen
// workers that run in parallel and would otherwise spam each other's output.
SearchResult think_result(Position& pos, const Limits& L,
                          std::atomic<bool>& stop,
                          int overhead, int threads) {
    TimeManager tm;
    tm.init(L, pos.side_to_move(), overhead);

    if (!lmr_ready) { init_lmr(); lmr_ready = true; }
    if (tt.size() == 0) tt.resize(64);
    tt.new_search();

    MoveList root0;
    generate_legal(pos, root0);
    if (root0.size == 0) return {0, 0};

    Move startBest = root0.moves[0];

    const int nThreads = std::max(1, std::min(256, threads));
    const int maxDepth = (L.depth > 0) ? std::min(L.depth, MAX_PLY) : MAX_PLY;
    const auto startTime = std::chrono::steady_clock::now();

    std::vector<Searcher> searchers(nThreads);
    std::vector<Position> positions(nThreads);
    std::vector<Move>     bestMoves(nThreads, startBest);
    std::vector<int>      bestScores(nThreads, 0);
    std::vector<int>      bestDepths(nThreads, 0);

    // Per-call null sink (not static, to avoid data races between concurrent
    // think_result calls in datagen threads).
    NullStreambuf null_buf;
    std::ostream  null_out(&null_buf);

    for (int t = 0; t < nThreads; ++t) {
        Searcher& s = searchers[t];
        s.stop    = &stop;
        s.hard_ms = tm.hard_ms;
        s.start   = startTime;
        s.nodes   = 1;
        s.aborted = false;
        std::memset(s.killers, 0, sizeof(s.killers));
        std::memset(s.history, 0, sizeof(s.history));
        positions[t].copy_from(pos);
    }

    std::vector<std::thread> helpers;
    helpers.reserve(nThreads > 0 ? nThreads - 1 : 0);
    for (int t = 1; t < nThreads; ++t) {
        helpers.emplace_back([&, t]() {
            smp_worker(searchers[t], positions[t], tm, maxDepth,
                       false, startBest, stop,
                       null_out, nullptr,
                       bestMoves[t], bestScores[t], bestDepths[t],
                       /*arm_crash=*/false);
        });
    }

    smp_worker(searchers[0], positions[0], tm, maxDepth,
               true, startBest, stop,
               null_out, nullptr,
               bestMoves[0], bestScores[0], bestDepths[0],
               /*arm_crash=*/false);

    for (auto& h : helpers)
        if (h.joinable()) h.join();

    Move best  = bestMoves[0];
    int  score = bestScores[0];
    if (best == 0) best = startBest;
    return {best, score};
}

} // namespace search
} // namespace king
