#include "search.hpp"
#include "movegen.hpp"
#include "bitboard.hpp"
#include "timeman.hpp"
#include "tt.hpp"
#include "see.hpp"
#include "crash.hpp"
#include "eval.hpp"
#include "syzygy.hpp"
#include "sparams.hpp"
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

// Fathom WDL constants (from tbprobe.h, mirrored here for clarity)
#ifndef TB_LOSS
#define TB_LOSS         0
#define TB_BLESSED_LOSS 1
#define TB_DRAW         2
#define TB_CURSED_WIN   3
#define TB_WIN          4
#define TB_RESULT_FAILED 0xFFFFFFFFu
#endif
#ifndef TB_GET_WDL
#define TB_GET_WDL(_res)  ((_res) & 0x0Fu)
#define TB_GET_FROM(_res) (((_res) >> 10) & 0x3Fu)
#define TB_GET_TO(_res)   (((_res) >>  4) & 0x3Fu)
#define TB_GET_PROMOTES(_res) (((_res) >> 16) & 0x7u)
#endif

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
                LMR[d][m] = (int)(sp::lmr_base / 100.0
                                  + std::log((double)d) * std::log((double)m)
                                        / (sp::lmr_div / 100.0));
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
static constexpr int SCORE_COUNTER     =    700'000; // countermove bonus band
static constexpr int SCORE_BAD_CAPTURE = -1'000'000; // SEE<0 capture base; tried after quiets

// ── Ordering-component toggles (bisection; default all on) ────────────────────
#ifndef ORD_CAPTHIST
#define ORD_CAPTHIST 1
#endif
#ifndef ORD_CONTHIST
#define ORD_CONTHIST 1
#endif
#ifndef ORD_COUNTER
#define ORD_COUNTER 1
#endif

// ── Singular Extensions (default ON — confirmed +28.7 Elo at 30+0.3 slow TC) ──
// A TT move is "singular" when a reduced-depth search excluding it fails below
// (ttScore - SE_MARGIN); such a move gets a +1 extension. Fast-TC SPRT
// UNDER-measures this (it needs search depth to pay off), so it is validated at
// the competition's slow TC. Build with -USE_SINGULAR removed only for A/B.
#ifndef SE_SINGULAR
#define SE_SINGULAR 1
#endif
#ifndef SE_MARGIN
#define SE_MARGIN 64
#endif

// ── Double / negative singular extensions (DSE — OFF until slow-TC SPRT) ──────
// On top of the +1 singular extension: a DRAMATICALLY singular TT move (excluded
// search fails below seBeta - DSE_MARGIN) gets +2 ("double extension"); a TT move
// that is NOT singular yet would itself fail high (ttScore >= beta) gets a -1
// "negative extension". DSE_CAP bounds consecutive double extensions along one
// line so a pathological position can't explode the tree (time-loss safety).
// Depth-dependent → validate at slow TC (like singular). Enable with -DSE_DSE=1.
#ifndef SE_DSE
#define SE_DSE 1   // double + negative singular extensions (depth-scaled margin): +19.7 Elo @60+0.6, LOS 99%
#endif
#ifndef DSE_MARGIN
#define DSE_MARGIN 16   // per-DEPTH coeff: double-ext only if seScore < seBeta - 16*depth (was flat 20 = far too eager)
#endif
#ifndef DSE_CAP
#define DSE_CAP 4       // tighter consecutive-double cap (was 6) to bound tree inflation
#endif

// ── Aggressive LMR (OFF until validated vs Tucano at slow TC) ─────────────────
// Targets the MEASURED weakness: KINg's effective branching factor (~2.5) is too
// high vs Tucano (~2.0) → ~4-6 plies shallower at equal time. This reduces late
// quiets EARLIER (depth>=3, moveCount>=4 vs 4/6) and by ONE more ply → narrower
// tree → deeper search. Validate vs Tucano at slow TC (gauntlet), then SPSA-refine.
#ifndef AGGR_LMR
#define AGGR_LMR 1   // default ON: part of the validated baseline (ship with this on)
#endif

// ── cutNode LMR (OFF until SPRT) ──────────────────────────────────────────────
// Thread the expected node type (cut node = a node expected to fail high for the
// opponent, i.e. produce a beta cutoff) through the search and reduce one extra
// ply on late quiets at cut nodes. Cut nodes need only refute, so a deeper search
// rarely changes the bound → safe to search shallower → narrower tree (lower EBF).
// NPS-neutral (one bool threaded). Enable -DCUTNODE=1. The bool is always plumbed
// (so the tree shape is unchanged when OFF); only the `r += cutNode` is gated.
#ifndef CUTNODE
#define CUTNODE 2   // default ON (=2): +11.6 Elo stacked with CORRHIST (LOS 98.3%)
#endif

// ── Aggressive reduced-depth pruning (OFF until SPRT) ─────────────────────────
// Structural EBF lever (NOT a param — those were SPRT-neutral). Gates shallow
// futility on the LMR-REDUCED depth (lmrDepth = depth - r): a late quiet that
// will be searched shallow should be pruned on that shallow depth, not the raw
// one → more late quiets pruned → narrower tree → deeper search. Also uncaps LMP
// (the moveCount≈depth² threshold self-limits, so high-depth LMP rarely fires but
// helps in wide middlegames). AGGR_PRUNE=0 ⇒ bit-identical to current. -DAGGR_PRUNE=1.
#ifndef AGGR_PRUNE
#define AGGR_PRUNE 0
#endif

// ── Best-thread voting for Lazy SMP (OFF until SPRT) ──────────────────────────
// Lazy SMP currently reports thread 0's move; helper threads only enrich the TT.
// Helpers reach different depths/scores, and thread 0 is not always best. With
// SMP_VOTE the final move is chosen across ALL threads by (deepest, then best
// score) — the data (bestDepths/bestScores) is already collected. ONLY affects
// Threads>1 (identical at Threads=1). Test at Threads=4/8. -DSMP_VOTE=1.
#ifndef SMP_VOTE
#define SMP_VOTE 1   // default ON: validated +5.8 @4thr (best-thread voting)
#endif

// ── Lazy SMP thread diversity (OFF until SPRT) ────────────────────────────────
// Today all helper threads run an IDENTICAL search → 8 threads reach the same
// depth as 1 (measured). With SMP_DIV, helpers skip a staggered subset of
// iterative-deepening depths so they work on DIFFERENT depths than the main
// thread → richer shared TT → main searches deeper. Only useful WITH SMP_VOTE
// (so a helper's better result is actually picked). -DSMP_DIV=1 (implies VOTE).
#ifndef SMP_DIV
#define SMP_DIV 1    // default ON: validated (helper depth diversity, with SMP_VOTE)
#endif

// ── SMP helper aspiration widening (OFF until SPRT) ───────────────────────────
// Helper threads use a 2x-wider aspiration window so they do NOT fail high/low in
// lockstep with the main thread → more varied re-searches → richer shared TT →
// main thread deepens. Bit-identical at Threads=1. -DSMP_ASPWIDE=1.
#ifndef SMP_ASPWIDE
#define SMP_ASPWIDE 0
#endif

// ── Lazy SMP per-thread LMR jitter ────────────────────────────────────────────
// SMP_DIV desyncs WHICH iterative-deepening depth a helper is on, but every thread
// searches a given subtree with IDENTICAL LMR → identical subtrees → redundant TT
// stores ("8 threads ~ 1"). With SMP_LMRJIT, helpers nudge LMR by +-1 ply (by id
// parity) so they expand a DIFFERENT late-move frontier than the main thread →
// genuinely different trees → non-redundant TT. Bit-identical at Threads=1.
#ifndef SMP_LMRJIT
#define SMP_LMRJIT 0
#endif

// ── History pruning (OFF until SPRT) ──────────────────────────────────────────
// Skip a late quiet whose combined history is very negative at shallow reduced
// depth — a SOTA pruning term KINg lacks. Gated lmrDepth∈[1,3]. -DHIST_PRUNE=1.
#ifndef HIST_PRUNE
#define HIST_PRUNE 1 // default ON: validated +24.9 Elo (biggest single search lever)
#endif

// ── Multicut on singular fail-high (OFF until SPRT) ───────────────────────────
// If the singular VERIFICATION search (which excludes the TT move) ALSO beats
// beta, then 2+ distinct moves beat beta → this is a cut node → return the proven
// bound (a multicut). Crash-safe: !aborted guard, returns a real lower bound that
// is >= beta. -DSE_MULTICUT=1.
#ifndef SE_MULTICUT
#define SE_MULTICUT 0
#endif

// ── Cross-type history malus (OFF until SPRT) ─────────────────────────────────
// On a quiet beta-cutoff, also penalise the capture-history of the captures that
// were searched first and failed to cut (they were a wasted try). Standard SOTA;
// improves capture ordering. -DXMALUS=1.
#ifndef XMALUS
#define XMALUS 0
#endif

// ── Correction history (OFF until SPRT) ──────────────────────────────────────
// A pawn(+king)-keyed table recording (searchScore − staticEval); it shifts the
// static eval of future similar pawn structures toward what search actually found
// → sharper RFP / NMP / futility / LMR decisions. NPS-neutral (one table lookup),
// grows at slow TC. Bounded ±64cp so it can never destabilise. Enable -DCORRHIST=1.
#ifndef CORRHIST
#define CORRHIST 1  // default ON: +11.6 Elo stacked with CUTNODE=2 (LOS 98.3%)
#endif
#define CORR_SIZE  16384   // entries per side (mask = CORR_SIZE-1)
#define CORR_GRAIN 256     // entry stored as correction_cp * GRAIN
#define CORR_SCALE 256     // EMA denominator (update step = weight/CORR_SCALE)
#define CORR_MAX   (CORR_GRAIN * 64)  // clamp: ±64cp max correction
#define CORR_TOTAL_MAX 96             // clamp on SUMMED multi-table correction (cp): guards over-correction

// ── ProbCut (default ON — confirmed +12.2 Elo at 30+0.3 slow TC) ─────────────
// Also depth-dependent: neutral at fast TC, positive at the competition TC.
#ifndef SE_PROBCUT
#define SE_PROBCUT 1
#endif

// ── History (gravity-updated, capped) ─────────────────────────────────────────
// Butterfly + capture + continuation histories all use the same int16 gravity
// scheme: a bonus pulls the score toward ±HIST_CAP without ever overflowing, so
// the table self-normalises instead of saturating like the old `+= depth*depth`.
static constexpr int HIST_CAP = 16384;

// Sentinel stored in the TT `eval` field when no static eval is meaningful
// (in-check nodes). Static evals never reach INT16_MIN, so it is unambiguous.
static constexpr int16_t TT_EVAL_NONE = INT16_MIN;

static inline int hist_bonus(int depth) {
    return std::min(sp::hist_max, sp::hist_quad * depth * depth + sp::hist_lin * depth - sp::hist_lin);
}
static inline void hist_update(int16_t& h, int bonus) {
    h += (int16_t)(bonus - (int)h * std::abs(bonus) / HIST_CAP);
}

// ── Per-ply search stack ───────────────────────────────────────────────────────
// Records, for each ply, the move chosen at that node (and the piece/target it
// moved) plus the static eval. Children read the PARENT entry (ss[ply-1]) for
// countermove / continuation-history indexing and the grandparent (ss[ply-2])
// static eval for the "improving" heuristic.
struct Stack {
    Move      currentMove = 0;   // move being searched at this ply
    PieceType movedPiece  = PAWN;// piece type that made currentMove
    Square    toSq        = A1;
    int       staticEval  = 0;
    Move      excluded    = 0;   // singular-extension excluded move (batch E)
    int       doubleExt   = 0;   // consecutive double singular extensions on this line (DSE cap)
};

// ── Eval cache (persistent per-thread, lockless) ─────────────────────────────
// A dedicated position->eval memo: a hit skips the NNUE forward pass. Distinct
// from the TT `eval` field (which qsearch leaves / re-searches constantly evict).
// The cached value is the RAW pre-correction static eval — a pure function of the
// position key — so entries stay valid forever and never need clearing (a
// different position simply fails the full 64-bit key check). Behaviour-preserving:
// the returned eval equals a recompute, so the search tree is unchanged.
#ifndef EVAL_CACHE
#define EVAL_CACHE 1
#endif
#ifndef TM_INSTAB
#define TM_INSTAB 1   // two-sided instability TM: extend when the best move is unsettled
#endif
#ifndef CHECK_SEE
#define CHECK_SEE 1   // SEE-gate the check extension: skip extending material-losing checks (+12.7 Elo @60+0.6)
#endif
#if EVAL_CACHE
struct EvCacheEntry { uint64_t key; int32_t eval; };
static constexpr int EVCACHE_BITS = 16;            // 64K entries ≈ 1 MB/thread
static constexpr int EVCACHE_SIZE = 1 << EVCACHE_BITS;
static constexpr int EVCACHE_MASK = EVCACHE_SIZE - 1;
// The cache itself is thread_local (declared inside eval_cached) — correct for both
// the per-search Lazy-SMP threads and concurrent datagen workers, with no shared
// global state and therefore no resize/torn-read data race.
#endif

// ── Searcher ──────────────────────────────────────────────────────────────────
struct Searcher {
    std::atomic<bool>* stop;
    int64_t            hard_ms;
    std::chrono::steady_clock::time_point start;
    uint64_t           nodes;
    bool               aborted;
    int                id = 0;   // Lazy SMP thread index (0 = main); drives diversity

    // ── Move-ordering tables ───────────────────────────────────────────────
    Move    killers[MAX_PLY][2];          // 2 killer (quiet) moves per ply
    int16_t history[2][64][64];           // [stm][from][to] butterfly history (quiets)
    int16_t captHist[2][6][64][6];        // [stm][attacker][to][victim] capture history
    Move    counterMove[12][64];          // [prevPiece12][prevTo] -> refutation reply
    // Continuation history: [prevPiece12][prevTo][curPiece12][curTo]. Indexed by
    // the parent (1-ply) and grandparent (2-ply) moves; ~1.2 MB/thread.
    int16_t contHist[12][64][12][64];
#ifdef SE_CONTHIST2
    // 4-ply (great-grandparent) continuation history; same layout, ~1.2 MB/thread.
    int16_t contHist2[12][64][12][64];
#endif
#if CORRHIST
    int16_t pawnCorr[2][CORR_SIZE];  // [stm][pawn_key & (CORR_SIZE-1)] static-eval correction
#if MULTICORR
    int16_t minorCorr[2][CORR_SIZE]; // [stm][minor_key & (CORR_SIZE-1)] knight+bishop correction
    int16_t majorCorr[2][CORR_SIZE]; // [stm][major_key & (CORR_SIZE-1)] rook+queen   correction
#endif
#endif
    Stack   ss[MAX_PLY + 4];

    bool times_up() {
        // Poll the clock every 1024 nodes normally, but every 256 once the budget
        // is small (low-clock panic, hard_ms can fall to ~50 ms). At very low time
        // on a busy/contended host a single node window can stall, so polling 4x
        // more often bounds the worst-case wall-time overshoot — crash/TIMEOUT = a
        // lost game. The extra clock reads are a negligible fraction of NPS, and at
        // fixed depth (hard_ms effectively infinite) the mask stays 1023, so a
        // depth-limited search is node-identical to before.
        const uint64_t mask = (hard_ms < 200) ? 255u : 1023u;
        if ((nodes & mask) == 0) {
            if (stop->load()) return true;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
            if (ms >= hard_ms) return true;
        }
        return false;
    }

    // Static eval via the per-thread cache: a hit skips the NNUE forward pass.
    // Returns the RAW (pre-correction) static eval, bit-identical to evaluate(pos).
    int eval_cached(Position& pos) {
#if EVAL_CACHE
        // thread_local: each OS thread (Lazy-SMP worker OR concurrent datagen worker)
        // owns its cache — no shared global, so no resize/torn-read race. Allocated
        // (zeroed) on first use per thread; the full-key check makes stale hits safe.
        static thread_local std::vector<EvCacheEntry> tl;
        if (tl.empty()) tl.assign(EVCACHE_SIZE, EvCacheEntry{0, 0});
        const uint64_t k = pos.key();
        EvCacheEntry& e = tl[k & EVCACHE_MASK];
        if (e.key == k) return e.eval;
        const int v = evaluate(pos);
        e.key = k; e.eval = (int32_t)v;
        return v;
#else
        return evaluate(pos);
#endif
    }

    // ── Quiescence search ──────────────────────────────────────────────────
    // Extends the leaf search through "noisy" moves (captures/promotions, and
    // all evasions while in check) so the static eval is only taken in a quiet
    // position. This removes the horizon effect inside capture sequences.
    int qsearch(Position& pos, int alpha, int beta, int ply,
                bool searchChecks = true) {
        if (aborted || times_up()) { aborted = true; return 0; }
        ++nodes;

        if (ply >= MAX_PLY) return evaluate(pos);

        const int  alphaOrig = alpha;
        const bool isPV       = (beta - alpha) > 1;

        // ── TT probe ─────────────────────────────────────────────────────────
        // Cheap cutoff + a hash move + a cached static eval, on the largest node
        // population in the tree. Cutoff only on non-PV nodes.
        TTEntry tte;
        const bool ttHit = tt.probe(pos.key(), tte);
        const Move ttMove = ttHit ? tte.move : 0;
        if (ttHit && !isPV) {
            int s = fromTT(tte.score, ply);
            Bound b = Bound(tte.genBound & 3);
            if (b == BOUND_EXACT
                || (b == BOUND_LOWER && s >= beta)
                || (b == BOUND_UPPER && s <= alpha))
                return s;
        }

        const bool inCheck = pos.in_check(pos.side_to_move());

        int standPat = 0;
        int best;
        if (inCheck) {
            // Cannot "stand pat" while in check: we must answer the check, so
            // every legal evasion is searched (not just captures).
            best = -INF;
        } else {
            // Reuse the TT's cached static eval when present (saves an evaluate()).
            standPat = (ttHit && tte.eval != TT_EVAL_NONE) ? (int)tte.eval : eval_cached(pos);
#ifdef SE_FIFTYSCALE
            // Scale the stand-pat score toward 0 as the 50-move clock nears 100.
            // The draw check fires at exactly 100, so the factor is in [0,1].
            standPat = standPat * (100 - pos.halfmove_clock()) / 100;
#endif
            if (standPat >= beta) {
                tt.store(pos.key(), ttMove, toTT(standPat, ply), (int16_t)standPat, 0, BOUND_LOWER);
                return standPat; // fail-high: opponent won't allow this
            }
            if (standPat > alpha) alpha = standPat;
            best = standPat;
        }

        MoveList ml;
        // Not in check, qsearch only searches captures/promotions (quiets are
        // skipped below), so generate just the NOISY moves — this avoids producing
        // the whole quiet list and shrinks the O(n^2) ordering sort from ~all moves
        // to ~the few captures (big win in open/endgame positions). In check we need
        // every evasion; with SE_QSCHECK we also need quiets for quiet-check search.
#if defined(SE_QSCHECK)
        generate_pseudo(pos, ml);
#else
        if (inCheck) generate_pseudo(pos, ml);
        else         generate_captures(pos, ml);
#endif

#if defined(QS_VERIFY) && !defined(SE_QSCHECK)
        // Correctness gate: generate_captures MUST equal the noisy subset of
        // generate_pseudo, in the same order (so qsearch ordering is unchanged).
        if (!inCheck) {
            MoveList full; generate_pseudo(pos, full);
            MoveList noisy;
            for (int i = 0; i < full.size; ++i) {
                Move mm = full.moves[i];
                bool ep    = type_of(mm) == EN_PASSANT;
                bool cap   = ep || pos.piece_on(to_sq(mm)) != NO_PIECE;
                bool promo = type_of(mm) == PROMO;
                if (cap || promo) noisy.add(mm);
            }
            if (noisy.size != ml.size) std::abort();
            for (int i = 0; i < ml.size; ++i)
                if (ml.moves[i] != noisy.moves[i]) std::abort();
        }
#endif

        // Precompute MVV-LVA keys for ordering; the TT move (if any) sorts first.
        int keys[256];
        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            if (ttMove != 0 && m == ttMove) { keys[i] = 1 << 24; continue; }
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

        Move bestMove = 0;
        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            const bool isEp      = (type_of(m) == EN_PASSANT);
            const Piece capPiece = pos.piece_on(to_sq(m));
            const bool isCapture = isEp || (capPiece != NO_PIECE);
            const bool isPromo   = (type_of(m) == PROMO);

            // When not in check, only consider captures and promotions — plus
            // (SE_QSCHECK) quiet moves that give check with SEE >= 0, searched
            // one extra ply deep (searchChecks=false in the child prevents
            // recursive check explosion).
            if (!inCheck && !isCapture && !isPromo) {
#ifdef SE_QSCHECK
                if (!searchChecks) continue;
                if (see(pos, m) < 0) continue;
                StateInfo stChk;
                pos.do_move(m, stChk);
                if (pos.in_check(Color(!pos.side_to_move()))) { pos.undo_move(m); continue; }
                if (!pos.in_check(pos.side_to_move())) { pos.undo_move(m); continue; }
                int score = -qsearch(pos, -beta, -alpha, ply + 1, false);
                pos.undo_move(m);
                if (aborted) return 0;
                if (score > best)  { best = score; bestMove = m; }
                if (score > alpha) alpha = score;
                if (alpha >= beta) break;
                continue;
#else
                continue;
#endif
            }

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
            tt.prefetch(pos.key());   // overlap child TT-probe latency
            int score = -qsearch(pos, -beta, -alpha, ply + 1, searchChecks);
            pos.undo_move(m);
            if (aborted) return 0;
            if (score > best)  { best = score; bestMove = m; }
            if (score > alpha) alpha = score;
            if (alpha >= beta) break; // beta cutoff
        }

        // If in check and no legal move was found, it is checkmate.
        if (inCheck && best == -INF) return -(MATE - ply);

        // Store the qsearch result (depth 0) so siblings/re-visits can reuse it.
        Bound bnd = (best >= beta)      ? BOUND_LOWER
                  : (best > alphaOrig)  ? BOUND_EXACT
                  :                       BOUND_UPPER;
        tt.store(pos.key(), bestMove, toTT(best, ply),
                 inCheck ? TT_EVAL_NONE : (int16_t)standPat, 0, bnd);

        return best;
    }

    // ── PVS negamax ───────────────────────────────────────────────────────────
    // Principal Variation Search: the first move is searched with the full
    // window; subsequent moves use a zero-window (scout) search. Only if the
    // scout exceeds alpha AND is inside beta is a costly re-search done.
    int negamax(Position& pos, int depth, int alpha, int beta, int ply, bool cutNode) {
        if (aborted || times_up()) { aborted = true; return 0; }
        ++nodes;

        // Hard ply cap: bounds every ply-indexed table (killers, search stack).
        // Reachable only via long checking/extension chains; returning the static
        // eval here is a safe emergency horizon.
        if (ply >= MAX_PLY) return evaluate(pos);

        const bool isPV = (beta - alpha) > 1; // wide window ⇒ PV node

        // ── Mate-distance pruning ──────────────────────────────────────────
        // A mate from this node is at best (MATE - ply); a loss at worst
        // -(MATE - ply). Tighten the window to those bounds — if it collapses
        // there is nothing left to search.
        if (ply > 0) {
            alpha = std::max(alpha, -(MATE - ply));
            beta  = std::min(beta,   MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }

        // ── Draw detection (repetition / 50-move / insufficient material) ─────
        // The search must recognise draws so it neither walks into a lost-on-the
        // board "draw" it thinks is fine nor misses forcing a draw when worse.
        // Repetition uses the first in-line repeat (the side to move can force it).
        // The 50-move rule is suppressed while in check, since a position can be
        // checkmate exactly on the 100th half-move (checkmate beats the draw claim);
        // in_check() is only evaluated in that rare case (short-circuit).
        if (ply > 0
            && (pos.is_repetition()
                || pos.insufficient_material()
                || (pos.halfmove_clock() >= 100
                    && !pos.in_check(pos.side_to_move()))))
            return 0;

        const int alphaOrig = alpha;

        // ── TT probe ───────────────────────────────────────────────────────
        Move  ttMove  = 0;
        int   ttScore = 0;            // ply-adjusted TT score (used by singular ext)
        Bound ttBound = BOUND_NONE;
        TTEntry tte;
        bool ttHit = tt.probe(pos.key(), tte);
        if (ttHit) {
            ttMove  = tte.move;
            ttScore = fromTT(tte.score, ply);
            ttBound = Bound(tte.genBound & 3);
            // Cutoff only on non-PV nodes with a deep-enough entry whose bound
            // is compatible with the window. (Keeps the PV exact/intact.) The
            // ss[ply].excluded == 0 guard skips the cutoff while verifying a
            // singular candidate, so the verification actually searches.
            if (!isPV && tte.depth >= depth && ss[ply].excluded == 0) {
                if (ttBound == BOUND_EXACT) return ttScore;
                if (ttBound == BOUND_LOWER && ttScore >= beta)  return ttScore;
                if (ttBound == BOUND_UPPER && ttScore <= alpha) return ttScore;
            }
        }

        if (depth <= 0) return qsearch(pos, alpha, beta, ply);

        const bool inCheck = pos.in_check(pos.side_to_move());
        const Color stm    = pos.side_to_move();

        // ── Syzygy WDL probe (non-root, in-search) ───────────────────────────
        // Conditions: TBs enabled, no castling rights, piece count ≤ TB_LARGEST,
        // depth >= 1 (already guaranteed by the depth<=0 guard above), ply > 0
        // so we are NOT at the root (root probing is done separately in think).
        // tb_probe_wdl already handles rule50 internally (returns FAILED if != 0
        // when castling==0; we still pass it and it's a correctness filter).
        if (ply > 0 && syzygy::enabled()
                && pos.castling_rights() == 0
                && static_cast<unsigned>(popcount(pos.occupied())) <= syzygy::largest()) {
            unsigned wdl = syzygy::probe_wdl(pos);
            if (wdl != TB_RESULT_FAILED) {
                // Convert WDL to a search score.
                // TB_WIN / TB_CURSED_WIN → winning score just below true mate
                // TB_LOSS / TB_BLESSED_LOSS → losing score just above true mate-loss
                // TB_DRAW → 0
                // We use MATE - MAX_PLY - 1 so TB scores are clearly below/above
                // normal mate scores while still counting as "forced" results.
                const int tb_base = MATE - MAX_PLY - 1;
                int score;
                if      (wdl == TB_WIN)          score =  tb_base - ply;
                else if (wdl == TB_CURSED_WIN)   score =  1;   // cursed: draws with perfect play
                else if (wdl == TB_BLESSED_LOSS) score = -1;   // blessed: loss, but 50-move draw
                else if (wdl == TB_LOSS)         score = -(tb_base - ply);
                else                             score =  0;   // TB_DRAW

                // We probe with rule50=0 (so the WDL is always available), so
                // scale decisive scores by 50-move proximity: a TB win/loss near
                // the 50-move limit is valued cautiously (approaches a draw).
                if (wdl == TB_WIN || wdl == TB_LOSS)
                    score = score * (100 - pos.halfmove_clock()) / 100;

                // Update TT with the TB result so siblings can use it.
                Bound tb_bound = (score >= beta) ? BOUND_LOWER
                               : (score <= alpha) ? BOUND_UPPER
                               : BOUND_EXACT;
                tt.store(pos.key(), 0, toTT(score, ply), TT_EVAL_NONE, (uint8_t)depth, tb_bound);

                // Cutoff (or return exact) based on window.
                if (tb_bound == BOUND_EXACT
                    || (tb_bound == BOUND_LOWER && score >= beta)
                    || (tb_bound == BOUND_UPPER && score <= alpha))
                    return score;

                // Soft bound: adjust alpha/beta for more accurate search.
                if (score > alpha) alpha = score;
                if (score < beta)  beta  = score;   // keep beta if we have an upper bound
            }
        }

        // ── Static evaluation (TT-cached, TT-refined) ─────────────────────────
        // Not meaningful while in check. Otherwise reuse the TT's cached eval
        // (saves an evaluate()), and refine the value used for PRUNING with the
        // TT score when its bound proves the position is better/worse than eval.
        int staticEval, evalForPruning;
        [[maybe_unused]] int rawEval = -INF;   // pre-correction static eval (for corrhist update)
        if (inCheck) {
            staticEval = evalForPruning = -INF;
        } else {
            staticEval = (ttHit && tte.eval != TT_EVAL_NONE) ? (int)tte.eval : eval_cached(pos);
#if CORRHIST
            rawEval = staticEval;
#if MULTICORR
            {
                const int cstm = pos.side_to_move();
                int corr = pawnCorr [cstm][pos.pawn_key()  & (CORR_SIZE - 1)]
                         + minorCorr[cstm][pos.minor_key() & (CORR_SIZE - 1)]
                         + majorCorr[cstm][pos.major_key() & (CORR_SIZE - 1)];
                corr /= CORR_GRAIN;                       // three same-grain ±64cp terms
                staticEval += std::clamp(corr, -CORR_TOTAL_MAX, CORR_TOTAL_MAX);
            }
#else
            staticEval += pawnCorr[pos.side_to_move()][pos.pawn_key() & (CORR_SIZE - 1)] / CORR_GRAIN;
#endif
#endif
#ifdef SE_FIFTYSCALE
            // Scale staticEval toward 0 near the 50-move draw threshold
            // (halfmove_clock() is in [0,99] here; the ==100 draw was caught above).
            staticEval = staticEval * (100 - pos.halfmove_clock()) / 100;
#endif
            evalForPruning = staticEval;
            if (ttHit) {
                int ts = fromTT(tte.score, ply);
                Bound tb = Bound(tte.genBound & 3);
                if (tb == BOUND_EXACT
                    || (tb == BOUND_LOWER && ts > evalForPruning)
                    || (tb == BOUND_UPPER && ts < evalForPruning))
                    evalForPruning = ts;
            }
        }
        ss[ply].staticEval = staticEval;
#if SE_DSE
        // Default: children inherit this node's double-extension count. The move
        // loop overrides per move (adds 1 when it double-extends); NMP / ProbCut
        // children (searched before the move loop) use this inherited value.
        ss[ply + 1].doubleExt = ss[ply].doubleExt;
#endif

        // ── "Improving" trend ─────────────────────────────────────────────────
        // Is our static eval higher than two plies ago (our previous turn)? If so
        // the position is improving and we can prune a touch more aggressively.
        const bool improving = !inCheck && ply >= 2
                             && ss[ply - 2].staticEval != -INF
                             && staticEval > ss[ply - 2].staticEval;

        // ── Reverse Futility Pruning (static null move) ───────────────────────
        // If the static eval already beats beta by a large margin, prune.
        if (!isPV && !inCheck && depth <= 8
            && beta < MATE - MAX_PLY && beta > -(MATE - MAX_PLY)
            && staticEval - sp::rfp_margin * depth >= beta)
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
            && evalForPruning >= beta   // only when the static eval already beats beta
            && beta < MATE - MAX_PLY && beta > -(MATE - MAX_PLY)) {
            int R = sp::nmp_base + depth / sp::nmp_div;
#ifdef SE_NMPEVAL
            // If our eval already beats beta by a large margin the position is
            // even more likely to be a safe prune — reduce a touch more. Bonus
            // in [0,3]; evalForPruning >= beta is implied at a fail-high node.
            R += std::max(0, std::min(3, (evalForPruning - beta) / 200));
#endif
            StateInfo nullSt;
            // Mark this ply as "no move" so the child doesn't index countermove /
            // continuation history off a stale sibling move.
            ss[ply].currentMove = 0;
            pos.do_null_move(nullSt);
            int nullScore = -negamax(pos, std::max(0, depth - 1 - R), -beta, -beta + 1, ply + 1, !cutNode);
            pos.undo_null_move();
            if (aborted) return 0;
            if (nullScore >= beta) {
                if (nullScore >= MATE - MAX_PLY) nullScore = beta; // don't trust an unproven mate
                return nullScore;                                  // fail-soft
            }
        }

#ifdef SE_PROBCUT
        // ── ProbCut ───────────────────────────────────────────────────────────
        // At non-PV, non-check nodes with depth >= 5, search a small set of good
        // captures at reduced depth to get a cheap fail-high. Only fire when beta
        // is far from a mate score so pbBeta stays in a sane range.
        if (!isPV && !inCheck && depth >= 5
                && beta < MATE - MAX_PLY && beta > -(MATE - MAX_PLY)) {
            const int pbBeta   = beta + 200;
            const int pbSeeMin = pbBeta - staticEval;
            MoveList pcml;
            generate_captures(pos, pcml);
            for (int pi = 0; pi < pcml.size; ++pi) {
                Move pm = pcml.moves[pi];
                const bool  pcIsEp  = (type_of(pm) == EN_PASSANT);
                const Piece pcCap   = pos.piece_on(to_sq(pm));
                const bool  pcIsCap = pcIsEp || (pcCap != NO_PIECE);
                if (!pcIsCap) continue;
                if (type_of(pm) == PROMO) continue;        // skip promo-captures
                if (see(pos, pm) < pbSeeMin) continue;     // only clearly-winning captures
                StateInfo pcSt;
                pos.do_move(pm, pcSt);
                if (pos.in_check(Color(!pos.side_to_move()))) { pos.undo_move(pm); continue; }
                const int pcDepth = std::max(0, depth - 4);
                int pcScore = -negamax(pos, pcDepth, -pbBeta, -pbBeta + 1, ply + 1, !cutNode);
                pos.undo_move(pm);
                if (aborted) return 0;
                if (pcScore >= pbBeta) return pbBeta;
            }
        }
#endif // SE_PROBCUT

        // ── Internal Iterative Reduction (IIR) ───────────────────────────────
        // If we have no TT move at this node (no prior search result to guide
        // ordering), do a shallower search by reducing depth by 1. This cheap
        // sacrifice pays off on the next visit when the TT move is available and
        // the full-depth search is better ordered.
        // Conditions: no TT move, depth >= 4, not in check.
        if (!ttMove && depth >= 4 && !inCheck) depth -= 1;

        MoveList ml;
        generate_pseudo(pos, ml);

        // ── Parent / grandparent move context (for countermove & cont-history) ─
        // The parent move (ply-1) was made by the opponent (!stm); the
        // grandparent (ply-2) by us (stm). pPiece12/gPiece12 are Piece (0..11)
        // indices; -1 means "no such move" (root / after a null move).
        int  pPiece12 = -1, pTo = 0, gPiece12 = -1, gTo = 0;
        Move counter  = 0;
        if (ply >= 1 && ss[ply - 1].currentMove != 0) {
            pPiece12 = make_piece(Color(!stm), ss[ply - 1].movedPiece);
            pTo      = ss[ply - 1].toSq;
            counter  = counterMove[pPiece12][pTo];
        }
        if (ply >= 2 && ss[ply - 2].currentMove != 0) {
            gPiece12 = make_piece(stm, ss[ply - 2].movedPiece);
            gTo      = ss[ply - 2].toSq;
        }
#ifdef SE_CONTHIST2
        int  ggPiece12 = -1, ggTo = 0;
        if (ply >= 4 && ss[ply - 4].currentMove != 0) {
            ggPiece12 = make_piece(stm, ss[ply - 4].movedPiece);
            ggTo      = ss[ply - 4].toSq;
        }
#endif

        // ── Move scoring (for ordering) ───────────────────────────────────────
        // Assign a score to each pseudo-legal move; we iterate in descending
        // score order using selection sort (lazy, one pass per iteration).
        //
        // Score buckets (non-overlapping):
        //   TT move:        2,000,000
        //   Good captures:   1,000,000 + MVV-LVA + capture-history
        //   Killer 1/2:        900,000 / 800,000
        //   Countermove:       700,000
        //   Quiet:           butterfly + continuation history
        //   Bad captures:   -1,000,000 + MVV-LVA + capture-history
        int scores[256];
        int seeVal[256];                       // SEE cached at ordering, reused by the prune
        constexpr int SEE_NONE = -2000000000;  // sentinel: SEE not computed for this move
        for (int i = 0; i < ml.size; ++i) {
            Move m = ml.moves[i];
            seeVal[i] = SEE_NONE;
            if (ttMove != 0 && m == ttMove) {
                scores[i] = SCORE_TT_MOVE;
            } else {
                const bool isEp      = (type_of(m) == EN_PASSANT);
                const Piece capPiece = pos.piece_on(to_sq(m));
                const bool isCapture = isEp || (capPiece != NO_PIECE);
                const bool isPromo   = (type_of(m) == PROMO);
                const Square from    = from_sq(m), to = to_sq(m);
                const PieceType mover = piece_type(pos.piece_on(from));

                if (isCapture || isPromo) {
                    // MVV-LVA: most-valuable-victim / least-valuable-attacker
                    PieceType victim   = isEp ? PAWN
                                       : isCapture ? piece_type(capPiece)
                                       : PAWN; // quiet promo: no real victim
                    int mvvlva = value_of(victim) * 16 - value_of(mover);
                    if (isPromo) mvvlva += value_of(QUEEN);
                    int ch = ORD_CAPTHIST ? (int)captHist[stm][mover][to][victim] : 0;
                    // Split captures by SEE: good/equal captures keep the high
                    // band; captures that lose material (SEE<0) drop below all
                    // quiets/history so they are tried last.  Promotions are
                    // always treated as good (kept high).
                    if (!isPromo && isCapture) {
                        // Compute SEE ONCE here (for the good/bad split) and cache
                        // it; the shallow-SEE prune below reuses it (same position),
                        // so a searched capture pays SEE once instead of twice.
                        int sv = see(pos, m);
                        seeVal[i] = sv;
                        scores[i] = (sv < 0 ? SCORE_BAD_CAPTURE : SCORE_CAPTURE) + mvvlva + ch;
                    } else {
                        scores[i] = SCORE_CAPTURE + mvvlva + ch;
                    }
                } else {
                    // Quiet move: killers, then countermove, then histories.
                    if (ply < MAX_PLY && m == killers[ply][0]) {
                        scores[i] = SCORE_KILLER1;
                    } else if (ply < MAX_PLY && m == killers[ply][1]) {
                        scores[i] = SCORE_KILLER2;
                    } else if (ORD_COUNTER && counter != 0 && m == counter) {
                        scores[i] = SCORE_COUNTER;
                    } else {
                        const int cp12 = make_piece(stm, mover);
                        int q = (int)history[stm][from][to];
                        if (ORD_CONTHIST && pPiece12 >= 0) {
                            q += (int)contHist[pPiece12][pTo][cp12][to];
                            if (gPiece12 >= 0) q += (int)contHist[gPiece12][gTo][cp12][to];
                        }
#ifdef SE_CONTHIST2
                        if (ggPiece12 >= 0) q += (int)contHist2[ggPiece12][ggTo][cp12][to];
#endif
                        scores[i] = q;
                    }
                }
            }
        }

        int  best      = -INF;
        Move bestMove  = 0;
        int  legal     = 0;
        int  moveCount = 0;  // post-legality counter (for LMR/LMP thresholds)
        bool firstMove = true;

        // Quiet/capture moves actually searched at this node — used to apply a
        // history MALUS to the moves that did NOT cause the cutoff.
        Move quietsTried[64]; int nQuiets = 0;
        Move capsTried[64];   int nCaps   = 0;

        for (int i = 0; i < ml.size; ++i) {
            // ── Selection sort: pick the highest-scored remaining move ─────
            int bi = i;
            for (int j = i + 1; j < ml.size; ++j)
                if (scores[j] > scores[bi]) bi = j;
            if (bi != i) {
                std::swap(ml.moves[i], ml.moves[bi]);
                std::swap(scores[i],   scores[bi]);
                std::swap(seeVal[i],   seeVal[bi]); // keep the SEE cache aligned with moves
            }

            Move m = ml.moves[i];

#ifdef SE_SINGULAR
            // Skip the excluded move while verifying a singular candidate.
            if (m == ss[ply].excluded) continue;
#endif

            // ── Classify move BEFORE do_move (board is still unmoved) ─────
            // isQuiet: not a capture, not en passant, not a promotion.
            const bool isEpPre      = (type_of(m) == EN_PASSANT);
            const Piece capPiecePre = pos.piece_on(to_sq(m));
            const bool isQuiet      = !(isEpPre || (capPiecePre != NO_PIECE)
                                                 || (type_of(m) == PROMO));

            // Reduced "lmr depth" for shallow pruning: a late quiet will be
            // searched at depth - r, so gate futility on that reduced depth.
            // Equals `depth` when AGGR_PRUNE is off ⇒ original behaviour.
            int lmrDepth = depth;
#if AGGR_PRUNE
            if (isQuiet && depth >= 3 && moveCount >= 4) {
                lmrDepth -= LMR[std::min(depth, 63)][std::min(moveCount, 63)];
                if (lmrDepth < 0) lmrDepth = 0;
            }
#endif

            // Mover piece type captured BEFORE do_move (board still unmoved),
            // for the search-stack / history indexing.
            const PieceType moverPt = piece_type(pos.piece_on(from_sq(m)));
            // Combined quiet-history (butterfly + 1/2-ply continuation), computed
            // BEFORE pruning so history-pruning AND the LMR reduction can use it.
            int quietHist = 0;
            if (isQuiet) {
                const int cp12 = make_piece(stm, moverPt);
                quietHist = (int)history[stm][from_sq(m)][to_sq(m)];
                if (pPiece12 >= 0) {
                    quietHist += (int)contHist[pPiece12][pTo][cp12][to_sq(m)];
                    if (gPiece12 >= 0) quietHist += (int)contHist[gPiece12][gTo][cp12][to_sq(m)];
                }
#ifdef SE_CONTHIST2
                if (ggPiece12 >= 0) quietHist += (int)contHist2[ggPiece12][ggTo][cp12][to_sq(m)];
#endif
            }

            // ── Late Move Pruning (LMP) ───────────────────────────────────
            // Skip very late quiet moves at low depth when not in check and
            // we already have a non-losing best score.  (moveCount is the
            // post-legality count so early moves always pass the threshold.)
            // Only prune at depth >= 3 to avoid hiding tactics at shallow nodes.
            // Prune later when improving (we can afford to look at more moves),
            // earlier when not — standard LMP "improving" modulation.
            if (!isPV && !inCheck && isQuiet
                    && depth >= 3
#if !AGGR_PRUNE
                    && depth <= 6
#endif
                    && moveCount >= (improving ? sp::lmp_imp + depth * depth : sp::lmp_noimp + depth * depth / 2)
                    && best > -(MATE - MAX_PLY)) {
                continue;
            }

            // ── Futility Pruning (frontier) ───────────────────────────────
            // At very shallow depths, if the (refined) eval plus a margin cannot
            // reach alpha, this quiet move almost certainly can't raise alpha.
            // Guard: only fire when the position isn't already losing to avoid
            // pruning in sharp positions where material eval underestimates
            // piece activity.
            if (!isPV && !inCheck && isQuiet && lmrDepth <= 6
                    && best > -(MATE - MAX_PLY)
                    && evalForPruning >= -150
                    && evalForPruning + sp::fut_base + sp::fut_mult * lmrDepth <= alpha) {
                continue;
            }

            // ── Shallow SEE pruning ───────────────────────────────────────
            // At low depth in a non-PV node, prune moves whose static exchange
            // is clearly bad: losing captures, or quiets that walk onto a
            // square where the piece is lost.  The margin grows with depth² so
            // we prune more conservatively as depth rises.  Guarded by
            // best > mate-loss so the first move (best == -INF) is never pruned.
            if (!isPV && !inCheck && depth <= 6
                    && best > -(MATE - MAX_PLY)) {
                // Reuse the SEE computed during move ordering (board unchanged
                // between scoring and here); compute on demand for non-captures.
                int sv = (seeVal[i] != SEE_NONE) ? seeVal[i] : see(pos, m);
                if (sv < -sp::see_margin * depth * depth) continue;
            }

#if HIST_PRUNE
            // ── History pruning ───────────────────────────────────────────
            // A late quiet with very negative history is unlikely to be best;
            // skip it at shallow reduced depth (placed after SEE so bad captures
            // are handled first). quietHist was computed above.
            if (!isPV && !inCheck && isQuiet && lmrDepth >= 1 && lmrDepth <= 3
                    && moveCount >= 4 && best > -(MATE - MAX_PLY)
                    && quietHist < -sp::hp_mult * lmrDepth * lmrDepth) {
                continue;
            }
#endif

#ifdef SE_SINGULAR
            // ── Singular Extension ────────────────────────────────────────
            // If the TT move's score is "singular" (a reduced search excluding
            // it fails below ttScore - SE_MARGIN), extend it by +1. Guards:
            // this is the TT move, not already verifying, deep enough node,
            // reliable TT entry (depth, non-UPPER, non-mate), ply headroom.
            // Crash-safe: seDepth=(depth-1)/2 < depth (no runaway); excluded is
            // cleared right after the call; the verification's TT cutoff/store
            // are skipped via the excluded==0 guards.
            int seExtension = 0;
            if (ttMove != 0 && m == ttMove
                    && ss[ply].excluded == 0
                    && depth >= 8
                    && ttHit && tte.depth >= depth - 3
                    && ttBound != BOUND_UPPER
                    && !is_mate_score(ttScore)
                    && ply < MAX_PLY - 2) {
                const int seBeta  = ttScore - sp::se_margin;
                const int seDepth = std::max(1, (depth - 1) / 2);
                ss[ply].excluded = ttMove;
                int seScore = negamax(pos, seDepth, seBeta - 1, seBeta, ply, cutNode);
                ss[ply].excluded = 0;
#if SE_MULTICUT
                // Multicut: the TT-move-excluded search ALSO beat beta → 2+ moves
                // beat beta → cut node → return the proven bound (>= beta).
                if (!aborted && seBeta >= beta && seScore >= beta) return seBeta;
#endif
                if (!aborted && seScore < seBeta) {
                    seExtension = 1;
#if SE_DSE
                    // Double extension: the TT move is dramatically better than
                    // every alternative (excluded search fails well below seBeta).
                    // Capped per line (DSE_CAP) so it can never explode the tree.
                    if (!isPV && seScore < seBeta - DSE_MARGIN * depth
                            && ss[ply].doubleExt < DSE_CAP)
                        seExtension = 2;
#endif
                }
#if SE_DSE
                // Negative extension: the TT move is NOT singular but would itself
                // fail high (its TT score already beats beta) — search it shallower.
                else if (!aborted && !isPV && ttScore >= beta)
                    seExtension = -1;
#endif
            }
#endif

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
            tt.prefetch(pos.key());   // overlap child TT-probe DRAM latency w/ loop body

            // Record into the search stack (children read this as their parent),
            // and remember the move for the history malus pass.
            ss[ply].currentMove = m;
            ss[ply].movedPiece  = moverPt;
            ss[ply].toSq        = to_sq(m);
            if (isQuiet) { if (nQuiets < 64) quietsTried[nQuiets++] = m; }
            else if (isEpPre || capPiecePre != NO_PIECE) {
                if (nCaps < 64) capsTried[nCaps++] = m;
            }

            // ── Does this move give check? (opponent now to move) ─────────
            const bool givesCheck = pos.in_check(pos.side_to_move());

            // ── Check extension (+ singular extension) ────────────────────
            // Extend by 1 ply when this move gives check or is singular. Checks
            // are forcing; the MAX_PLY guard in the recursion prevents runaway.
            bool chkExtend = givesCheck && ply < MAX_PLY - 1;
#if CHECK_SEE
            // SEE-gate: don't spend a ply on a check that loses material (a spite/
            // sac check). Capturing checks reuse the cached seeVal (free); quiet
            // checks have no parent SEE here and are treated as non-losing.
            if (chkExtend && seeVal[i] != SEE_NONE && seeVal[i] < 0) chkExtend = false;
#endif
#if SE_DSE
            // DSE: the singular result drives a -1..+2 extension; the check
            // extension applies only when singular didn't fire. newDepth is
            // clamped >= 1 (safety), and the per-line double-extension count
            // propagates to the child so DSE_CAP can bound consecutive doubles.
            const int chkExt    = chkExtend ? 1 : 0;
            const int extension = (seExtension != 0) ? seExtension : chkExt;
            // newDepth==0 is the NORMAL leaf->qsearch transition; only guard
            // against a theoretical negative (the -1 negative extension can only
            // fire at depth>=8, so newDepth>=6 there — this never actually binds).
            int newDepth = depth - 1 + extension;
            if (newDepth < 0) newDepth = 0;
            ss[ply + 1].doubleExt = ss[ply].doubleExt + (seExtension == 2 ? 1 : 0);
#elif defined(SE_SINGULAR)
            const int extension =
                std::min(1, (chkExtend ? 1 : 0) + seExtension);
            const int newDepth = depth - 1 + extension;
#else
            const int extension = chkExtend ? 1 : 0;
            const int newDepth = depth - 1 + extension;
#endif
            int score;

            if (firstMove) {
                // First (best) move: full-window search, no reduction. PV node ⇒
                // its first child is the PV continuation (cutNode=false); at a
                // non-PV (scout) node the first child is the opposite type.
                score = -negamax(pos, newDepth, -beta, -alpha, ply + 1,
                                 isPV ? false : !cutNode);
            } else {
                // ── Late Move Reduction (LMR) ─────────────────────────────
                int r = 0;
#if AGGR_LMR
                if (depth >= 3 && moveCount >= 4 && isQuiet
                        && !inCheck && !givesCheck) {
#else
                if (depth >= 4 && moveCount >= 6 && isQuiet
                        && !inCheck && !givesCheck) {
#endif
                    r = LMR[std::min(depth, 63)][std::min(moveCount, 63)];
                    if (isPV) r -= 1;                       // reduce less on PV
                    if (!improving) r += 1;                 // reduce more when not improving
                    r -= std::clamp(quietHist / sp::lmr_hist_div, -2, 2); // trust history
#if AGGR_LMR
                    r += 1;                                 // one more ply: lower EBF
#endif
#if CUTNODE == 1
                    if (cutNode) r += 1;                    // expected cut node: reduce one extra ply
#elif CUTNODE == 2
                    if (cutNode && !ttMove) r += 1;         // only the poorly-ordered cut nodes (no TT move)
#endif
#if SMP_LMRJIT
                    // Lazy-SMP tree-shape diversity: odd-id helpers reduce 1 ply LESS,
                    // even-id 1 ply MORE → different late-move frontier than main. The
                    // clamps below bound it; main (id 0) is untouched → Thr=1 identical.
                    if (id > 0) r += (id & 1) ? -1 : +1;
#endif
                    if (r < 0) r = 0;
                    if (r > newDepth - 1) r = newDepth - 1; // never below 1 ply
                    if (r < 0) r = 0;
                }
#ifdef SE_BADCAPLMR
                // Mild LMR for late SEE-negative captures (scored < 0 by the
                // ordering pass; good captures score >= SCORE_CAPTURE, quiets
                // were handled above). Promotions keep SCORE_CAPTURE so are safe.
                else if (!isQuiet && scores[i] < 0 && depth >= 4
                         && moveCount >= 4 && !inCheck && !givesCheck) {
                    r = 1;
                    if (r > newDepth - 1) r = newDepth - 1;
                    if (r < 0) r = 0;
                }
#endif

                // Reduced zero-window scout. A reduced search is always a cut
                // node (we expect it to fail low); an unreduced scout inherits the
                // opposite type from the parent.
                score = -negamax(pos, newDepth - r, -alpha - 1, -alpha, ply + 1,
                                 (r > 0) ? true : !cutNode);

                // Failed high while reduced → re-search at full depth (zero window).
                if (!aborted && score > alpha && r > 0)
                    score = -negamax(pos, newDepth, -alpha - 1, -alpha, ply + 1, !cutNode);

                // Still beating alpha but below beta → full-window re-search (PV child).
                if (!aborted && score > alpha && score < beta)
                    score = -negamax(pos, newDepth, -beta, -alpha, ply + 1, false);
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
                // ── Beta cutoff: gravity-update the move-ordering histories ──
                // Board is fully restored here (undo_move already ran), so
                // piece_on(from) is the mover and piece_on(to) the captured
                // victim again — safe to read for indexing.
                const int bonus = hist_bonus(depth);

                // Quiet-history helpers (butterfly + 1/2-ply continuation).
                auto quietBonus = [&](Move mv, int bo) {
                    const Square f = from_sq(mv), t = to_sq(mv);
                    const int cp12 = make_piece(stm, piece_type(pos.piece_on(f)));
                    hist_update(history[stm][f][t], bo);
                    if (pPiece12 >= 0) hist_update(contHist[pPiece12][pTo][cp12][t], bo);
                    if (gPiece12 >= 0) hist_update(contHist[gPiece12][gTo][cp12][t], bo);
#ifdef SE_CONTHIST2
                    if (ggPiece12 >= 0) hist_update(contHist2[ggPiece12][ggTo][cp12][t], bo / 2);
#endif
                };
                auto capBonus = [&](Move mv, int bo) {
                    const Square f = from_sq(mv), t = to_sq(mv);
                    const PieceType att = piece_type(pos.piece_on(f));
                    const PieceType vic = (type_of(mv) == EN_PASSANT)
                                        ? PAWN : piece_type(pos.piece_on(t));
                    hist_update(captHist[stm][att][t][vic], bo);
                };

                const bool isEpPost      = (type_of(m) == EN_PASSANT);
                const Piece capPiecePost = pos.piece_on(to_sq(m));
                const bool isCapturePost = isEpPost || (capPiecePost != NO_PIECE);
                const bool isPromoPost   = (type_of(m) == PROMO);

                if (!isCapturePost && !isPromoPost) {
                    // Killers (keep 2 distinct per ply).
                    if (ply < MAX_PLY && m != killers[ply][0]) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                    }
                    // Countermove: this quiet refutes the parent move.
                    if (pPiece12 >= 0) counterMove[pPiece12][pTo] = m;
                    // Reward the cutting quiet; punish the quiets that didn't cut.
                    quietBonus(m, bonus);
                    for (int q = 0; q < nQuiets; ++q)
                        if (quietsTried[q] != m) quietBonus(quietsTried[q], -bonus);
#if XMALUS
                    // A quiet refuted this node, so the captures we searched first
                    // and that did NOT cut were a waste — penalise their capture
                    // history too (cross-type malus, SOTA). They are all != m.
                    for (int c = 0; c < nCaps; ++c) capBonus(capsTried[c], -bonus);
#endif
                } else if (isCapturePost) {
                    // Reward the cutting capture; punish the others.
                    capBonus(m, bonus);
                    for (int c = 0; c < nCaps; ++c)
                        if (capsTried[c] != m) capBonus(capsTried[c], -bonus);
                }
                break; // beta cutoff
            }
        }

        if (legal == 0)
            return inCheck ? -(MATE - ply) : 0;

        // ── TT store (cache the static eval for future probes) ────────────────
        Bound bound = (best <= alphaOrig) ? BOUND_UPPER
                    : (best >= beta)       ? BOUND_LOWER
                    :                        BOUND_EXACT;
#if CORRHIST
        // Update pawn correction history toward the (bound-consistent) search
        // result on quiet, non-check, non-mate, non-singular-verification nodes.
        if (!inCheck && rawEval != -INF && bestMove != 0 && ss[ply].excluded == 0
                && !is_mate_score(best)
                && type_of(bestMove) != PROMO && type_of(bestMove) != EN_PASSANT
                && pos.piece_on(to_sq(bestMove)) == NO_PIECE
                && !(bound == BOUND_LOWER && best <= rawEval)
                && !(bound == BOUND_UPPER && best >= rawEval)) {
            const int idx    = (int)(pos.pawn_key() & (CORR_SIZE - 1));
            const int c      = pos.side_to_move();
            const int target = (best - rawEval) * CORR_GRAIN;
            const int w      = std::min(depth + 1, 16);
            const int e      = pawnCorr[c][idx] + (target - pawnCorr[c][idx]) * w / CORR_SCALE;
            pawnCorr[c][idx] = (int16_t)std::clamp(e, -CORR_MAX, CORR_MAX);
#if MULTICORR
            const int mi      = (int)(pos.minor_key() & (CORR_SIZE - 1));
            const int em      = minorCorr[c][mi] + (target - minorCorr[c][mi]) * w / CORR_SCALE;
            minorCorr[c][mi]  = (int16_t)std::clamp(em, -CORR_MAX, CORR_MAX);
            const int ma      = (int)(pos.major_key() & (CORR_SIZE - 1));
            const int ej      = majorCorr[c][ma] + (target - majorCorr[c][ma]) * w / CORR_SCALE;
            majorCorr[c][ma]  = (int16_t)std::clamp(ej, -CORR_MAX, CORR_MAX);
#endif
        }
#endif
#ifdef SE_SINGULAR
        // Don't pollute the TT with the result of a singular-verification search
        // (it deliberately omits the best move).
        if (ss[ply].excluded == 0)
#endif
        // TT stores the RAW (pre-correction) static eval; the correction is
        // re-applied fresh on each probe (storing the corrected value would
        // compound the correction over repeated probes).
        tt.store(pos.key(), bestMove, toTT(best, ply),
#if CORRHIST
                 inCheck ? TT_EVAL_NONE : (int16_t)rawEval,
#else
                 inCheck ? TT_EVAL_NONE : (int16_t)staticEval,
#endif
                 (uint8_t)depth, bound);

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
    int  stableCnt = 0;   // consecutive depths where the root best move was unchanged

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

#if SE_DSE
        // Root moves are never "double-extended into": reset the per-line double-
        // extension counter so each root search starts the DSE_CAP count clean.
        s.ss[0].doubleExt = 0;
        s.ss[1].doubleExt = 0;
#endif
        for (int i = 0; i < root.size; ++i) {
            StateInfo st;
            pos.do_move(root.moves[i], st);
            tt.prefetch(pos.key());
            int score;
            if (firstMove) {
                // Root is a PV node; its first child is the PV continuation.
                score = -s.negamax(pos, depth - 1, -beta, -alpha, 1, false);
            } else {
                // Scout children of the root expect to fail low → cut nodes.
                score = -s.negamax(pos, depth - 1, -alpha - 1, -alpha, 1, true);
                if (!s.aborted && score > alpha && score < beta) {
                    score = -s.negamax(pos, depth - 1, -beta, -alpha, 1, false);
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
#if SMP_DIV
        // Helper-thread diversity: skip a staggered subset of depths so helpers
        // search at DIFFERENT depths than main → richer shared TT → main deepens.
        if (s.id > 0) {
            static const int sSize[]  = {1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4};
            static const int sPhase[] = {0,1,0,1,2,3,0,1,2,3,4,5,0,1,2,3,4,5,6,7};
            int i = (s.id - 1) % 20;
            if (((depth + sPhase[i]) / sSize[i]) % 2) continue;
        }
#endif
#if SMP_ASPWIDE
        int delta = sp::asp_delta * (s.id > 0 ? 2 : 1); // helpers: wider window, varied re-searches
#else
        int delta = sp::asp_delta;
#endif
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

        // ── Time-management signals (vs the PREVIOUS committed iteration) ─────
        const bool moveStable = (depth > 1 && iterBest == best);
        if (moveStable) ++stableCnt; else stableCnt = 0;
        const int scoreDrop = prevScore - scoreThisDepth; // >0 ⇒ eval just fell

        // Commit the completed depth.
        prevScore = scoreThisDepth;
        best      = iterBest;
        bestScore = scoreThisDepth;
        bestDepth = depth;

        // Store the completed root result (EXACT — full window, full search).
        // Shared TT: lockless XOR scheme makes concurrent stores safe.
        tt.store(pos.key(), best, toTT(scoreThisDepth, 0), TT_EVAL_NONE, (uint8_t)depth, BOUND_EXACT);

        if (isMain) {
            if (arm_crash) crash::arm_fallback(to_uci(best).c_str());

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - s.start)
                          .count();
            if (out_mtx) out_mtx->lock();
            out << "info depth " << depth;
            if (is_mate_score(scoreThisDepth)) {
                // UCI "score mate N": N is in MOVES (not plies), signed.
                int mate_plies = MATE - std::abs(scoreThisDepth);
                int mate_moves = (mate_plies + 1) / 2;
                out << " score mate " << (scoreThisDepth > 0 ? mate_moves : -mate_moves);
            } else {
                out << " score cp " << scoreThisDepth;
            }
            out << " nodes " << s.nodes
                << " time " << ms
                << " pv " << to_uci(best) << std::endl;
            if (out_mtx) out_mtx->unlock();

            // Soft-time management is the MAIN thread's job only. Dynamic soft
            // limit: stop earlier when the best move has been stable for a while
            // (we're confident — bank the time), spend longer when the score just
            // dropped (fail-low panic — resolve it). Never exceed the hard limit.
            double factor = 1.0;
#if TM_INSTAB
            // Two-sided instability TM: bank time when the best move is settled, and
            // EXTEND when it's unsettled (best move just changed, or eval fell) so
            // critical positions get the depth they need instead of stopping at ~soft.
            // The hard limit below is the flag-safety backstop, so extending is safe.
            if (depth >= 6 && stableCnt >= 3) factor *= 0.60;   // very stable: bank time
            if (depth >= 6 && stableCnt == 0) factor *= 1.7;    // best move just changed
            if (scoreDrop >= 30)              factor *= 1.6;    // eval fell (fail-low panic)
            factor = std::clamp(factor, 0.5, 4.0);              // raised ceiling (was 2.5)
#else
            if (depth >= 6 && stableCnt >= 3) factor *= 0.65;
            if (scoreDrop >= 30) factor *= 1.6;
            factor = std::clamp(factor, 0.5, 2.5);
#endif
            int64_t softLimit = (int64_t)(tm.soft_ms * factor);
            if (softLimit > tm.hard_ms) softLimit = tm.hard_ms;
            if (ms >= softLimit) { stop.store(true); break; }
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
#ifdef TUNE
    init_lmr();  // rebuild every search: lmr_base/lmr_div may change via setoption
    lmr_ready = true;
#else
    if (!lmr_ready) { init_lmr(); lmr_ready = true; }
#endif

    // Capture the search start BEFORE any pre-search work (TT sizing, root move
    // generation, Syzygy root probe) so that overhead is charged against the time
    // budget — under flag=loss the clock effectively starts at "go", not at the
    // first node. Reused as each Searcher's `start` below.
    const auto startTime = std::chrono::steady_clock::now();

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

    // ── Root Syzygy probe ────────────────────────────────────────────────────
    // If TBs are enabled and the root position is probe-eligible, try to find
    // a TB-optimal move. We use the DTZ probe (probe_root) which accounts for
    // the 50-move rule. If it succeeds and gives us a clear move, we use it as
    // startBest (the PVS root search will also see it first, and in many cases
    // we can short-circuit entirely for a clear win/loss).
    if (syzygy::enabled()
            && pos.castling_rights() == 0
            && static_cast<unsigned>(popcount(pos.occupied())) <= syzygy::largest()) {
        unsigned res = syzygy::probe_root(pos);
        if (res != TB_RESULT_FAILED) {
            unsigned wdl  = TB_GET_WDL(res);
            unsigned from = TB_GET_FROM(res);
            unsigned to   = TB_GET_TO(res);
            unsigned promo = TB_GET_PROMOTES(res);

            // Build the king Move from TB result.
            // TB_PROMOTES: 0=none,1=queen,2=rook,3=bishop,4=knight
            // Our PROMO flag uses KNIGHT..QUEEN offset from KNIGHT.
            Move tbMove = 0;
            if (from < 64 && to < 64) {
                if (promo == 0) {
                    tbMove = make_move(Square(from), Square(to));
                } else {
                    // Map Fathom promo to our PieceType
                    PieceType pt = QUEEN;
                    if      (promo == 2) pt = ROOK;
                    else if (promo == 3) pt = BISHOP;
                    else if (promo == 4) pt = KNIGHT;
                    tbMove = make_move(Square(from), Square(to), PROMO, pt);
                }
                // Verify tbMove is legal (safety check)
                bool legal = false;
                for (int i = 0; i < root0.size; ++i) {
                    if (root0.moves[i] == tbMove) { legal = true; break; }
                }
                if (legal) {
                    startBest = tbMove;
                    crash::arm_fallback(to_uci(startBest).c_str());
                    // For a decisive TB result (WIN or LOSS), emit an info line
                    // so the GUI knows we're using TBs. For cursed wins / blessed
                    // losses / draws we still search (the 50-move counter matters).
                    if (wdl == TB_WIN || wdl == TB_LOSS) {
                        const int tb_score = (wdl == TB_WIN)
                            ? (MATE - MAX_PLY - 1)
                            : -(MATE - MAX_PLY - 1);
                        if (out_mtx) out_mtx->lock();
                        out << "info depth 0 score cp " << tb_score
                            << " tb 1 pv " << to_uci(startBest) << "\n";
                        out.flush();
                        if (out_mtx) out_mtx->unlock();
                    }
                }
            }
        }
    }

    const int nThreads = std::max(1, std::min(256, threads));
    const int maxDepth = (L.depth > 0) ? std::min(L.depth, MAX_PLY) : MAX_PLY;
    // (startTime was captured at the top of think(), before pre-search overhead.)

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
        s.id      = t;
        s.hard_ms = tm.hard_ms;
        s.start   = startTime;
        s.nodes   = 1; // first stop/time check fires at nodes=2048
        s.aborted = false;
        std::memset(s.killers, 0, sizeof(s.killers));
        std::memset(s.history, 0, sizeof(s.history));
        std::memset(s.captHist, 0, sizeof(s.captHist));
        std::memset(s.counterMove, 0, sizeof(s.counterMove));
#if CORRHIST
        std::memset(s.pawnCorr, 0, sizeof(s.pawnCorr));
#if MULTICORR
        std::memset(s.minorCorr, 0, sizeof(s.minorCorr));
        std::memset(s.majorCorr, 0, sizeof(s.majorCorr));
#endif
#endif
        std::memset(s.contHist, 0, sizeof(s.contHist));
#ifdef SE_CONTHIST2
        std::memset(s.contHist2, 0, sizeof(s.contHist2));
#endif
        std::memset(s.ss, 0, sizeof(s.ss));
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

    // ── Select the move to play ──────────────────────────────────────────────
    int bestIdx = 0;
#if SMP_VOTE
    // Best-thread voting: pick the thread that reached the greatest completed
    // depth, breaking ties by the higher score (prefers shorter mates / avoids a
    // shallower thread's stale result). Helpers often out-search thread 0.
    for (int t = 1; t < nThreads; ++t) {
        if (bestMoves[t] == 0) continue;
        if (bestDepths[t] > bestDepths[bestIdx]
            || (bestDepths[t] == bestDepths[bestIdx] && bestScores[t] > bestScores[bestIdx]))
            bestIdx = t;
    }
#endif
    Move best = bestMoves[bestIdx];
    if (best == 0) best = bestMoves[0];
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

#ifdef TUNE
    init_lmr();  // rebuild every search: lmr_base/lmr_div may change via setoption
    lmr_ready = true;
#else
    if (!lmr_ready) { init_lmr(); lmr_ready = true; }
#endif
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
        s.id      = t;
        s.hard_ms = tm.hard_ms;
        s.start   = startTime;
        s.nodes   = 1;
        s.aborted = false;
        std::memset(s.killers, 0, sizeof(s.killers));
        std::memset(s.history, 0, sizeof(s.history));
        std::memset(s.captHist, 0, sizeof(s.captHist));
        std::memset(s.counterMove, 0, sizeof(s.counterMove));
#if CORRHIST
        std::memset(s.pawnCorr, 0, sizeof(s.pawnCorr));
#if MULTICORR
        std::memset(s.minorCorr, 0, sizeof(s.minorCorr));
        std::memset(s.majorCorr, 0, sizeof(s.majorCorr));
#endif
#endif
        std::memset(s.contHist, 0, sizeof(s.contHist));
#ifdef SE_CONTHIST2
        std::memset(s.contHist2, 0, sizeof(s.contHist2));
#endif
        std::memset(s.ss, 0, sizeof(s.ss));
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
