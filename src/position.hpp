#pragma once
#include <cstdint>
#include <string>
#include <cassert>
#include <vector>
#include "types.hpp"
#include "bitboard.hpp"
#ifdef EVAL_NNUE
#include "nnue.hpp"
#endif

namespace king {

enum CastlingRight {
    NO_CASTLING = 0,
    WHITE_OO    = 1,
    WHITE_OOO   = 2,
    BLACK_OO    = 4,
    BLACK_OOO   = 8,
    ANY_CASTLING = 15
};

// State saved before a move so it can be undone (and for incremental key/ep/etc).
struct StateInfo {
    Piece      captured;       // piece removed by the move (NO_PIECE if none); for EN_PASSANT = enemy pawn
    Square     prev_ep;
    uint8_t    prev_castling;
    int        prev_halfmove;
    int        prev_fullmove;
    uint64_t   prev_key;
    StateInfo* previous;
    // NOTE: no NNUE accumulator snapshot here. The accumulator uses COPY-MAKE: a
    // per-ply ring in Position (acc_stack_) where do_move computes the child slot
    // from the parent and undo_move just steps the ring index back — so the parent
    // accumulator is never overwritten and needs no per-StateInfo snapshot/copy.
};

class Position {
public:
    Position();
    Position(const Position&) = delete;
    Position& operator=(const Position&) = delete;

    // FEN I/O
    void        set_fen(const std::string& fen);
    std::string fen() const;

    // Piece queries
    Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
    Bitboard pieces(PieceType pt)          const { return by_type_[pt]; }
    Bitboard pieces(Color c)               const { return by_color_[c]; }
    Bitboard occupied()                    const { return by_color_[WHITE] | by_color_[BLACK]; }
    Piece    piece_on(Square s)            const { return board_[s]; }
    Square   king_sq(Color c)              const { Bitboard bb = pieces(c, KING); assert(bb); return lsb(bb); }

    // Attack queries
    Bitboard attackers_to(Square s, Bitboard occ) const;
    bool     in_check(Color c) const;

    // State
    Color    side_to_move()    const { return stm_; }
    uint8_t  castling_rights() const { return castling_; }
    Square   ep_square()       const { return ep_; }
    int      halfmove_clock()  const { return halfmove_; }
    uint64_t key()             const { return key_; }
    uint64_t pawn_key()        const { return pawn_key_; }  // pawns+kings hash (correction history)
    uint64_t compute_key()     const;

    // Draw detection
    bool is_repetition()        const;
    bool insufficient_material() const;
    bool is_draw()               const { return is_repetition() || halfmove_ >= 100 || insufficient_material(); }

    // Controlled deep clone (the generic copy ctor is deleted because st_/root_
    // are self-referential). Copies all board/scalar state and the repetition
    // history, then re-roots the StateInfo chain: the clone's root_ becomes a
    // fresh root (previous == nullptr) and st_ points at it. The clone can then
    // do/undo moves completely independently of the source — used by Lazy SMP to
    // give each search thread its own Position with the game history intact.
    void copy_from(const Position& o);

    // Make / unmake
    void do_move(Move m, StateInfo& st);
    void undo_move(Move m);

    // Null-move make/unmake (search-internal; does not touch hist_)
    void do_null_move(StateInfo& st);
    void undo_null_move();

    // Incremental helpers — also used by T7 make/unmake
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

#ifdef EVAL_NNUE
    // Live NNUE accumulator = the top of a per-ply COPY-MAKE ring (acc_stack_).
    // do_move computes the child slot from the parent in one fused pass and steps
    // acc_ply_ forward; undo_move just steps it back (the parent slot is left
    // intact, so unmake is free). Refreshed from scratch in set_fen / copy_from.
    const nnue::Accumulator& accumulator() const { return acc_stack_[acc_ply_ & ACC_MASK]; }
    void refresh_accumulator() { nnue::refresh(acc_stack_[acc_ply_ & ACC_MASK], *this); }
    // Rebuild only one perspective (after a king move of that side).
    void refresh_accumulator(Color persp) { nnue::refresh_perspective(acc_stack_[acc_ply_ & ACC_MASK], *this, persp); }
#endif

private:
    Bitboard by_type_[6]{};
    Bitboard by_color_[2]{};
    Piece    board_[64];
    Color    stm_;
    uint8_t  castling_;
    Square   ep_;
    int      halfmove_;
    int      fullmove_;
    uint64_t key_;
    uint64_t pawn_key_ = 0;   // Zobrist of pawns + kings only (correction-history key)

    StateInfo  root_;          // root state (previous == nullptr)
    StateInfo* st_ = &root_;   // current state node

    std::vector<uint64_t> hist_; // Zobrist key history for repetition detection

#ifdef EVAL_NNUE
    // Tracked king squares — the per-perspective NNUE bucket determinants. Only
    // load-bearing for KB>1 (king-bucketed nets); cheap to keep for the KB=1
    // production net. Maintained in set_fen / copy_from / do_move / undo_move.
    Square ksq_[2] = { E1, E8 };
    // Copy-make accumulator ring, indexed by (acc_ply_ & ACC_MASK). do_move writes
    // the next slot from the current one then ++acc_ply_; undo_move just --acc_ply_
    // (the parent slot is untouched → free unmake). ACC_SIZE only needs to exceed
    // the deepest do/undo NESTING (search ≤ MAX_PLY); the power-of-two ring lets
    // forward game replay (uci `position … moves`) grow acc_ply_ unboundedly with
    // zero overflow risk. Heap-backed so a Position stays small for vector<Position>.
    static constexpr int ACC_SIZE = 256;
    static constexpr int ACC_MASK = ACC_SIZE - 1;
    std::vector<nnue::Accumulator> acc_stack_;
    int acc_ply_ = 0;
#endif
};

} // namespace king
