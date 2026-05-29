#pragma once
#include "types.hpp"
#include "position.hpp"

namespace king {

// Fixed-capacity move buffer. 256 is the safe upper bound for any legal chess
// position (the practical max number of moves in a position is well under 256).
struct MoveList {
    Move moves[256];
    int  size = 0;
    void add(Move m) { moves[size++] = m; }
};

// Pseudo-legal generation: king-safety is NOT filtered, EXCEPT castling, which is
// fully legality-checked here (squares empty + not moving through/into check).
void generate_pseudo(const Position& pos, MoveList& list);

// Legality test via make/unmake: after do_move the mover is the non-side-to-move;
// the move is legal iff that mover's king is not in check.
bool is_legal(Position& pos, Move m);

// Fully legal generation: generate_pseudo then keep only is_legal moves.
void generate_legal(Position& pos, MoveList& list);

} // namespace king
