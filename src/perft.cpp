#include "perft.hpp"
#include "movegen.hpp"

namespace king {

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList ml;
    generate_pseudo(pos, ml);

    uint64_t nodes = 0;
    for (int i = 0; i < ml.size; ++i) {
        StateInfo st;
        pos.do_move(ml.moves[i], st);
        // The mover is now the non-side-to-move; the move is legal iff
        // that mover's king is not in check.
        if (!pos.in_check(Color(!pos.side_to_move()))) {
            nodes += (depth == 1) ? 1 : perft(pos, depth - 1);
        }
        pos.undo_move(ml.moves[i]);
    }
    return nodes;
}

} // namespace king
