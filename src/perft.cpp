#include "perft.hpp"
#include "movegen.hpp"
#include <cstdio>

namespace king {

// Convert a square to its two-character algebraic name ("a1".."h8").
static char file_char(Square s) { return char('a' + file_of(s)); }
static char rank_char(Square s) { return char('1' + rank_of(s)); }

// Write the UCI string for a move into buf (must be at least 6 bytes).
// Returns the number of characters written.
static int move_to_uci(Move m, char buf[6]) {
    Square from = from_sq(m);
    Square to   = to_sq(m);
    buf[0] = file_char(from);
    buf[1] = rank_char(from);
    buf[2] = file_char(to);
    buf[3] = rank_char(to);
    if (type_of(m) == PROMO) {
        static const char promo_char[] = "nbrq";
        // promo_pt returns KNIGHT=1..QUEEN=4; index into nbrq at (pt - KNIGHT)
        buf[4] = promo_char[promo_pt(m) - KNIGHT];
        buf[5] = '\0';
        return 5;
    }
    buf[4] = '\0';
    return 4;
}

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

void perft_divide(Position& pos, int depth) {
    MoveList ml;
    generate_legal(pos, ml);

    uint64_t total = 0;
    for (int i = 0; i < ml.size; ++i) {
        char buf[6];
        move_to_uci(ml.moves[i], buf);

        uint64_t cnt = 1;
        if (depth > 1) {
            StateInfo st;
            pos.do_move(ml.moves[i], st);
            cnt = perft(pos, depth - 1);
            pos.undo_move(ml.moves[i]);
        }

        std::printf("%s: %llu\n", buf, (unsigned long long)cnt);
        total += cnt;
    }
    std::printf("Total: %llu\n", (unsigned long long)total);
}

} // namespace king
