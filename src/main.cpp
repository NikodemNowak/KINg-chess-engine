#include "uci.hpp"
#include "perft.hpp"
#include "position.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#include "zobrist.hpp"
#include "crash.hpp"
#include "tune.hpp"
#include "datagen.hpp"
#include <iostream>
#include <string>

using namespace king;

int main(int argc, char** argv) {
    crash::install_handlers();
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();

    try {
        if (argc >= 2 && std::string(argv[1]) == "tune") {
            return king::run_tuner(argc, argv);
        }
        if (argc >= 4 && std::string(argv[1]) == "datagen") {
            return king::run_datagen(argc, argv);
        }
        if (argc >= 3 && std::string(argv[1]) == "perft") {
            int depth = std::stoi(argv[2]);
            std::string fen = (argc >= 4)
                ? argv[3]
                : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            Position p;
            p.set_fen(fen);
            std::cout << perft(p, depth) << std::endl;
            return 0;
        }
        uci::loop();
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "fatal: unknown" << std::endl;
        return 1;
    }
    return 0;
}
