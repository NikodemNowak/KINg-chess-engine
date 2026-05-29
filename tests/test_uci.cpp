#include "doctest/doctest.h"
#include "uci.hpp"
#include <sstream>
#include <string>
using namespace king;

static std::string run_uci(const std::string& input) {
    std::istringstream in(input);
    std::ostringstream out;
    uci::run(in, out);
    return out.str();
}

TEST_CASE("uci handshake") {
    auto o = run_uci("uci\nquit\n");
    CHECK(o.find("uciok") != std::string::npos);
    CHECK(o.find("id name KINg") != std::string::npos);
}

TEST_CASE("isready") {
    CHECK(run_uci("isready\nquit\n").find("readyok") != std::string::npos);
}

TEST_CASE("go yields a legal bestmove") {
    auto o = run_uci("position startpos\ngo depth 3\nquit\n");
    CHECK(o.find("bestmove ") != std::string::npos);
    CHECK(o.find("bestmove 0000") == std::string::npos);
}

TEST_CASE("position moves then go") {
    auto o = run_uci("position startpos moves e2e4 e7e5\ngo depth 2\nquit\n");
    CHECK(o.find("bestmove ") != std::string::npos);
}

TEST_CASE("garbage tolerated, no crash") {
    CHECK_NOTHROW(run_uci("foobar\n\nxyz 123\nposition startpos moves e2e4\ngo movetime 50\nquit\n"));
}

TEST_CASE("fen position") {
    auto o = run_uci("position fen 4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1\ngo depth 3\nquit\n");
    CHECK(o.find("bestmove e4d5") != std::string::npos);
}
