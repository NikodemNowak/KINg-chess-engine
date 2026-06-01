#include "crash.hpp"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace king {
namespace crash {

static char              g_line[24];
static volatile sig_atomic_t g_armed = 0;

static void on_fatal(int /*sig*/) {
    if (g_armed) {
        std::fputs(g_line, stdout);
        std::fflush(stdout);
    }
    std::_Exit(1);
}

void arm_fallback(const char* uci_move) {
    // Format "bestmove <uci>\n" into g_line (bounded)
    const char prefix[] = "bestmove ";
    std::size_t plen = sizeof(prefix) - 1; // without null terminator
    std::size_t mlen = std::strlen(uci_move);
    if (plen + mlen + 2 > sizeof(g_line) - 1) return; // won't fit; leave previous value
    std::memcpy(g_line, prefix, plen);
    std::memcpy(g_line + plen, uci_move, mlen);
    g_line[plen + mlen]     = '\n';
    g_line[plen + mlen + 1] = '\0';
    g_armed = 1;
}

void install_handlers() {
    std::signal(SIGSEGV, on_fatal);
    std::signal(SIGABRT, on_fatal);
    std::signal(SIGFPE,  on_fatal);
    std::signal(SIGILL,  on_fatal);   // illegal instruction (e.g. AVX2 on an unexpected CPU)
    std::signal(SIGTERM, on_fatal);   // arbiter/OS kill: still emit our pre-armed best move
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);    // broken stdout pipe must not kill us before flush
#endif
}

} // namespace crash
} // namespace king
