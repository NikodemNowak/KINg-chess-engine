#include "crash.hpp"
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace king {
namespace crash {

// Double-buffered so the async signal handler always sees a COMPLETE line: we fill
// the inactive buffer, then flip g_idx (a single sig_atomic_t write) to publish it.
// A fatal signal arriving mid-rewrite therefore reads either the previous or the new
// bestmove — never a torn/half-written one, and never an empty buffer. crash == lost
// game, so the emergency fallback line must always be valid and emittable.
static char                  g_line[2][24];
static volatile sig_atomic_t g_idx   = 0;
static volatile sig_atomic_t g_armed = 0;

static void on_fatal(int /*sig*/) {
    if (g_armed) {
        std::fputs(g_line[g_idx], stdout);
        std::fflush(stdout);
    }
    std::_Exit(1);
}

void arm_fallback(const char* uci_move) {
    // Format "bestmove <uci>\n" into the INACTIVE buffer (bounded), then publish by
    // flipping g_idx — so a fatal signal can never observe a half-written line.
    const char prefix[] = "bestmove ";
    std::size_t plen = sizeof(prefix) - 1; // without null terminator
    std::size_t mlen = std::strlen(uci_move);
    if (plen + mlen + 2 > sizeof(g_line[0]) - 1) return; // won't fit; keep previous
    int   next = g_idx ^ 1;
    char* buf  = g_line[next];
    std::memcpy(buf, prefix, plen);
    std::memcpy(buf + plen, uci_move, mlen);
    buf[plen + mlen]     = '\n';
    buf[plen + mlen + 1] = '\0';
    // Compiler barrier: the buffer must be fully written before we publish the new
    // index. Same-thread signal handler, so no CPU fence is needed.
    std::atomic_signal_fence(std::memory_order_release);
    g_idx   = static_cast<sig_atomic_t>(next);
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
