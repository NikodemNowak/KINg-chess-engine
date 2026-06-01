#include "doctest/doctest.h"
#include "timeman.hpp"
#include "types.hpp"
using namespace king;
TEST_CASE("sudden death 10+0") {
  Limits L; L.time[WHITE]=600000;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.soft_ms > 0);
  CHECK(tm.hard_ms > tm.soft_ms);
  CHECK(tm.hard_ms < (int64_t)(600000*0.45));   // never gamble too much on one move
}
TEST_CASE("panic under 1s") {
  Limits L; L.time[WHITE]=500;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.hard_ms >= 50);
  CHECK(tm.hard_ms < 500);
}
TEST_CASE("movetime fixed") {
  // movetime uses ~all the allotted time; only a tiny guard (min(overhead,20))
  // is subtracted, so with overhead=300 we still get 980ms out of 1000ms.
  Limits L; L.movetime=1000;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.hard_ms >= 980);
  CHECK(tm.hard_ms <= 1000);
  CHECK(tm.soft_ms == tm.hard_ms);
}
TEST_CASE("increment") {
  Limits L; L.time[WHITE]=60000; L.inc[WHITE]=1000;
  TimeManager tm; tm.init(L, WHITE, 50);
  CHECK(tm.soft_ms > 0);
  CHECK(tm.hard_ms > tm.soft_ms);
  CHECK(tm.hard_ms < 60000);
}
TEST_CASE("never pathological on tiny clock") {
  Limits L; L.time[WHITE]=100;
  TimeManager tm; tm.init(L, WHITE, 300);   // overhead exceeds clock
  CHECK(tm.hard_ms >= 1);
  CHECK(tm.soft_ms >= 1);
}
TEST_CASE("low clock must not collapse to instant move") {
  // Regression: with the default 200ms overhead, a 1-9s clock used to make the
  // soft limit collapse to ~1ms (subtracting the full overhead from a tiny soft
  // budget), so the engine played instant, losing moves. The soft target must
  // stay sensible while the hard limit remains flag-safe (< 45% of the clock).
  int clocks[] = {1000, 2000, 5000, 8000};
  for (int t : clocks) {
    Limits L; L.time[WHITE]=t;
    TimeManager tm; tm.init(L, WHITE, 200);
    CHECK(tm.soft_ms >= 20);
    CHECK(tm.hard_ms >= tm.soft_ms);
    CHECK(tm.hard_ms < (int64_t)(t * 0.45));
  }
}
TEST_CASE("infinite") {
  Limits L; L.infinite=true;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.hard_ms > 1000000);
}
