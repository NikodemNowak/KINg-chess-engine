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
  Limits L; L.movetime=1000;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.hard_ms <= 700);
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
TEST_CASE("infinite") {
  Limits L; L.infinite=true;
  TimeManager tm; tm.init(L, WHITE, 300);
  CHECK(tm.hard_ms > 1000000);
}
