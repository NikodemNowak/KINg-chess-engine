#include "doctest/doctest.h"
#include "types.hpp"
TEST_CASE("smoke") { CHECK(king::sanity() == 42); }
