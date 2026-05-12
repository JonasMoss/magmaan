#include <doctest/doctest.h>

#include "latva/version.hpp"

TEST_CASE("version string is non-empty and matches the constants") {
  CHECK(latva::version_major == 0);
  CHECK(latva::version_minor == 0);
  CHECK(latva::version_patch == 1);
  CHECK(latva::version() == "0.0.1");
}
