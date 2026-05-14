#include <doctest/doctest.h>

#include "magmaan/version.hpp"

TEST_CASE("version string is non-empty and matches the constants") {
  CHECK(magmaan::version_major == 0);
  CHECK(magmaan::version_minor == 0);
  CHECK(magmaan::version_patch == 1);
  CHECK(magmaan::version() == "0.0.1");
}
