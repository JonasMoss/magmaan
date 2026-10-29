#include <doctest/doctest.h>
#include <string>

#include "magmaan/version.hpp"

TEST_CASE("version string is non-empty and matches the constants") {
  CHECK(magmaan::version_major == 0);
  CHECK(magmaan::version_minor == 2);
  CHECK(magmaan::version_patch == 3);
  CHECK(magmaan::version() == "0.2.3");
  CHECK(magmaan::version() == std::to_string(magmaan::version_major) + "." +
        std::to_string(magmaan::version_minor) + "." +
        std::to_string(magmaan::version_patch));
}
