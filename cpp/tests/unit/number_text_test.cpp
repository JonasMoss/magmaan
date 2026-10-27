#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <limits>
#include <locale>
#include <random>
#include <string>

#include "magmaan/parse/number_text.hpp"

#if !defined(__APPLE__)
TEST_CASE("portable decimal parsing matches finite charconv tokens") {
  using magmaan::parse::detail::decimal_from_chars_portable;
  for (const std::string text : {"0", "-0", ".5", "1.", "1e-310", "4.9406564584124654e-324",
                                 "1.7976931348623157e308", "1e309", "1e-999", "1e+", "0x12",
                                 "12suffix", "+2", " 2", ".", "-"}) {
    double expected = 123;
    double actual = 123;
    const auto a = std::from_chars(text.data(), text.data() + text.size(), expected);
    const auto b = decimal_from_chars_portable(text.data(), text.data() + text.size(), actual);
    INFO(text);
    CHECK(a.ec == b.ec);
    CHECK(a.ptr - text.data() == b.ptr - text.data());
    CHECK(std::bit_cast<std::uint64_t>(actual) == std::bit_cast<std::uint64_t>(expected));
  }
}
#endif

TEST_CASE("portable decimal conversion round trips across finite binary64 values") {
  using magmaan::parse::detail::decimal_from_chars_portable;
  using magmaan::parse::detail::decimal_to_text_portable;
  const auto check = [](double value) {
    for (int precision : {0, std::numeric_limits<double>::max_digits10}) {
      const auto text = decimal_to_text_portable(value, precision);
      double parsed = 0;
      const auto result = decimal_from_chars_portable(text.data(), text.data() + text.size(), parsed);
      INFO(value, " -> ", text);
      CHECK(result.ec == std::errc{});
      CHECK(result.ptr == text.data() + text.size());
      CHECK(std::bit_cast<std::uint64_t>(parsed) == std::bit_cast<std::uint64_t>(value));
    }
  };
  for (double value : {0., -0., 0.1, -0.125, 1e-5, 1e20, std::numeric_limits<double>::min(),
                       std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max()}) check(value);
  std::mt19937_64 random(20261027);
  for (int i = 0; i < 1000; ++i) {
    const auto value = std::bit_cast<double>(random());
    if (std::isfinite(value)) check(value);
  }
  CHECK(decimal_to_text_portable(0.1) == "0.1");
  CHECK(decimal_to_text_portable(-0.125) == "-0.125");
  CHECK(decimal_to_text_portable(1e-5) == "1e-05");
}

TEST_CASE("portable decimal conversion ignores the global decimal separator") {
  const std::string previous = std::setlocale(LC_NUMERIC, nullptr);
  bool comma = false;
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "German_Germany.1252", "fr_FR.UTF-8", "en_DK.utf8"}) {
    if (std::setlocale(LC_NUMERIC, name) != nullptr &&
        std::string(std::localeconv()->decimal_point) == ",") { comma = true; break; }
  }
  const auto text = magmaan::parse::detail::decimal_to_text_portable(0.125);
  double parsed = 0;
  const std::string literal = "0.125";
  const auto result = magmaan::parse::detail::decimal_from_chars_portable(
      literal.data(), literal.data() + literal.size(), parsed);
  std::setlocale(LC_NUMERIC, previous.c_str());
  if (!comma) { MESSAGE("No installed comma-decimal locale; locale probe skipped"); return; }
  CHECK(text == "0.125");
  CHECK(result.ec == std::errc{});
  CHECK(parsed == 0.125);
}
