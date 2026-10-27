#pragma once

#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace magmaan::parse::detail {

// production: num_lit = digit+ ('.' digit*)? exponent? | '.' digit+ exponent?
// Decimal tokens only. The classic locale keeps model syntax independent of
// the process locale on macOS runtimes without floating-point charconv.
inline std::from_chars_result decimal_from_chars_portable(
    const char* first, const char* last, double& value) {
  const auto digit = [](char c) { return c >= '0' && c <= '9'; };
  auto end = first;
  if (end != last && *end == '-') ++end;
  bool any = false;
  bool nonzero = false;
  const auto digits = [&] {
    while (end != last && digit(*end)) {
      any = true;
      nonzero = nonzero || *end != '0';
      ++end;
    }
  };
  digits();
  if (end != last && *end == '.') { ++end; digits(); }
  if (!any) return {first, std::errc::invalid_argument};
  if (end != last && (*end == 'e' || *end == 'E')) {
    auto exponent = end + 1;
    if (exponent != last && (*exponent == '+' || *exponent == '-')) ++exponent;
    const auto start = exponent;
    while (exponent != last && digit(*exponent)) ++exponent;
    if (exponent != start) end = exponent;
  }
  std::istringstream input(std::string(first, end));
  input.imbue(std::locale::classic());
  double parsed = 0;
  input >> std::noskipws >> parsed;
  if (input.fail() || !std::isfinite(parsed) || (parsed == 0 && nonzero))
    return {end, std::errc::result_out_of_range};
  value = parsed;
  return {end, {}};
}

inline std::from_chars_result decimal_from_chars(
    const char* first, const char* last, double& value) {
#if defined(__APPLE__)
  return decimal_from_chars_portable(first, last, value);
#else
  return std::from_chars(first, last, value);
#endif
}

inline std::string decimal_to_text_portable(double value, int precision = 0) {
  const auto scientific = [value](int digits) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::scientific << std::setprecision(digits - 1) << value;
    return output.str();
  };
  if (precision != 0 || !std::isfinite(value)) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(precision == 0 ? 17 : precision) << value;
    return output.str();
  }
  // Find the shortest correctly rounded significant-digit sequence that
  // round-trips, then choose fixed/scientific notation by length (fixed on ties).
  std::string text;
  for (int digits = 1; digits <= std::numeric_limits<double>::max_digits10; ++digits) {
    text = scientific(digits);
    double parsed = 0;
    const auto result = decimal_from_chars_portable(text.data(), text.data() + text.size(), parsed);
    if (result.ec == std::errc{} && parsed == value &&
        std::signbit(parsed) == std::signbit(value)) break;
  }
  const auto e = text.find('e');
  int exponent = 0;
  auto exp_start = text.data() + e + 1;
  if (*exp_start == '+') ++exp_start;
  std::from_chars(exp_start, text.data() + text.size(), exponent);
  const bool negative = text.front() == '-';
  auto digits = text.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
  if (const auto dot = digits.find('.'); dot != std::string::npos) digits.erase(dot, 1);
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  const int point = exponent + 1;
  std::string fixed = negative ? "-" : "";
  if (point <= 0) fixed += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
  else if (static_cast<std::size_t>(point) >= digits.size())
    fixed += digits + std::string(static_cast<std::size_t>(point) - digits.size(), '0');
  else fixed += digits.substr(0, static_cast<std::size_t>(point)) + "." +
                digits.substr(static_cast<std::size_t>(point));
  std::string compact = negative ? "-" : "";
  compact += digits.front();
  if (digits.size() > 1) compact += "." + digits.substr(1);
  compact += text.substr(e);
  return fixed.size() <= compact.size() ? fixed : compact;
}

inline std::string decimal_to_text(double value, int precision = 0) {
#if defined(__APPLE__)
  return decimal_to_text_portable(value, precision);
#else
  char buffer[64];
  const auto result = precision == 0
      ? std::to_chars(buffer, buffer + sizeof(buffer), value)
      : std::to_chars(buffer, buffer + sizeof(buffer), value,
                      std::chars_format::general, precision);
  return std::string(buffer, result.ptr);
#endif
}

}  // namespace magmaan::parse::detail
