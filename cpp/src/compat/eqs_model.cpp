#include "magmaan/compat/eqs/model.hpp"

#include <charconv>
#include <limits>
#include <variant>

#include "magmaan/parse/op.hpp"

namespace magmaan::compat::eqs {
namespace {
std::string number(double value) {
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                   std::chars_format::general,
                                   std::numeric_limits<double>::max_digits10);
  return std::string(buffer, result.ptr);
}
}  // namespace

std::string to_lavaan_syntax(const parse::FlatPartable& flat) {
  std::string result;
  for (const auto& row : flat.rows) {
    result.append(row.lhs);
    result += " ";
    result.append(parse::to_string(row.op));
    result += " ";
    if (row.mod_idx != 0) {
      const auto& modifier = flat.mods[row.mod_idx];
      if (const auto* fixed = std::get_if<parse::FixedValue>(&modifier))
        result += number(fixed->value) + "*";
      else if (const auto* start = std::get_if<parse::StartValue>(&modifier))
        result += "start(" + number(start->value) + ")*";
      else if (std::holds_alternative<parse::Free>(modifier)) result += "NA*";
    }
    result.append(row.rhs);
    result += "\n";
  }
  return result;
}
}  // namespace magmaan::compat::eqs
