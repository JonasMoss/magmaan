#pragma once

#include <cstdint>

namespace magmaan {

// A half-open range [begin, end) into the model source string. line/col
// describe the position of `begin` (1-based, in characters within the line;
// tabs count as one column).
struct SourceSpan {
  std::uint32_t begin = 0;
  std::uint32_t end   = 0;
  std::uint32_t line  = 1;
  std::uint32_t col   = 1;

  friend bool operator==(const SourceSpan&, const SourceSpan&) = default;
};

}
