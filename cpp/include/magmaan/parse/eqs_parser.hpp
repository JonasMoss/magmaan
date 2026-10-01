#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::parse {

class EqsParser {
 public:
  // production: eqs_model = eqs_section+ ('/END')?
  // Explicit single-group covariance-model sections only. observed_names maps
  // V1, V2, ... to data-column names; empty keeps canonical V-number names.
  // Rows retain EQS source spans and own all normalized/mapped name bytes.
  static parse_expected<FlatPartable> parse(
      std::string_view source,
      const std::vector<std::string>& observed_names = {});
};

}  // namespace magmaan::parse
