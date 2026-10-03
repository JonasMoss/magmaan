#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::parse {

enum class MplusClass : std::uint8_t { Schema, DataDescription, Reported, Rejected };

struct MplusDiagnostic {
  MplusClass klass;
  SourceSpan span;
  std::string rule;
  std::string message;
};

struct MplusInput {
  std::string source;  // All spans index this owned, unmodified copy.
  std::vector<std::string> names;
  std::vector<std::string> analysis;
  std::vector<std::string> type_settings;
  bool nomeanstructure = false;
  bool nocovariances = false;
  std::string estimator;
  std::string information;
  SourceSpan model_body;
  std::vector<MplusDiagnostic> notes;
};

struct MplusModel {
  MplusInput input;
  FlatPartable flat;
  std::vector<MplusDiagnostic> notes;
};

class MplusParser {
 public:
  // production: input_file ::= line*
  static parse_expected<MplusInput> read(std::string_view source);
  // production: model_body ::= model_statement*
  static parse_expected<MplusModel> parse(std::string_view source);
};

}  // namespace magmaan::parse
