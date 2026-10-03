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

struct MplusGroup { std::string label; std::string code; };
struct MplusGroupSection { std::string label; SourceSpan body; };

enum class MplusFormatKind : std::uint8_t { Field, Skip, Tab, Record };
struct MplusFormatItem {
  MplusFormatKind kind;
  std::int32_t width = 0;
  std::int32_t decimals = 0;
};
struct MplusDataFile { std::string path, label; };
struct MplusMissingRule {
  std::vector<std::string> variables;
  std::vector<double> values;
};
struct MplusDataPlan {
  std::vector<MplusDataFile> files;
  std::vector<MplusFormatItem> format;
  std::string matrix_type;  // Empty means INDIVIDUAL.
  bool means = false, standard_deviations = false, listwise = false;
  bool file_groups = false;
  std::int32_t n_groups = 1;
  std::vector<std::int32_t> n_observations;
  std::string missing_symbol;
  std::vector<MplusMissingRule> missing;
};

struct MplusInput {
  MplusDataPlan data_plan;
  std::string source;  // All spans index this owned, unmodified copy.
  std::vector<std::string> names;
  std::vector<std::string> analysis;
  std::vector<std::string> type_settings;
  bool nomeanstructure = false;
  bool nocovariances = false;
  std::string grouping_variable;
  std::vector<MplusGroup> groups;
  std::vector<MplusGroupSection> group_sections;
  std::string invariance;
  std::string estimator;
  std::string information;
  SourceSpan model_body;
  std::vector<MplusDiagnostic> notes;
};

struct MplusGeneratedRow { std::string lhs, rhs; Op op; std::int32_t group; };

struct MplusModel {
  MplusInput input;
  FlatPartable flat;
  std::vector<MplusGeneratedRow> generated_rows;
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
