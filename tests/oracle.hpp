#pragma once

// Test-only fixture infrastructure. Loads `corpus.json`, loads per-fixture
// JSON files, and diffs C++ pipeline output against expected output with
// field-pointing error messages.
//
// Regen mode: when the MAGMAAN_REGEN_FIXTURES environment variable is set,
// the diff functions instead WRITE the current pipeline output to the
// fixture path and report a non-fatal "regenerated" message. This is the
// supported way to update fixtures after an intentional grammar change.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/error.hpp"
#include "magmaan/parse/expr_format.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/lexer.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/parse/token.hpp"
#include "magmaan/source_span.hpp"

namespace magmaan::test {

inline std::string fixtures_dir() {
  return MAGMAAN_FIXTURES_DIR;
}

struct CorpusEntry {
  std::string id;
  std::string model;
  std::string notes;
  // Optional multi-group + mean-structure metadata. Defaults match
  // single-group, meanstructure=false (preserves behavior for existing
  // entries). When set, the R script fits with `group = group_var` and
  // `meanstructure = TRUE` accordingly.
  int          n_groups      = 1;
  std::string  group_var;     // e.g., "school"
  bool         meanstructure = false;
};

struct NegativeCorpusEntry {
  std::string id;
  std::string input;
  std::string expected_kind;   // matches ParseError::Kind name
};

inline std::vector<NegativeCorpusEntry> load_lexer_negatives() {
  std::vector<NegativeCorpusEntry> out;
  const std::string path = fixtures_dir() + "/lexer/negative_corpus.json";
  std::ifstream in(path);
  if (!in) return out;
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.contains("cases")) return out;
  for (const auto& c : j["cases"]) {
    NegativeCorpusEntry e;
    if (c.contains("id"))            e.id            = c["id"].get<std::string>();
    if (c.contains("input"))         e.input         = c["input"].get<std::string>();
    if (c.contains("expected_kind")) e.expected_kind = c["expected_kind"].get<std::string>();
    out.push_back(std::move(e));
  }
  return out;
}

inline std::vector<CorpusEntry> load_corpus() {
  std::vector<CorpusEntry> out;
  const std::string path = fixtures_dir() + "/corpus.json";
  std::ifstream in(path);
  if (!in) return out;
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.contains("models")) return out;
  for (const auto& m : j["models"]) {
    CorpusEntry e;
    if (m.contains("id"))            e.id            = m["id"].get<std::string>();
    if (m.contains("model"))         e.model         = m["model"].get<std::string>();
    if (m.contains("notes"))         e.notes         = m["notes"].get<std::string>();
    if (m.contains("n_groups"))      e.n_groups      = m["n_groups"].get<int>();
    if (m.contains("group_var"))     e.group_var     = m["group_var"].get<std::string>();
    if (m.contains("meanstructure")) e.meanstructure = m["meanstructure"].get<bool>();
    out.push_back(std::move(e));
  }
  return out;
}

inline bool regen_mode() {
  const char* v = std::getenv("MAGMAAN_REGEN_FIXTURES");
  return v != nullptr && v[0] != '\0' && std::string_view(v) != "0";
}

// === JSON shape for lexer/{id}.tokens.json ==================================
//
// {
//   "_meta": { "format_version": 1, "fixture_kind": "lexer.tokens",
//              "corpus_id": "...", "tool": "regen-from-lexer" },
//   "input": "...",
//   "tokens": [
//     {"kind":"Identifier","text":"f", "begin":0, "end":1, "line":1,"col":1},
//     ...
//   ]
// }
// Negative fixtures (lexer error expected) replace "tokens" with:
//   "expected_error": {"kind":"...","begin":..,"end":..,"line":..,"col":..}

inline std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
  return out;
}

inline std::string format_token_json(const parse::Token& t) {
  std::string s = "{\"kind\":";
  s += json_escape(parse::to_string(t.kind));
  s += ",\"text\":";
  s += json_escape(t.text);
  s += ",\"begin\":" + std::to_string(t.span.begin);
  s += ",\"end\":"   + std::to_string(t.span.end);
  s += ",\"line\":"  + std::to_string(t.span.line);
  s += ",\"col\":"   + std::to_string(t.span.col);
  s += "}";
  return s;
}

inline std::string format_lexer_fixture(std::string_view corpus_id,
                                        std::string_view input,
                                        std::span<const parse::Token> tokens,
                                        const std::optional<ParseError>& err) {
  std::string s;
  s += "{\n";
  s += "  \"_meta\": {\"format_version\": 1, \"fixture_kind\": \"lexer.tokens\", ";
  s += "\"corpus_id\": " + json_escape(corpus_id) + ", ";
  s += "\"tool\": \"regen-from-lexer\"},\n";
  s += "  \"input\": " + json_escape(input) + ",\n";
  if (err.has_value()) {
    s += "  \"expected_error\": {";
    // Map ParseError::Kind to its name. Keep this in sync with error.hpp.
    auto err_name = [](ParseError::Kind k) -> std::string_view {
      switch (k) {
        case ParseError::Kind::UnexpectedChar:      return "UnexpectedChar";
        case ParseError::Kind::UnknownOperator:     return "UnknownOperator";
        case ParseError::Kind::UnsupportedOperator: return "UnsupportedOperator";
        case ParseError::Kind::UnterminatedString:  return "UnterminatedString";
        case ParseError::Kind::MalformedNumber:     return "MalformedNumber";
        case ParseError::Kind::ExpectedLhs:         return "ExpectedLhs";
        case ParseError::Kind::ExpectedOperator:    return "ExpectedOperator";
        case ParseError::Kind::ExpectedRhsTerm:     return "ExpectedRhsTerm";
        case ParseError::Kind::ModifierEvalFailed:  return "ModifierEvalFailed";
        case ParseError::Kind::GroupVecMismatch:    return "GroupVecMismatch";
      }
      return "?";
    };
    s += "\"kind\":" + json_escape(err_name(err->kind));
    s += ",\"begin\":" + std::to_string(err->span.begin);
    s += ",\"end\":"   + std::to_string(err->span.end);
    s += ",\"line\":"  + std::to_string(err->span.line);
    s += ",\"col\":"   + std::to_string(err->span.col);
    s += "}\n";
  } else {
    s += "  \"tokens\": [\n";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      s += "    " + format_token_json(tokens[i]);
      if (i + 1 < tokens.size()) s += ",";
      s += "\n";
    }
    s += "  ]\n";
  }
  s += "}\n";
  return s;
}

inline bool write_fixture(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out << content;
  return out.good();
}

inline std::optional<std::string> read_fixture(const std::string& path) {
  std::ifstream in(path);
  if (!in) return std::nullopt;
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Load a `raw: [{X:[[...]], mask?:[[...]]}]` JSON grid into data::RawData.
// One object per block; `X` is row-major (n_b × p_b). `mask` is optional:
// when every block omits it, RawData::mask is left empty, which is what
// data::sample_stats_from_raw() requires for complete data. A null `X` entry
// becomes NaN. Shared by the FIML and lavaan-parity golden tests.
inline magmaan::data::RawData raw_from_fixture(const nlohmann::json& exp) {
  magmaan::data::RawData raw;
  const auto& blocks = exp["raw"];
  raw.X.reserve(blocks.size());
  for (const auto& block : blocks) {
    const auto& Xj = block["X"];
    const Eigen::Index n = static_cast<Eigen::Index>(Xj.size());
    const Eigen::Index p = n > 0 ? static_cast<Eigen::Index>(Xj[0].size()) : 0;
    const bool has_mask = block.contains("mask") && !block["mask"].is_null();

    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M;
    if (has_mask) M.resize(n, p);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < p; ++c) {
        const auto& x = Xj[static_cast<std::size_t>(r)]
                          [static_cast<std::size_t>(c)];
        X(r, c) = x.is_null() ? std::numeric_limits<double>::quiet_NaN()
                              : x.get<double>();
        if (has_mask)
          M(r, c) = static_cast<std::uint8_t>(
              block["mask"][static_cast<std::size_t>(r)]
                           [static_cast<std::size_t>(c)].get<int>());
      }
    }
    raw.X.push_back(std::move(X));
    if (has_mask) raw.mask.push_back(std::move(M));
  }
  // Either every block carries a mask or none do; a partial set is malformed.
  if (raw.mask.size() != raw.X.size()) raw.mask.clear();
  return raw;
}

// Read a fixture's `gamma_hat` field as one block-diagonal Eigen::MatrixXd of
// size `Σ_b block_sizes[b]` × itself.
//
// Layouts (G3b, Tranche C):
//   • single-group: JSON value is a flat 2D matrix (array of rows). The
//     result is exactly that matrix (one block, fills the entire output).
//   • multi-group: JSON value is an array of {block:int, matrix:[[...]]}
//     objects. The output stacks each block on the diagonal in `block`
//     order, with `block_sizes[b]` giving the expected per-block size
//     (= p_b + p_b* when has_means, p_b* when cov-only). Off-diagonal
//     entries between blocks stay zero (lavaan's `gamma_hat` is per-group
//     so the block-diagonal lift is the natural global-frame embedding).
//
// Returns `std::nullopt` if `gamma_hat` is missing/null, or `std::expected`-
// style error message on shape mismatch. The bool overload returns success
// only — callers use the `errorMessage` to surface mismatches.
inline std::optional<Eigen::MatrixXd>
read_gamma_hat_blockdiag(const nlohmann::json&             gamma_hat,
                         const std::vector<Eigen::Index>&  block_sizes,
                         std::string*                      err = nullptr) {
  auto set_err = [&](const std::string& m) {
    if (err) *err = m;
    return std::optional<Eigen::MatrixXd>{};
  };
  if (gamma_hat.is_null()) return std::nullopt;
  Eigen::Index total = 0;
  for (auto sz : block_sizes) total += sz;
  if (total <= 0) return set_err("read_gamma_hat_blockdiag: zero total size");

  if (gamma_hat.is_array() && !gamma_hat.empty() &&
      gamma_hat[0].is_object() && gamma_hat[0].contains("matrix")) {
    // Multi-group: array of {block, matrix}.
    if (gamma_hat.size() != block_sizes.size())
      return set_err("read_gamma_hat_blockdiag: gamma_hat has " +
                     std::to_string(gamma_hat.size()) + " blocks, expected " +
                     std::to_string(block_sizes.size()));
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(total, total);
    Eigen::Index offset = 0;
    for (std::size_t b = 0; b < gamma_hat.size(); ++b) {
      const auto& M = gamma_hat[b]["matrix"];
      const Eigen::Index n = static_cast<Eigen::Index>(M.size());
      if (n != block_sizes[b])
        return set_err("read_gamma_hat_blockdiag: block " + std::to_string(b) +
                       " size " + std::to_string(n) + " ≠ expected " +
                       std::to_string(block_sizes[b]));
      for (Eigen::Index r = 0; r < n; ++r) {
        const auto& row = M[static_cast<std::size_t>(r)];
        for (Eigen::Index c = 0; c < n; ++c) {
          out(offset + r, offset + c) =
              row[static_cast<std::size_t>(c)].get<double>();
        }
      }
      offset += n;
    }
    return out;
  }

  // Single-group: flat 2D matrix.
  if (block_sizes.size() != 1)
    return set_err("read_gamma_hat_blockdiag: gamma_hat is single-block but "
                   "block_sizes has " + std::to_string(block_sizes.size()) +
                   " entries");
  const Eigen::Index n = static_cast<Eigen::Index>(gamma_hat.size());
  if (n != block_sizes[0])
    return set_err("read_gamma_hat_blockdiag: gamma_hat is " +
                   std::to_string(n) + "×… ≠ expected " +
                   std::to_string(block_sizes[0]));
  Eigen::MatrixXd out(n, n);
  for (Eigen::Index r = 0; r < n; ++r) {
    const auto& row = gamma_hat[static_cast<std::size_t>(r)];
    for (Eigen::Index c = 0; c < n; ++c) {
      out(r, c) = row[static_cast<std::size_t>(c)].get<double>();
    }
  }
  return out;
}

// === Lexer-stage helpers ====================================================

struct LexerScan {
  std::vector<parse::Token> tokens;
  std::optional<ParseError> error;
};

inline LexerScan scan_all(std::string_view src) {
  parse::Lexer lex(src);
  LexerScan out;
  for (;;) {
    auto t = lex.next();
    if (!t.has_value()) {
      out.error = t.error();
      return out;
    }
    out.tokens.push_back(*t);
    if (t->kind == parse::TokenKind::EndOfFile) return out;
  }
}

// Returns empty string on match; otherwise a multi-line, field-pointing diff.
// The diff walks the entire token stream — never short-circuits.
inline std::string diff_lexer_fixture(std::string_view corpus_id,
                                      std::string_view input,
                                      const LexerScan& got,
                                      const nlohmann::json& expected) {
  std::ostringstream d;
  auto bad = [&](std::string_view what) {
    if (d.tellp() == 0) d << "lexer fixture mismatch [" << corpus_id << "]:\n";
    d << "  " << what << "\n";
  };

  if (expected.contains("expected_error")) {
    if (!got.error.has_value()) {
      bad("expected an error, lexer succeeded");
      return d.str();
    }
    const auto& e = expected["expected_error"];
    auto check = [&](const char* field, auto got_val, auto exp_val) {
      if (got_val != exp_val) {
        std::ostringstream m;
        m << "expected_error." << field << ": got=" << got_val
          << ", expected=" << exp_val;
        bad(m.str());
      }
    };
    if (e.contains("kind")) {
      // Compare via name string.
      auto err_name = [](ParseError::Kind k) -> std::string {
        switch (k) {
          case ParseError::Kind::UnexpectedChar:      return "UnexpectedChar";
          case ParseError::Kind::UnknownOperator:     return "UnknownOperator";
          case ParseError::Kind::UnsupportedOperator: return "UnsupportedOperator";
          case ParseError::Kind::UnterminatedString:  return "UnterminatedString";
          case ParseError::Kind::MalformedNumber:     return "MalformedNumber";
          default: return "?";
        }
      };
      check("kind", err_name(got.error->kind), e["kind"].get<std::string>());
    }
    if (e.contains("begin")) check("begin", got.error->span.begin, e["begin"].get<std::uint32_t>());
    if (e.contains("end"))   check("end",   got.error->span.end,   e["end"].get<std::uint32_t>());
    if (e.contains("line"))  check("line",  got.error->span.line,  e["line"].get<std::uint32_t>());
    if (e.contains("col"))   check("col",   got.error->span.col,   e["col"].get<std::uint32_t>());
    return d.str();
  }

  if (got.error.has_value()) {
    std::ostringstream m;
    m << "lexer error not expected: kind=" << static_cast<int>(got.error->kind)
      << " at line " << got.error->span.line << ":" << got.error->span.col
      << " — " << got.error->detail;
    bad(m.str());
    return d.str();
  }

  if (!expected.contains("tokens") || !expected["tokens"].is_array()) {
    bad("fixture has neither 'tokens' nor 'expected_error'");
    return d.str();
  }
  const auto& exp = expected["tokens"];
  const std::size_t n = std::min<std::size_t>(got.tokens.size(), exp.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto& g = got.tokens[i];
    const auto& e = exp[i];
    auto field_str = [&](const char* name, std::string g_val, std::string e_val) {
      if (g_val != e_val) {
        std::ostringstream m;
        m << "token[" << i << "]." << name << ": got=" << json_escape(g_val)
          << ", expected=" << json_escape(e_val);
        bad(m.str());
      }
    };
    auto field_u32 = [&](const char* name, std::uint32_t g_val, std::uint32_t e_val) {
      if (g_val != e_val) {
        std::ostringstream m;
        m << "token[" << i << "]." << name << ": got=" << g_val
          << ", expected=" << e_val;
        bad(m.str());
      }
    };
    if (e.contains("kind"))  field_str("kind",  std::string(parse::to_string(g.kind)),
                                       e["kind"].get<std::string>());
    if (e.contains("text"))  field_str("text",  std::string(g.text),
                                       e["text"].get<std::string>());
    if (e.contains("begin")) field_u32("begin", g.span.begin, e["begin"].get<std::uint32_t>());
    if (e.contains("end"))   field_u32("end",   g.span.end,   e["end"].get<std::uint32_t>());
    if (e.contains("line"))  field_u32("line",  g.span.line,  e["line"].get<std::uint32_t>());
    if (e.contains("col"))   field_u32("col",   g.span.col,   e["col"].get<std::uint32_t>());
  }
  if (got.tokens.size() != exp.size()) {
    std::ostringstream m;
    m << "token count: got=" << got.tokens.size() << ", expected=" << exp.size();
    bad(m.str());
  }
  // Touch input so a fixture editing /input/ alone fails clearly.
  if (expected.contains("input") && expected["input"].get<std::string>() != input) {
    bad("fixture's 'input' does not match the corpus model string");
  }
  return d.str();
}

// === flat.partable JSON translation =========================================
//
// Mirrors the shape produced by tests/tools/regen_oracle.R. Comparing OUR
// FlatPartable against lavaan's output reduces to:
//   nlohmann::json got = flat_partable_to_json(parsed);
//   nlohmann::json exp = json::parse(read_fixture(...));
//   if (got != exp) FAIL with both dumped.

inline nlohmann::json modifier_atom_to_json(const parse::ModifierAtom& a) {
  nlohmann::json j;
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::FixedValue>) {
          j["kind"] = "fixed";
          j["value"] = v.value;
        } else if constexpr (std::is_same_v<T, parse::Free>) {
          j["kind"] = "free";
        } else if constexpr (std::is_same_v<T, parse::Label>) {
          j["kind"] = "label";
          j["text"] = std::string(v.text);
        } else if constexpr (std::is_same_v<T, parse::StartValue>) {
          j["kind"] = "start";
          j["value"] = v.value;
        } else if constexpr (std::is_same_v<T, parse::EqualRef>) {
          j["kind"] = "equal";
          j["target"] = std::string(v.text);
        }
      },
      a);
  return j;
}

inline nlohmann::json modifier_to_json(const parse::Modifier& m) {
  nlohmann::json j;
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::FixedValue>) {
          j["kind"] = "fixed";
          j["value"] = v.value;
        } else if constexpr (std::is_same_v<T, parse::Free>) {
          j["kind"] = "free";
        } else if constexpr (std::is_same_v<T, parse::Label>) {
          j["kind"] = "label";
          j["text"] = std::string(v.text);
        } else if constexpr (std::is_same_v<T, parse::StartValue>) {
          j["kind"] = "start";
          j["value"] = v.value;
        } else if constexpr (std::is_same_v<T, parse::EqualRef>) {
          j["kind"] = "equal";
          j["target"] = std::string(v.text);
        } else if constexpr (std::is_same_v<T, parse::GroupVec>) {
          j["kind"] = "group";
          nlohmann::json atoms = nlohmann::json::array();
          for (const auto& a : v.per_group) atoms.push_back(modifier_atom_to_json(a));
          j["atoms"] = std::move(atoms);
        }
      },
      m);
  return j;
}

// (expr_to_canonical / format_number_canonical / binop_prec / binop_char are
// provided by magmaan/parse/expr_format.hpp — public header so lavaanify and
// other layers can use the same canonicalization.)

using parse::expr_to_canonical;

inline std::string constraint_op_to_string(parse::ConstraintKind k) {
  switch (k) {
    case parse::ConstraintKind::Eq:     return "==";
    case parse::ConstraintKind::Lt:     return "<";
    case parse::ConstraintKind::Gt:     return ">";
    case parse::ConstraintKind::Define: return ":=";
  }
  return "?";
}

inline nlohmann::json constraint_to_json(const parse::Constraint& c) {
  nlohmann::json j;
  j["op"] = constraint_op_to_string(c.kind);
  if (c.kind == parse::ConstraintKind::Define) {
    j["lhs"] = std::string(c.name);
  } else {
    j["lhs"] = expr_to_canonical(c.lhs);
  }
  j["rhs"] = expr_to_canonical(c.rhs);
  return j;
}

inline std::string op_to_lavaan_string(parse::Op op) {
  switch (op) {
    case parse::Op::Measurement:   return "=~";
    case parse::Op::Regression:    return "~";
    case parse::Op::Covariance:    return "~~";
    case parse::Op::Threshold:     return "|";
    case parse::Op::ResponseScale: return "~*~";
    case parse::Op::Intercept:     return "~1";
    case parse::Op::DefineParam:   return ":=";
    case parse::Op::EqConstraint:  return "==";
    case parse::Op::LtConstraint:  return "<";
    case parse::Op::GtConstraint:  return ">";
    case parse::Op::Composite:     return "<~";
  }
  return "?";
}

inline nlohmann::json flat_partable_to_json(std::string_view input,
                                            std::string_view corpus_id,
                                            const parse::FlatPartable& fp) {
  nlohmann::json j;
  j["_meta"] = {
      {"format_version", 1},
      {"fixture_kind",   "flat.partable"},
      {"corpus_id",      std::string(corpus_id)},
      {"tool",           "magmaan::Parser"},
      // intentionally omit lavaan_version on our side; the oracle owns it
  };
  j["input"] = std::string(input);

  nlohmann::json rows = nlohmann::json::array();
  for (const auto& r : fp.rows) {
    nlohmann::json row;
    row["lhs"]   = std::string(r.lhs);
    row["op"]    = op_to_lavaan_string(r.op);
    row["rhs"]   = std::string(r.rhs);
    row["block"] = r.block;
    if (r.mod_idx != 0) row["modifier"] = modifier_to_json(fp.mods[r.mod_idx]);
    rows.push_back(std::move(row));
  }
  j["rows"] = std::move(rows);

  nlohmann::json cons = nlohmann::json::array();
  for (const auto& c : fp.constraints) cons.push_back(constraint_to_json(c));
  j["constraints"] = std::move(cons);
  return j;
}

// Strip the `_meta` block from a flat fixture before comparing — its tool
// name and lavaan_version legitimately differ between OUR output and the
// oracle's. Everything else (input, rows, constraints) must match.
inline nlohmann::json strip_meta(nlohmann::json j) {
  if (j.is_object() && j.contains("_meta")) j.erase("_meta");
  return j;
}

}  // namespace magmaan::test
