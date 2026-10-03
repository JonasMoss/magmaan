#include "magmaan/parse/mplus_parser.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace magmaan::parse {
namespace {

// production: letter ::= [A-Za-z]
bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
// production: digit ::= [0-9]
bool digit(char c) { return c >= '0' && c <= '9'; }
// production: BLANK ::= ' ' | '\t'
bool blank(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
std::string upper(std::string_view s) {
  std::string out(s);
  for (char& c : out) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  return out;
}
// production: name ::= letter (letter | digit | '_')*
bool valid_name(std::string_view s) {
  return !s.empty() && letter(s.front()) && std::all_of(s.begin(), s.end(), [](char c) {
    return letter(c) || digit(c) || c == '_';
  });
}
std::vector<std::string> words(std::string_view s) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size();) {
    if (blank(s[i]) || s[i] == ',') { ++i; continue; }
    auto start = i;
    while (i < s.size() && !blank(s[i]) && s[i] != ',') ++i;
    out.emplace_back(s.substr(start, i - start));
  }
  return out;
}
std::string resolve(std::string_view raw, const std::vector<std::string>& choices) {
  const auto key = upper(raw);
  std::string found;
  for (const auto& choice : choices) {
    if (key == choice) return choice;
    if (key.size() >= 4 && choice.starts_with(key)) {
      if (!found.empty()) return {};
      found = choice;
    }
  }
  return found;
}
std::string setting(std::string_view raw, std::string_view table) {
  const auto key = upper(raw);
  for (const auto& entry : words(table)) {
    const auto slash = entry.find('/');
    const auto full = entry.substr(0, slash);
    if (key == full || (slash != std::string::npos && key == entry.substr(slash + 1)))
      return full;
  }
  return {};
}

struct OptionSpec { std::string name; MplusClass klass; std::string rule; int later = 0; };
struct Section { std::string command; std::string qualifier; std::size_t begin, body, end; };
struct Item { std::string first, last; };

class Reader {
 public:
  MplusInput out;
  std::string clean;
  std::vector<std::size_t> lines{0};
  std::vector<MplusDiagnostic> rejects;
  std::vector<Section> sections;
  std::vector<Item> use;
  SourceSpan use_span, nomean_span;
  bool has_use = false, grouping = false, data_groups = false, mixture = false;

  explicit Reader(std::string_view s) { out.source = s; clean = s; }

  SourceSpan span(std::size_t begin, std::size_t end) const {
    const auto it = std::upper_bound(lines.begin(), lines.end(), begin);
    const auto row = static_cast<std::size_t>(it - lines.begin() - 1);
    return {static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end),
            static_cast<std::uint32_t>(row + 1), static_cast<std::uint32_t>(begin - lines[row] + 1)};
  }
  void diagnostic(MplusClass klass, SourceSpan at, std::string rule, std::string message) {
    (klass == MplusClass::Rejected ? rejects : out.notes).push_back(
        {klass, at, std::move(rule), std::move(message)});
  }
  void reject(SourceSpan at, std::string rule, std::string message) {
    // Preserve the specific rule explanation and make every reader rejection
    // actionable, including malformed list/command boundaries.
    const auto found = out.source.substr(at.begin, std::min<std::size_t>(at.end - at.begin, 120));
    if (message.find("Mplus") == std::string::npos)
      message += "; Mplus requires valid names, complete command heads and terminated options, and supports additional analysis/data families; magmaan cannot reproduce this input in increment 1; write a valid supported single-group continuous specification instead";
    diagnostic(MplusClass::Rejected, at, std::move(rule), "found '" + found + "': " + message);
  }

  // production: line_content ::= (token | BLANK | block_comment)+ line_comment? | line_comment
  void comments() {
    for (std::size_t i = 0; i < clean.size(); ++i) if (clean[i] == '\n') lines.push_back(i + 1);
    char quote = 0;
    for (std::size_t i = 0; i < clean.size();) {
      if (clean[i] == '\n') quote = 0;
      if (quote) { if (clean[i] == quote) quote = 0; ++i; continue; }
      if (clean[i] == '\'' || clean[i] == '"') { quote = clean[i++]; continue; }
      if (clean[i] != '!') { ++i; continue; }
      const auto begin = i;
      if (i + 1 < clean.size() && clean[i + 1] == '*') {
        const auto close = clean.find("*!", i + 2);
        const auto end = close == std::string::npos ? clean.size() : close + 2;
        const auto at = span(begin, end);
        const auto line_start = lines[at.line - 1];
        bool after_code = false;
        for (auto j = line_start; j < begin; ++j) if (!blank(clean[j])) after_code = true;
        if (close == std::string::npos) reject(at, "LX05", "unterminated block comment");
        else if (after_code && span(close, end).line != at.line)
          reject(at, "LX05", "block comment opened after code must close on its opening line");
        i = end;
      } else {
        i = clean.find('\n', begin);
        if (i == std::string::npos) i = clean.size();
      }
      for (auto j = begin; j < i; ++j) if (clean[j] != '\n' && clean[j] != '\r') clean[j] = ' ';
    }
  }

  // production: command_head ::= LINE_START BLANK* command_word ':'
  void heads() {
    const std::vector<std::string> commands = {"TITLE", "DATA", "VARIABLE", "DEFINE", "ANALYSIS", "MODEL", "OUTPUT", "SAVEDATA", "PLOT", "MONTECARLO"};
    std::set<std::string> seen;
    for (auto line_start : lines) {
      auto i = line_start;
      while (i < clean.size() && (clean[i] == ' ' || clean[i] == '\t' || clean[i] == '\r')) ++i;
      const auto begin = i;
      if (i == clean.size() || !letter(clean[i])) continue;
      while (i < clean.size() && (letter(clean[i]) || digit(clean[i]) || clean[i] == '_')) ++i;
      const auto raw = clean.substr(begin, i - begin);
      auto colon = i;
      while (colon < clean.size() && clean[colon] != '\n' && clean[colon] != ':' &&
             (letter(clean[colon]) || digit(clean[colon]) || clean[colon] == '_' ||
              clean[colon] == '-' || clean[colon] == ' ' || clean[colon] == '\t')) ++colon;
      if (colon == clean.size() || clean[colon] != ':') continue;
      const auto command = resolve(raw, commands);
      // TITLE is free text; only a recognized line-start command ends it.
      if (command.empty() && !sections.empty() && sections.back().command == "TITLE") continue;
      if (!sections.empty()) sections.back().end = begin;
      const auto qualifier_words = words(std::string_view(clean).substr(i, colon - i));
      std::string qualifier;
      for (const auto& w : qualifier_words) { if (!qualifier.empty()) qualifier += ' '; qualifier += upper(w); }
      if (command.empty()) reject(span(begin, colon + 1), "LX03", "unknown or ambiguous command: " + raw);
      if (!qualifier.empty() && command != "MODEL" && command != "DATA")
        reject(span(begin, colon + 1), "LX01", "unexpected command qualifier");
      const auto key = command + " " + qualifier;
      const auto model_kind = resolve(qualifier, {"CONSTRAINT", "INDIRECT", "TEST", "PRIORS", "POPULATION", "COVERAGE", "MISSING"});
      const bool model_label = command == "MODEL" && !qualifier.empty() && model_kind.empty() &&
          qualifier.find('-') == std::string::npos;
      if (!model_label && !seen.insert(key).second)
        reject(span(begin, colon + 1), "LX01", "repeated command: " + key);
      sections.push_back({command, qualifier, begin, colon + 1, clean.size()});
    }
    if (sections.empty()) reject(span(0, clean.size()), "LX01", "input has no command heads");
    else {
      for (std::size_t i = 0; i < sections.front().begin; ++i) if (!blank(clean[i])) {
        reject(span(i, sections.front().begin), "LX01", "content before first command"); break;
      }
    }
    for (const auto& required : {"DATA", "VARIABLE"})
      if (std::none_of(sections.begin(), sections.end(), [&](const Section& s) { return s.command == required && s.qualifier.empty(); }))
        reject(span(0, 0), "LX01", std::string("required command: ") + required);
  }

  static std::vector<OptionSpec> specs(const std::string& command) {
    std::vector<OptionSpec> result;
    const auto add = [&](std::string_view names, MplusClass klass, const char* rule, int later = 0) {
      for (auto& name : words(names)) result.push_back({std::move(name), klass, rule, later});
    };
    using enum MplusClass;
    if (command == "DATA") {
      add("FILE", DataDescription, "CL02");
      add("FORMAT NOBSERVATIONS NGROUPS LISTWISE TYPE", DataDescription, "CL03");
      add("VARIANCES", Reported, "CL05"); add("SWMATRIX", Rejected, "CL04");
    } else if (command == "VARIABLE") {
      add("NAMES", Schema, "CL07"); add("USEVARIABLES", Schema, "CL08");
      add("MISSING", DataDescription, "CL09"); add("GROUPING", Schema, "CL11");
      add("CATEGORICAL", Rejected, "CL10", 3); add("IDVARIABLE AUXILIARY", Reported, "CL12");
      add("USEOBSERVATIONS SUBPOPULATION", Rejected, "CL13");
      add("CENSORED NOMINAL COUNT DSURVIVAL TSCORES SURVIVAL TIMECENSORED LAGGED TINTERVAL", Rejected, "CL14");
      add("FREQWEIGHT CONSTRAINT PATTERN STRATIFICATION CLUSTER WEIGHT WTSCALE BWEIGHT B2WEIGHT B3WEIGHT BWTSCALE REPWEIGHTS FINITE CLASSES KNOWNCLASS TRAINING WITHIN BETWEEN", Rejected, "CL15");
    } else if (command == "ANALYSIS") {
      add("TYPE", Schema, "CL17"); add("MODEL", Schema, "CL19"); add("ESTIMATOR", Reported, "CL18");
      add("PARAMETERIZATION", Rejected, "CL22", 3); add("INFORMATION", Reported, "CL25");
      add("DISTRIBUTION MATRIX", Reported, "CL23"); add("LINK", Rejected, "CL23");
      add("ALIGNMENT", Rejected, "CL21");
      add("ROTATION ROWSTANDARDIZATION PARALLEL REPSE MULTIPLIER BASEHAZARD RSTARTS RITERATIONS RCONVERGENCE ASTARTS AITERATIONS ACONVERGENCE SIMPLICITY TOLERANCE METRIC", Rejected, "CL24");
      add("CHOLESKY ALGORITHM INTEGRATION MCSEED ADAPTIVE BOOTSTRAP LRTBOOTSTRAP STARTS STITERATIONS STCONVERGENCE STSCALE STSEED OPTSEED K-1STARTS LRTSTARTS H1STARTS DIFFTEST COVERAGE ADDFREQUENCY ITERATIONS SDITERATIONS H1ITERATIONS MITERATIONS MCITERATIONS MUITERATIONS CONVERGENCE H1CONVERGENCE LOGCRITERION RLOGCRITERION MCONVERGENCE MCCONVERGENCE MUCONVERGENCE MIXC MIXU LOGHIGH LOGLOW UCELLSIZE VARIANCE POINT CHAINS BSEED STVALUES PREDICTOR BCONVERGENCE BITERATIONS FBITERATIONS THIN MDITERATIONS KOLMOGOROV PRIOR INTERACTIVE PROCESSORS NESTED", Reported, "CL25");
    }
    return result;
  }

  // production: grouping_value ::= name '(' grouping_pair (list_sep grouping_pair)* ')'
  void read_grouping(std::string_view value, SourceSpan at) {
    const auto fail = [&](std::string detail) {
      reject(at, "MG03", detail + "; Mplus supports data-dependent group forms, but magmaan imports explicit integer codes only; recode the grouping variable to integers in R and write g (1 = g1 2 = g2) instead");
    };
    const auto open = value.find('('), close = value.find(')');
    if (open == std::string_view::npos || close == std::string_view::npos || close < open) { fail("missing code = label pairs"); return; }
    const auto vars = words(value.substr(0, open));
    if (vars.size() != 1 || !valid_name(vars[0]) || !words(value.substr(close+1)).empty()) { fail("several grouping variables or trailing tokens"); return; }
    out.grouping_variable = vars[0];
    std::vector<std::pair<double, MplusGroup>> groups;
    std::set<std::string> labels;
    auto body = value.substr(open+1, close-open-1);
    for (std::size_t i=0; i<body.size();) {
      while (i<body.size() && (blank(body[i]) || body[i]==',')) ++i;
      if (i==body.size()) break;
      const auto begin=i;
      if (body[i]=='-' || body[i]=='+') ++i;
      while (i<body.size() && (digit(body[i]) || body[i]=='.')) ++i;
      double code=0;
      auto numeric=body.substr(begin,i-begin);
      const auto parsed=std::from_chars(numeric.data(),numeric.data()+numeric.size(),code);
      if (numeric.empty() || parsed.ec!=std::errc{} || parsed.ptr!=numeric.data()+numeric.size() || !std::isfinite(code) || std::trunc(code)!=code || std::abs(code)>2147483647) { fail("non-integral or invalid grouping code; Mplus requires integer grouping values and suggests DEFINE, which magmaan does not import"); return; }
      while (i<body.size() && blank(body[i])) ++i;
      if (i==body.size() || body[i]!='=') { fail("count, value list or range without labels; Mplus determines count-form g1, g2 labels from ascending data values"); return; }
      ++i; while (i<body.size() && blank(body[i])) ++i;
      const auto label_begin=i;
      while (i<body.size() && (letter(body[i]) || digit(body[i]) || body[i]=='_')) ++i;
      const std::string label(body.substr(label_begin,i-label_begin));
      if (!valid_name(label)) { fail("invalid group label"); return; }
      if (!labels.insert(upper(label)).second || std::any_of(groups.begin(),groups.end(),[&](const auto& g) {return g.first==code;})) { fail("duplicate grouping code or label"); return; }
      groups.push_back({code,{label,std::to_string(static_cast<long long>(code))}});
    }
    if (groups.empty()) { fail("empty GROUPING declaration"); return; }
    std::sort(groups.begin(),groups.end(),[](const auto& a,const auto& b) {return a.first<b.first;});
    for (auto& group:groups) out.groups.push_back(std::move(group.second));
  }

  // production: name_generator ::= name | name '-' name
  std::vector<Item> items(std::string_view value, SourceSpan at, const char* rule) {
    std::vector<Item> result;
    for (std::size_t i = 0; i < value.size();) {
      if (blank(value[i]) || value[i] == ',') { ++i; continue; }
      const auto begin = i;
      while (i < value.size() && (letter(value[i]) || digit(value[i]) || value[i] == '_')) ++i;
      std::string first(value.substr(begin, i - begin));
      if (!valid_name(first)) { reject(at, rule, "invalid variable name or list"); return {}; }
      while (i < value.size() && blank(value[i])) ++i;
      std::string last;
      if (i < value.size() && value[i] == '-') {
        ++i;
        while (i < value.size() && blank(value[i])) ++i;
        const auto start = i;
        while (i < value.size() && (letter(value[i]) || digit(value[i]) || value[i] == '_')) ++i;
        last = value.substr(start, i - start);
        if (!valid_name(last)) { reject(at, rule, "invalid range endpoint"); return {}; }
      }
      if (i < value.size() && !blank(value[i]) && value[i] != ',') {
        // Whitespace before a new name was already consumed above.
        if (!letter(value[i]) || last.size()) { reject(at, rule, "invalid list separator"); return {}; }
      }
      result.push_back({std::move(first), std::move(last)});
    }
    if (result.empty()) reject(at, rule, "empty variable list");
    return result;
  }

  // production: names_value ::= name_generator (list_sep name_generator)*
  void names(std::string_view value, SourceSpan at) {
    std::set<std::string> seen;
    const auto append = [&](std::string name) {
      if (out.names.size() >= 100000) { reject(at, "NM02", "NAMES generates more than 100000 variables; Mplus expands these ranges, but magmaan bounds expansion; list a smaller analysis schema instead"); return false; }
      if (!seen.insert(upper(name)).second) {
        reject(at, "NM02", "duplicate NAMES variable: " + name); return false;
      }
      out.names.push_back(std::move(name));
      return true;
    };
    for (const auto& item : items(value, at, "NM02")) {
      if (item.last.empty()) { if (!append(item.first)) return; continue; }
      // A range endpoint cannot fit an accepted physical line beyond 90 bytes.
      // Avoid amplifying an already-rejected enormous token during expansion.
      if (item.first.size() > 90 || item.last.size() > 90) {
        reject(at, "LX02", "NAMES range endpoint exceeds physical line limit"); continue;
      }
      auto a = item.first.size(), b = item.last.size();
      while (a && digit(item.first[a - 1])) --a;
      while (b && digit(item.last[b - 1])) --b;
      if (a < item.first.size() && b < item.last.size() && upper(item.first.substr(0, a)) == upper(item.last.substr(0, b))) {
        std::uint64_t first = 0, last = 0;
        const auto av = std::string_view(item.first).substr(a), bv = std::string_view(item.last).substr(b);
        const auto ar = std::from_chars(av.data(), av.data() + av.size(), first);
        const auto br = std::from_chars(bv.data(), bv.data() + bv.size(), last);
        if (ar.ec != std::errc{} || br.ec != std::errc{} || last < first || last - first > 10000) {
          reject(at, "NM02", "invalid or oversized NAMES range (maximum 10001)"); continue;
        }
        const auto width = av.size();
        for (auto n = first;; ++n) {
          auto suffix = std::to_string(n);
          if (!append(item.first.substr(0, a) + std::string(width > suffix.size() ? width - suffix.size() : 0, '0') + suffix)) return;
          if (n == last) break;
        }
      } else if (item.first.size() == item.last.size() &&
                 upper(item.first.substr(0, item.first.size() - 1)) == upper(item.last.substr(0, item.last.size() - 1)) &&
                 letter(item.first.back()) && letter(item.last.back())) {
        const char first = upper(item.first).back(), last = upper(item.last).back();
        if (last < first) { reject(at, "NM02", "backward NAMES range"); continue; }
        for (char c = first; c <= last; ++c) {
          const char suffix = item.first.back() >= 'a' && item.first.back() <= 'z' ? static_cast<char>(c - 'A' + 'a') : c;
          if (!append(item.first.substr(0, item.first.size() - 1) + suffix)) return;
        }
      } else reject(at, "NM02", "unsupported NAMES range shape '" + item.first + "-" + item.last + "': Mplus does not generate a1b, a2b, a3b from a1b-a3b; it produces names such as A01, A02, A03; magmaan avoids silently renaming columns; list the names instead");
    }
  }

  // production: usevariables_value ::= ('ALL' list_sep)? variable_range (list_sep variable_range)*
  void select() {
    if (!has_use) { out.analysis = out.names; return; }
    std::map<std::string, std::size_t> positions;
    for (std::size_t i = 0; i < out.names.size(); ++i) positions.emplace(upper(out.names[i]), i);
    std::set<std::size_t> selected;
    const auto append_range = [&](std::size_t first, std::size_t last) {
      for (auto i = first; i <= last; ++i) {
        if (!selected.insert(i).second) reject(use_span, "NM03", "duplicate USEVARIABLES name '" + out.names[i] + "'; Mplus requires a unique analysis list; remove the duplicate");
        else out.analysis.push_back(out.names[i]);
      }
    };
    const bool transformations = std::any_of(sections.begin(),sections.end(),[](const Section& section) { return section.command == "DEFINE" || (section.command == "DATA" && !section.qualifier.empty()); });
    for (std::size_t n = 0; n < use.size(); ++n) {
      const auto& item = use[n];
      if (n == 0 && upper(item.first) == "ALL" && item.last.empty()) {
        if (!out.names.empty()) append_range(0, out.names.size() - 1);
        continue;
      }
      auto a = positions.find(upper(item.first));
      auto b = item.last.empty() ? a : positions.find(upper(item.last));
      if (a == positions.end() || b == positions.end()) { reject(use_span, "NM03", "unknown USEVARIABLES name '" + (a == positions.end() ? item.first : item.last) + "'; Mplus resolves analysis names after transformations" + std::string(transformations ? "; DEFINE or a DATA transformation is present, so the name is probably created there" : "; the name is not in NAMES") + "; magmaan reads the supplied schema only; compute transformations in R and list the resulting names"); continue; }
      if (a->second > b->second) { reject(use_span, "NM03", "backward USEVARIABLES range"); continue; }
      append_range(a->second, b->second);
    }
  }

  // production: settings_value ::= setting+
  std::vector<std::string> settings(std::string_view value, std::string_view table, SourceSpan at, const std::string& rule) {
    std::vector<std::string> result;
    for (const auto& raw : words(value)) {
      auto full = setting(raw, table);
      if (full.empty()) reject(at, "LX03", "unknown setting '" + raw + "' for " + rule + "; Mplus accepts exact words or documented stems; accepted words/stems: " + std::string(table) + "; write one of these instead");
      else result.push_back(std::move(full));
    }
    if (result.empty() && words(value).empty()) reject(at, rule, "missing setting");
    return result;
  }

  // production: format_value ::= 'FREE' | format_group | format_item+
  void data_format(std::string_view value, SourceSpan at) {
    if (upper(value) == "FREE") return;
    std::size_t i = 0;
    const auto fail = [&] { reject(at,"DA01","malformed FORMAT '" + std::string(value) + "'; Mplus reads Fw.d/w.d, X, Tn, / and repeated groups; write FREE or a valid bounded fixed format instead"); };
    const auto skip = [&] { while (i<value.size() && (blank(value[i]) || value[i]==',')) ++i; };
    const auto integer = [&]() -> int {
      const auto start=i;
      while (i<value.size() && digit(value[i])) ++i;
      int result=0;
      if (start==i) return -1;
      const auto p=std::from_chars(value.data()+start,value.data()+i,result);
      return p.ec==std::errc{} && result<=10000 ? result : -1;
    };
    // production: format_group ::= '(' format_item (','? format_item)* ')'
    const auto group = [&](auto&& self, bool nested, int depth) -> bool {
      if (depth>32) return false;
      while (true) {
        skip();
        if (i==value.size()) return !nested;
        if (value[i]==')') { if (!nested) return false; ++i; return true; }
        int repeat=1, width=-1;
        const bool prefix=digit(value[i]);
        if (prefix) { repeat=integer(); if (repeat<1) return false; }
        if (i==value.size()) return false;
        const auto begin=out.data_plan.format.size();
        auto c=value[i]; if(c>='a' && c<='z') c=static_cast<char>(c-'a'+'A');
        if (c=='(') { ++i; if (!self(self,true,depth+1) || out.data_plan.format.size()==begin) return false; }
        else if (c=='X') { ++i; out.data_plan.format.push_back({MplusFormatKind::Skip,repeat,0}); repeat=1; }
        else if (c=='T') { if(prefix) return false; ++i; width=integer(); if(width<1) return false; out.data_plan.format.push_back({MplusFormatKind::Tab,width,0}); }
        else if (c=='/') { ++i; out.data_plan.format.push_back({MplusFormatKind::Record,0,0}); }
        else {
          if(c=='F') { ++i; width=integer(); }
          else if(c=='.' && prefix) {width=repeat;repeat=1;}
          else return false;
          if(width<1) return false;
          int decimals=0;
          if(i<value.size() && value[i]=='.') {++i;decimals=integer();}
          else if(c!='F') return false;
          if(decimals<0 || decimals>width) return false;
          out.data_plan.format.push_back({MplusFormatKind::Field,width,decimals});
        }
        const auto length=out.data_plan.format.size()-begin;
        if (length*static_cast<std::size_t>(repeat)>100000-out.data_plan.format.size()) return false;
        for(int r=1;r<repeat;++r) for(std::size_t j=0;j<length;++j) out.data_plan.format.push_back(out.data_plan.format[begin+j]);
      }
    };
    if(!group(group,false,0) || out.data_plan.format.empty() ||
        std::none_of(out.data_plan.format.begin(),out.data_plan.format.end(),[](const auto& f){return f.kind==MplusFormatKind::Field;})) fail();
  }

  // production: missing_value ::= '.' | '*' | 'BLANK' | (('ALL' | variable_range+) '(' missing_numbers ')')+
  void data_missing(std::string_view value, SourceSpan at) {
    const auto fail = [&] { reject(at,"DA03","malformed MISSING '" + std::string(value) + "'; Mplus accepts one global symbol or numeric flags per NAMES list/ALL; write * or ALL (-9) with optional numeric ranges instead"); };
    auto text=upper(value);
    if(text=="." || text=="*" || text=="BLANK") {out.data_plan.missing_symbol=text;return;}
    std::size_t i=0;
    while(i<value.size()) {
      while(i<value.size() && (blank(value[i]) || value[i]==',')) ++i;
      if(i==value.size()) break;
      const auto open=value.find('(',i), close=value.find(')',open);
      if(open==std::string_view::npos || close==std::string_view::npos) {fail();return;}
      auto vars=items(value.substr(i,open-i),at,"DA03");
      MplusMissingRule rule;
      // Keep variable ranges until the whole NAMES schema is available.
      for(const auto& v:vars) rule.variables.push_back(v.first+(v.last.empty()?"":"-"+v.last));
      auto numbers=value.substr(open+1,close-open-1);
      std::size_t j=0;
      const auto number = [&]() -> std::optional<double> {
        while(j<numbers.size() && blank(numbers[j])) ++j;
        const auto begin=j;
        if(j<numbers.size() && (numbers[j]=='-' || numbers[j]=='+')) ++j;
        while(j<numbers.size() && (digit(numbers[j]) || numbers[j]=='.')) ++j;
        double n=0; auto token=numbers.substr(begin,j-begin);
        if(token.starts_with('+')) token.remove_prefix(1);
        auto p=std::from_chars(token.data(),token.data()+token.size(),n);
        if(token.empty() || p.ec!=std::errc{} || p.ptr!=token.data()+token.size() || !std::isfinite(n)) return {};
        return n;
      };
      while(j<numbers.size()) {
        while(j<numbers.size() && (blank(numbers[j]) || numbers[j]==',')) ++j;
        if(j==numbers.size()) break;
        auto a=number(); if(!a) {fail();return;}
        const auto after_number=j;
        while(j<numbers.size() && blank(numbers[j])) ++j;
        const bool separated=j>after_number;
        if(j<numbers.size() && numbers[j]=='-') {
          ++j; auto b=number(); if(!b || *b<*a || *b-*a>100000 || std::trunc(*b-*a)!=*b-*a) {fail();return;}
          for(double n=*a;n<=*b;n+=1) rule.values.push_back(n);
        } else rule.values.push_back(*a);
        if(j<numbers.size() && !separated && !blank(numbers[j]) && numbers[j]!=',') {fail();return;}
      }
      if(rule.variables.empty() || rule.values.empty()) {fail();return;}
      out.data_plan.missing.push_back(std::move(rule));i=close+1;
    }
    if(out.data_plan.missing.empty()) fail();
  }

  // production: data_file ::= ('(' group_label ')' assign)? file_path
  void data_file(std::string_view value, SourceSpan at) {
    MplusDataFile file;
    if(value.starts_with('(')) {
      const auto close=value.find(')'); if(close==std::string_view::npos) {reject(at,"CL02","missing FILE label terminator; Mplus expects FILE (label) = path; write that form instead");return;}
      file.label=std::string(value.substr(1,close-1));
      if(!valid_name(file.label)) reject(at,"MG01","invalid FILE group label; Mplus requires a name; write FILE (g1) = path instead");
      value.remove_prefix(close+1);
      while(!value.empty() && blank(value.front())) value.remove_prefix(1);
      if(value.starts_with('=')) value.remove_prefix(1);
      else if(upper(value).starts_with("IS ")) value.remove_prefix(3);
      else {reject(at,"CL02","FILE label lacks assignment; Mplus expects FILE (label) = path; add = instead");return;}
    }
    while(!value.empty() && blank(value.front())) value.remove_prefix(1);
    while(!value.empty() && blank(value.back())) value.remove_suffix(1);
    if(value.empty()) {reject(at,"CL02","empty FILE path; Mplus reads a separate data file; supply a path instead");return;}
    if(value.front()=='\'' || value.front()=='"') {
      if(value.size()<2 || value.back()!=value.front()) {reject(at,"CL02","unclosed FILE quote; Mplus quotes paths with blanks; close the quote instead");return;}
      value=value.substr(1,value.size()-2);
    } else if(value.find_first_of(" \t\r\n")!=std::string_view::npos) reject(at,"CL02","unquoted FILE path with blanks; Mplus requires quotes; quote the path instead");
    file.path=value;out.data_plan.files.push_back(std::move(file));
  }

  // production: data_type ::= ('INDIVIDUAL' | 'COVARIANCE' | 'CORRELATION' | 'FULLCOV' | 'FULLCORR' | 'MEANS' | 'STDEVIATIONS')+
  void validate_data() {
    auto& plan=out.data_plan;const auto at=span(0,0);
    const auto fail=[&](std::string rule,std::string message){reject(at,std::move(rule),std::move(message)+"; Mplus requires a consistent data description; write one matching FILE/TYPE/NOBSERVATIONS plan instead");};
    if(!plan.files.empty()) {
      plan.file_groups=!plan.files.front().label.empty();
      std::set<std::string> labels;
      for(const auto& f:plan.files) if(f.label.empty()==plan.file_groups || (plan.file_groups && !labels.insert(upper(f.label)).second)) fail("MG01","mixed or duplicate FILE group labels");
      if(!plan.file_groups && plan.files.size()!=1) fail("CL02","repeated unlabelled FILE");
      if(plan.file_groups) {
        if(grouping) fail("MG01","FILE groups combined with GROUPING");
        if(data_groups && plan.n_groups!=static_cast<int>(plan.files.size())) fail("MG02","NGROUPS disagrees with FILE count");
        plan.n_groups=static_cast<int>(plan.files.size());
        for(const auto& f:plan.files) out.groups.push_back({f.label,""});
      }
    }
    if(data_groups && !plan.file_groups) {
      if(plan.matrix_type.empty() || grouping) fail("MG02","NGROUPS without summary data or combined with GROUPING");
      else for(int i=1;i<=plan.n_groups;++i) out.groups.push_back({"g"+std::to_string(i),""});
    }
    if(plan.file_groups || (data_groups && !out.groups.empty())) {
      out.grouping_variable=".mplus_group";
      if(std::any_of(out.names.begin(),out.names.end(),[](const auto& n){return upper(n)==".MPLUS_GROUP";})) fail("MG01","reserved .mplus_group column in NAMES");
    }
    if(!plan.matrix_type.empty()) {
      if(!plan.format.empty()) fail("DA02","fixed FORMAT with summary data");
      if(plan.n_observations.size()!=static_cast<std::size_t>(plan.n_groups)) fail("DA02","summary NOBSERVATIONS count differs from group count");
      out.nomeanstructure=!plan.means;
    } else if(plan.means || plan.standard_deviations) fail("CL03","MEANS/STDEVIATIONS without a covariance or correlation type");
    if(plan.standard_deviations && plan.matrix_type!="CORRELATION" && plan.matrix_type!="FULLCORR") fail("CL03","STDEVIATIONS without correlation");
    if(!plan.n_observations.empty() && plan.n_observations.size()!=static_cast<std::size_t>(plan.n_groups)) fail("CL03","NOBSERVATIONS count differs from data group count");
    if(plan.missing_symbol=="BLANK" && plan.format.empty()) fail("DA03","BLANK missing flag with free format");
    for(auto& rule:plan.missing) {
      std::vector<std::string> variables;
      for(const auto& v:rule.variables) {
        if(upper(v)=="ALL") {variables.insert(variables.end(),out.names.begin(),out.names.end());continue;}
        const auto dash=v.find('-'); auto first=v.substr(0,dash),last=dash==std::string::npos?first:v.substr(dash+1);
        auto a=std::find_if(out.names.begin(),out.names.end(),[&](const auto& n){return upper(n)==upper(first);});
        auto b=std::find_if(out.names.begin(),out.names.end(),[&](const auto& n){return upper(n)==upper(last);});
        if(a==out.names.end() || b==out.names.end() || b<a) {fail("DA03","unknown or reversed MISSING variable range");continue;}
        variables.insert(variables.end(),a,b+1);
      }
      rule.variables=std::move(variables);
    }
  }

  // production: option ::= option_name (assign? option_value)? ';'
  void option(const Section& section, std::size_t begin, std::size_t end, std::set<std::string>& seen) {
    while (begin < end && blank(clean[begin])) ++begin;
    if (begin == end) return;
    auto name_end = begin;
    while (name_end < end && !blank(clean[name_end]) && clean[name_end] != '=' && clean[name_end] != '(') ++name_end;
    const auto raw = clean.substr(begin, name_end - begin);
    const auto at = span(begin, end + 1);
    const auto table = specs(section.command);
    std::vector<std::string> choices;
    for (const auto& spec : table) choices.push_back(spec.name);
    auto name = resolve(raw, choices);
    if (name.empty()) {
      std::string accepted;
      for (const auto& choice : choices) { if (!accepted.empty()) accepted += ", "; accepted += choice; }
      reject(at, "LX03", "unknown or ambiguous option '" + raw + "'; Mplus resolves complete names or unique prefixes of at least four letters; accepted options: " + accepted + "; write a complete accepted name instead"); return;
    }
    const auto spec = *std::find_if(table.begin(), table.end(), [&](const OptionSpec& s) { return s.name == name; });
    if (spec.klass==MplusClass::DataDescription) diagnostic(spec.klass,at,spec.rule,name+" is parsed into the data plan; mplus_data() reads it");
    if (!seen.insert(name).second && !(section.command == "DATA" && name == "FILE")) reject(at, "LX01", "repeated option: " + name);
    auto value_begin = name_end;
    while (value_begin < end && blank(clean[value_begin])) ++value_begin;
    if (value_begin < end && clean[value_begin] == '=') ++value_begin;
    else {
      auto j = value_begin;
      while (j < end && letter(clean[j])) ++j;
      const auto assign = upper(std::string_view(clean).substr(value_begin, j - value_begin));
      if (assign == "IS" || assign == "ARE") value_begin = j;
    }
    while (value_begin < end && blank(clean[value_begin])) ++value_begin;
    const auto value = std::string_view(clean).substr(value_begin, end - value_begin);
    if (name == "GROUPING") { grouping = true; read_grouping(value, at); }
    if (name == "CLASSES" || name == "KNOWNCLASS") mixture = true;
    if (name == "NGROUPS") data_groups = true;
    if (name == "AUXILIARY" && value.find('(') != std::string_view::npos) {
      reject(at, "CL15", "AUXILIARY modifiers change the analysis"); return;
    }
    auto after_name = name_end;
    while (after_name < end && blank(clean[after_name])) ++after_name;
    if (section.command == "DATA" && name == "FILE") {
      data_file(std::string_view(clean).substr(after_name,end-after_name).starts_with('(') ?
          std::string_view(clean).substr(after_name,end-after_name) : value,at); return;
    }
    if (name == "PARAMETERIZATION" && section.command == "ANALYSIS") {
      const auto values = settings(value, "DELTA THETA LOGIT LOGLINEAR/LOGLIN PROBABILITY/PROB RESCOVARIANCES/RESCOV", at, "CL22");
      if (values.size() != 1) reject(at, "CL22", "expected one PARAMETERIZATION setting");
      for (const auto& v : values) reject(at, "CL22", v +
          (v == "DELTA" || v == "THETA" ? " not yet supported; planned for increment 3; supply continuous outcomes instead" : " is outside scope"));
      return;
    }
    if (name == "CATEGORICAL" && value.find('(') != std::string_view::npos) {
      reject(at, "CL10", "CATEGORICAL category-set forms are outside scope"); return;
    }
    if (spec.klass == MplusClass::Rejected) {
      reject(at, spec.rule, name + (spec.rule == "CL13" ? " selects cases in Mplus; magmaan cannot ignore sample selection; select cases in R before fitting" : spec.later ? " not yet supported; planned for increment " + std::to_string(spec.later) + "; supply an increment-1 continuous single-group model instead" : " is outside the supported model families"));
      return;
    }
    if (section.command == "VARIABLE") {
      if (name == "NAMES") names(value, at);
      else if (name == "USEVARIABLES") { has_use = true; use_span = at; use = items(value, at, "NM03"); }
    }
    if(section.command == "VARIABLE" && name == "MISSING") {data_missing(value,at);return;}
    if(section.command == "DATA") {
      if(name == "FORMAT") {data_format(value,at);return;}
      if(name == "NGROUPS" || name == "NOBSERVATIONS") {
        std::vector<std::int32_t> counts;
        for(const auto& word:words(value)) {
          int n=0;auto p=std::from_chars(word.data(),word.data()+word.size(),n);
          if(p.ec!=std::errc{} || p.ptr!=word.data()+word.size() || n<1 || (name=="NGROUPS" && n>1000)) reject(at,"CL03","invalid "+name+"; Mplus requires positive integer counts; write one count per group instead");
          else counts.push_back(n);
        }
        if(counts.empty()) reject(at,"CL03","empty "+name+"; Mplus requires counts; supply positive integers instead");
        if(name=="NGROUPS") {if(counts.size()!=1) reject(at,"MG02","NGROUPS requires one count; write one positive integer instead");else out.data_plan.n_groups=counts[0];}
        else out.data_plan.n_observations=std::move(counts);
        return;
      }
      if(name=="TYPE" || name=="LISTWISE") {
        const auto values=settings(value,name=="TYPE"?"INDIVIDUAL/IND COVARIANCE/COVA CORRELATION/CORR FULLCOV FULLCORR MEANS STDEVIATIONS/STD MONTECARLO/MONTE IMPUTATION/IMP":"ON OFF",at,spec.rule);
        if(name=="LISTWISE") {if(values.size()!=1) reject(at,"CL03","LISTWISE requires one setting; write ON or OFF instead");else out.data_plan.listwise=values[0]=="ON";}
        else {
          bool individual=false;
          for(const auto& v:values) {
            if(v=="MONTECARLO" || v=="IMPUTATION") reject(at,"CL04",v+" reads multiple datasets in Mplus; magmaan does not import them; supply one INDIVIDUAL dataset instead");
            else if(v=="MEANS") out.data_plan.means=true;
            else if(v=="STDEVIATIONS") out.data_plan.standard_deviations=true;
            else if(v=="INDIVIDUAL") individual=true;
            else if(!out.data_plan.matrix_type.empty()) reject(at,"CL03","multiple summary matrix types; Mplus requires one; write COVARIANCE or CORRELATION instead");
            else out.data_plan.matrix_type=v;
          }
          if(individual && values.size()!=1) reject(at,"CL03","INDIVIDUAL combined with summary TYPE; Mplus requires one data family; write INDIVIDUAL alone instead");
        }
        return;
      }
    }
    if (section.command == "ANALYSIS") {
      if (name == "TYPE") {
        auto type_value = value;
        const auto type_words = words(value);
        if (!type_words.empty() && upper(type_words.front()) == "EFA") type_value = value.substr(0, 3);
        for (auto& v : settings(type_value, "GENERAL/GEN BASIC/BAS RANDOM/RAND COMPLEX/COM MIXTURE/MIX TWOLEVEL/TWO THREELEVEL/THREE CROSSCLASSIFIED/CROSS EFA MISSING MEANSTRUCTURE H1", at, "CL17")) {
          if (v == "GENERAL" || v == "MISSING" || v == "MEANSTRUCTURE" || v == "H1") {
            out.type_settings.push_back(v);
            if (v != "GENERAL") diagnostic(MplusClass::Reported, at, "CL17", v + " is a legacy no-op");
          } else reject(at, "CL17", v + " analysis is outside scope");
        }
      } else if (name == "MODEL") {
        for (const auto& v : settings(value, "NOMEANSTRUCTURE/NOMEAN NOCOVARIANCES/NOCOV CONFIGURAL/CONFIG METRIC SCALAR ALLFREE/ALL", at, "CL19")) {
          if (v == "NOMEANSTRUCTURE") { out.nomeanstructure = true; nomean_span = at; }
          else if (v == "NOCOVARIANCES") out.nocovariances = true;
          else if (v != "ALLFREE") {
            if (!out.invariance.empty()) reject(at, "IV04", "Mplus runs multiple invariance models for a setting list; magmaan returns one model; write exactly one of CONFIGURAL, METRIC or SCALAR instead");
            out.invariance = v;
          }
          else reject(at, v == "ALLFREE" ? "CL21" : "CL20", v + (v == "ALLFREE" ? " is outside scope" : " not yet supported; planned for increment 2; fit groups separately in R instead"));
        }
      } else if (name == "ESTIMATOR" || name == "INFORMATION" || name == "DISTRIBUTION" || name == "MATRIX") {
        const std::string_view settings_table = name == "ESTIMATOR" ? "ML MLM MLMV MLR MLF MUML WLS WLSM WLSMV ULS ULSMV GLS BAYES" :
            name == "INFORMATION" ? "OBSERVED/OBS EXPECTED/EXP COMBINATION/COMB" :
            name == "DISTRIBUTION" ? "NORMAL/NORM SKEWNORMAL/SKEW TDISTRIBUTION/TDIST SKEWT" : "COVARIANCE/COVA CORRELATION/CORR";
        const auto values = settings(value, settings_table, at, spec.rule);
        if (values.size() != 1) reject(at, spec.rule, "expected one " + name + " setting");
        for (const auto& v : values) {
          if (name == "ESTIMATOR") {
            out.estimator = v;
            if (v == "BAYES" || v == "MUML") reject(at, "CL18", v + " estimator is outside scope");
          } else if (name == "INFORMATION") out.information = v;
          else if ((name == "DISTRIBUTION" && v != "NORMAL") || (name == "MATRIX" && v != "COVARIANCE"))
            reject(at, "CL23", v + " is outside scope");
        }
      }
    }
    if (spec.klass == MplusClass::Reported)
      diagnostic(spec.klass, at, spec.rule, name + " is recognized but not imported" +
          (name == "ADDFREQUENCY" ? "; changes polychoric inputs" : ""));
  }

  // production: option_body ::= option*
  void options(const Section& section) {
    std::set<std::string> seen;
    auto begin = section.body;
    char quote = 0;
    for (auto i = begin; i < section.end; ++i) {
      if (quote) { if (clean[i] == quote) quote = 0; continue; }
      if (clean[i] == '\'' || clean[i] == '"') { quote = clean[i]; continue; }
      if (clean[i] == ';') {
        if (section.command == "OUTPUT" || section.command == "SAVEDATA" || section.command == "PLOT") {
          auto start = begin;
          while (start < i && blank(clean[start])) ++start;
          if (start < i) diagnostic(MplusClass::Reported, span(start, i + 1), "CL30", section.command + " option is not imported");
        } else option(section, begin, i, seen);
        begin = i + 1;
      }
    }
    while (begin < section.end && blank(clean[begin])) ++begin;
    if (begin < section.end) reject(span(begin, section.end), "LX01", "option requires terminating semicolon");
  }

  // production: command_body ::= title_body | option_body | model_body | opaque_body
  void bodies() {
    // GROUPING may follow a MODEL label section; classify labels after all options.
    for (const auto& section : sections) if (section.qualifier.empty() &&
        (section.command == "DATA" || section.command == "VARIABLE" || section.command == "ANALYSIS" ||
         section.command == "OUTPUT" || section.command == "SAVEDATA" || section.command == "PLOT")) options(section);
    validate_data();
    for (const auto& section : sections) {
      const auto at = span(section.begin, section.end);
      const auto& c = section.command;
      if (c == "TITLE") diagnostic(MplusClass::Reported, at, "CL01", "TITLE is not imported");
      else if (c == "DEFINE") reject(at, "CL16", "DEFINE transforms or creates variables in Mplus; magmaan does not reproduce data transformations; compute them in R before fitting and remove DEFINE");
      else if (c == "MONTECARLO") reject(at, "CL29", "MONTECARLO is outside scope");
      else if (c == "DATA" && !section.qualifier.empty()) reject(at, "CL06", "DATA transformations are outside scope");
      else if (c == "MODEL") {
        if (section.qualifier.empty()) out.model_body = span(section.body, section.end);
        else {
          const auto dash = section.qualifier.find('-');
          const auto q = resolve(std::string_view(section.qualifier).substr(0, dash),
              {"CONSTRAINT", "INDIRECT", "TEST", "PRIORS", "POPULATION", "COVERAGE", "MISSING"});
          if (q == "TEST") diagnostic(MplusClass::Reported, at, "CL28", "MODEL TEST is not imported");
          else if (q == "CONSTRAINT" || q == "INDIRECT") reject(at, "CL27", "MODEL '" + q + "' not yet supported; planned for increment 4; write explicit BY/ON parameters instead");
          else if (q == "PRIORS") reject(at, "CL32", "MODEL PRIORS penalties are outside scope");
          else if (q == "POPULATION" || q == "COVERAGE" || q == "MISSING" ||
                   section.qualifier.starts_with("POPULATION-") || section.qualifier.starts_with("COVERAGE-") || section.qualifier.starts_with("MISSING-"))
            reject(at, "CL29", "simulation MODEL command is outside scope");
          else if (!out.groups.empty() && std::any_of(out.groups.begin(),out.groups.end(),[&](const auto& g) {return upper(g.label)==section.qualifier;}))
            out.group_sections.push_back({section.qualifier,span(section.body,section.end)});
          else if (!out.groups.empty()) reject(at,"MG06","unknown or multi-label MODEL group '" + section.qualifier + "'; Mplus requires one declared group label per section; write one GROUPING label after MODEL instead");
          else reject(at, mixture ? "MS10" : grouping || data_groups ? "CL26" : "CL31", mixture ? "MODEL '" + section.qualifier + "' is a mixture class section in Mplus; mixtures are outside scope; supply a single-group continuous model instead" : grouping || data_groups ? "MODEL '" + section.qualifier + "' is a group section in Mplus; not yet supported; planned for increment " + std::string(data_groups ? "5" : "2") + "; fit groups separately in R instead" : "MODEL '" + section.qualifier + "' may denote longitudinal invariance in Mplus; outside scope; write an unqualified MODEL section instead");
        }
      }
    }
    if (!out.grouping_variable.empty() && out.grouping_variable!=".mplus_group") {
      const auto name=std::find_if(out.names.begin(),out.names.end(),[&](const auto& n) {return upper(n)==upper(out.grouping_variable);});
      if (name==out.names.end()) reject(span(0,0),"MG03","grouping variable is absent from NAMES; Mplus uses the data schema; add it to NAMES instead");
      else out.grouping_variable=*name;
    }
    if (!out.invariance.empty() && out.groups.empty()) reject(span(0,0),"IV01","Mplus invariance shortcuts require GROUPING; magmaan cannot expand a single-group shortcut; add explicit GROUPING or write ordinary BY statements instead");
    for (std::size_t row = 0; row < lines.size(); ++row) {
      const auto begin = lines[row];
      const auto end = row + 1 < lines.size() ? lines[row + 1] - 1 : clean.size();
      auto content_begin = begin;
      while (content_begin < end && blank(clean[content_begin])) ++content_begin;
      bool title = false;
      for (const auto& section : sections) if (section.command == "TITLE" && content_begin >= section.begin && content_begin < section.end) title = true;
      if (!title && end > begin + 90 && std::any_of(clean.begin() + static_cast<std::ptrdiff_t>(begin + 90), clean.begin() + static_cast<std::ptrdiff_t>(end), [](char c) { return !blank(c); }))
        reject(span(begin + 90, end), "LX02", "content extends beyond column 90; Mplus truncates physical lines, which magmaan rejects; wrap the statement before column 91");
    }
    select();
    if (out.nomeanstructure && out.data_plan.matrix_type.empty() && out.information != "EXPECTED")
      reject(nomean_span, "MS11", "Mplus ignores NOMEANSTRUCTURE under its default observed information and keeps the means; magmaan does not reproduce this ignored setting; add INFORMATION = EXPECTED; or remove NOMEANSTRUCTURE");
  }

  parse_expected<MplusInput> finish() {
    const auto order = [](const MplusDiagnostic& a, const MplusDiagnostic& b) { return a.span.begin < b.span.begin; };
    std::stable_sort(out.notes.begin(), out.notes.end(), order);
    std::stable_sort(rejects.begin(), rejects.end(), order);
    if (!rejects.empty()) {
      std::string detail;
      for (const auto& d : rejects) {
        if (!detail.empty()) detail += '\n';
        detail += std::to_string(d.span.line) + ":" + std::to_string(d.span.col) + " [" + d.rule + "] " + d.message;
      }
      return std::unexpected(ParseError{ParseError::Kind::RejectedConstruct, rejects.front().span, std::move(detail)});
    }
    if (!out.grouping_variable.empty()) std::erase_if(out.analysis,[&](const auto& n) {return upper(n)==upper(out.grouping_variable);});
    return std::move(out);
  }
};
}  // namespace

// production: input_file ::= line*
parse_expected<MplusInput> MplusParser::read(std::string_view source) {
  if (source.size() >= std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(ParseError{ParseError::Kind::RejectedConstruct, {}, "1:1 [LX01] source exceeds span capacity"});
  Reader reader(source);
  reader.comments();
  reader.heads();
  reader.bodies();
  return reader.finish();
}
}  // namespace magmaan::parse
