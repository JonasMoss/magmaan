#include "magmaan/spec/composite_expand.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::spec {

namespace {

PartableError comp_err(PartableError::Kind k, std::string detail) {
  return PartableError{k, std::move(detail)};
}

// Deep-clone a constraint's expression tree. `parse::Expr` holds move-only
// `unique_ptr` children, so `std::vector<Constraint>` is not copyable — the
// expanded FlatPartable must rebuild each constraint by hand. The `Param` /
// `name` string_views are copied as-is (they keep borrowing from the input).
parse::Expr clone_expr(const parse::Expr& e) {
  return std::visit(
      [](const auto& node) -> parse::Expr {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, parse::BinNode>) {
          return parse::Expr{parse::BinNode{
              node.op, std::make_unique<parse::Expr>(clone_expr(*node.lhs)),
              std::make_unique<parse::Expr>(clone_expr(*node.rhs))}};
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          return parse::Expr{parse::UnNode{
              node.op, std::make_unique<parse::Expr>(clone_expr(*node.arg))}};
        } else {
          return parse::Expr{node};  // Num / Param — trivially copyable leaves
        }
      },
      e);
}

parse::Constraint clone_constraint(const parse::Constraint& c) {
  return parse::Constraint{c.kind, c.name, clone_expr(c.lhs), clone_expr(c.rhs),
                           c.span};
}

// A composite under construction. `name`/`ind`/`exc` own their strings (they
// flow into the returned CompositeInfo); the `_sv` views are what the
// synthesized FlatRows borrow — `name_sv`/`ind_sv` borrow from the input
// FlatPartable, `exc_sv` points into CompositeExpansion::flat.source_text.
struct CompositeBuild {
  std::string                   name;
  std::string_view              name_sv;
  std::vector<std::string>      ind;
  std::vector<std::string_view> ind_sv;
  std::vector<std::string>      exc;
  std::vector<std::string_view> exc_sv;
};

// All latents in the expanded model — needed to pin every excrescent latent
// orthogonal to every other latent.
struct LatentRef {
  std::string      name;
  std::string_view sv;
  bool             excrescent = false;
};

}  // namespace

partable_expected<CompositeExpansion>
expand_composites(const parse::FlatPartable& flat) {
  using parse::Op;

  // --- 1. collect composites, first-appearance order, capturing views -------
  std::vector<CompositeBuild> comps;
  std::unordered_map<std::string, std::size_t> comp_idx;
  for (const auto& r : flat.rows) {
    if (r.op != Op::Composite) continue;
    const std::string lhs(r.lhs);
    auto it = comp_idx.find(lhs);
    std::size_t ci = 0;
    if (it == comp_idx.end()) {
      ci = comps.size();
      comp_idx.emplace(lhs, ci);
      CompositeBuild cb;
      cb.name    = lhs;
      cb.name_sv = r.lhs;
      comps.push_back(std::move(cb));
    } else {
      ci = it->second;
    }
    comps[ci].ind.emplace_back(r.rhs);
    comps[ci].ind_sv.push_back(r.rhs);
  }

  // With no composites the steps below simply rebuild `flat` verbatim (an
  // effective no-op); `spec::build` never calls us in that case anyway.
  CompositeExpansion out;

  // --- 2. structural validation ---------------------------------------------
  std::unordered_map<std::string, std::string> indicator_owner;
  for (const auto& c : comps) {
    if (c.ind.size() < 2) {
      return std::unexpected(comp_err(
          PartableError::Kind::CompositeTooFewIndicators,
          "composite '" + c.name + "' has " + std::to_string(c.ind.size()) +
              " indicator(s); a composite needs at least 2"));
    }
    for (const auto& ind : c.ind) {
      auto [it, inserted] = indicator_owner.emplace(ind, c.name);
      if (!inserted) {
        return std::unexpected(comp_err(
            PartableError::Kind::CompositeOverlap,
            "indicator '" + ind + "' is shared by composites '" + it->second +
                "' and '" + c.name +
                "'; composite indicator sets must be disjoint"));
      }
    }
  }
  for (const auto& c : comps) {
    if (indicator_owner.count(c.name) != 0) {
      return std::unexpected(comp_err(
          PartableError::Kind::CompositeOverlap,
          "composite '" + c.name +
              "' is itself a composite indicator; nested composites are not "
              "supported"));
    }
  }
  for (const auto& r : flat.rows) {
    if (r.op == Op::Measurement && comp_idx.count(std::string(r.lhs)) != 0) {
      return std::unexpected(comp_err(
          PartableError::Kind::CompositeOverlap,
          "'" + std::string(r.lhs) +
              "' is declared both as a composite (<~) and a latent factor "
              "(=~)"));
    }
  }

  // --- 2b. connectivity --------------------------------------------------- -
  // The covariances among a composite's own indicators are unrestricted, so
  // they carry no information about the weights — only a relation to an
  // *external* variable identifies them. Each composite must therefore appear
  // in at least one `~` or `~~` row together with a variable that is neither
  // itself nor one of its own indicators.
  for (const auto& c : comps) {
    const std::unordered_set<std::string> own(c.ind.begin(), c.ind.end());
    bool connected = false;
    for (const auto& r : flat.rows) {
      if (r.op != Op::Regression && r.op != Op::Covariance) continue;
      const bool lhs_is_c = (r.lhs == c.name);
      const bool rhs_is_c = (r.rhs == c.name);
      if (!lhs_is_c && !rhs_is_c) continue;
      const std::string_view other = lhs_is_c ? r.rhs : r.lhs;
      if (other == c.name) continue;                       // `C ~~ C` variance
      if (own.count(std::string(other)) != 0) continue;    // own indicator
      connected = true;
      break;
    }
    if (!connected) {
      return std::unexpected(comp_err(
          PartableError::Kind::UnidentifiedComposite,
          "composite '" + c.name +
              "' is not related to any variable outside its own indicators; "
              "the Henseler-Ogasawara specification needs at least one `~` or "
              "`~~` link to an external variable to identify its weights"));
    }
  }

  // --- 3. synthesize excrescent latent names into the owned source buffer ---
  std::string synth;
  struct Slice { std::size_t off; std::size_t len; };
  std::vector<std::vector<Slice>> exc_slice(comps.size());
  for (std::size_t ci = 0; ci < comps.size(); ++ci) {
    CompositeBuild& c = comps[ci];
    const std::size_t k = c.ind.size();
    for (std::size_t j = 1; j < k; ++j) {
      std::string name = ".exc." + c.name + "." + std::to_string(j);
      exc_slice[ci].push_back(Slice{synth.size(), name.size()});
      synth.append(name);
      synth.push_back(' ');  // separator keeps the interned tokens distinct
      c.exc.push_back(std::move(name));
    }
  }
  out.flat.source_text.assign(synth.begin(), synth.end());
  const char* base = out.flat.source_text.data();
  for (std::size_t ci = 0; ci < comps.size(); ++ci) {
    for (const Slice& s : exc_slice[ci]) {
      comps[ci].exc_sv.push_back(std::string_view(base + s.off, s.len));
    }
  }

  // --- 4. modifier table: originals + the three synthesized fixes -----------
  out.flat.mods = flat.mods;
  const std::uint32_t m_zero = static_cast<std::uint32_t>(out.flat.mods.size());
  out.flat.mods.push_back(parse::Modifier{parse::FixedValue{0.0}});
  const std::uint32_t m_one = static_cast<std::uint32_t>(out.flat.mods.size());
  out.flat.mods.push_back(parse::Modifier{parse::FixedValue{1.0}});
  const std::uint32_t m_free = static_cast<std::uint32_t>(out.flat.mods.size());
  out.flat.mods.push_back(parse::Modifier{parse::Free{}});

  // Constraints carry move-only Expr trees — deep-clone (string_views still
  // borrow from `flat`, per the header lifetime note).
  for (const auto& c : flat.constraints) {
    out.flat.constraints.push_back(clone_constraint(c));
  }

  // --- 5. per-composite H-O blocks (=~ loadings, zero residuals, variances) -
  std::vector<std::vector<parse::FlatRow>> ho_block(comps.size());
  for (std::size_t ci = 0; ci < comps.size(); ++ci) {
    const CompositeBuild& c = comps[ci];
    const std::size_t k = c.ind.size();
    std::vector<std::string_view> lat;  // latent column views: emergent, then excrescent
    lat.reserve(k);
    lat.push_back(c.name_sv);
    for (std::string_view e : c.exc_sv) lat.push_back(e);

    std::vector<parse::FlatRow>& block = ho_block[ci];
    // K×K loading block, column-major (emergent column first).
    for (std::size_t j = 0; j < k; ++j) {
      for (std::size_t i = 0; i < k; ++i) {
        std::uint32_t mod = m_free;
        if (j == 0) {
          mod = (i + 1 == k) ? m_one : m_free;  // emergent: last loading = 1
        } else if (i + 1 < j) {
          mod = m_zero;                          // excrescent cascading zero
        } else if (i + 1 == j) {
          mod = m_one;                           // excrescent scale fix
        }
        block.push_back(parse::FlatRow{lat[j], Op::Measurement, c.ind_sv[i],
                                       1u, mod, {}});
      }
    }
    // Zero indicator residuals — an exact weighted sum has no measurement error.
    for (std::size_t i = 0; i < k; ++i) {
      block.push_back(parse::FlatRow{c.ind_sv[i], Op::Covariance, c.ind_sv[i],
                                     1u, m_zero, {}});
    }
    // Explicit free latent variances, so std.lv / auto.var leave them alone.
    for (std::size_t j = 0; j < k; ++j) {
      block.push_back(parse::FlatRow{lat[j], Op::Covariance, lat[j], 1u,
                                     m_free, {}});
    }
  }

  // --- 6. global orthogonality: every excrescent latent ⊥ every other latent -
  std::vector<LatentRef> all_lat;
  std::unordered_map<std::string, std::size_t> lat_seen;
  auto add_lat = [&](std::string nm, std::string_view sv, bool excrescent) {
    if (lat_seen.count(nm) != 0) return;
    lat_seen.emplace(nm, all_lat.size());
    all_lat.push_back(LatentRef{std::move(nm), sv, excrescent});
  };
  for (const auto& r : flat.rows) {
    if (r.op == Op::Measurement) add_lat(std::string(r.lhs), r.lhs, false);
  }
  for (const CompositeBuild& c : comps) {
    add_lat(c.name, c.name_sv, false);
    for (std::size_t j = 0; j < c.exc.size(); ++j) {
      add_lat(c.exc[j], c.exc_sv[j], true);
    }
  }
  std::vector<parse::FlatRow> ortho;
  for (std::size_t a = 0; a < all_lat.size(); ++a) {
    for (std::size_t b = a + 1; b < all_lat.size(); ++b) {
      if (!all_lat[a].excrescent && !all_lat[b].excrescent) continue;
      ortho.push_back(parse::FlatRow{all_lat[a].sv, Op::Covariance,
                                     all_lat[b].sv, 1u, m_zero, {}});
    }
  }

  // --- 7. assemble the expanded row list ------------------------------------
  // Non-composite rows pass through in place; each composite's H-O block is
  // spliced in at the position of its first `<~` row; the `<~` rows are dropped.
  std::vector<bool> emitted(comps.size(), false);
  for (const auto& r : flat.rows) {
    if (r.op == Op::Composite) {
      const std::size_t ci = comp_idx.at(std::string(r.lhs));
      if (!emitted[ci]) {
        emitted[ci] = true;
        for (const parse::FlatRow& hr : ho_block[ci]) out.flat.rows.push_back(hr);
      }
      continue;
    }
    out.flat.rows.push_back(r);  // borrows views from `flat`
  }
  for (const parse::FlatRow& orow : ortho) out.flat.rows.push_back(orow);

  // --- 8. composite metadata ------------------------------------------------
  for (const CompositeBuild& c : comps) {
    CompositeInfo info;
    info.composite  = c.name;
    info.indicators = c.ind;
    info.excrescent = c.exc;
    out.composites.push_back(std::move(info));
  }
  return out;
}

}  // namespace magmaan::spec
