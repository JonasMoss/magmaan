#pragma once

// Internal: variable classification + canonical orderings, shared by
// `lavaanify` (which reads the parsed FlatPartable) and `from_lavaan_partable`
// (which reads an already-resolved lavaan-shaped partable). Both need the same
// answer: which name is an indicator / ov.y / ov.x / ov.misc / latent, and the
// observed / extended-latent orderings that `matrix_rep` consumes by var id.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"  // VarRole

namespace magmaan::spec::detail {

// Insertion-order-preserving set of strings (lavaan's variable ordering
// follows order of appearance in the model syntax).
struct OrderedSet {
  std::vector<std::string> items;

  bool contains(std::string_view s) const noexcept {
    for (const auto& it : items) if (it == s) return true;
    return false;
  }
  void insert(std::string_view s) {
    if (!contains(s)) items.emplace_back(s);
  }
};

//   lv     : LHS of `=~` or native `<~`
//   ov_ind : RHS of `=~` or native `<~` (observed indicators)
//   ov_y   : LHS of `~` / `~1` (endogenous observed) — minus latents
//   ov_x   : RHS of `~` (exogenous observed) — minus latents, indicators, ov.y
//   ov_misc: observed mentioned only in `~~` rows (no other role)
struct VarSets {
  OrderedSet lv;
  OrderedSet ov_ind;
  OrderedSet ov_y;
  OrderedSet ov_x;
  OrderedSet ov_misc;
};

// Classify variables from any sequence of rows exposing `.op`, `.lhs`, `.rhs`
// (a `parse::FlatPartable::rows` or a lavaan-shaped partable's formula rows;
// constraint rows are harmless — their `.lhs`/`.rhs` are expression text, never
// a `=~`/`~`/`~~`/`~1` op, so they fall through every bucket).
template <class RowSeq>
VarSets classify_vars(const RowSeq& rows) {
  VarSets v;
  // Every LHS of `=~` is a latent. Collected in a pass of its own, *before*
  // the indicator pass, so that a latent which is itself measured by a
  // higher-order factor (`higher =~ lower`) is recognized as a latent no
  // matter where its own `lower =~ ...` rows sit in the syntax.
  for (const auto& r : rows) {
    if (r.op == parse::Op::Measurement || r.op == parse::Op::Composite)
      v.lv.insert(r.lhs);
  }
  // RHS of `=~` is an observed indicator — UNLESS it is itself a latent, in
  // which case the row is a higher-order measurement: the RHS stays latent
  // and `matrix_rep` lowers the row to a latent-on-latent Β path.
  for (const auto& r : rows) {
    if ((r.op == parse::Op::Measurement || r.op == parse::Op::Composite) &&
        !v.lv.contains(r.rhs))
      v.ov_ind.insert(r.rhs);
  }
  for (const auto& r : rows) {
    if (r.op == parse::Op::Regression || r.op == parse::Op::Intercept) {
      if (!v.lv.contains(r.lhs)) v.ov_y.insert(r.lhs);
    }
  }
  for (const auto& r : rows) {
    if (r.op == parse::Op::Regression) {
      if (!v.lv.contains(r.rhs) && !v.ov_ind.contains(r.rhs) &&
          !v.ov_y.contains(r.rhs)) {
        v.ov_x.insert(r.rhs);
      }
    }
  }
  for (const auto& r : rows) {
    if (r.op != parse::Op::Covariance) continue;
    for (std::string_view name : {std::string_view(r.lhs), std::string_view(r.rhs)}) {
      if (name.empty()) continue;
      if (!v.lv.contains(name) && !v.ov_ind.contains(name) &&
          !v.ov_y.contains(name) && !v.ov_x.contains(name)) {
        v.ov_misc.insert(name);
      }
    }
  }
  for (const auto& r : rows) {
    if (r.op != parse::Op::Threshold && r.op != parse::Op::ResponseScale) continue;
    if (!r.lhs.empty() && !v.lv.contains(r.lhs) && !v.ov_ind.contains(r.lhs) &&
        !v.ov_y.contains(r.lhs) && !v.ov_x.contains(r.lhs)) {
      v.ov_misc.insert(r.lhs);
    }
  }
  return v;
}

// Assigns a stable id to every distinct variable (id = order of first
// appearance scanning [ov.ind, ov.y, ov.x, ov.misc, lv]) and builds the
// canonical observed / extended-latent orderings (var ids). `reduced` = "the
// model has any `~` row" — then ov.y / ov.x are promoted into the extended
// latent set (phantom latents).
struct VarInventory {
  std::vector<std::string>    names;        // id → name
  std::vector<VarRole>        roles;        // id → first-bucket role
  std::vector<std::int8_t>    is_user_lv;   // id → 1 if in the `lv` set
  std::vector<std::int32_t>   ov_order;     // ids, canonical observed order
  std::vector<std::int32_t>   lv_ext_order; // ids, extended-latent order
  std::unordered_map<std::string, std::int32_t> id_of;

  std::int32_t lookup(std::string_view name) const {
    auto it = id_of.find(std::string(name));
    return (it == id_of.end()) ? -1 : it->second;
  }
};

inline VarInventory build_var_inventory(const VarSets& v, bool reduced) {
  VarInventory inv;
  auto ensure = [&](const std::string& name, VarRole role) -> std::int32_t {
    auto it = inv.id_of.find(name);
    if (it != inv.id_of.end()) return it->second;
    const std::int32_t id = static_cast<std::int32_t>(inv.names.size());
    inv.names.push_back(name);
    inv.roles.push_back(role);
    inv.is_user_lv.push_back(0);
    inv.id_of.emplace(name, id);
    return id;
  };
  for (const auto& s : v.ov_ind.items)  ensure(s, VarRole::Indicator);
  for (const auto& s : v.ov_y.items)    ensure(s, VarRole::EndoOv);
  for (const auto& s : v.ov_x.items)    ensure(s, VarRole::ExoOv);
  for (const auto& s : v.ov_misc.items) ensure(s, VarRole::MiscOv);
  for (const auto& s : v.lv.items) {
    const std::int32_t id = ensure(s, VarRole::Latent);
    inv.is_user_lv[static_cast<std::size_t>(id)] = 1;
  }

  auto append_unique = [&](std::vector<std::int32_t>& dst, const std::string& name) {
    const std::int32_t id = inv.id_of.at(name);
    for (auto x : dst) if (x == id) return;
    dst.push_back(id);
  };
  // Observed order — mirrors matrix_rep's historic classify():
  //   Reduced & no user latents : ov.y, then ov.x.
  //   else                      : ov.ind, ov.y, ov.x, ov.misc.
  if (reduced && v.lv.items.empty()) {
    for (const auto& s : v.ov_y.items) append_unique(inv.ov_order, s);
    for (const auto& s : v.ov_x.items) append_unique(inv.ov_order, s);
  } else {
    for (const auto& s : v.ov_ind.items)  append_unique(inv.ov_order, s);
    for (const auto& s : v.ov_y.items)    append_unique(inv.ov_order, s);
    for (const auto& s : v.ov_x.items)    append_unique(inv.ov_order, s);
    for (const auto& s : v.ov_misc.items) append_unique(inv.ov_order, s);
  }
  // Extended-latent order: user latents, then (Reduced only) ov.y, ov.x.
  for (const auto& s : v.lv.items) append_unique(inv.lv_ext_order, s);
  if (reduced) {
    for (const auto& s : v.ov_y.items) append_unique(inv.lv_ext_order, s);
    for (const auto& s : v.ov_x.items) append_unique(inv.lv_ext_order, s);
  }
  return inv;
}

}  // namespace magmaan::spec::detail
