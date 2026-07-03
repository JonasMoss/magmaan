#pragma once

// Private SAM helper: split a joint lavaanified model into per-block measurement
// CFA sub-specs and a structural sub-spec (latents promoted to observed). Works
// on the lavaan-shaped `LavaanParTable` projection — the same name-based surface
// lavaan's own `sam()` subsets — so `from_lavaan_partable` re-derives variable
// roles, orderings, and the equality reparameterization for each sub-model
// without re-firing lavaanify auto-add. Single group only.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "magmaan/compat/lavaan/partable_view.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::estimate::frontier::detail {

using compat::lavaan::LavaanParTable;

struct MeasurementBlockSpec {
  LavaanParTable           pt;
  std::vector<std::string> latents;
  std::vector<std::string> indicators;
};

struct SamPartition {
  std::vector<MeasurementBlockSpec> mm;            // one per block (1 if global)
  LavaanParTable                    structural;    // latents promoted to observed
  std::vector<std::string>          latent_order;  // all user latents, first-seen
};

// Copy the rows of `full` selected by `keep` into a fresh table, renumbering the
// surviving positive `free` indices to a contiguous 1..k (distinct original
// index → distinct new index, preserving auto-equality that rides on a shared
// index) and resetting id/block/group to a single group.
inline LavaanParTable
subset_rows(const LavaanParTable& full, const std::vector<char>& keep) {
  LavaanParTable out;
  std::unordered_map<std::int32_t, std::int32_t> free_map;
  std::int32_t next_free = 0;
  const std::size_t n = full.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (!keep[i]) continue;
    std::int32_t f = full.free[i];
    if (f > 0) {
      auto it = free_map.find(f);
      if (it == free_map.end()) f = free_map[full.free[i]] = ++next_free;
      else                      f = it->second;
    }
    out.id.push_back(static_cast<std::int32_t>(out.id.size()) + 1);
    out.user.push_back(full.user[i]);
    out.lhs.push_back(full.lhs[i]);
    out.op.push_back(full.op[i]);
    out.rhs.push_back(full.rhs[i]);
    out.block.push_back(1);
    out.group.push_back(1);
    out.free.push_back(f);
    out.exo.push_back(full.exo[i]);
    out.ustart.push_back(full.ustart[i]);
    out.label.push_back(full.label[i]);
    out.plabel.push_back(full.plabel[i]);
  }
  out.group_labels = full.group_labels.empty()
                         ? std::vector<std::string>{}
                         : std::vector<std::string>{full.group_labels[0]};
  return out;
}

// Append a fresh free row `lhs op rhs` (start value `ustart`) to `pt`, using the
// next free index. Used to free a fixed latent variance (free.fixed.var) or add
// an exogenous-latent covariance (add.exo.cov) in the structural sub-spec.
inline void
append_free_row(LavaanParTable& pt, const std::string& lhs, parse::Op op,
                const std::string& rhs, double ustart) {
  std::int32_t next_free = 0;
  for (auto f : pt.free) next_free = std::max(next_free, f);
  pt.id.push_back(static_cast<std::int32_t>(pt.id.size()) + 1);
  pt.user.push_back(0);
  pt.lhs.push_back(lhs);
  pt.op.push_back(op);
  pt.rhs.push_back(rhs);
  pt.block.push_back(1);
  pt.group.push_back(1);
  pt.free.push_back(next_free + 1);
  pt.exo.push_back(0);
  pt.ustart.push_back(ustart);
  pt.label.push_back("");
  pt.plabel.push_back("");
}

// Union-find over latent indices for measurement-block detection.
struct UnionFind {
  std::vector<int> parent;
  explicit UnionFind(int n) : parent(static_cast<std::size_t>(n)) {
    for (int i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i;
  }
  int find(int a) {
    std::size_t ua = static_cast<std::size_t>(a);
    while (parent[ua] != a) {
      parent[ua] = parent[static_cast<std::size_t>(parent[ua])];
      a = parent[ua];
      ua = static_cast<std::size_t>(a);
    }
    return a;
  }
  void unite(int a, int b) {
    parent[static_cast<std::size_t>(find(a))] = find(b);
  }
};

// Partition `full` (a single-group joint model) into measurement blocks and a
// structural sub-spec. `global == true` forces one measurement block over all
// user latents.
inline SamPartition
partition_model(const LavaanParTable& full, bool global, bool meanstructure) {
  const std::size_t n = full.size();

  // 1. User latents (lhs of `=~`), first-seen order, and their indicators.
  std::vector<std::string> latents;
  std::unordered_map<std::string, int> latent_idx;
  std::unordered_map<std::string, std::vector<std::string>> indicators_of;
  std::unordered_map<std::string, std::vector<int>> latents_of_indicator;
  for (std::size_t i = 0; i < n; ++i) {
    if (full.op[i] != parse::Op::Measurement) continue;
    const std::string& lv = full.lhs[i];
    if (!latent_idx.count(lv)) {
      latent_idx[lv] = static_cast<int>(latents.size());
      latents.push_back(lv);
    }
    indicators_of[lv].push_back(full.rhs[i]);
    latents_of_indicator[full.rhs[i]].push_back(latent_idx[lv]);
  }
  const int L = static_cast<int>(latents.size());
  std::unordered_set<std::string> latent_set(latents.begin(), latents.end());

  // Indicator ownership (union over cross-loadings handled below).
  auto is_indicator = [&](const std::string& v) {
    return latents_of_indicator.count(v) > 0;
  };

  // 2. Latent adjacency: shared indicator (cross-loading) or cross-block
  //    residual covariance between indicators of different latents.
  UnionFind uf(std::max(L, 1));
  for (const auto& [ind, lvs] : latents_of_indicator) {
    for (std::size_t k = 1; k < lvs.size(); ++k) uf.unite(lvs[0], lvs[k]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (full.op[i] != parse::Op::Covariance) continue;
    const std::string& a = full.lhs[i];
    const std::string& b = full.rhs[i];
    if (a == b || !is_indicator(a) || !is_indicator(b)) continue;
    uf.unite(latents_of_indicator[a][0], latents_of_indicator[b][0]);
  }
  if (global) for (int i = 1; i < L; ++i) uf.unite(0, i);

  // 3. Group latents into blocks by component root, preserving first-seen order.
  std::unordered_map<int, int> block_of_root;
  std::vector<std::vector<int>> block_latents;
  for (int i = 0; i < L; ++i) {
    int r = uf.find(i);
    auto it = block_of_root.find(r);
    if (it == block_of_root.end()) {
      block_of_root[r] = static_cast<int>(block_latents.size());
      block_latents.emplace_back();
    }
    block_latents[static_cast<std::size_t>(block_of_root[r])].push_back(i);
  }

  SamPartition part;
  part.latent_order = latents;

  // 4. Per-block measurement sub-spec.
  for (const auto& blk : block_latents) {
    std::unordered_set<std::string> blk_lv, blk_ind;
    for (int li : blk) {
      const std::string& lvn = latents[static_cast<std::size_t>(li)];
      blk_lv.insert(lvn);
      for (const auto& ind : indicators_of[lvn]) blk_ind.insert(ind);
    }
    std::vector<char> keep(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const parse::Op op = full.op[i];
      const std::string& lhs = full.lhs[i];
      const std::string& rhs = full.rhs[i];
      if (op == parse::Op::Measurement) {
        keep[i] = blk_lv.count(lhs) ? 1 : 0;
      } else if (op == parse::Op::Covariance) {
        const bool ind_pair = blk_ind.count(lhs) && blk_ind.count(rhs);
        const bool lv_pair = blk_lv.count(lhs) && blk_lv.count(rhs);
        keep[i] = (ind_pair || lv_pair) ? 1 : 0;
      } else if (op == parse::Op::Intercept && meanstructure) {
        keep[i] = blk_ind.count(lhs) ? 1 : 0;   // indicator intercepts (Nu)
      }
    }
    MeasurementBlockSpec mb;
    mb.pt = subset_rows(full, keep);
    for (int li : blk) {
      const std::string& lvn = latents[static_cast<std::size_t>(li)];
      mb.latents.push_back(lvn);
      for (const auto& ind : indicators_of[lvn]) mb.indicators.push_back(ind);
    }
    part.mm.push_back(std::move(mb));
  }

  // 5. Structural sub-spec: regressions and covariances among latents (and, with
  //    a mean structure, latent intercepts). Latents carry no `=~` rows here, so
  //    from_lavaan_partable classifies them as observed.
  {
    std::vector<char> keep(n, 0);
    std::unordered_set<std::string> endo_lv;   // latents that are regressed (lhs of ~)
    for (std::size_t i = 0; i < n; ++i) {
      const parse::Op op = full.op[i];
      const std::string& lhs = full.lhs[i];
      const std::string& rhs = full.rhs[i];
      if (op == parse::Op::Regression && latent_set.count(lhs) &&
          latent_set.count(rhs)) {
        keep[i] = 1;
        endo_lv.insert(lhs);
      } else if (op == parse::Op::Covariance && latent_set.count(lhs) &&
                 latent_set.count(rhs)) {
        keep[i] = 1;
      } else if (op == parse::Op::Intercept && meanstructure &&
                 latent_set.count(lhs)) {
        keep[i] = 1;
      }
    }
    LavaanParTable st = subset_rows(full, keep);

    // free.fixed.var: ensure every latent has a free self-variance row.
    for (const auto& lv : latents) {
      bool has_var = false;
      for (std::size_t i = 0; i < st.size(); ++i) {
        if (st.op[i] == parse::Op::Covariance && st.lhs[i] == lv &&
            st.rhs[i] == lv) {
          has_var = true;
          if (st.free[i] == 0) {   // was fixed for identification (std.lv)
            std::int32_t nf = 0;
            for (auto f : st.free) nf = std::max(nf, f);
            st.free[i] = nf + 1;
            st.ustart[i] = std::numeric_limits<double>::quiet_NaN();
          }
          break;
        }
      }
      if (!has_var) append_free_row(st, lv, parse::Op::Covariance, lv, 1.0);
    }

    // add.exo.cov: free covariances among exogenous latents (never regressed).
    std::vector<std::string> exo_lv;
    for (const auto& lv : latents)
      if (!endo_lv.count(lv)) exo_lv.push_back(lv);
    for (std::size_t a = 0; a < exo_lv.size(); ++a) {
      for (std::size_t b = a + 1; b < exo_lv.size(); ++b) {
        bool has_cov = false;
        for (std::size_t i = 0; i < st.size(); ++i) {
          if (st.op[i] == parse::Op::Covariance &&
              ((st.lhs[i] == exo_lv[a] && st.rhs[i] == exo_lv[b]) ||
               (st.lhs[i] == exo_lv[b] && st.rhs[i] == exo_lv[a]))) {
            has_cov = true;
            break;
          }
        }
        if (!has_cov)
          append_free_row(st, exo_lv[a], parse::Op::Covariance, exo_lv[b], 0.0);
      }
    }
    part.structural = std::move(st);
  }

  return part;
}

}  // namespace magmaan::estimate::frontier::detail
