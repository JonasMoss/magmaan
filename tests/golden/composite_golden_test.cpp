#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "../oracle.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/measures/composite_weights.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

namespace {

const std::vector<std::string> kCompositeFixtures = {
    "0001_pure_composite_hs",
    "0002_composite_factor_hs",
    "0003_composite_structural_hs",
};

Eigen::MatrixXd matrix_from_json(const nlohmann::json &j) {
  const Eigen::Index nr = static_cast<Eigen::Index>(j.size());
  const Eigen::Index nc = nr > 0 ? static_cast<Eigen::Index>(j[0].size()) : 0;
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index r = 0; r < nr; ++r)
    for (Eigen::Index c = 0; c < nc; ++c)
      out(r, c) = j[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                      .get<double>();
  return out;
}

magmaan::data::SampleStats sample_stats_from_fixture(const nlohmann::json &fx) {
  magmaan::data::SampleStats samp;
  for (const auto &b : fx["sample_cov"]) {
    samp.S.push_back(matrix_from_json(b["matrix"]));
    samp.n_obs.push_back(fx["n_obs"].get<std::int64_t>());
  }
  return samp;
}

bool close(double a, double b, double tol) {
  return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

std::string op_string(magmaan::parse::Op op) {
  switch (op) {
  case magmaan::parse::Op::Measurement:
    return "=~";
  case magmaan::parse::Op::Composite:
    return "<~";
  case magmaan::parse::Op::Regression:
    return "~";
  case magmaan::parse::Op::Covariance:
    return "~~";
  case magmaan::parse::Op::Intercept:
    return "~1";
  case magmaan::parse::Op::DefineParam:
    return ":=";
  case magmaan::parse::Op::EqConstraint:
    return "==";
  case magmaan::parse::Op::LtConstraint:
    return "<";
  case magmaan::parse::Op::GtConstraint:
    return ">";
  case magmaan::parse::Op::Threshold:
    return "|";
  case magmaan::parse::Op::ResponseScale:
    return "~*~";
  }
  return "?";
}

struct Built {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::model::MatrixRep rep;
};

std::optional<std::size_t> find_row(const Built &b, const nlohmann::json &row) {
  const std::string lhs = row["lhs"].get<std::string>();
  const std::string op = row["op"].get<std::string>();
  const std::string rhs = row["rhs"].get<std::string>();
  const int group = row.value("group", 1);
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    if (b.pt.group[i] != group)
      continue;
    if (b.names.row_lhs[i] != lhs || b.names.row_rhs[i] != rhs)
      continue;
    if (op_string(b.pt.op[i]) != op)
      continue;
    return i;
  }
  return std::nullopt;
}

std::map<std::string, const magmaan::measures::composite::CompositeWeights *>
weight_map(
    const std::vector<magmaan::measures::composite::CompositeWeights> &w) {
  std::map<std::string, const magmaan::measures::composite::CompositeWeights *>
      out;
  for (const auto &cw : w)
    out[cw.composite] = &cw;
  return out;
}

struct NormalizedWeights {
  Eigen::VectorXd weight;
  Eigen::VectorXd se;
  Eigen::MatrixXd vcov;
};

std::optional<NormalizedWeights>
normalize_marker(const magmaan::measures::composite::CompositeWeights &cw) {
  if (cw.weight.size() == 0 || std::abs(cw.weight(0)) < 1e-10)
    return std::nullopt;
  const Eigen::Index k = cw.weight.size();
  NormalizedWeights out;
  out.weight = cw.weight / cw.weight(0);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(k, k);
  for (Eigen::Index j = 0; j < k; ++j) {
    J(j, j) += 1.0 / cw.weight(0);
    J(j, 0) -= cw.weight(j) / (cw.weight(0) * cw.weight(0));
  }
  out.vcov = J * cw.vcov * J.transpose();
  out.se.resize(k);
  for (Eigen::Index j = 0; j < k; ++j)
    out.se(j) = out.vcov(j, j) > 0.0 ? std::sqrt(out.vcov(j, j)) : 0.0;
  return out;
}

std::int32_t var_id(const magmaan::spec::LatentNames &names,
                    std::string_view name) {
  for (std::size_t i = 0; i < names.var_name.size(); ++i)
    if (names.var_name[i] == name)
      return static_cast<std::int32_t>(i);
  return -1;
}

double observed_variance(const Built &b, const Eigen::MatrixXd &sigma,
                         std::string_view name) {
  const std::int32_t id = var_id(b.names, name);
  REQUIRE(id >= 0);
  const std::int32_t pos = b.pt.ov_pos[static_cast<std::size_t>(id)];
  REQUIRE(pos >= 0);
  return sigma(pos, pos);
}

Eigen::MatrixXd indicator_sigma(const Built &b, const Eigen::MatrixXd &sigma,
                                const std::vector<std::string> &indicators) {
  const Eigen::Index k = static_cast<Eigen::Index>(indicators.size());
  Eigen::MatrixXd out(k, k);
  for (Eigen::Index r = 0; r < k; ++r) {
    const auto rid = var_id(b.names, indicators[static_cast<std::size_t>(r)]);
    REQUIRE(rid >= 0);
    const auto rp = b.pt.ov_pos[static_cast<std::size_t>(rid)];
    REQUIRE(rp >= 0);
    for (Eigen::Index c = 0; c < k; ++c) {
      const auto cid = var_id(b.names, indicators[static_cast<std::size_t>(c)]);
      REQUIRE(cid >= 0);
      const auto cp = b.pt.ov_pos[static_cast<std::size_t>(cid)];
      REQUIRE(cp >= 0);
      out(r, c) = sigma(rp, cp);
    }
  }
  return out;
}

}  // namespace

// Skipped until the complete-data ML composite implementation matches
// lavaan's native `<~` W-matrix semantics. Keeping the assertions wired here
// makes the remaining parity gap concrete and ready to unskip.
TEST_CASE("composite ML goldens — lavaan parity for fitted public surface" *
          doctest::skip()) {
  const std::string dir = magmaan::test::fixtures_dir() + "/composite";

  int total = 0, passed = 0;
  std::vector<std::string> failures;

  for (const auto &id : kCompositeFixtures) {
    const std::string path = dir + "/" + id + ".fit.json";
    auto raw = magmaan::test::read_fixture(path);
    if (!raw.has_value()) {
      failures.push_back(id + ": missing fixture");
      continue;
    }
    auto exp = nlohmann::json::parse(*raw, nullptr, false);
    if (exp.is_discarded()) {
      failures.push_back(id + ": invalid JSON");
      continue;
    }
    ++total;

    const std::string model = exp["input"].get<std::string>();
    auto fp = magmaan::parse::Parser::parse(model);
    if (!fp.has_value()) {
      failures.push_back(id + ": parse — " + fp.error().detail);
      continue;
    }

    Built b;
    auto pt_or = magmaan::spec::build(*fp, {}, nullptr, &b.names);
    if (!pt_or.has_value()) {
      failures.push_back(id + ": build — " + pt_or.error().detail);
      continue;
    }
    b.pt = std::move(*pt_or);
    auto rep_or = magmaan::model::build_matrix_rep(b.pt);
    if (!rep_or.has_value()) {
      failures.push_back(id + ": matrix_rep — " + rep_or.error().detail);
      continue;
    }
    b.rep = std::move(*rep_or);

    auto samp = sample_stats_from_fixture(exp);
    magmaan::optim::LbfgsOptions fit_opts;
    fit_opts.max_iter = 5000;
    auto est_or = magmaan::test::fit(
        b.pt, b.rep, samp, {}, magmaan::estimate::Backend::Lbfgs, fit_opts);
    if (!est_or.has_value()) {
      failures.push_back(id + ": fit — " + est_or.error().detail);
      continue;
    }
    const auto &est = *est_or;

    auto info_or =
        magmaan::inference::information_expected(b.pt, b.rep, samp, est);
    if (!info_or.has_value()) {
      failures.push_back(id + ": information_expected — " +
                         info_or.error().detail);
      continue;
    }
    auto vcov_or = magmaan::inference::vcov(*info_or, b.pt);
    if (!vcov_or.has_value()) {
      failures.push_back(id + ": vcov — " + vcov_or.error().detail);
      continue;
    }
    const Eigen::MatrixXd &vcov = *vcov_or;

    auto ev_or = magmaan::model::ModelEvaluator::build(b.pt, b.rep);
    if (!ev_or.has_value()) {
      failures.push_back(id + ": evaluator — " + ev_or.error().detail);
      continue;
    }
    auto sm_or = ev_or->sigma(est.theta);
    if (!sm_or.has_value()) {
      failures.push_back(id + ": sigma — " + sm_or.error().detail);
      continue;
    }

    bool ok = true;
    auto fail = [&](const std::string &msg) {
      failures.push_back(id + ": " + msg);
      ok = false;
    };

    const Eigen::MatrixXd sigma_l =
        matrix_from_json(exp["implied_sigma"][0]["matrix"]);
    const double max_sigma_diff =
        (sm_or->sigma[0] - sigma_l).cwiseAbs().maxCoeff();
    if (max_sigma_diff > 2e-5) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "max |Sigma - lavaan| = %.3e",
                    max_sigma_diff);
      fail(buf);
    }

    const double chi2 = magmaan::inference::chi2_stat(samp, est);
    auto df_or = magmaan::inference::df_stat(b.pt, samp);
    if (!df_or.has_value()) {
      fail("df_stat — " + df_or.error().detail);
    } else {
      if (*df_or != exp["df"].get<int>()) {
        fail("df = " + std::to_string(*df_or) +
             ", lavaan = " + std::to_string(exp["df"].get<int>()));
      }
      if (!close(chi2, exp["chi2"].get<double>(), 2e-5)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "chi2 = %.8f, lavaan = %.8f", chi2,
                      exp["chi2"].get<double>());
        fail(buf);
      }
      const auto bl = magmaan::measures::baseline_chi2(samp);
      const auto fm = magmaan::measures::fit_measures(chi2, *df_or, bl, samp);
      auto fx_or = magmaan::measures::fit_extras(b.pt, b.rep, samp, est);
      if (!fx_or.has_value()) {
        fail("fit_extras — " + fx_or.error().detail);
      } else {
        const auto &fx = *fx_or;
        struct Check {
          const char *name;
          double ours;
          const char *key;
          double tol;
        };
        const Check checks[] = {
            {"cfi", fm.cfi, "cfi", 2e-5},
            {"tli", fm.tli, "tli", 2e-5},
            {"rmsea", fm.rmsea, "rmsea", 2e-5},
            {"rmsea.ci.lower", fm.rmsea_ci_lower, "rmsea_ci_lower", 2e-4},
            {"rmsea.ci.upper", fm.rmsea_ci_upper, "rmsea_ci_upper", 2e-4},
            {"rmsea.pvalue", fm.rmsea_pvalue, "rmsea_pvalue", 2e-5},
            {"srmr", fx.srmr, "srmr", 2e-4},
            {"logl", fx.logl, "logl", 2e-5},
            {"unrestricted_logl", fx.unrestricted_logl, "unrestricted_logl",
             2e-5},
            {"aic", fx.aic, "aic", 2e-5},
            {"bic", fx.bic, "bic", 2e-5},
            {"bic2", fx.bic2, "bic2", 2e-5},
        };
        for (const auto &c : checks) {
          if (!exp.contains(c.key) || exp[c.key].is_null())
            continue;
          const double lavaan = exp[c.key].get<double>();
          if (std::isfinite(lavaan) && std::isfinite(c.ours) &&
              !close(c.ours, lavaan, c.tol)) {
            char buf[180];
            std::snprintf(buf, sizeof(buf), "%s = %.8f, lavaan = %.8f", c.name,
                          c.ours, lavaan);
            fail(buf);
          }
        }
      }
    }

    auto cw_or = magmaan::measures::composite::composite_weights(b.pt, b.names,
                                                                 est, vcov);
    if (!cw_or.has_value()) {
      fail("composite_weights — " + cw_or.error().detail);
    }
    auto slv_or =
        magmaan::measures::standardize::standardize_lv(b.pt, b.rep, est, vcov);
    auto sall_or =
        magmaan::measures::standardize::standardize_all(b.pt, b.rep, est, vcov);
    if (!slv_or.has_value())
      fail("standardize_lv — " + slv_or.error().detail);
    if (!sall_or.has_value())
      fail("standardize_all — " + sall_or.error().detail);

    if (cw_or.has_value()) {
      const auto by_name = weight_map(*cw_or);
      for (const auto &wexp : exp["weights"]) {
        const std::string comp = wexp["lhs"].get<std::string>();
        const std::string ind = wexp["rhs"].get<std::string>();
        auto it = by_name.find(comp);
        if (it == by_name.end()) {
          fail("missing composite weights for " + comp);
          continue;
        }
        const auto &cw = *it->second;
        auto nw_or = normalize_marker(cw);
        if (!nw_or.has_value()) {
          fail("marker-normalization failed for " + comp);
          continue;
        }
        const auto hit =
            std::find(cw.indicators.begin(), cw.indicators.end(), ind);
        if (hit == cw.indicators.end()) {
          fail("missing indicator " + ind + " for " + comp);
          continue;
        }
        const Eigen::Index k = static_cast<Eigen::Index>(
            std::distance(cw.indicators.begin(), hit));
        const auto &nw = *nw_or;
        if (!close(nw.weight(k), wexp["est"].get<double>(), 5e-4)) {
          char buf[180];
          std::snprintf(buf, sizeof(buf), "%s<~%s weight = %.8f, lavaan = %.8f",
                        comp.c_str(), ind.c_str(), nw.weight(k),
                        wexp["est"].get<double>());
          fail(buf);
        }
        if (!close(nw.se(k), wexp["se"].get<double>(), 3e-3)) {
          char buf[180];
          std::snprintf(buf, sizeof(buf), "%s<~%s se = %.8f, lavaan = %.8f",
                        comp.c_str(), ind.c_str(), nw.se(k),
                        wexp["se"].get<double>());
          fail(buf);
        }

        const Eigen::MatrixXd Scc =
            indicator_sigma(b, sm_or->sigma[0], cw.indicators);
        const double var_c = (nw.weight.transpose() * Scc * nw.weight)(0, 0);
        const double sd_c = std::sqrt(var_c);
        const double std_lv = nw.weight(k) / sd_c;
        const double std_all =
            nw.weight(k) *
            std::sqrt(observed_variance(b, sm_or->sigma[0], ind)) / sd_c;
        if (!close(std_lv, wexp["std_lv"].get<double>(), 8e-4)) {
          fail(comp + "<~" + ind + " std.lv mismatch");
        }
        if (!close(std_all, wexp["std_all"].get<double>(), 8e-4)) {
          fail(comp + "<~" + ind + " std.all mismatch");
        }
      }
    }

    if (slv_or.has_value() && sall_or.has_value()) {
      const auto &slv = *slv_or;
      const auto &sall = *sall_or;
      const auto by_name =
          cw_or.has_value()
              ? weight_map(*cw_or)
              : std::map<
                    std::string,
                    const magmaan::measures::composite::CompositeWeights *>{};
      for (const auto &row : exp["rows"]) {
        auto idx = find_row(b, row);
        if (!idx.has_value()) {
          fail("missing row " + row["lhs"].get<std::string>() +
               row["op"].get<std::string>() + row["rhs"].get<std::string>());
          continue;
        }
        const auto i = *idx;
        if (b.pt.free[i] <= 0)
          continue;
        const Eigen::Index t = b.pt.free[i] - 1;
        double raw_est = est.theta(t);
        bool compare_raw_se = true;
        const std::string rhs = row["rhs"].get<std::string>();
        if (auto wit = by_name.find(rhs); wit != by_name.end()) {
          raw_est *= wit->second->weight(0);
          compare_raw_se = false;
        }
        if (!close(raw_est, row["est"].get<double>(), 5e-4)) {
          char buf[180];
          std::snprintf(buf, sizeof(buf), "%s%s%s est = %.8f, lavaan = %.8f",
                        row["lhs"].get<std::string>().c_str(),
                        row["op"].get<std::string>().c_str(),
                        row["rhs"].get<std::string>().c_str(), raw_est,
                        row["est"].get<double>());
          fail(buf);
        }
        if (compare_raw_se) {
          const double se = std::sqrt(std::max(0.0, vcov(t, t)));
          if (!close(se, row["se"].get<double>(), 3e-3)) {
            fail(row["lhs"].get<std::string>() + row["op"].get<std::string>() +
                 row["rhs"].get<std::string>() + " se mismatch");
          }
        }
        if (!close(slv.theta(t), row["std_lv"].get<double>(), 8e-4)) {
          fail(row["lhs"].get<std::string>() + row["op"].get<std::string>() +
               row["rhs"].get<std::string>() + " std.lv mismatch");
        }
        if (!close(sall.theta(t), row["std_all"].get<double>(), 8e-4)) {
          fail(row["lhs"].get<std::string>() + row["op"].get<std::string>() +
               row["rhs"].get<std::string>() + " std.all mismatch");
        }
      }
    }

    if (ok)
      ++passed;
  }

  MESSAGE("composite goldens: " << passed << " / " << total << " pass");
  for (const auto &f : failures)
    MESSAGE("  FAIL " << f);
  CHECK(passed == total);
}
