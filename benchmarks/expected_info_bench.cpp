// `inference::information_expected_per_case_blocks`: the T-array form vs the
// whitened-Jacobian form.
//
// I_b[a,c] = ½·tr(Σ_b⁻¹ Δ_a Σ_b⁻¹ Δ_c) + ν_a' Σ_b⁻¹ ν_c  (per case).
//
// The historical implementation materialized T[k][b] = Σ_b⁻¹ · unvech(J[:,k])
// for every free parameter k and every block b — an n_free × n_blocks array of
// p×p matrices, all live at once — then reduced each (a,c) pair with an
// elementwise `(T_a' .* T_c).sum()` over every block. In a multi-group model a
// given parameter usually touches one group, so all but one block per parameter
// is an explicitly stored p×p of zeros, and the pair loop contracts against
// those zeros anyway.
//
// Both terms are the same bilinear form — the normal-theory inner product
// induced by Σ_b — which `gmm::BlockWeight`'s NormalTheory kind already carries
// in factored form. With Y_b the NT-whitened stacked [dμ/dθ ; dvech(Σ)/dθ] for
// block b, I_b = Y_b' Y_b exactly. One q_b × n_free whitened Jacobian per block,
// built and discarded in turn, and one BLAS-3 syrk instead of the hand-rolled
// n_free²·n_blocks·p² reduction.
//
// `old_per_case_blocks` below is a verbatim transcription of the historical
// algorithm, kept local to this benchmark so the comparison does not depend on
// checking out an old revision. The two arms are run as a paired, rotated
// comparison via `timing/timing.hpp`, and their checksums are printed: equal
// checksums are the tripwire that both arms computed the same information.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#include "timing/timing.hpp"

namespace {

using magmaan::bench::Arm;
using magmaan::bench::must;
using magmaan::bench::TimingOptions;
using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;

constexpr Eigen::Index vech_len(Eigen::Index p) { return p * (p + 1) / 2; }
constexpr Eigen::Index vech_index(Eigen::Index p, Eigen::Index r,
                                  Eigen::Index c) {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

// The historical T-array algorithm, transcribed. `t_bytes` receives the peak
// bytes held by the T array.
std::vector<Eigen::MatrixXd> old_per_case_blocks(const ModelEvaluator& ev,
                                                 const Eigen::VectorXd& theta,
                                                 std::size_t n_blocks,
                                                 double* t_bytes) {
  const std::size_t n_free = ev.n_free();
  auto sm  = ev.sigma(theta);
  auto J_  = ev.dsigma_dtheta(theta);
  auto Jm_ = ev.dmu_dtheta(theta);
  const Eigen::MatrixXd& J   = *J_;
  const Eigen::MatrixXd& Jmu = *Jm_;
  const bool has_means = (Jmu.size() > 0);

  std::vector<Eigen::MatrixXd> SigmaInv(n_blocks);
  std::vector<Eigen::Index>    p_dim(n_blocks, 0), vech_off(n_blocks, 0);
  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::MatrixXd Sb = 0.5 * (sm->sigma[b] + sm->sigma[b].transpose());
    const Eigen::Index p = Sb.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sb);
    SigmaInv[b] = llt.solve(Eigen::MatrixXd::Identity(p, p));
    p_dim[b]    = p;
    vech_off[b] = running;
    running += vech_len(p);
  }

  std::vector<std::vector<Eigen::MatrixXd>> T(
      n_free, std::vector<Eigen::MatrixXd>(n_blocks));
  Eigen::MatrixXd M;
  double bytes = 0.0;
  for (std::size_t k = 0; k < n_free; ++k) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p = p_dim[b];
      M.setZero(p, p);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off[b] + vech_index(p, r, c),
                             static_cast<Eigen::Index>(k));
          M(r, c) = v;
          if (r != c) M(c, r) = v;
        }
      }
      T[k][b].noalias() = SigmaInv[b] * M;
      bytes += static_cast<double>(p * p) * 8.0;
    }
  }
  *t_bytes = bytes;

  std::vector<Eigen::Index> mu_off(n_blocks, 0);
  std::vector<std::vector<Eigen::VectorXd>> eta;
  if (has_means) {
    Eigen::Index rp = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) { mu_off[b] = rp; rp += p_dim[b]; }
    eta.assign(n_free, std::vector<Eigen::VectorXd>(n_blocks));
    for (std::size_t k = 0; k < n_free; ++k)
      for (std::size_t b = 0; b < n_blocks; ++b)
        eta[k][b].noalias() = SigmaInv[b] * Jmu.col(static_cast<Eigen::Index>(k))
                                                .segment(mu_off[b], p_dim[b]);
  }

  const Eigen::Index nf = static_cast<Eigen::Index>(n_free);
  std::vector<Eigen::MatrixXd> out(n_blocks, Eigen::MatrixXd::Zero(nf, nf));
  for (std::size_t a = 0; a < n_free; ++a) {
    for (std::size_t c = a; c < n_free; ++c) {
      for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        double pb = (T[a][blk].transpose().array() * T[c][blk].array()).sum();
        if (has_means) {
          pb += 2.0 * Jmu.col(static_cast<Eigen::Index>(a))
                          .segment(mu_off[blk], p_dim[blk])
                          .dot(eta[c][blk]);
        }
        const double v = 0.5 * pb;
        out[blk](static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(c)) = v;
        if (a != c)
          out[blk](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(a)) = v;
      }
    }
  }
  return out;
}

double checksum_of(const std::vector<Eigen::MatrixXd>& blocks) {
  double acc = 0.0;
  for (const auto& b : blocks) acc += b.trace() + b.sum();
  return acc;
}

std::string one_factor_syntax(int p) {
  std::string s = "f =~ x1";
  for (int i = 2; i <= p; ++i) s += " + x" + std::to_string(i);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ps = {6, 12, 24, 48};
  const std::vector<int> gs = {1, 4};
  if (argc > 1) {
    ps.clear();
    for (int i = 1; i < argc; ++i) ps.push_back(std::atoi(argv[i]));
  }

  TimingOptions opt;
  opt.reps    = 9;
  opt.warmups = 2;

  std::printf("%4s %3s %7s %11s %11s %8s %10s %9s %9s %9s\n", "p", "G", "n_free",
              "old_ms", "new_ms", "speedup", "old_T_MB", "new_Y_MB", "maxdiff",
              "chksum");
  for (int G : gs) {
    for (int p : ps) {
      auto fp = Parser::parse(one_factor_syntax(p));
      if (!fp.has_value()) continue;
      magmaan::spec::BuildOptions bo;
      bo.n_groups = G;
      auto pt = build(*fp, bo);
      if (!pt.has_value()) continue;
      auto mr = build_matrix_rep(*pt);
      if (!mr.has_value()) continue;
      auto evr = ModelEvaluator::build(*pt, *mr);
      if (!evr.has_value()) continue;
      const auto& ev = *evr;

      const Eigen::Index nf = static_cast<Eigen::Index>(ev.n_free());
      const Eigen::VectorXd theta = Eigen::VectorXd::Constant(nf, 0.7);
      auto m0 = ev.sigma(theta);
      if (!m0.has_value()) continue;

      std::mt19937 rng(20260918u + static_cast<unsigned>(p));
      std::normal_distribution<double> nd(0.0, 0.01);
      SampleStats samp;
      for (std::size_t b = 0; b < m0->sigma.size(); ++b) {
        Eigen::MatrixXd S = m0->sigma[b];
        for (Eigen::Index i = 0; i < S.rows(); ++i)
          for (Eigen::Index j = 0; j <= i; ++j) {
            S(i, j) += nd(rng);
            S(j, i) = S(i, j);
          }
        samp.S.push_back(S);
        samp.n_obs.push_back(1000);
      }

      magmaan::estimate::Estimates est;
      est.theta = theta;

      const std::size_t nb = m0->sigma.size();
      double t_bytes = 0.0;
      const auto ref = old_per_case_blocks(ev, theta, nb, &t_bytes);
      const auto got = must(magmaan::inference::information_expected_per_case_blocks(
                                *pt, *mr, samp, est),
                            "information_expected_per_case_blocks");

      double maxdiff = 0.0, scale = 1.0;
      for (std::size_t b = 0; b < nb; ++b) {
        maxdiff = std::max(maxdiff, (ref[b] - got[b]).cwiseAbs().maxCoeff());
        scale   = std::max(scale, ref[b].cwiseAbs().maxCoeff());
      }

      // Paired, rotated comparison — see timing/timing.hpp on why arm order
      // matters here.
      const std::vector<Arm> arms = {
          Arm{"old", [&] {
                double bb = 0.0;
                return checksum_of(old_per_case_blocks(ev, theta, nb, &bb));
              }},
          Arm{"new", [&] {
                return checksum_of(
                    must(magmaan::inference::information_expected_per_case_blocks(
                             *pt, *mr, samp, est),
                         "information_expected_per_case_blocks"));
              }},
      };
      const auto stats = magmaan::bench::time_arms(arms, opt);
      const double old_ms = stats[0].median_ns / 1e6;
      const double new_ms = stats[1].median_ns / 1e6;

      // New peak: one q_b × n_free whitened Jacobian live at a time.
      const double y_bytes = static_cast<double>(vech_len(p) * nf) * 8.0;
      // `StageStats::checksum` is the sum over the calibrated batch, and the two
      // arms calibrate to different batch sizes, so it must be divided by
      // `batch` before the arms are comparable. (Raw checksums differ by the
      // batch ratio, which looks alarming and means nothing.)
      const double chk_old = stats[0].checksum / std::max(1, stats[0].batch);
      const double chk_new = stats[1].checksum / std::max(1, stats[1].batch);
      const double chk_rel = std::abs(chk_old - chk_new) /
                             std::max(1.0, std::abs(chk_old));

      std::printf("%4d %3d %7lld %11.4f %11.4f %7.2fx %10.2f %9.2f %9.1e %9.1e\n",
                  p, G, static_cast<long long>(nf), old_ms, new_ms,
                  old_ms / new_ms, t_bytes / 1048576.0, y_bytes / 1048576.0,
                  maxdiff / scale, chk_rel);
    }
  }
  return 0;
}
