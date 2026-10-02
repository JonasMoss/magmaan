#include "ordinal_test_helpers.hpp"

namespace ordinal_test_detail {

Eigen::MatrixXd ordinal_expected_counts(const Eigen::VectorXd& th_i,
                                        const Eigen::VectorXd& th_j,
                                        double rho,
                                        double total) {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd out(th_i.size() + 1, th_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == out.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < out.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == out.cols()) ? inf : th_j(b);
      out(a, b) = total * magmaan::data::ordinal_bvn_rect_prob(
          lo_i, hi_i, lo_j, hi_j, rho);
    }
  }
  return out;
}

double h_score_pair_objective(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho,
    const magmaan::data::PolychoricHScoreOptions& options) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double t = counts(a, b) / (total * p);
      auto h = magmaan::data::eval_polychoric_h_score(t, options);
      if (!h.has_value()) return std::numeric_limits<double>::quiet_NaN();
      out += p * h->phi;
    }
  }
  return out;
}

double dpd_pair_objective(const Eigen::MatrixXd& counts,
                          const Eigen::VectorXd& th_i,
                          const Eigen::VectorXd& th_j,
                          double rho,
                          double alpha) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double fhat = counts(a, b) / total;
      const double pa = std::pow(p, alpha);
      out += p * pa - ((1.0 + alpha) / alpha) * fhat * pa;
    }
  }
  return out;
}

Eigen::MatrixXd ordinal_pair_score_rows_from_counts(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho) {
  std::int64_t n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<std::int64_t>(std::llround(counts(a, b)));
    }
  }
  Eigen::VectorXi xi(n);
  Eigen::VectorXi xj(n);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        xi(row) = static_cast<int>(a);
        xj(row) = static_cast<int>(b);
        ++row;
      }
    }
  }

  auto scores = magmaan::data::ordinal_pair_scores(xi, xj, rho, th_i, th_j);
  REQUIRE(scores.has_value());
  Eigen::MatrixXd out(n, th_i.size() + th_j.size() + 1);
  out.leftCols(th_i.size()) = scores->threshold_i;
  out.middleCols(th_i.size(), th_j.size()) = scores->threshold_j;
  out.col(out.cols() - 1) = scores->rho;
  return out;
}

Eigen::MatrixXd ordinal_data_from_pair_counts(const Eigen::MatrixXd& counts) {
  Eigen::Index n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<Eigen::Index>(std::llround(counts(a, b)));
    }
  }
  Eigen::MatrixXd out(n, 2);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        out(row, 0) = static_cast<double>(a + 1);
        out(row, 1) = static_cast<double>(b + 1);
        ++row;
      }
    }
  }
  return out;
}

double std_normal_cdf(double x) noexcept {
  if (x == std::numeric_limits<double>::infinity()) return 1.0;
  if (x == -std::numeric_limits<double>::infinity()) return 0.0;
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double std_normal_pdf(double x) noexcept {
  if (!std::isfinite(x)) return 0.0;
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}



GammaDiagInfluenceProbe gamma_diag_influence_probe_2var(
    const Eigen::MatrixXi& Xcat,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::MatrixXd& R) {
  REQUIRE(Xcat.cols() == 2);
  REQUIRE(levels.size() == 2);
  const Eigen::Index n = Xcat.rows();
  Eigen::Index nth = 0;
  std::array<Eigen::Index, 2> th_start{};
  std::array<Eigen::Index, 2> th_len{};
  std::array<Eigen::VectorXd, 2> th_by_var;
  for (Eigen::Index j = 0; j < 2; ++j) {
    th_start[static_cast<std::size_t>(j)] = nth;
    th_len[static_cast<std::size_t>(j)] =
        static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
    th_by_var[static_cast<std::size_t>(j)] =
        thresholds.segment(nth, th_len[static_cast<std::size_t>(j)]);
    nth += th_len[static_cast<std::size_t>(j)];
  }
  const Eigen::Index mdim = nth + 1;

  Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
  constexpr double prob_floor = 1.4901161193847656e-8;
  const double inf = std::numeric_limits<double>::infinity();
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < 2; ++j) {
      const int c = Xcat(r, j);
      if (c < 0) continue;
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -inf : thj(c - 1);
      const double hi = (c == thj.size()) ? inf : thj(c);
      const double pr = std::max(prob_floor, std_normal_cdf(hi) - std_normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC_TH(r, base + c) += std_normal_pdf(thj(c)) / pr;
      if (c > 0) SC_TH(r, base + c - 1) -= std_normal_pdf(thj(c - 1)) / pr;
    }
  }

  std::vector<Eigen::Index> obs_rows;
  obs_rows.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index r = 0; r < n; ++r) {
    if (Xcat(r, 1) >= 0 && Xcat(r, 0) >= 0) obs_rows.push_back(r);
  }
  Eigen::VectorXi xi(static_cast<Eigen::Index>(obs_rows.size()));
  Eigen::VectorXi xj(static_cast<Eigen::Index>(obs_rows.size()));
  for (Eigen::Index r = 0; r < xi.size(); ++r) {
    const Eigen::Index src = obs_rows[static_cast<std::size_t>(r)];
    xi(r) = Xcat(src, 1);
    xj(r) = Xcat(src, 0);
  }
  auto ps = magmaan::data::ordinal_pair_scores(
      xi, xj, R(1, 0), th_by_var[1], th_by_var[0]);
  REQUIRE(ps.has_value());

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(nth) = SC_TH;
  SC.col(nth).setZero();
  for (Eigen::Index r = 0; r < xi.size(); ++r) {
    SC(obs_rows[static_cast<std::size_t>(r)], nth) = ps->rho(r);
  }
  const Eigen::MatrixXd INNER = SC.transpose() * SC;

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(mdim, mdim);
  const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
  for (Eigen::Index j = 0; j < 2; ++j) {
    const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
    const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
    B.block(s, s, l, l) = INNER_TH.block(s, s, l, l);
  }
  B.block(nth, th_start[1], 1, th_len[1]) =
      ps->rho.transpose() * ps->threshold_i;
  B.block(nth, th_start[0], 1, th_len[0]) =
      ps->rho.transpose() * ps->threshold_j;
  B(nth, nth) = ps->rho.squaredNorm();

  const Eigen::MatrixXd B_inv = B.inverse();
  GammaDiagInfluenceProbe out;
  out.SC = SC;
  out.B = B;
  out.Gamma = static_cast<double>(n) * B_inv * INNER * B_inv.transpose();
  out.Gamma = 0.5 * (out.Gamma + out.Gamma.transpose()).eval();
  out.b_case.reserve(static_cast<std::size_t>(n));

  Eigen::MatrixXd pair_a21_i = Eigen::MatrixXd::Zero(n, ps->threshold_i.cols());
  Eigen::MatrixXd pair_a21_j = Eigen::MatrixXd::Zero(n, ps->threshold_j.cols());
  for (Eigen::Index r = 0; r < xi.size(); ++r) {
    const Eigen::Index dst = obs_rows[static_cast<std::size_t>(r)];
    pair_a21_i.row(dst) = ps->rho(r) * ps->threshold_i.row(r);
    pair_a21_j.row(dst) = ps->rho(r) * ps->threshold_j.row(r);
  }
  for (Eigen::Index r = 0; r < n; ++r) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index j = 0; j < 2; ++j) {
      const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
      const Eigen::VectorXd sv = SC_TH.row(r).segment(s, l).transpose();
      bi.block(s, s, l, l) = sv * sv.transpose();
    }
    bi(nth, nth) = SC(r, nth) * SC(r, nth);
    bi.block(nth, th_start[1], 1, th_len[1]) = pair_a21_i.row(r);
    bi.block(nth, th_start[0], 1, th_len[0]) = pair_a21_j.row(r);
    out.b_case.push_back(std::move(bi));
  }
  return out;
}

Eigen::VectorXd finite_diff_gamma_diag_case_influence(
    const GammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps) {
  const Eigen::Index n = probe.SC.rows();
  const Eigen::MatrixXd A = probe.B / static_cast<double>(n);
  const Eigen::MatrixXd V =
      (probe.SC.transpose() * probe.SC) / static_cast<double>(n);
  const Eigen::MatrixXd A_inv = A.inverse();
  const Eigen::MatrixXd Gamma = A_inv * V * A_inv.transpose();
  const Eigen::VectorXd s_i = probe.SC.row(row).transpose();
  const Eigen::MatrixXd A_eps =
      A + eps * (probe.b_case[static_cast<std::size_t>(row)] - A);
  const Eigen::MatrixXd V_eps = V + eps * (s_i * s_i.transpose() - V);
  const Eigen::MatrixXd A_eps_inv = A_eps.inverse();
  const Eigen::MatrixXd Gamma_eps =
      A_eps_inv * V_eps * A_eps_inv.transpose();
  return (Gamma_eps.diagonal() - Gamma.diagonal()) / eps;
}

Eigen::MatrixXd finite_diff_gamma_case_influence(
    const GammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps) {
  const Eigen::Index n = probe.SC.rows();
  const Eigen::MatrixXd A = probe.B / static_cast<double>(n);
  const Eigen::MatrixXd V =
      (probe.SC.transpose() * probe.SC) / static_cast<double>(n);
  const Eigen::MatrixXd A_inv = A.inverse();
  const Eigen::MatrixXd Gamma = A_inv * V * A_inv.transpose();
  const Eigen::VectorXd s_i = probe.SC.row(row).transpose();
  const Eigen::MatrixXd A_eps =
      A + eps * (probe.b_case[static_cast<std::size_t>(row)] - A);
  const Eigen::MatrixXd V_eps = V + eps * (s_i * s_i.transpose() - V);
  const Eigen::MatrixXd A_eps_inv = A_eps.inverse();
  const Eigen::MatrixXd Gamma_eps =
      A_eps_inv * V_eps * A_eps_inv.transpose();
  return (Gamma_eps - Gamma) / eps;
}



MixedGammaDiagInfluenceProbe mixed_gamma_diag_influence_probe(
    const Eigen::MatrixXd& X,
    const std::vector<std::int32_t>& ordered,
    const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds,
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& R) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  Eigen::Index nth = 0;
  Eigen::Index n_cont = 0;
  std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
  std::vector<Eigen::Index> th_len(static_cast<std::size_t>(p), 0);
  std::vector<Eigen::Index> cont_pos(static_cast<std::size_t>(p), -1);
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) {
      th_start[static_cast<std::size_t>(j)] = nth;
      th_len[static_cast<std::size_t>(j)] =
          static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
      nth += th_len[static_cast<std::size_t>(j)];
    } else {
      cont_pos[static_cast<std::size_t>(j)] = n_cont++;
    }
  }
  const Eigen::Index s1 = nth + 2 * n_cont;
  const Eigen::Index n_assoc = p * (p - 1) / 2;
  const Eigen::Index mdim = s1 + n_assoc;

  Eigen::MatrixXi Xcat = Eigen::MatrixXi::Constant(n, p, -1);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
  std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) {
      th_by_var[static_cast<std::size_t>(j)] =
          thresholds.segment(th_start[static_cast<std::size_t>(j)],
                             th_len[static_cast<std::size_t>(j)]);
      for (Eigen::Index r = 0; r < n; ++r) {
        Xcat(r, j) = static_cast<int>(X(r, j)) - 1;
      }
    } else {
      U.col(j) = (X.col(j).array() - mean(j)) / std::sqrt(R(j, j));
    }
  }

  Eigen::MatrixXd SC1 = Eigen::MatrixXd::Zero(n, s1);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[static_cast<std::size_t>(j)] == 0) continue;
      const int c = Xcat(r, j);
      const auto& thj = th_by_var[static_cast<std::size_t>(j)];
      const double lo = (c == 0) ? -std::numeric_limits<double>::infinity()
                                 : thj(c - 1);
      const double hi = (c == thj.size())
                            ? std::numeric_limits<double>::infinity()
                            : thj(c);
      const double pr =
          std::max(1.4901161193847656e-8, std_normal_cdf(hi) - std_normal_cdf(lo));
      const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
      if (c < thj.size()) SC1(r, base + c) += std_normal_pdf(thj(c)) / pr;
      if (c > 0) SC1(r, base + c - 1) -= std_normal_pdf(thj(c - 1)) / pr;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) continue;
    const Eigen::Index cp = cont_pos[static_cast<std::size_t>(j)];
    const double v = R(j, j);
    SC1.col(nth + cp) = (X.col(j).array() - mean(j)) / v;
    SC1.col(nth + n_cont + cp) =
        ((X.col(j).array() - mean(j)).square() - v) / (2.0 * v * v);
  }

  const Eigen::MatrixXd INNER1 = SC1.transpose() * SC1;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(mdim, mdim);
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] != 0) {
      const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
      B.block(s, s, l, l) = INNER1.block(s, s, l, l);
    } else {
      const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(j)];
      const Eigen::Index va =
          nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
      B(mu, mu) = INNER1(mu, mu);
      B(va, va) = INNER1(va, va);
      B(mu, va) = B(va, mu) = INNER1(mu, va);
    }
  }

  Eigen::MatrixXd SC_ASSOC = Eigen::MatrixXd::Zero(n, n_assoc);
  std::vector<Eigen::MatrixXd> pair_a21_case;
  pair_a21_case.reserve(static_cast<std::size_t>(n_assoc));
  Eigen::Index assoc = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      Eigen::MatrixXd a21_case = Eigen::MatrixXd::Zero(n, s1);
      if (oi && oj) {
        auto ps = magmaan::data::ordinal_pair_scores(
            Xcat.col(i), Xcat.col(j), R(i, j),
            th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        REQUIRE(ps.has_value());
        SC_ASSOC.col(assoc) = ps->rho;
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        B.block(s1 + assoc, si, 1, ps->threshold_i.cols()) =
            ps->rho.transpose() * ps->threshold_i;
        B.block(s1 + assoc, sj, 1, ps->threshold_j.cols()) =
            ps->rho.transpose() * ps->threshold_j;
        a21_case.block(0, si, n, ps->threshold_i.cols()) =
            ps->rho.asDiagonal() * ps->threshold_i;
        a21_case.block(0, sj, n, ps->threshold_j.cols()) =
            ps->rho.asDiagonal() * ps->threshold_j;
      } else if (oi || oj) {
        const Eigen::Index o = oi ? i : j;
        const Eigen::Index c = oi ? j : i;
        Eigen::VectorXi cat(n);
        for (Eigen::Index r = 0; r < n; ++r) cat(r) = Xcat(r, o);
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        auto ps = magmaan::data::polyserial_pair_scores(
            cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
        REQUIRE(ps.has_value());
        SC_ASSOC.col(assoc) = ps->rho;
        const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
        const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(c)];
        const Eigen::Index va =
            nth + n_cont + cont_pos[static_cast<std::size_t>(c)];
        B.block(s1 + assoc, so, 1, ps->thresholds.cols()) =
            ps->rho.transpose() * ps->thresholds;
        B(s1 + assoc, mu) = ps->rho.dot(ps->mu_unit) / sd;
        B(s1 + assoc, va) = ps->rho.dot(ps->var_unit) / R(c, c);
        a21_case.block(0, so, n, ps->thresholds.cols()) =
            ps->rho.asDiagonal() * ps->thresholds;
        a21_case.col(mu) =
            (ps->rho.array() * ps->mu_unit.array() / sd).matrix();
        a21_case.col(va) =
            (ps->rho.array() * ps->var_unit.array() / R(c, c)).matrix();
      } else {
        auto sc = magmaan::data::continuous_pair_normal_scores(
            X.col(i), X.col(j), mean(i), mean(j), R(i, i), R(j, j), R(i, j));
        REQUIRE(sc.has_value());
        const Eigen::MatrixXd& S = sc->score_contributions;
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        const Eigen::VectorXd s_rho = sdi * sdj * S.col(4);
        SC_ASSOC.col(assoc) = s_rho;
        const Eigen::Index mui = nth + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index muj = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index vai =
            nth + n_cont + cont_pos[static_cast<std::size_t>(i)];
        const Eigen::Index vaj =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::VectorXd ch_var_i =
            S.col(2) + (rho * sdj / (2.0 * sdi)) * S.col(4);
        const Eigen::VectorXd ch_var_j =
            S.col(3) + (rho * sdi / (2.0 * sdj)) * S.col(4);
        B(s1 + assoc, mui) = s_rho.dot(S.col(0));
        B(s1 + assoc, muj) = s_rho.dot(S.col(1));
        B(s1 + assoc, vai) = s_rho.dot(ch_var_i);
        B(s1 + assoc, vaj) = s_rho.dot(ch_var_j);
        a21_case.col(mui) = (s_rho.array() * S.col(0).array()).matrix();
        a21_case.col(muj) = (s_rho.array() * S.col(1).array()).matrix();
        a21_case.col(vai) = (s_rho.array() * ch_var_i.array()).matrix();
        a21_case.col(vaj) = (s_rho.array() * ch_var_j.array()).matrix();
      }
      B(s1 + assoc, s1 + assoc) = SC_ASSOC.col(assoc).squaredNorm();
      pair_a21_case.push_back(std::move(a21_case));
      ++assoc;
    }
  }

  Eigen::MatrixXd SC(n, mdim);
  SC.leftCols(s1) = SC1;
  SC.rightCols(n_assoc) = SC_ASSOC;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(mdim, mdim);
  Eigen::Index row = 0;
  for (Eigen::Index k = 0; k < nth; ++k) H(row++, k) = 1.0;
  for (Eigen::Index j = 0; j < p; ++j)
    if (ordered[static_cast<std::size_t>(j)] == 0)
      H(row++, nth + cont_pos[static_cast<std::size_t>(j)]) = -1.0;
  for (Eigen::Index j = 0; j < p; ++j)
    if (ordered[static_cast<std::size_t>(j)] == 0)
      H(row++, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) = 1.0;
  assoc = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      if (oi && oj) {
        H(row, s1 + assoc) = 1.0;
      } else if (oi || oj) {
        const Eigen::Index c = oi ? j : i;
        const double sd = std::sqrt(R(c, c));
        const double rho = R(i, j) / sd;
        H(row, s1 + assoc) = sd;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(c)]) =
            rho / (2.0 * sd);
      } else {
        const double sdi = std::sqrt(R(i, i));
        const double sdj = std::sqrt(R(j, j));
        const double rho = R(i, j) / (sdi * sdj);
        H(row, s1 + assoc) = sdi * sdj;
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(i)]) =
            rho * sdj / (2.0 * sdi);
        H(row, nth + n_cont + cont_pos[static_cast<std::size_t>(j)]) =
            rho * sdi / (2.0 * sdj);
      }
      ++row;
      ++assoc;
    }
  }

  MixedGammaDiagInfluenceProbe out;
  out.SC = std::move(SC);
  out.B = std::move(B);
  out.H = std::move(H);
  const Eigen::MatrixXd B_inv = out.B.inverse();
  out.Gamma = static_cast<double>(n) * out.H * B_inv *
              (out.SC.transpose() * out.SC) * B_inv.transpose() *
              out.H.transpose();
  out.Gamma = 0.5 * (out.Gamma + out.Gamma.transpose()).eval();
  out.b_case.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index r = 0; r < n; ++r) {
    Eigen::MatrixXd bi = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[static_cast<std::size_t>(j)] != 0) {
        const Eigen::Index s = th_start[static_cast<std::size_t>(j)];
        const Eigen::Index l = th_len[static_cast<std::size_t>(j)];
        const Eigen::VectorXd sv = out.SC.row(r).segment(s, l).transpose();
        bi.block(s, s, l, l) = sv * sv.transpose();
      } else {
        const Eigen::Index mu = nth + cont_pos[static_cast<std::size_t>(j)];
        const Eigen::Index va =
            nth + n_cont + cont_pos[static_cast<std::size_t>(j)];
        bi(mu, mu) = out.SC(r, mu) * out.SC(r, mu);
        bi(mu, va) = out.SC(r, mu) * out.SC(r, va);
        bi(va, mu) = out.SC(r, va) * out.SC(r, mu);
        bi(va, va) = out.SC(r, va) * out.SC(r, va);
      }
    }
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      bi(s1 + k, s1 + k) = out.SC(r, s1 + k) * out.SC(r, s1 + k);
      bi.block(s1 + k, 0, 1, s1) =
          pair_a21_case[static_cast<std::size_t>(k)].row(r);
    }
    out.b_case.push_back(std::move(bi));
  }
  return out;
}

Eigen::VectorXd finite_diff_mixed_gamma_diag_case_influence(
    const MixedGammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps) {
  const Eigen::Index n = probe.SC.rows();
  const Eigen::MatrixXd A = probe.B / static_cast<double>(n);
  const Eigen::MatrixXd V =
      (probe.SC.transpose() * probe.SC) / static_cast<double>(n);
  const Eigen::MatrixXd A_inv = A.inverse();
  const Eigen::MatrixXd Gamma =
      probe.H * A_inv * V * A_inv.transpose() * probe.H.transpose();
  const Eigen::VectorXd s_i = probe.SC.row(row).transpose();
  const Eigen::MatrixXd A_eps =
      A + eps * (probe.b_case[static_cast<std::size_t>(row)] - A);
  const Eigen::MatrixXd V_eps = V + eps * (s_i * s_i.transpose() - V);
  const Eigen::MatrixXd A_eps_inv = A_eps.inverse();
  const Eigen::MatrixXd Gamma_eps =
      probe.H * A_eps_inv * V_eps * A_eps_inv.transpose() *
      probe.H.transpose();
  return (Gamma_eps.diagonal() - Gamma.diagonal()) / eps;
}

Eigen::MatrixXd finite_diff_mixed_gamma_case_influence(
    const MixedGammaDiagInfluenceProbe& probe,
    Eigen::Index row,
    double eps) {
  const Eigen::Index n = probe.SC.rows();
  const Eigen::MatrixXd A = probe.B / static_cast<double>(n);
  const Eigen::MatrixXd V =
      (probe.SC.transpose() * probe.SC) / static_cast<double>(n);
  const Eigen::MatrixXd A_inv = A.inverse();
  const Eigen::MatrixXd Gamma =
      probe.H * A_inv * V * A_inv.transpose() * probe.H.transpose();
  const Eigen::VectorXd s_i = probe.SC.row(row).transpose();
  const Eigen::MatrixXd A_eps =
      A + eps * (probe.b_case[static_cast<std::size_t>(row)] - A);
  const Eigen::MatrixXd V_eps = V + eps * (s_i * s_i.transpose() - V);
  const Eigen::MatrixXd A_eps_inv = A_eps.inverse();
  const Eigen::MatrixXd Gamma_eps =
      probe.H * A_eps_inv * V_eps * A_eps_inv.transpose() *
      probe.H.transpose();
  return (Gamma_eps - Gamma) / eps;
}

double symmetric_condition_number(const Eigen::MatrixXd& x) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (x + x.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  const double max_eval = es.eigenvalues().cwiseAbs().maxCoeff();
  const double min_eval = es.eigenvalues().cwiseAbs().minCoeff();
  if (!(min_eval > 0.0) || !std::isfinite(max_eval)) {
    return std::numeric_limits<double>::infinity();
  }
  return max_eval / min_eval;
}

bool matrix_matches_with_nan(const Eigen::MatrixXd& lhs,
                             const Eigen::MatrixXd& rhs,
                             double tol) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) return false;
  for (Eigen::Index i = 0; i < lhs.rows(); ++i) {
    for (Eigen::Index j = 0; j < lhs.cols(); ++j) {
      const double a = lhs(i, j);
      const double b = rhs(i, j);
      if (!std::isfinite(a) || !std::isfinite(b)) {
        if (std::isfinite(a) || std::isfinite(b)) return false;
        continue;
      }
      if (std::abs(a - b) > tol) return false;
    }
  }
  return true;
}

}  // namespace


namespace ordinal_test_detail {
// Synthetic one-factor 3-category block for multi-group profiling tests.
Eigen::MatrixXd ordinal_test_block(std::uint32_t seed,
                                   Eigen::Index n,
                                   const std::array<double, 4>& loading,
                                   double cut1,
                                   double cut2) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < 4; ++j) {
      const double lj = loading[static_cast<std::size_t>(j)];
      const double eps = std::sqrt(1.0 - lj * lj) * norm(rng);
      const double y = lj * eta + eps;
      X(i, j) = 1.0 + (y > cut1) + (y > cut2);
    }
  }
  return X;
}

}  // namespace ordinal_test_detail
