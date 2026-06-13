#!/usr/bin/env Rscript

# Goldens for the hand-rolled special functions in
# src/detail_distribution_math.hpp (regularized incomplete gamma/beta, their
# inverses, Student-t CDF/quantile, and the F upper tail). Oracle is base R
# (pgamma/pbeta/pt/pf/pchisq + qbeta/qgamma/qt); no extra package required.
#
# Each case is {fn, <args>, expected}. The grids place arguments through the
# body AND into deep tails (via quantile placement) because that is where the
# continued-fraction convergence and the bisection iteration caps matter. The
# two infinite-df F branches are validated through pchisq since R's pf cannot
# take Inf degrees of freedom.

suppressPackageStartupMessages({
  library(jsonlite)
})

cases <- list()
add <- function(...) cases[[length(cases) + 1L]] <<- list(...)

# Probability ladder spanning both tails; avoids exact 0/1 (those are handled by
# the kernels' closed-form endpoints, not the iterative path).
p_grid <- c(1e-8, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999,
            1 - 1e-6, 1 - 1e-8)
# Trimmed ladder for the quantile (inverse-CDF) targets whose deepest tails are
# ill-conditioned: the kernel inverts the CDF by bisection, so where the density
# is near-flat (e.g. t(1) at p=1e-8) the recoverable precision is limited. The
# CDFs themselves keep the full p_grid via quantile placement.
pq_grid <- c(1e-6, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1 - 1e-6)

# ---- regularized incomplete gamma: gamma_p / gamma_q ------------------------
# gamma_p(a, x) == pgamma(x, a);  gamma_q(a, x) == pgamma(x, a, lower=FALSE).
for (a in c(0.5, 1, 3, 30)) {
  # Body points plus tail points placed by qgamma so x reaches the far tails.
  xs <- sort(unique(c(qgamma(p_grid, shape = a), a, a + 1)))
  xs <- xs[is.finite(xs) & xs > 0]
  for (x in xs) {
    add(fn = "gamma_p", a = a, x = x, expected = pgamma(x, shape = a))
    add(fn = "gamma_q", a = a, x = x,
        expected = pgamma(x, shape = a, lower.tail = FALSE))
  }
}

# ---- regularized incomplete beta: regularized_beta --------------------------
# regularized_beta(a, b, x) == pbeta(x, a, b).
for (a in c(0.5, 1, 2, 5)) {
  for (b in c(0.5, 1, 2, 5)) {
    xs <- sort(unique(c(0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99,
                        qbeta(p_grid, a, b))))
    xs <- xs[is.finite(xs) & xs > 0 & xs < 1]
    for (x in xs) {
      add(fn = "regularized_beta", a = a, b = b, x = x,
          expected = pbeta(x, a, b))
    }
  }
}

# ---- inverse regularized beta: inverse_regularized_beta ---------------------
# inverse_regularized_beta(p, a, b) == qbeta(p, a, b).
for (a in c(0.5, 1, 2, 5)) {
  for (b in c(0.5, 1, 2, 5)) {
    for (p in p_grid) {
      add(fn = "inverse_regularized_beta", p = p, a = a, b = b,
          expected = qbeta(p, a, b))
    }
  }
}

# ---- inverse scaled gamma: inverse_gamma_p ----------------------------------
# inverse_gamma_p(p, shape, scale) == qgamma(p, shape, scale = scale).
for (shape in c(0.5, 1, 3, 30)) {
  for (scale in c(0.5, 1, 2)) {
    for (p in p_grid) {
      add(fn = "inverse_gamma_p", p = p, shape = shape, scale = scale,
          expected = qgamma(p, shape = shape, scale = scale))
    }
  }
}

# ---- Student-t CDF and quantile ---------------------------------------------
# student_t_cdf(x, df) == pt(x, df);  student_t_quantile(p, df) == qt(p, df).
for (df in c(1, 3, 10, 30)) {
  xs <- sort(unique(c(0, qt(p_grid, df))))
  xs <- xs[is.finite(xs)]
  for (x in xs) {
    add(fn = "student_t_cdf", x = x, df = df, expected = pt(x, df))
  }
  for (p in pq_grid) {
    add(fn = "student_t_quantile", p = p, df = df, expected = qt(p, df))
  }
}

# ---- F upper tail: f_upper_tail ---------------------------------------------
# f_upper_tail(x, d1, d2) == pf(x, d1, d2, lower=FALSE), finite df.
for (d1 in c(1, 3, 10, 30)) {
  for (d2 in c(1, 3, 10, 30)) {
    xs <- sort(unique(c(0.25, 1, 4, qf(1 - p_grid, d1, d2))))
    xs <- xs[is.finite(xs) & xs > 0]
    for (x in xs) {
      add(fn = "f_upper_tail", x = x, d1 = d1, d2 = d2,
          expected = pf(x, d1, d2, lower.tail = FALSE))
    }
  }
}

# Infinite-df limits (the d1_inf / d2_inf branches of f_upper_tail). R's pf
# rejects Inf df, so use the chi-square limits the branches reduce to:
#   d1 -> Inf:  f_upper_tail(x, Inf, d2) == pchisq(d2 / x, d2)            (lower)
#   d2 -> Inf:  f_upper_tail(x, d1, Inf) == pchisq(d1 * x, d1, upper)
for (d in c(1, 3, 10, 30)) {
  for (x in c(0.25, 0.5, 1, 2, 4, 8)) {
    add(fn = "f_upper_tail_d1inf", x = x, d2 = d,
        expected = pchisq(d / x, df = d))
    add(fn = "f_upper_tail_d2inf", x = x, d1 = d,
        expected = pchisq(d * x, df = d, lower.tail = FALSE))
  }
}

fixture <- list(
  `_meta` = list(
    format_version = 1,
    fixture_kind = "distribution_math",
    oracle = paste0("base R ", getRversion(),
                    " pgamma/pbeta/pt/pf/pchisq + qbeta/qgamma/qt")
  ),
  cases = cases
)

args_file <- sub("^--file=", "",
                 commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
script_dir <- dirname(normalizePath(args_file, mustWork = TRUE))
root <- normalizePath(file.path(script_dir, "..", "fixtures"), mustWork = TRUE)
write_json(
  fixture,
  file.path(root, "distribution_math.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
cat(sprintf("wrote %d cases to %s\n", length(cases),
            file.path(root, "distribution_math.json")))
