# Informal probe: is a permutation measurement-invariance test a use case for the
# closed-form (Guttman) CFA estimators?
#
# This is not a support claim or a benchmark. It asks two questions on a 2-group,
# 2-factor, 6-indicator congeneric CFA with metric-invariant loadings and
# deliberately HETEROGENEOUS factor variances (the Behrens-Fisher case for
# measurement invariance):
#
#   (1) cost   -- what does one *valid* permutation cost for the closed-form map
#                 versus magmaan's own ML, as p grows?
#   (2) level  -- which studentizer keeps the permutation Wald calibrated?
#
# The statistic is a Wald test of the metric restriction in the MARKER metric
# (default lavaanify fixes the first loading per group, so the free loadings are
# directly comparable across groups and no Omega-metric projection is needed):
#
#     W(labels) = (A theta)' (A Omega A')^-1 (A theta),   A = "free loadings equal"
#
# computed by magmaan_core$inference_wald_test_theta() and fed either the
# closed-form map's delta-method Omega or ML's expected-information inverse. Both
# estimators see the identical partable, free-parameter order, restriction matrix,
# permutations, and studentizer policy, so the comparison is symmetric.
#
# Studentizer arms. A permutation test needs T to be ONE measurable function of the
# labels, applied identically to the observed and permuted labellings. Freezing
# Omega at the *observed* fit violates this (the observed statistic is
# self-studentized, the permuted ones are not) and is spuriously conservative, so
# the frozen arms below take their bread from the label-invariant POOLED covariance:
#
#   full   Omega(labels) = J(S(labels)) Gamma(X(labels)) J(S(labels))'/N   [B+1 Jacobians]
#   pbread Omega(labels) = J(S_pool)    Gamma(X(labels)) J(S_pool)'/N      [1 Jacobian]
#   pboth  Omega         = J(S_pool)    Gamma(X_pool)    J(S_pool)'/N      [1 Jacobian, constant]
#
# Only `full` is asymptotically pivotal, so by Chung & Romano (2013, Ann. Statist.)
# only `full` should survive the loss of exchangeability that factor-variance
# heterogeneity induces. `pbread` is the only arm that is cheap for the map, because
# the map's Jacobian J is its dominant cost while Gamma is nearly free.
#
# Finding (see docs/backlog/speculative.md): `pbread` buys essentially nothing over
# `pboth` -- the pivotality lives in the bread, not the meat -- so a valid permutation
# Wald must rebuild J per permutation. That inverts the estimator's cost profile: the
# map is ~12x faster than ML per point fit and ~20x slower per valid permutation.

suppressMessages(library(magmaan))
core <- magmaan::magmaan_core

parse_args <- function(args) {
  out <- list(reps = 600L, perms = 199L, seed = 20260709L, what = "both")
  for (arg in args) {
    if (grepl("^--reps=", arg)) out$reps <- as.integer(sub("^--reps=", "", arg))
    if (grepl("^--perms=", arg)) out$perms <- as.integer(sub("^--perms=", "", arg))
    if (grepl("^--seed=", arg)) out$seed <- as.integer(sub("^--seed=", "", arg))
    if (grepl("^--what=", arg)) out$what <- sub("^--what=", "", arg)  # cost|level|both
  }
  out
}

labs <- c("a", "b")

cfa_syntax <- function(k, m) {
  paste(vapply(seq_len(k), function(g) {
    idx <- ((g - 1L) * m + 1L):(g * m)
    sprintf("f%d =~ %s", g, paste0("x", idx, collapse = " + "))
  }, character(1)), collapse = "\n")
}

# Metric-invariant loadings; groups differ in factor variance; `dif` plants a
# single loading difference on x2 in group b.
sim_2group <- function(n, k, m, load = 0.7, phi1 = 1, phi2 = 1, dif = 0) {
  p <- k * m
  Lam <- matrix(0, p, k)
  for (g in seq_len(k)) Lam[((g - 1L) * m + 1L):(g * m), g] <- load
  cPsi <- chol(diag(1 - load^2, p))
  gen <- function(n, phi, L) {
    (matrix(rnorm(n * k), n, k) * sqrt(phi)) %*% t(L) +
      matrix(rnorm(n * p), n, p) %*% cPsi
  }
  Lam2 <- Lam
  Lam2[2L, 1L] <- Lam2[2L, 1L] + dif
  Y <- rbind(gen(n, phi1, Lam), gen(n, phi2, Lam2))
  colnames(Y) <- paste0("x", seq_len(p))
  Y
}

# A: every free loading equal across groups (marker metric, group 1 the reference).
make_A <- function(pt) {
  npar <- max(pt$free)
  ld <- pt[pt$free > 0 & pt$op == "=~", ]
  ref <- ld[ld$group == 1L, ]
  rows <- list()
  for (g in sort(unique(ld$group))) {
    if (g == 1L) next
    gg <- ld[ld$group == g, ]
    for (i in seq_len(nrow(ref))) {
      j <- which(gg$lhs == ref$lhs[i] & gg$rhs == ref$rhs[i])
      r <- numeric(npar)
      r[ref$free[i]] <- -1
      r[gg$free[j]] <- 1
      rows[[length(rows) + 1L]] <- r
    }
  }
  do.call(rbind, rows)
}

ss_split <- function(proto, Xa, Xb) {
  proto$X <- setNames(list(Xa, Xb), labs)
  proto$S <- list(cov(Xa) * (nrow(Xa) - 1L) / nrow(Xa),
                  cov(Xb) * (nrow(Xb) - 1L) / nrow(Xb))
  proto$mean <- list(colMeans(Xa), colMeans(Xb))
  proto$nobs <- c(nrow(Xa), nrow(Xb))
  proto
}

fit_map <- function(pt, ss) {
  tryCatch(fit_noniterative_cfa(pt, ss, "guttman_gls_aligned", "standardized"),
           error = function(e) NULL)
}
# ctx_from_fit(fit) supplies S/nobs/partable (the bread); `data` supplies only the
# empirical fourth-moment meat. That split is what makes the `pbread` arm expressible.
omega_map <- function(fit, ss) {
  if (is.null(fit)) return(NULL)
  tryCatch(noniterative_cfa_grouped_inference(fit, "uls", "empirical", ss)$vcov,
           error = function(e) NULL)
}
fit_ml <- function(pt, ss) {
  f <- tryCatch(core$estimate_ml(pt, ss), error = function(e) NULL)
  if (is.null(f) || is.null(f$theta) || !all(is.finite(f$theta))) return(NULL)
  f
}
omega_ml <- function(fit) {
  if (is.null(fit)) return(NULL)
  tryCatch(core$inference_vcov(core$inference_information_expected(fit), fit),
           error = function(e) NULL)
}
wald <- function(theta, A, Om) {
  if (is.null(theta) || is.null(Om)) return(NA_real_)
  w <- tryCatch(core$inference_wald_test_theta(theta, A, Om), error = function(e) NULL)
  if (is.null(w)) NA_real_ else w$chi2
}

# ---------------------------------------------------------------------------
# (1) cost of one valid permutation: refit + restudentize
# ---------------------------------------------------------------------------
bench <- function(thunk, min_total = 0.25) {
  thunk()
  reps <- 2L
  repeat {
    t <- system.time(for (i in seq_len(reps)) thunk())[["elapsed"]]
    if (t >= min_total || reps > 2e4L) break
    reps <- reps * 2L
  }
  t / reps * 1000
}

cost_table <- function(seed) {
  set.seed(seed)
  cat(sprintf("%-4s | %9s %9s | %9s %9s | %7s %8s\n",
              "p", "map_point", "map_full", "ml_point", "ml_full", "r_point", "r_full"))
  cat(strrep("-", 68), "\n")
  for (cell in list(c(2L, 3L), c(3L, 5L), c(5L, 5L))) {
    k <- cell[1L]; m <- cell[2L]; p <- k * m; n <- 300L
    Y <- sim_2group(n, k, m, phi1 = 1, phi2 = 2)
    dat <- data.frame(Y, g = rep(labs, each = n), stringsAsFactors = FALSE)
    spec <- model_spec(cfa_syntax(k, m), group = "g", group_labels = labs)
    pt <- spec$partable
    ss <- df_to_data(dat, spec, group = "g", missing = "error")
    gp <- bench(function() fit_map(pt, ss))
    gf <- bench(function() omega_map(fit_map(pt, ss), ss))
    mp <- bench(function() fit_ml(pt, ss))
    mf <- bench(function() omega_ml(fit_ml(pt, ss)))
    cat(sprintf("%-4d | %9.3f %9.3f | %9.3f %9.3f | %7.1f %8.2f\n",
                p, gp, gf, mp, mf, mp / gp, mf / gf))
  }
  cat("\nms/call. *_point = refit only; *_full = refit + rebuild Omega = ONE VALID\n",
      "permutation. r_* = ML / map; >1 favours the map.\n", sep = "")
}

# ---------------------------------------------------------------------------
# (2) level and power of the permutation Wald under the three studentizers
# ---------------------------------------------------------------------------
one_rep <- function(seed, n, phi1, phi2, dif, B) {
  set.seed(seed)
  k <- 2L; m <- 3L
  Y <- sim_2group(n, k, m, phi1 = phi1, phi2 = phi2, dif = dif)
  dat <- data.frame(Y, g = rep(labs, each = n), stringsAsFactors = FALSE)
  spec <- model_spec(cfa_syntax(k, m), group = "g", group_labels = labs)
  pt <- spec$partable
  ss0 <- df_to_data(dat, spec, group = "g", missing = "error")
  A <- make_A(pt)
  N <- 2L * n

  # label-invariant bread source: both groups carry the pooled covariance
  Spool <- cov(Y) * (N - 1L) / N
  ss_pool <- ss0
  ss_pool$S <- list(Spool, Spool)
  ss_pool$mean <- list(colMeans(Y), colMeans(Y))
  ss_pool$X <- setNames(list(Y, Y), labs)
  ss_pool$nobs <- c(n, n)
  g_pool <- fit_map(pt, ss_pool)
  if (is.null(g_pool)) return(NULL)
  Om_pboth <- omega_map(g_pool, ss_pool)
  if (is.null(Om_pboth)) return(NULL)

  g0 <- fit_map(pt, ss0)
  m0 <- fit_ml(pt, ss0)
  if (is.null(g0) || is.null(m0)) return(NULL)
  W_full <- wald(g0$theta, A, omega_map(g0, ss0))
  W_pbread <- wald(g0$theta, A, omega_map(g_pool, ss0))
  W_pboth <- wald(g0$theta, A, Om_pboth)
  W_ml <- wald(m0$theta, A, omega_ml(m0))

  P <- matrix(NA_real_, B, 4L)
  for (b in seq_len(B)) {
    j <- sample.int(N)
    ssb <- ss_split(ss0, Y[j[seq_len(n)], , drop = FALSE],
                    Y[j[-seq_len(n)], , drop = FALSE])
    gb <- fit_map(pt, ssb)
    if (!is.null(gb)) {
      P[b, 1L] <- wald(gb$theta, A, omega_map(gb, ssb))
      P[b, 2L] <- wald(gb$theta, A, omega_map(g_pool, ssb))
      P[b, 3L] <- wald(gb$theta, A, Om_pboth)
    }
    mb <- fit_ml(pt, ssb)
    if (!is.null(mb)) P[b, 4L] <- wald(mb$theta, A, omega_ml(mb))
  }
  pp <- function(Wobs, col) (1 + sum(col >= Wobs, na.rm = TRUE)) / (1 + sum(!is.na(col)))
  c(map_full = pp(W_full, P[, 1L]), map_pbread = pp(W_pbread, P[, 2L]),
    map_pboth = pp(W_pboth, P[, 3L]), ml_full = pp(W_ml, P[, 4L]))
}

level_table <- function(reps, B, seed) {
  ncores <- max(1L, parallel::detectCores() - 2L)
  cells <- list(
    list(tag = "n=50   phi=(1,1) dif=0",    n = 50L,  phi1 = 1, phi2 = 1, dif = 0),
    list(tag = "n=50   phi=(1,3) dif=0",    n = 50L,  phi1 = 1, phi2 = 3, dif = 0),
    list(tag = "n=100  phi=(1,1) dif=0",    n = 100L, phi1 = 1, phi2 = 1, dif = 0),
    list(tag = "n=100  phi=(1,3) dif=0",    n = 100L, phi1 = 1, phi2 = 3, dif = 0),
    list(tag = "n=300  phi=(1,3) dif=0",    n = 300L, phi1 = 1, phi2 = 3, dif = 0),
    list(tag = "n=100  phi=(1,2) dif=0.25", n = 100L, phi1 = 1, phi2 = 2, dif = 0.25)
  )
  cat(sprintf("reps=%d perms=%d cores=%d alpha=.05\n\n", reps, B, ncores))
  cat(sprintf("%-26s | %-31s | %8s\n", "", "closed-form map", "ML"))
  cat(sprintf("%-26s | %9s %9s %9s | %8s\n", "cell", "full", "pbread", "pboth", "full"))
  cat(strrep("-", 72), "\n")
  for (ci in seq_along(cells)) {
    cl <- cells[[ci]]
    res <- parallel::mclapply(seq_len(reps), function(i) {
      tryCatch(one_rep(seed + 31000L * ci + i, cl$n, cl$phi1, cl$phi2, cl$dif, B),
               error = function(e) NULL)
    }, mc.cores = ncores)
    res <- do.call(rbind, Filter(Negate(is.null), res))
    rej <- colMeans(res <= 0.05, na.rm = TRUE)
    cat(sprintf("%-26s | %9.3f %9.3f %9.3f | %8.3f   [%d]\n", cl$tag,
                rej["map_full"], rej["map_pbread"], rej["map_pboth"],
                rej["ml_full"], nrow(res)))
  }
  cat("\nrejection rate at alpha=.05. rows 1-5 = LEVEL (target .05); row 6 = POWER.\n")
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (opts$what %in% c("cost", "both")) {
  cat("== cost of one valid permutation ==\n")
  cost_table(opts$seed)
  cat("\n")
}
if (opts$what %in% c("level", "both")) {
  cat("== permutation Wald: level and power ==\n")
  level_table(opts$reps, opts$perms, opts$seed)
}
