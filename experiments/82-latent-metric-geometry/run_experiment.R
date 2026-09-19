#!/usr/bin/env Rscript
# Experiment 82: latent-metric parameterization -- geometry, cost, and inference.
#
# Supersedes experiments/_archive/02-latent-metric-identification, which timed
# marker vs std_lv on the golden-fixture models and reported a fit-only ratio of
# 0.839 that decayed to 0.996 end-to-end once the back-conversion was counted.
# Three things made that inconclusive, and this experiment fixes each:
#
#   1. No marker-strength axis. The fixture models carry whatever loadings they
#      happen to carry, but the entire marker/std_lv difference is controlled by
#      how strongly the marker indicator loads: the marker chart pins the scaling
#      gauge with lambda_1, so the gauge-fixing slice becomes tangent to the
#      gauge orbit as lambda_1 -> 0 and the chart degenerates. `lambda1` is the
#      primary design factor here.
#   2. Wall time only, at models small enough that timing is noise. magmaan
#      reports `f_evals`/`g_evals`, which measure optimizer work directly. On
#      real HS data the same model costs 62 f_evals under marker, 22 under
#      std_lv, 36 under effect coding -- a 3x spread that microsecond timings of
#      a sub-millisecond fit cannot resolve.
#   3. Timing only, no statistical quality. A chart cannot move chi-square, df,
#      or the LRT (all gauge-invariant), but it does move conditioning, the shape
#      of theta-hat's sampling distribution, and Wald tests. None of that was
#      measured anywhere in the repo.
#
# It also adds effect coding (previously only a timing arm), a model-size axis so
# the back-convert question from exp 02 can actually be answered, and the
# Bates & Watts (1980) curvature decomposition, which is what makes "is there a
# better parameterization?" a decidable question rather than a matter of taste.
#
# ARMS
#
# geometry  Deterministic, at the population moments; no replication needed
#           because these are population quantities. Per (chart, lambda1, p):
#           the pseudo-condition number of the expected information, and the
#           Bates-Watts split of the second-derivative array into
#           parameter-effects (PE) and intrinsic (IN) curvature. PE is
#           chart-dependent and reducible; IN is a property of the model manifold
#           and is invariant, so agreement of IN across charts is a built-in
#           correctness check on the whole computation. PE/IN is the headroom: a
#           chart with PE ~ IN has nothing left to gain from reparameterization.
#
# cost      Optimizer work and wall time per (chart, lambda1, p, backend), plus
#           the back-convert cost, so the fit-only and end-to-end ratios can be
#           read as a function of p. exp 02's wash was a small-p artifact: the
#           back-convert is O(p) while the fit is superlinear.
#
# inference Finite-sample, per (chart, lambda1, p): skewness of a comparable
#           loading estimate, Wald vs LRT rejection for a restriction that is
#           TRUE in the population, and the improper-solution rate. The LRT is
#           gauge-invariant, so the across-chart LRT spread is a second built-in
#           correctness check -- it must be numerical noise. Any Wald/LRT gap is
#           then attributable to the chart alone.
#
# The three charts are implemented twice on purpose. The cost and inference arms
# drive magmaan's own `model_spec(std_lv=, effect_coding=)` fits. The geometry arm
# needs Sigma(theta) at arbitrary theta in unconstrained coordinates (effect
# coding's raw parameter vector lives on a constraint surface, so differencing it
# directly would leave the surface), so it uses explicit closed-form chart maps --
# which are then validated against magmaan's `model_implied` at the population
# point, and the run aborts if they disagree.

suppressWarnings(suppressMessages(library(magmaan)))

source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))

set_single_threaded_math()

`%||%` <- function(x, y) if (is.null(x)) y else x

# Per-call wall time for sub-millisecond work.
#
# `system.time()` / `proc.time()` quantise to ~1 ms on Linux, so timing a single
# 225 us fit returns 0 or 0.001 -- measured directly: ten timings of the same fit
# give min 0.000, median 0.001, max 0.007. That is QUANTISATION, not variance, and
# it is why exp 02's per-fit timings could not resolve the difference it was
# looking for. Batching fixes it outright: 200 calls of that same fit divide out
# to 225.1 us with three digits of agreement across batches.
#
# Calibrate a batch size k so one batch runs at least `min_time`, then report the
# median and IQR of per-call time over `batches` batches. Warm up first so page
# faults and allocator growth land outside the measurement.
time_per_call <- function(f, min_time = 0.05, batches = 5L, warmup = 2L) {
  for (i in seq_len(warmup)) f()
  k <- 1L
  repeat {
    t0 <- Sys.time(); for (i in seq_len(k)) f()
    el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (el >= min_time || k >= 8192L) break
    k <- min(8192L, max(2L, as.integer(k * max(2, min_time / max(el, 1e-7)))))
  }
  ts <- vapply(seq_len(batches), function(b) {
    t0 <- Sys.time(); for (i in seq_len(k)) f()
    as.numeric(difftime(Sys.time(), t0, units = "secs")) / k
  }, numeric(1))
  c(median = stats::median(ts), iqr = stats::IQR(ts), batch_k = k)
}

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Latent-metric parameterization: geometry, optimizer cost, inference quality.\n\n",
    "Options:\n",
    "  --smoke            Quick run. Default.\n",
    "  --full             Paper-grade run.\n",
    "  --arms LIST        geometry,cost,inference. Default: all three.\n",
    "  --charts LIST      marker,std_lv,effect. Default: all three.\n",
    "  --bc-p LIST        p values for the back-conversion arm. Default: 6,12,24,48\n",
    "  --conv-n LIST      Sample sizes for the convergence arm.\n",
    "                     Default: 50,75,100,150,400\n",
    "  --lambda1 LIST     Marker-indicator loadings. Default: 0.3,0.5,0.7,0.9\n",
    "  --p LIST           Indicator counts. Default smoke: 6,12; full: 6,12,24\n",
    "  --curv-p LIST      p values for the O(p^4) curvature split. Default: 6,12\n",
    "  --n N              Sample size for the cost/inference arms. Default: 400\n",
    "  --reps N           Replications. Smoke: 20; full: 1000.\n",
    "  --backends LIST    Optimizers. Default: nlopt-lbfgs,port\n",
    "  --seed-base N      Base RNG seed. Default: 20260919\n",
    "  --results-dir PATH Output directory. Default: results\n",
    "  --help             Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  o <- list(smoke = TRUE,
            arms = c("geometry", "cost", "inference", "backconvert", "convergence"),
            charts = c("marker", "std_lv", "effect"),
            lambda1 = c(0.3, 0.5, 0.7, 0.9), p = NULL, curv_p = c(6L, 12L),
            bc_p = c(6L, 12L, 24L, 48L), conv_n = c(50L, 75L, 100L, 150L, 400L),
            n = 400L, reps = NULL, backends = c("nlopt-lbfgs", "port"),
            seed_base = 20260919L, results_dir = NULL)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]; nxt <- function() { i <<- i + 1L; args[[i]] }
    if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
    else if (a == "--smoke") o$smoke <- TRUE
    else if (a == "--full") o$smoke <- FALSE
    else if (a == "--arms") o$arms <- parse_csv_arg(nxt())
    else if (a == "--charts") o$charts <- parse_csv_arg(nxt())
    else if (a == "--lambda1") o$lambda1 <- parse_csv_numeric(nxt())
    else if (a == "--p") o$p <- as.integer(parse_csv_numeric(nxt()))
    else if (a == "--curv-p") o$curv_p <- as.integer(parse_csv_numeric(nxt()))
    else if (a == "--bc-p") o$bc_p <- as.integer(parse_csv_numeric(nxt()))
    else if (a == "--conv-n") o$conv_n <- as.integer(parse_csv_numeric(nxt()))
    else if (a == "--n") o$n <- as.integer(nxt())
    else if (a == "--reps") o$reps <- as.integer(nxt())
    else if (a == "--backends") o$backends <- parse_csv_arg(nxt())
    else if (a == "--seed-base") o$seed_base <- as.integer(nxt())
    else if (a == "--results-dir") o$results_dir <- nxt()
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (is.null(o$p)) o$p <- if (o$smoke) c(6L, 12L) else c(6L, 12L, 24L)
  if (is.null(o$reps)) o$reps <- if (o$smoke) 20L else 1000L
  bad <- setdiff(o$arms, c("geometry", "cost", "inference", "backconvert", "convergence"))
  if (length(bad)) stop("unknown arm(s): ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(o$charts, c("marker", "std_lv", "effect"))
  if (length(bad)) stop("unknown chart(s): ", paste(bad, collapse = ","), call. = FALSE)
  if (any(o$p < 4L)) stop("--p must be >= 4 (df > 0 for a one-factor model)", call. = FALSE)
  o
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
res_dir <- if (is.null(opts$results_dir)) results_dir(create = TRUE) else opts$results_dir
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------- population --
# One-factor CFA. The marker indicator loads `lambda1`; everything else loads
# 0.7. phi = 1 and psi_i = 1 - lambda_i^2, so every indicator has unit variance
# and lambda1 is the ONLY thing that varies across cells. That isolates gauge
# transversality: the marker chart anchors the scaling gauge with lambda1, so its
# conditioning should degrade as lambda1 falls while std_lv (anchored on phi) and
# effect coding (anchored on sum(lambda)) should not.
population <- function(p, lambda1) {
  lam <- c(lambda1, rep(0.7, p - 1L))
  list(lambda = lam, psi = 1 - lam^2, phi = 1,
       Sigma = tcrossprod(lam) + diag(1 - lam^2))
}

ov <- function(p) paste0("x", seq_len(p))
syntax_1f <- function(p) paste("f =~", paste(ov(p), collapse = " + "))

# Population theta in each chart's own unconstrained coordinates.
pop_theta <- function(pop, chart) {
  lam <- pop$lambda; p <- length(lam)
  switch(chart,
    marker = c(lam[-1L] / lam[1L], lam[1L]^2, pop$psi),
    std_lv = c(lam, pop$psi),
    # sum(lambda) == p pins the gauge; drop lambda_p and recover it from the
    # constraint so the chart is unconstrained and differentiable in place.
    effect = { c_ <- p / sum(lam); c(((lam * c_))[-p], 1 / c_^2, pop$psi) })
}

# Closed-form chart maps. theta -> vech(Sigma). Validated against magmaan below.
sigma_of <- function(theta, p, chart) {
  psi <- theta[(p + 1L):(2L * p)]
  M <- switch(chart,
    marker = { m <- c(1, theta[1L:(p - 1L)]); theta[p] * tcrossprod(m) },
    std_lv = { lam <- theta[1L:p]; tcrossprod(lam) },
    effect = { lam <- c(theta[1L:(p - 1L)], p - sum(theta[1L:(p - 1L)]))
               theta[p] * tcrossprod(lam) })
  M + diag(psi)
}
vech <- function(S) S[lower.tri(S, diag = TRUE)]

dup_matrix <- function(p) {
  q <- p * (p + 1L) / 2L; D <- matrix(0, p * p, q); k <- 1L
  for (j in seq_len(p)) for (i in j:p) {
    D[(j - 1L) * p + i, k] <- 1; D[(i - 1L) * p + j, k] <- 1; k <- k + 1L
  }
  D
}

# --------------------------------------------------------------- arm: geometry
# Bates & Watts (1980), Relative Curvature Measures of Nonlinearity, JRSS-B 42.
# QR-decompose the metric-whitened Jacobian, rotate the second-derivative array
# into that frame, normalise by R1^-1 so the tangent basis is orthonormal, then
# split: the first n rows are tangent to the model manifold (parameter-effects
# curvature, chart-dependent and in principle removable), the remaining q - n are
# normal to it (intrinsic curvature, a property of the manifold and invariant).
#
# Reported as Frobenius aggregates of the two sub-arrays rather than Bates-Watts'
# max-direction gamma. The Frobenius norm is invariant to orthogonal changes of
# the tangent and normal bases, which is what cross-chart comparison needs, but
# the absolute values are NOT gamma and must not be compared against their
# 0.3/sqrt(F) negligibility threshold. Evaluated at the population point, so the
# rho = s*sqrt(p) scaling is omitted; it is common to all charts and cancels.
bates_watts <- function(p, chart, theta) {
  q <- p * (p + 1L) / 2L; n <- length(theta)
  Sig <- sigma_of(theta, p, chart); Si <- solve(Sig)
  D <- dup_matrix(p)
  V <- 0.5 * t(D) %*% kronecker(Si, Si) %*% D        # Gamma_NT(Sigma)^-1 for vech
  ev <- eigen(V, symmetric = TRUE)
  Vh <- ev$vectors %*% diag(sqrt(pmax(ev$values, 0))) %*% t(ev$vectors)

  h1 <- 1e-5
  J <- vapply(seq_len(n), function(k) {
    e <- numeric(n); e[k] <- h1
    (vech(sigma_of(theta + e, p, chart)) - vech(sigma_of(theta - e, p, chart))) / (2 * h1)
  }, numeric(q))
  Vd <- Vh %*% J                                     # q x n

  h2 <- 1e-4
  Hw <- matrix(0, q, n * n)
  for (i in seq_len(n)) for (j in i:n) {
    ei <- numeric(n); ej <- numeric(n); ei[i] <- h2; ej[j] <- h2
    d2 <- (vech(sigma_of(theta + ei + ej, p, chart)) -
           vech(sigma_of(theta + ei - ej, p, chart)) -
           vech(sigma_of(theta - ei + ej, p, chart)) +
           vech(sigma_of(theta - ei - ej, p, chart))) / (4 * h2^2)
    w <- Vh %*% d2
    Hw[, (j - 1L) * n + i] <- w
    Hw[, (i - 1L) * n + j] <- w
  }

  qr1 <- qr(Vd)
  Q <- qr.Q(qr1, complete = TRUE)                    # q x q
  R1 <- qr.R(qr1)[seq_len(n), seq_len(n), drop = FALSE]
  Ri <- solve(R1)
  Bq <- crossprod(Q, Hw)                             # q x n^2, one matmul
  pe2 <- 0; in2 <- 0
  for (k in seq_len(q)) {
    M <- t(Ri) %*% matrix(Bq[k, ], n, n) %*% Ri
    if (k <= n) pe2 <- pe2 + sum(M^2) else in2 <- in2 + sum(M^2)
  }
  info <- crossprod(Vd)                              # expected information
  e <- sort(abs(eigen(info, symmetric = TRUE, only.values = TRUE)$values),
            decreasing = TRUE)
  e <- e[e > 1e-12 * e[1L]]
  list(PE = sqrt(pe2), IN = sqrt(in2), pcond_info = e[1L] / e[length(e)],
       npar = n, nmoment = q)
}

# Data whose sample covariance IS the population Sigma to machine precision, so a
# fit lands exactly on the population point and the validation below has no
# sampling slack to hide behind. magmaan's ML sample covariance uses the n
# divisor, so the whitening must too.
exact_cov_data <- function(Sigma, n, seed = 1L) {
  p <- ncol(Sigma)
  set.seed(seed)
  Z <- scale(matrix(stats::rnorm(n * p), n, p), center = TRUE, scale = FALSE)
  W <- backsolve(chol(crossprod(Z) / n), diag(p))
  X <- Z %*% W %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  as.data.frame(X)
}

# Cross-check the closed-form chart maps against magmaan, so the geometry arm is
# not an unvalidated hand-roll. Two independent assertions, both order-free (the
# partable's theta ordering is deliberately not assumed):
#   (a) sigma_of(pop_theta(chart)) == population Sigma -- the closed-form chart
#       map and its population theta are mutually consistent;
#   (b) magmaan's own implied Sigma, fitted under the same convention to
#       exact-moment data, == population Sigma.
# Together they pin both maps to the same manifold point.
validate_chart_map <- function(p, chart, theta, pop) {
  own <- max(abs(vech(sigma_of(theta, p, chart)) - vech(pop$Sigma)))
  syn <- syntax_1f(p)
  spec <- switch(chart,
    marker = model_spec(syn),
    std_lv = model_spec(syn, std_lv = TRUE),
    effect = model_spec(syn, effect_coding = TRUE, auto_fix_first = FALSE))
  S <- pop$Sigma; dimnames(S) <- list(ov(p), ov(p))
  dfr <- exact_cov_data(S, max(50L * p, 500L))
  fit <- tryCatch(magmaan:::fit_ml(spec, df_to_data(dfr, spec)), error = function(e) NULL)
  if (is.null(fit)) return(c(own = own, magmaan = NA_real_))
  imp <- tryCatch(unlist(magmaan:::model_implied(fit)), error = function(e) NULL)
  if (is.null(imp)) return(c(own = own, magmaan = NA_real_))
  c(own = own, magmaan = max(abs(as.numeric(imp) - as.numeric(pop$Sigma))))
}

run_geometry <- function(opts) {
  ps <- intersect(opts$p, opts$curv_p)
  if (!length(ps)) ps <- opts$curv_p
  rows <- list()
  for (p in ps) for (l1 in opts$lambda1) {
    pop <- population(p, l1)
    for (ch in opts$charts) {
      th <- pop_theta(pop, ch)
      bw <- tryCatch(bates_watts(p, ch, th), error = function(e) NULL)
      if (is.null(bw)) next
      v <- validate_chart_map(p, ch, th, pop)
      rows[[length(rows) + 1L]] <- data.frame(
        p = p, lambda1 = l1, chart = ch, npar = bw$npar, nmoment = bw$nmoment,
        pcond_info = bw$pcond_info, curv_pe = bw$PE, curv_in = bw$IN,
        pe_over_in = bw$PE / bw$IN,
        curv_total = sqrt(bw$PE^2 + bw$IN^2),
        validate_own_map = v[["own"]], validate_vs_magmaan = v[["magmaan"]],
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

# ------------------------------------------------------------------- arm: cost
# Back-convert std_lv / effect coordinates to the marker chart, which is what a
# library would have to do to use a better internal chart while still reporting
# the parameterization the user asked for. Pure O(p) algebra on the loadings.
backconvert_to_marker <- function(theta, p, chart) {
  switch(chart,
    marker = theta,
    std_lv = { lam <- theta[1L:p]
               c(lam[-1L] / lam[1L], lam[1L]^2, theta[(p + 1L):(2L * p)]) },
    effect = { lam <- c(theta[1L:(p - 1L)], p - sum(theta[1L:(p - 1L)]))
               phi <- theta[p]
               c(lam[-1L] / lam[1L], phi * lam[1L]^2, theta[(p + 1L):(2L * p)]) })
}

run_cost <- function(opts) {
  rows <- list()
  for (p in opts$p) for (l1 in opts$lambda1) {
    pop <- population(p, l1); syn <- syntax_1f(p)
    S <- pop$Sigma; dimnames(S) <- list(ov(p), ov(p))
    for (r in seq_len(opts$reps)) {
      set.seed(opts$seed_base + 1000L * p + as.integer(1000 * l1) + r)
      X <- tryCatch(MASS::mvrnorm(opts$n, rep(0, p), pop$Sigma), error = function(e) NULL)
      if (is.null(X)) next
      colnames(X) <- ov(p); dfr <- as.data.frame(X)
      for (ch in opts$charts) for (be in opts$backends) {
        spec <- switch(ch,
          marker = model_spec(syn),
          std_lv = model_spec(syn, std_lv = TRUE),
          effect = model_spec(syn, effect_coding = TRUE, auto_fix_first = FALSE))
        dd <- tryCatch(df_to_data(dfr, spec), error = function(e) NULL)
        if (is.null(dd)) next
        fit <- tryCatch(magmaan:::fit_ml(spec, dd, optimizer = be), error = function(e) NULL)
        if (is.null(fit)) next
        # Batched, so these are real microsecond measurements rather than 1 ms
        # quantisation noise. `spec` timing includes df_to_data because that is
        # what a caller pays per fit.
        ts <- time_per_call(function() { s <- switch(ch,
                marker = model_spec(syn),
                std_lv = model_spec(syn, std_lv = TRUE),
                effect = model_spec(syn, effect_coding = TRUE, auto_fix_first = FALSE))
              df_to_data(dfr, s) })
        tf <- time_per_call(function() magmaan:::fit_ml(spec, dd, optimizer = be))
        tb <- time_per_call(function() backconvert_to_marker(fit$theta, p, ch))
        rows[[length(rows) + 1L]] <- data.frame(
          p = p, lambda1 = l1, chart = ch, backend = be, replicate = r,
          n = opts$n, converged = isTRUE(fit$converged), fmin = fit$fmin,
          f_evals = fit$f_evals %||% NA_integer_,
          g_evals = fit$g_evals %||% NA_integer_,
          spec_sec = ts[["median"]], fit_sec = tf[["median"]],
          fit_sec_iqr = tf[["iqr"]], fit_batch_k = tf[["batch_k"]],
          backconvert_sec = tb[["median"]],
          pipeline_sec = ts[["median"]] + tf[["median"]] + tb[["median"]],
          stringsAsFactors = FALSE)
      }
    }
  }
  do.call(rbind, rows)
}

# ------------------------------------------------------------ arm: backconvert
# Using a better chart internally is only viable if converting back to the user's
# chart is both EXACT and CHEAP. exp 02 measured only the cost, and only for point
# estimates. Both halves matter, and the second one is where the accounting gets
# interesting:
#
#   * point estimates are O(p) -- a loading rescale, nothing more;
#   * the vcov needs the delta-method sandwich J V J', which is O(p^3) dense.
#
# A user who asks for the marker chart wants marker standard errors, not just
# marker point estimates, so the honest per-fit overhead is the second one. That is
# the term exp 02 never counted, and it is the term that can grow faster than the
# fit it is trying to save.
#
# Jacobian of std_lv -> marker: mu_i = lambda_i / lambda_1, phi = lambda_1^2, psi
# unchanged. Sparse in principle (each mu_i touches lambda_i and lambda_1 only), so
# a structure-aware implementation would be O(p^2); the dense product below is what
# a general library actually ships, and is timed as such.
jacobian_stdlv_to_marker <- function(theta, p) {
  lam <- theta[1L:p]; n <- 2L * p
  J <- matrix(0, n, n)
  for (i in 2L:p) {
    J[i - 1L, 1L] <- -lam[i] / lam[1L]^2
    J[i - 1L, i]  <- 1 / lam[1L]
  }
  J[p, 1L] <- 2 * lam[1L]
  J[(p + 1L):n, (p + 1L):n] <- diag(p)
  J
}

run_backconvert <- function(opts) {
  rows <- list()
  for (p in opts$bc_p) for (l1 in opts$lambda1) {
    pop <- population(p, l1); syn <- syntax_1f(p)
    set.seed(opts$seed_base + 31L + 1000L * p + as.integer(1000 * l1))
    X <- tryCatch(MASS::mvrnorm(opts$n, rep(0, p), pop$Sigma), error = function(e) NULL)
    if (is.null(X)) next
    colnames(X) <- ov(p); dfr <- as.data.frame(X)

    sp_m <- model_spec(syn)
    fm <- tryCatch(magmaan:::fit_ml(sp_m, df_to_data(dfr, sp_m)), error = function(e) NULL)
    if (is.null(fm)) next
    Sig_marker <- tryCatch(matrix(as.numeric(unlist(magmaan:::model_implied(fm))), p, p),
                           error = function(e) NULL)

    for (ch in setdiff(opts$charts, "marker")) {
      spec <- switch(ch,
        std_lv = model_spec(syn, std_lv = TRUE),
        effect = model_spec(syn, effect_coding = TRUE, auto_fix_first = FALSE))
      dd <- tryCatch(df_to_data(dfr, spec), error = function(e) NULL)
      if (is.null(dd)) next
      fit <- tryCatch(magmaan:::fit_ml(spec, dd), error = function(e) NULL)
      if (is.null(fit)) next

      # Accuracy first: this is the disqualifying check. Back-convert the fitted
      # theta into marker coordinates, push it through the closed-form marker map,
      # and require the SAME implied Sigma the native marker fit produced. If the
      # round trip is not exact, no amount of speed rescues the substitution.
      # Compared in Sigma space on purpose, because each chart's partable orders
      # its free parameters differently.
      th_own <- pop_theta(pop, ch)   # chart's own ordering, for the map
      bc <- tryCatch(backconvert_to_marker(th_own, p, ch), error = function(e) NULL)
      roundtrip <- if (is.null(bc) || is.null(Sig_marker)) NA_real_ else
        max(abs(sigma_of(bc, p, "marker") - pop$Sigma))

      t_point <- time_per_call(function() backconvert_to_marker(th_own, p, ch))
      # vcov transform, timed on magmaan's own expected information at the fit.
      core <- magmaan::magmaan_core
      V <- tryCatch({
        info <- core$inference_information_expected(fit)
        as.matrix(core$inference_vcov_fit(info, fit))
      }, error = function(e) NULL)
      # std_lv only, deliberately. `jacobian_stdlv_to_marker` is that chart's
      # Jacobian and is simply wrong for effect coding, whose free vector is
      # (lambda_1..lambda_{p-1}, phi, psi) of length 2p+1 rather than 2p. Timing a
      # knowingly-wrong matrix would still give the right wall time, since the
      # cost is the dense O(p^3) product and not the matrix contents, but shipping
      # it invites someone to read the numbers as a transform that works. Effect
      # coding's Jacobian has the same density and near-identical dimension, so its
      # vcov cost is the std_lv figure to within a percent.
      t_vcov <- NA_real_; vdim <- NA_integer_
      if (identical(ch, "std_lv") && !is.null(V) && nrow(V) == 2L * p) {
        J <- jacobian_stdlv_to_marker(th_own, p)
        tv <- time_per_call(function() J %*% V %*% t(J))
        t_vcov <- tv[["median"]]; vdim <- nrow(V)
      }
      t_fit <- time_per_call(function() magmaan:::fit_ml(spec, dd))
      rows[[length(rows) + 1L]] <- data.frame(
        p = p, lambda1 = l1, chart = ch, n = opts$n,
        roundtrip_max_abs_sigma = roundtrip,
        fit_sec = t_fit[["median"]],
        bc_point_sec = t_point[["median"]],
        bc_vcov_sec = t_vcov, vcov_dim = vdim,
        bc_point_pct_of_fit = 100 * t_point[["median"]] / t_fit[["median"]],
        bc_vcov_pct_of_fit = 100 * t_vcov / t_fit[["median"]],
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

# ----------------------------------------------------------- arm: convergence
# Conditioning should bite hardest where the problem is hard. The geometry arm is a
# population-level statement and says nothing about whether a badly conditioned
# chart actually FAILS more often at small n, which is the practically important
# question and the one the exp 03 archive only touched across three cases.
#
# Every chart sees the SAME dataset per (p, lambda1, n, replicate), so the charts
# can be compared per-draw: a draw where marker fails and std_lv succeeds is a
# concrete win rather than an aggregate one.
run_convergence <- function(opts) {
  rows <- list()
  for (p in opts$p) for (l1 in opts$lambda1) for (nn in opts$conv_n) {
    pop <- population(p, l1); syn <- syntax_1f(p)
    for (r in seq_len(opts$reps)) {
      set.seed(opts$seed_base + 91L + 1000L * p + 17L * nn + as.integer(1000 * l1) + r)
      X <- tryCatch(MASS::mvrnorm(nn, rep(0, p), pop$Sigma), error = function(e) NULL)
      if (is.null(X)) next
      colnames(X) <- ov(p); dfr <- as.data.frame(X)
      for (ch in opts$charts) for (be in opts$backends) {
        spec <- switch(ch,
          marker = model_spec(syn),
          std_lv = model_spec(syn, std_lv = TRUE),
          effect = model_spec(syn, effect_coding = TRUE, auto_fix_first = FALSE))
        dd <- tryCatch(df_to_data(dfr, spec), error = function(e) NULL)
        if (is.null(dd)) { errored <- TRUE; fit <- NULL } else {
          fit <- tryCatch(magmaan:::fit_ml(spec, dd, optimizer = be),
                          error = function(e) NULL)
        }
        vars <- if (is.null(fit)) numeric(0) else {
          pt <- fit$partable; sel <- pt$op == "~~" & pt$lhs == pt$rhs
          suppressWarnings(as.numeric(pt$est[sel]))
        }
        rows[[length(rows) + 1L]] <- data.frame(
          p = p, lambda1 = l1, n = nn, chart = ch, backend = be, replicate = r,
          errored = is.null(fit),
          converged = if (is.null(fit)) FALSE else isTRUE(fit$converged),
          fmin = if (is.null(fit)) NA_real_ else fit$fmin,
          f_evals = if (is.null(fit)) NA_integer_ else (fit$f_evals %||% NA_integer_),
          min_est_var = if (length(vars)) min(vars, na.rm = TRUE) else NA_real_,
          improper = if (length(vars)) any(vars <= 0, na.rm = TRUE) else NA,
          stringsAsFactors = FALSE)
      }
    }
  }
  do.call(rbind, rows)
}

# -------------------------------------------------------------- arm: inference
# The restriction lambda_{p-1} == lambda_p is TRUE in the population (both 0.7),
# so Wald and LRT rejection rates are Type-I error. The LRT tests the same
# submanifold in every chart, so its across-chart spread must be numerical noise;
# any Wald spread is then attributable to the chart.
run_inference <- function(opts) {
  rows <- list()
  for (p in opts$p) for (l1 in opts$lambda1) {
    pop <- population(p, l1)
    nm <- ov(p); a <- nm[p - 1L]; b <- nm[p]
    free <- setdiff(nm, c(a, b))
    syn_h1 <- paste("f =~", paste(nm, collapse = " + "))
    syn_h0 <- paste("f =~", paste(c(free, paste0("eq*", a), paste0("eq*", b)),
                                  collapse = " + "))
    for (r in seq_len(opts$reps)) {
      set.seed(opts$seed_base + 7L + 1000L * p + as.integer(1000 * l1) + r)
      X <- tryCatch(MASS::mvrnorm(opts$n, rep(0, p), pop$Sigma), error = function(e) NULL)
      if (is.null(X)) next
      colnames(X) <- nm; dfr <- as.data.frame(X)
      for (ch in opts$charts) {
        mk <- function(s) switch(ch,
          marker = model_spec(s),
          std_lv = model_spec(s, std_lv = TRUE),
          effect = model_spec(s, effect_coding = TRUE, auto_fix_first = FALSE))
        f1 <- tryCatch({ sp <- mk(syn_h1); magmaan:::fit_ml(sp, df_to_data(dfr, sp)) },
                       error = function(e) NULL)
        f0 <- tryCatch({ sp <- mk(syn_h0); magmaan:::fit_ml(sp, df_to_data(dfr, sp)) },
                       error = function(e) NULL)
        if (is.null(f1) || is.null(f0)) next
        m1 <- tryCatch(fit_measures(f1), error = function(e) NULL)
        m0 <- tryCatch(fit_measures(f0), error = function(e) NULL)
        lrt_p <- NA_real_
        if (!is.null(m1) && !is.null(m0)) {
          dchi <- m0$chisq - m1$chisq; ddf <- m0$df - m1$df
          if (is.finite(dchi) && is.finite(ddf) && ddf > 0)
            lrt_p <- stats::pchisq(dchi, ddf, lower.tail = FALSE)
        }
        # Pull the loading by NAME, not by theta index: each chart orders its
        # free parameters differently and the point is to compare charts.
        # `loading_a` is chart-specific on purpose -- that is the quantity whose
        # sampling distribution the chart is allowed to reshape.
        pt <- f1$partable
        sel_l <- pt$op == "=~" & pt$rhs == a
        sel_v <- pt$op == "~~" & pt$lhs == pt$rhs
        vars <- suppressWarnings(as.numeric(pt$est[sel_v]))
        rows[[length(rows) + 1L]] <- data.frame(
          p = p, lambda1 = l1, chart = ch, replicate = r, n = opts$n,
          converged = isTRUE(f1$converged),
          loading_a = if (any(sel_l)) suppressWarnings(as.numeric(pt$est[sel_l]))[1L]
                      else NA_real_,
          chisq_h1 = if (is.null(m1)) NA_real_ else m1$chisq,
          df_h1 = if (is.null(m1)) NA_integer_ else m1$df,
          lrt_p = lrt_p,
          # Improperness read off the estimated variances, not the implied
          # covariance: exp 03 found std_lv RELOCATES a Heywood case from the
          # latent variance to the observed ones rather than removing it, so the
          # detector has to look at every variance parameter in the chart.
          min_est_var = if (length(vars)) min(vars, na.rm = TRUE) else NA_real_,
          improper = if (length(vars)) any(vars <= 0, na.rm = TRUE) else NA,
          stringsAsFactors = FALSE)
      }
    }
  }
  do.call(rbind, rows)
}

# ------------------------------------------------------------------------ run --
cat(sprintf("experiment 82: mode=%s arms=%s charts=%s p=%s lambda1=%s reps=%d n=%d\n",
            if (opts$smoke) "smoke" else "full",
            paste(opts$arms, collapse = ","), paste(opts$charts, collapse = ","),
            paste(opts$p, collapse = ","), paste(opts$lambda1, collapse = ","),
            opts$reps, opts$n))

if ("geometry" %in% opts$arms) {
  g <- run_geometry(opts)
  write_csv(g, file.path(res_dir, "geometry.csv"))
  cat(sprintf("geometry: %d rows -> geometry.csv\n", nrow(g)))
  for (nmv in c("validate_own_map", "validate_vs_magmaan")) {
    bad <- g[[nmv]]; bad <- bad[is.finite(bad)]
    if (length(bad)) cat(sprintf("  %-20s max|d| = %.2e%s\n", nmv, max(bad),
        if (max(bad) > 1e-6) "   *** MISMATCH ***" else ""))
  }
  inv <- tapply(g$curv_in, list(g$p, g$lambda1), function(x)
    if (length(x) > 1L) diff(range(x)) / mean(x) else 0)
  cat(sprintf("  IN invariance across charts (max relative spread): %.2e\n",
              max(inv, na.rm = TRUE)))
}
if ("cost" %in% opts$arms) {
  cc <- run_cost(opts)
  write_csv(cc, file.path(res_dir, "cost.csv"))
  cat(sprintf("cost: %d rows -> cost.csv\n", nrow(cc)))
}
if ("backconvert" %in% opts$arms) {
  bcr <- run_backconvert(opts)
  write_csv(bcr, file.path(res_dir, "backconvert.csv"))
  cat(sprintf("backconvert: %d rows -> backconvert.csv\n", nrow(bcr)))
  rt <- bcr$roundtrip_max_abs_sigma; rt <- rt[is.finite(rt)]
  if (length(rt)) cat(sprintf("  round-trip exactness (max|dSigma|): %.2e%s\n", max(rt),
      if (max(rt) > 1e-8) "   *** NOT EXACT ***" else ""))
}
if ("convergence" %in% opts$arms) {
  cv <- run_convergence(opts)
  write_csv(cv, file.path(res_dir, "convergence.csv"))
  cat(sprintf("convergence: %d rows -> convergence.csv\n", nrow(cv)))
}
if ("inference" %in% opts$arms) {
  ii <- run_inference(opts)
  write_csv(ii, file.path(res_dir, "inference.csv"))
  cat(sprintf("inference: %d rows -> inference.csv\n", nrow(ii)))
}

write_metadata(file.path(res_dir, "metadata.csv"),
  values = list(mode = if (opts$smoke) "smoke" else "full",
                arms = paste(opts$arms, collapse = ","),
                charts = paste(opts$charts, collapse = ","),
                p = paste(opts$p, collapse = ","),
                curv_p = paste(opts$curv_p, collapse = ","),
                bc_p = paste(opts$bc_p, collapse = ","),
                conv_n = paste(opts$conv_n, collapse = ","),
                lambda1 = paste(opts$lambda1, collapse = ","),
                n = opts$n, reps = opts$reps,
                backends = paste(opts$backends, collapse = ","),
                seed_base = opts$seed_base,
                # Provenance only. Deliberately not a path: invariant 3 in
                # AGENTS.md forbids an experiment leaf referencing a sibling
                # experiment from code, and a metadata string value is code.
                supersedes = "archived experiment 02 (latent metric identification)"),
  packages = c("magmaan", "MASS"))
cat("metadata -> metadata.csv\n")
