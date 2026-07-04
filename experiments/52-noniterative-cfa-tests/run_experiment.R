#!/usr/bin/env Rscript

# Finite-sample validation of residual-based delta-method inference for the
# closed-form (non-iterative) Guttman CFA estimator, against magmaan's own ML.
#
# The machinery under test (estimate::frontier / robust::frontier, exposed as
# magmaan_core$noniterative_cfa_{fit,inference,wald,difference}_impl) gives any
# closed-form sigma -> theta CFA estimator three post-fit outputs: delta-method
# standard errors, a residual goodness-of-fit test, and a difference / pseudo-LRT
# test. This experiment asks whether those outputs are calibrated in finite
# samples, and at what efficiency cost relative to ML, across three data
# generators sharing one 3-factor CFA population:
#
#   normal   -- multivariate normal (NT Gamma is correct).
#   ig       -- Foldnes-Olsson independent generator, non-normal continuous with
#               the covariance held exactly at the model-implied Sigma.
#   ordinal  -- five-category ordinal codes, Pearson-calibrated so the observed
#               covariance is an exact 3-factor model (a diagonal rescaling of a
#               factor model is a factor model), then treated as continuous.
#
# Because every generator holds the CFA exactly (correct spec), goodness-of-fit
# and coverage measure Type-I / nominal calibration; a second misspecified arm
# (one omitted cross-loading in the population) measures GOF power.
#
# The base message: the delta-method inference is calibrated on all three
# generators *when the empirical Gamma is used*; the normal-theory Gamma is
# calibrated only under normality (the ULS/NTML residual-GOF analogue of the
# Satorra-Bentler robustness story, mirroring Dhaene-Rosseel 2024).
#
# The study sweeps named reliability `regimes` (see regime_reliabilities). The
# `bal.*` regimes hold all three indicators at a common reliability (0.3/0.5/0.7)
# and reproduce the uniform reliability sweep. The `het.*` regimes hold the
# average reliability at 0.5 but spread it across the three positions, placing a
# weak indicator (position 3) next to a strong marker: at matched reliability,
# weak loadings and noisy residuals are the same experiment (both estimators are
# scale-equivariant), so heterogeneity is the only "weak loadings" manipulation
# that is not already the reliability sweep. It stresses the closed form's
# per-variable communality step, which should pull ML ahead specifically on the
# weak indicator's loading. We track loadings split by position (p2 / p3), whether
# the inference stays calibrated, and graceful degradation (Heywood / improper
# rate and convergence: the closed form cannot fail to converge, unlike ML).

suppressWarnings(suppressMessages(library(magmaan)))
source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))

core <- magmaan::magmaan_core

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Validate delta-method SE / GOF / difference-test calibration for the\n",
    "closed-form Guttman CFA estimator against magmaan ML.\n\n",
    "Options:\n",
    "  --smoke            One quick slice (n=250, reps=40, normal+ordinal). Default.\n",
    "  --full             Full generator x reliability x n grid at --reps reps.\n",
    "  --reps N           Replications per cell (full). Default: 1000.\n",
    "  --n LIST           Comma-separated sample sizes. Default: 50,100,250,500,1000.\n",
    "  --regimes LIST     Reliability regimes (see regime_reliabilities).\n",
    "                     Default: bal.3,bal.5,bal.7,het.mod,het.str.\n",
    "  --generators LIST  Subset of normal,ig,ordinal. Default: all.\n",
    "  --specs LIST       Subset of correct,misspec. Default: both.\n",
    "  --ref-n N          Reference draw for the ordinal population target. Default: 200000.\n",
    "  --alpha X          Test level for rejection / coverage. Default: 0.05.\n",
    "  --cores N          mclapply cores. Default: detectCores()-1.\n",
    "  --seed-base N      Base seed. Default: 20260704.\n",
    "  --results-dir PATH Output directory. Default: results.\n",
    "  --help             Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    smoke = TRUE, full = FALSE,
    reps = 1000L,
    n = c(50L, 100L, 250L, 500L, 1000L),
    regimes = c("bal.3", "bal.5", "bal.7", "het.mod", "het.str"),
    generators = c("normal", "ig", "ordinal"),
    specs = c("correct", "misspec"),
    ref_n = 200000L,
    alpha = 0.05,
    cores = max(1L, parallel::detectCores() - 1L),
    seed_base = 20260704L,
    results_dir = experiment_path("results")
  )
  i <- 1L
  explicit <- character(0)
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value for ", args[[i - 1L]], call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help") { usage(); quit(status = 0) }
    else if (a == "--smoke") { opts$smoke <- TRUE; opts$full <- FALSE }
    else if (a == "--full") { opts$full <- TRUE; opts$smoke <- FALSE }
    else if (a == "--reps") { opts$reps <- as.integer(take()); explicit <- c(explicit, "reps") }
    else if (a == "--n") { opts$n <- as.integer(parse_csv_arg(take())); explicit <- c(explicit, "n") }
    else if (a == "--regimes") { opts$regimes <- parse_csv_arg(take()); explicit <- c(explicit, "regimes") }
    else if (a == "--generators") { opts$generators <- parse_csv_arg(take()); explicit <- c(explicit, "generators") }
    else if (a == "--specs") { opts$specs <- parse_csv_arg(take()); explicit <- c(explicit, "specs") }
    else if (a == "--ref-n") { opts$ref_n <- as.integer(take()); explicit <- c(explicit, "ref_n") }
    else if (a == "--alpha") opts$alpha <- as.numeric(take())
    else if (a == "--cores") opts$cores <- as.integer(take())
    else if (a == "--seed-base") opts$seed_base <- as.integer(take())
    else if (a == "--results-dir") opts$results_dir <- take()
    else stop("unknown option: ", a, call. = FALSE)
    i <- i + 1L
  }
  opts$explicit <- explicit
  bad <- setdiff(opts$generators, c("normal", "ig", "ordinal"))
  if (length(bad)) stop("unknown generators: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$specs, c("correct", "misspec"))
  if (length(bad)) stop("unknown specs: ", paste(bad, collapse = ","), call. = FALSE)
  opts
}

opts <- parse_args(commandArgs(TRUE))
set_single_threaded_math()

# Smoke mode fills only the fields the user did not set explicitly, so an
# individual override (e.g. --generators ig) still exercises that path quickly.
if (isTRUE(opts$smoke)) {
  if (!"reps" %in% opts$explicit) opts$reps <- 40L
  if (!"n" %in% opts$explicit) opts$n <- 250L
  if (!"regimes" %in% opts$explicit) opts$regimes <- c("bal.5", "het.str")
  if (!"generators" %in% opts$explicit) opts$generators <- c("normal", "ordinal")
  if (!"ref_n" %in% opts$explicit) opts$ref_n <- 40000L
}
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
z_crit <- stats::qnorm(1 - opts$alpha / 2)

# ---------------------------------------------------------------------------
# Population: three correlated factors, simple structure, one factor pair
# (f1,f3) uncorrelated so `f1 ~~ 0*f3` is a true restriction for the difference
# test. Misspec adds an omitted cross-loading x4 ~ f1 to the DGP only.
#
# A `regime` is a per-position reliability triple applied to every factor's three
# indicators (position 1 = marker, 2, 3). At matched reliability, weak loadings
# and noisy residuals give the same correlation matrix (both estimators are
# scale-equivariant), so the only manipulation that is *not* redundant with the
# reliability sweep is heterogeneity: a weak indicator among strong ones. The
# `bal.*` regimes reproduce the (uniform) reliability sweep; the `het.*` regimes
# hold the average reliability at 0.5 and spread it, putting a weak indicator
# (position 3) next to a strong marker (position 1). Residual variances are held
# at 1 and the DGP loading realizes the target reliability lambda = sqrt(r/(1-r)),
# so a low-reliability position is literally a weak loading.
# ---------------------------------------------------------------------------

factors <- c("f1", "f2", "f3")
ov <- paste0("x", 1:9)
factor_of <- rep(1:3, each = 3)                 # indicator -> factor
pos_of <- ((seq_len(9) - 1L) %% 3L) + 1L        # indicator -> within-factor position
cross_frac <- 0.35                               # misspec cross-loading, x4 ~ f1

regime_reliabilities <- list(
  "bal.3"   = c(0.3, 0.3, 0.3),   # uniform low   (reliability sweep)
  "bal.5"   = c(0.5, 0.5, 0.5),   # uniform mid
  "bal.7"   = c(0.7, 0.7, 0.7),   # uniform high
  "het.mod" = c(0.65, 0.5, 0.35), # moderate spread, average 0.5
  "het.str" = c(0.80, 0.5, 0.20)) # strong spread,   average 0.5

model_h1 <- "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf3 =~ x7 + x8 + x9\n"
model_h0 <- paste0(model_h1, "f1 ~~ 0*f3\n")

make_population <- function(spec, regime) {
  r <- regime_reliabilities[[regime]]
  if (is.null(r)) stop("unknown regime: ", regime, call. = FALSE)
  lam_pos <- sqrt(r / (1 - r))                   # DGP loading per position
  lambda_all <- lam_pos[pos_of]
  L <- matrix(0, 9, 3)
  L[cbind(1:9, factor_of)] <- lambda_all
  Phi <- matrix(c(1, .3, 0,
                  .3, 1, .3,
                  0, .3, 1), 3, 3, byrow = TRUE)
  theta <- rep(1, 9)
  Ldgp <- L
  if (identical(spec, "misspec")) Ldgp[4, 1] <- cross_frac * mean(lam_pos)
  Sigma <- Ldgp %*% Phi %*% t(Ldgp) + diag(theta, 9)
  if (min(eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values) <= 1e-8) {
    stop("population covariance not PD for spec=", spec, call. = FALSE)
  }
  dimnames(Sigma) <- list(ov, ov)
  list(Sigma = Sigma, Lambda = L, Phi = Phi, theta = theta,
       spec = spec, regime = regime, avg_reliability = mean(r))
}

# Validate requested regimes now that the table exists.
{
  bad <- setdiff(opts$regimes, names(regime_reliabilities))
  if (length(bad)) stop("unknown regimes: ", paste(bad, collapse = ","), call. = FALSE)
}

# ---------------------------------------------------------------------------
# Parameter bookkeeping: map each free partable row to its analysis target and
# a reporting group. Guttman and ML both lavaanify `model_h1`, so they share the
# free-index layout captured here.
# ---------------------------------------------------------------------------

pt_h1 <- core$lavaan_lavaanify(model_h1)
pt_h0 <- core$lavaan_lavaanify(model_h0)
free_rows <- which(pt_h1$free > 0)
free_idx <- pt_h1$free[free_rows]
stopifnot(identical(sort(free_idx), seq_along(free_idx)))

param_group <- vapply(free_rows, function(r) {
  op <- pt_h1$op[r]; lhs <- pt_h1$lhs[r]; rhs <- pt_h1$rhs[r]
  if (op == "=~") "loadings"
  else if (op == "~~" && lhs %in% factors && rhs %in% factors) "factor_cov"
  else "residual"
}, character(1))

# Within-factor position of each free loading (2 = middle indicator, 3 = last;
# markers at position 1 are fixed). Under het regimes position 3 is the weak
# indicator, so loadings_p3 isolates the weak-loading estimate.
param_pos <- vapply(free_rows, function(r) {
  if (pt_h1$op[r] == "=~") pos_of[as.integer(sub("x", "", pt_h1$rhs[r]))] else 0L
}, integer(1))

# A reporting group is a logical mask over the free parameters.
group_defs <- list(
  loadings    = param_group == "loadings",
  loadings_p2 = param_group == "loadings" & param_pos == 2L,
  loadings_p3 = param_group == "loadings" & param_pos == 3L,
  factor_cov  = param_group == "factor_cov",
  residual    = param_group == "residual",
  all         = rep(TRUE, length(free_rows)))
group_levels <- names(group_defs)

group_reduce <- function(vals) {
  vapply(group_defs, function(m) mean(vals[m]), numeric(1))
}

# Per-cell analysis targets come from population_target() (the Guttman map of the
# actual population covariance, which equals ML on-model); the startup block
# checks that closed form against ML on Sigma directly.

# ---------------------------------------------------------------------------
# Generators. Each (generator, spec) is calibrated once; per-cell draws are
# batched for ig/ordinal and drawn per-replicate for normal. The population
# target theta for coverage/risk is the Guttman map of the population covariance
# (= ML on-model): Sigma for normal/ig, the reference-draw covariance for the
# rescaled ordinal covariance.
# ---------------------------------------------------------------------------

# Asymmetric five-category thresholds: the codes are right-skewed with a floor,
# so the treated-as-continuous fourth moments are genuinely non-normal. This is
# what makes the empirical Gamma matter on the ordinal arm (symmetric categories
# are too near-normal to stress the normal-theory Gamma).
ordinal_probs <- {
  tau <- c(-0.5, 0.2, 0.8, 1.5)
  diff(c(0, stats::pnorm(tau), 1))
}

calibrate_generator <- function(generator, spec, regime) {
  pop <- make_population(spec, regime)
  Sigma <- pop$Sigma
  if (generator == "normal") {
    return(list(pop = pop, kind = "normal", chol = chol(Sigma)))
  }
  if (generator == "ig") {
    cal <- core$sim_ig_calibrate(
      Sigma,
      target_skewness = rep(2, 9),
      target_excess_kurtosis = rep(7, 9),
      root = "symmetric", generator_family = "pearson")
    return(list(pop = pop, kind = "ig", cal = cal))
  }
  # ordinal: Pearson-calibrate the latent so observed code correlation = cov2cor(Sigma)
  marginals <- stats::setNames(rep(list(ordinal_probs), 9), ov)
  cal <- core$sim_ordcorr_calibrate(
    stats::cov2cor(Sigma), marginals, metric = "pearson_codes")
  err <- max(abs(cal$achieved_corr - stats::cov2cor(Sigma)))
  if (err > 5e-3) {
    warning(sprintf("ordinal calibration (%s): max achieved-corr error %.4g", spec, err))
  }
  list(pop = pop, kind = "ordinal", cal = cal, calib_err = err)
}

# Draw a batch of `reps` datasets for one cell; returns a function idx -> matrix.
batch_sampler <- function(gcal, n, reps, seed) {
  if (gcal$kind == "normal") {
    return(function(i) {
      set.seed(seed + i)
      X <- matrix(stats::rnorm(n * 9), n, 9) %*% gcal$chol
      colnames(X) <- ov; X
    })
  }
  if (gcal$kind == "ig") {
    batch <- core$sim_ig_draw(gcal$cal, n = n, reps = reps, seed_base = seed)
    return(function(i) { X <- batch$draws[[i]]; colnames(X) <- ov; X })
  }
  batch <- core$sim_ordcorr_draw(gcal$cal, n = n, reps = reps, seed_base = seed)
  function(i) {
    X <- matrix(as.numeric(batch$draws[[i]]$X), nrow = n)
    colnames(X) <- ov; X
  }
}

# Population target theta for a (generator, spec): Guttman map of the population
# covariance (= ML on-model). For normal/ig this is Sigma exactly; for ordinal
# we take a large reference draw's covariance (the rescaled-but-exact factor
# covariance of the treated-as-continuous codes).
population_target <- function(gcal, seed) {
  if (gcal$kind %in% c("normal", "ig")) {
    Spop <- gcal$pop$Sigma
  } else {
    draw <- batch_sampler(gcal, opts$ref_n, 1L, seed)(1L)
    Spop <- stats::cov(draw)
    dimnames(Spop) <- list(ov, ov)
  }
  gfit <- tryCatch(
    core$noniterative_cfa_fit_impl(pt_h1, list(S = Spop, nobs = opts$ref_n), "guttman"),
    error = function(e) NULL)
  theta <- if (is.null(gfit)) rep(NA_real_, length(free_idx)) else gfit$theta[free_idx]
  list(Spop = Spop, theta = theta)
}

# ---------------------------------------------------------------------------
# Metric layout. Each replicate returns a fixed-name numeric vector so a cell is
# aggregated with colMeans(na.rm). Coverage / risk / difference metrics are only
# defined under the correct spec (where the analysis target is the estimand);
# GOF rejection is recorded under both (Type-I under correct, power under misspec).
# ---------------------------------------------------------------------------

se_variants   <- c("g_ntml_nt", "g_ntml_emp", "g_uls_nt", "g_uls_emp", "ml_nt", "ml_rob")
gof_variants  <- c("g_ntml_nt", "g_ntml_emp", "g_uls_nt", "g_uls_emp")
gof_ptypes    <- c("scaled", "meanvar", "scaled_shifted", "mixture")
ml_gof_ptypes <- c("nt", "sb", "mv", "ss")
diff_variants <- c("ntml_nt", "ntml_emp")
diff_ptypes   <- c("scaled", "mixture")
estimators    <- c("guttman", "ml")

metric_names <- c(
  as.vector(outer(se_variants, group_levels, function(v, g) paste0("cov_", v, "_", g))),
  as.vector(outer(estimators, group_levels, function(e, g) paste0("mse_", e, "_", g))),
  as.vector(outer(gof_variants, gof_ptypes, function(v, p) paste0("rej_", v, "_", p))),
  paste0("rej_ml_", ml_gof_ptypes),
  as.vector(outer(diff_variants, diff_ptypes, function(v, p) paste0("rej_diff_", v, "_", p))),
  paste0("gofT_", gof_variants),
  "conv_guttman", "conv_ml", "improper_guttman", "improper_ml"
)

worker <- function(i, sampler, target, spec) {
  out <- stats::setNames(rep(NA_real_, length(metric_names)), metric_names)
  X <- tryCatch(sampler(i), error = function(e) NULL)
  if (is.null(X)) return(out)
  Xdf <- as.data.frame(X)
  ss <- tryCatch(core$data_sample_stats_from_raw(list(X)), error = function(e) NULL)
  if (is.null(ss)) return(out)

  ## Guttman closed-form fit
  gfit <- tryCatch(core$noniterative_cfa_fit_impl(pt_h1, ss, "guttman"),
                   error = function(e) NULL)
  out["conv_guttman"] <- as.numeric(!is.null(gfit))
  if (!is.null(gfit)) {
    est_g <- gfit$theta[free_idx]
    out["improper_guttman"] <- as.numeric(any(est_g[param_group == "residual"] < 0))
    infs <- list(
      g_ntml_nt  = tryCatch(core$noniterative_cfa_inference_impl(gfit, "guttman", "ntml", "nt", NULL), error = function(e) NULL),
      g_ntml_emp = tryCatch(core$noniterative_cfa_inference_impl(gfit, "guttman", "ntml", "empirical", X), error = function(e) NULL),
      g_uls_nt   = tryCatch(core$noniterative_cfa_inference_impl(gfit, "guttman", "uls", "nt", NULL), error = function(e) NULL),
      g_uls_emp  = tryCatch(core$noniterative_cfa_inference_impl(gfit, "guttman", "uls", "empirical", X), error = function(e) NULL))
    for (v in names(infs)) {
      inf <- infs[[v]]
      if (is.null(inf)) next
      out[paste0("gofT_", v)] <- inf$T
      pv <- c(scaled = inf$p_scaled, meanvar = inf$p_meanvar,
              scaled_shifted = inf$p_scaled_shifted, mixture = inf$p_mixture)
      for (p in gof_ptypes) out[paste0("rej_", v, "_", p)] <- as.numeric(pv[[p]] < opts$alpha)
      if (identical(spec, "correct")) {
        ci <- as.numeric(abs(est_g - target) <= z_crit * inf$se[free_idx])
        gr <- group_reduce(ci)
        for (g in group_levels) out[paste0("cov_", v, "_", g)] <- gr[[g]]
      }
    }
    if (identical(spec, "correct")) {
      gr <- group_reduce((est_g - target)^2)
      for (g in group_levels) out[paste0("mse_guttman_", g)] <- gr[[g]]
      g0 <- tryCatch(core$noniterative_cfa_fit_impl(pt_h0, ss, "guttman"), error = function(e) NULL)
      if (!is.null(g0)) {
        dtests <- list(
          ntml_nt  = tryCatch(core$noniterative_cfa_difference_impl(g0, gfit, 1L, "guttman", "ntml", "nt", NULL, NULL), error = function(e) NULL),
          ntml_emp = tryCatch(core$noniterative_cfa_difference_impl(g0, gfit, 1L, "guttman", "ntml", "empirical", X, X), error = function(e) NULL))
        for (dv in names(dtests)) {
          d <- dtests[[dv]]
          if (is.null(d)) next
          out[paste0("rej_diff_", dv, "_scaled")]  <- as.numeric(d$p_scaled < opts$alpha)
          out[paste0("rej_diff_", dv, "_mixture")] <- as.numeric(d$p_mixture < opts$alpha)
        }
      }
    }
  }

  ## ML reference (magmaan's own iterative fit)
  mlfit <- tryCatch(suppressMessages(magmaan::magmaan(model_h1, Xdf, estimator = "ML")),
                    error = function(e) NULL)
  ok_ml <- !is.null(mlfit) && isTRUE(mlfit$converged)
  out["conv_ml"] <- as.numeric(ok_ml)
  if (ok_ml) {
    est_m <- mlfit$partable$est[free_rows]
    out["improper_ml"] <- as.numeric(any(est_m[param_group == "residual"] < 0))
    if (identical(spec, "correct")) {
      se_nt <- tryCatch(sqrt(diag(stats::vcov(mlfit, regime = "model", data = Xdf)))[free_idx],
                        error = function(e) rep(NA_real_, length(free_idx)))
      se_rb <- tryCatch(sqrt(diag(stats::vcov(mlfit, regime = "robust", data = Xdf)))[free_idx],
                        error = function(e) rep(NA_real_, length(free_idx)))
      for (vv in c("ml_nt", "ml_rob")) {
        se <- if (vv == "ml_nt") se_nt else se_rb
        gr <- group_reduce(as.numeric(abs(est_m - target) <= z_crit * se))
        for (g in group_levels) out[paste0("cov_", vv, "_", g)] <- gr[[g]]
      }
      gr <- group_reduce((est_m - target)^2)
      for (g in group_levels) out[paste0("mse_ml_", g)] <- gr[[g]]
    }
    fm <- tryCatch(magmaan::fit_measures(mlfit), error = function(e) NULL)
    if (!is.null(fm)) out["rej_ml_nt"] <- as.numeric(fm[["pvalue"]] < opts$alpha)
    ft <- tryCatch(magmaan::fmg_tests(mlfit, data = Xdf, tests = c("sb", "mv", "ss")),
                   error = function(e) NULL)
    if (!is.null(ft)) {
      pget <- function(pre) {
        idx <- grep(paste0("^", pre), ft$label)
        if (length(idx)) ft$p_value[idx[1]] else NA_real_
      }
      out["rej_ml_sb"] <- as.numeric(pget("sb") < opts$alpha)
      out["rej_ml_mv"] <- as.numeric(pget("mv") < opts$alpha)
      out["rej_ml_ss"] <- as.numeric(pget("ss") < opts$alpha)
    }
  }
  out
}

# ---------------------------------------------------------------------------
# Startup validation: Guttman and ML share the free-index layout, and the
# Guttman map of Sigma reproduces the analysis target on-model.
# ---------------------------------------------------------------------------

local({
  # On-model, Guttman(Sigma) and ML(Sigma) recover the same exact parameters;
  # check on the most heterogeneous regime (the hardest case for the closed form).
  pop <- make_population("correct", "het.str")
  ss <- list(S = pop$Sigma, nobs = 100000L)
  gt <- core$noniterative_cfa_fit_impl(pt_h1, ss, "guttman")$theta[free_idx]
  ml <- core$fit_fit(pt_h1, ss)$theta[free_idx]
  if (max(abs(gt - ml)) > 1e-4) {
    stop("Guttman(Sigma) != ML(Sigma) on-model (max err ",
         signif(max(abs(gt - ml)), 3), "); population may not be an exact factor model",
         call. = FALSE)
  }
  X <- matrix(stats::rnorm(500 * 9), 500, 9) %*% chol(pop$Sigma); colnames(X) <- ov
  mlfit <- suppressMessages(magmaan::magmaan(model_h1, as.data.frame(X), estimator = "ML"))
  if (!identical(as.integer(mlfit$partable$free), as.integer(pt_h1$free))) {
    stop("ML partable free-index layout differs from lavaanify(model_h1)", call. = FALSE)
  }
})

# ---------------------------------------------------------------------------
# Run the grid.
# ---------------------------------------------------------------------------

design <- expand.grid(generator = opts$generators, spec = opts$specs,
                      regime = opts$regimes, n = opts$n,
                      stringsAsFactors = FALSE)
design <- design[order(design$generator, design$spec, design$regime, design$n), ]

message(sprintf("Grid: %d cells (%s) x regime(%s) x n(%s) x spec(%s); reps=%d cores=%d",
                nrow(design), paste(opts$generators, collapse = ","),
                paste(opts$regimes, collapse = ","),
                paste(opts$n, collapse = ","), paste(opts$specs, collapse = ","),
                opts$reps, opts$cores))

gen_cache <- new.env(parent = emptyenv())
gen_setup <- function(generator, spec, regime, gi, si, ri) {
  key <- paste(generator, spec, regime, sep = "|")
  if (!is.null(gen_cache[[key]])) return(gen_cache[[key]])
  message(sprintf("  calibrating %s / %s / %s ...", generator, spec, regime))
  gcal <- calibrate_generator(generator, spec, regime)
  tgt <- population_target(gcal, opts$seed_base + gi * 1000003L + si * 300007L + ri * 90001L + 777L)
  setup <- list(gcal = gcal, target = tgt$theta,
                avg_reliability = gcal$pop$avg_reliability,
                calib_err = if (is.null(gcal$calib_err)) NA_real_ else gcal$calib_err)
  gen_cache[[key]] <- setup
  setup
}

cov_rows <- list(); mse_rows <- list(); gof_rows <- list()
diff_rows <- list(); diag_rows <- list()
push <- function(lst, row) { lst[[length(lst) + 1L]] <- row; lst }

t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(design))) {
  generator <- design$generator[ci]; spec <- design$spec[ci]
  regime <- design$regime[ci]; n <- design$n[ci]
  gi <- match(generator, c("normal", "ig", "ordinal"))
  si <- match(spec, c("correct", "misspec"))
  ri <- match(regime, names(regime_reliabilities))
  setup <- gen_setup(generator, spec, regime, gi, si, ri)
  cell_seed <- opts$seed_base + gi * 1000003L + si * 300007L + ri * 90001L + n * 101L
  sampler <- batch_sampler(setup$gcal, n, opts$reps, cell_seed)

  message(sprintf("[%d/%d] %s / %s / %s / n=%d ...",
                  ci, nrow(design), generator, spec, regime, n))
  res <- parallel::mclapply(seq_len(opts$reps), function(i)
    worker(i, sampler, setup$target, spec),
    mc.cores = opts$cores, mc.preschedule = TRUE)
  ok <- vapply(res, function(r) is.numeric(r) && length(r) == length(metric_names), logical(1))
  M <- do.call(rbind, res[ok])
  cm <- colMeans(M, na.rm = TRUE)
  base <- list(generator = generator, spec = spec, regime = regime,
               avg_reliability = setup$avg_reliability, n = n, reps = sum(ok))

  for (v in se_variants) for (g in group_levels)
    cov_rows <- push(cov_rows, data.frame(base, variant = v, group = g,
                                          coverage = cm[[paste0("cov_", v, "_", g)]]))
  for (e in estimators) for (g in group_levels) {
    mse <- cm[[paste0("mse_", e, "_", g)]]
    mse_rows <- push(mse_rows, data.frame(base, estimator = e, group = g,
                                          mse = mse, rmse = sqrt(mse)))
  }
  for (v in gof_variants) for (p in gof_ptypes)
    gof_rows <- push(gof_rows, data.frame(base, variant = v, ptype = p,
                                          reject_rate = cm[[paste0("rej_", v, "_", p)]]))
  for (p in ml_gof_ptypes)
    gof_rows <- push(gof_rows, data.frame(base, variant = "ml", ptype = p,
                                          reject_rate = cm[[paste0("rej_ml_", p)]]))
  for (dv in diff_variants) for (p in diff_ptypes)
    diff_rows <- push(diff_rows, data.frame(base, variant = dv, ptype = p,
                                            reject_rate = cm[[paste0("rej_diff_", dv, "_", p)]]))
  diag_rows <- push(diag_rows, data.frame(
    base,
    conv_guttman = cm[["conv_guttman"]], conv_ml = cm[["conv_ml"]],
    improper_guttman = cm[["improper_guttman"]], improper_ml = cm[["improper_ml"]],
    gofT_ntml_nt = cm[["gofT_g_ntml_nt"]], gofT_uls_emp = cm[["gofT_g_uls_emp"]],
    calib_err = setup$calib_err))

  el <- proc.time()[["elapsed"]] - t_start
  message(sprintf("    done (%.0fs, conv g=%.3f ml=%.3f, improper g=%.3f ml=%.3f)",
                  el, cm[["conv_guttman"]], cm[["conv_ml"]],
                  cm[["improper_guttman"]], cm[["improper_ml"]]))
}

# ---------------------------------------------------------------------------
# Write outputs.
# ---------------------------------------------------------------------------

coverage   <- do.call(rbind, cov_rows)
risk       <- do.call(rbind, mse_rows)
gof        <- do.call(rbind, gof_rows)
difftest   <- do.call(rbind, diff_rows)
diagnostics <- do.call(rbind, diag_rows)

paths <- c(
  coverage   = file.path(opts$results_dir, "coverage.csv"),
  risk       = file.path(opts$results_dir, "risk.csv"),
  gof        = file.path(opts$results_dir, "gof.csv"),
  difftest   = file.path(opts$results_dir, "difftest.csv"),
  diagnostics = file.path(opts$results_dir, "diagnostics.csv"))
write_csv(coverage, paths[["coverage"]])
write_csv(risk, paths[["risk"]])
write_csv(gof, paths[["gof"]])
write_csv(difftest, paths[["difftest"]])
write_csv(diagnostics, paths[["diagnostics"]])

meta <- metadata_frame(
  values = list(
    mode = if (opts$smoke) "smoke" else "full",
    reps = opts$reps, n = opts$n, generators = opts$generators, specs = opts$specs,
    ref_n = opts$ref_n, alpha = opts$alpha, seed_base = opts$seed_base,
    cores = opts$cores, regimes = opts$regimes, cross_frac = cross_frac),
  packages = "magmaan")
meta_path <- file.path(opts$results_dir, "metadata.csv")
write_csv(meta, meta_path)

cat("Wrote:\n"); for (p in c(paths, metadata = meta_path)) cat("  ", p, "\n", sep = "")
