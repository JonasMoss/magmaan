#!/usr/bin/env Rscript
# Experiment 54: measurement invariance for the closed-form Guttman CFA.
#
# The closed-form estimator gained two invariance tests (see the note
# constrained_noniterative_cfa.tex):
#   * metric  -- the Omega-metric minimum-distance projection onto equal
#                loadings, an exact chi^2 Wald;
#   * scalar  -- the reference-group mean map, a linearized pseudo-inverse Wald
#                on the mean residual orthogonal to the loadings.
# This experiment asks whether those two tests are calibrated (Type-I near the
# nominal level) and powered, across normal and non-normal continuous data, and
# whether the empirical (distribution-free) Gamma is needed off normality, the
# same way it was for the single-group goodness-of-fit in experiment 52. A
# guarded ML likelihood-ratio arm (configural/metric/scalar refits plus a
# robust nested test) runs alongside where it is available.
#
# Population: two groups, two correlated factors, three indicators each, marker
# identification, mean structure with the reference group's latent means fixed
# at 0 and the second group's latent means free. Three scenarios drive the two
# tests:
#   invariant   -- equal loadings, equal intercepts, group-2 latent-mean shift
#                  inside col(Lambda). Both tests should hold their level.
#   metric_viol -- one group-2 loading differs. The metric test should reject.
#   scalar_viol -- equal loadings, one group-2 intercept shifted OUTSIDE
#                  col(Lambda). The scalar test should reject; metric holds.

suppressWarnings(suppressMessages(library(magmaan)))
source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))
set_single_threaded_math()
core <- magmaan::magmaan_core

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Type-I and power for the closed-form metric / scalar invariance Wald,\n",
    "vs an ML likelihood-ratio reference, on normal / independent-generator data.\n\n",
    "Options:\n",
    "  --smoke            One quick slice (n=250, reps=40, normal only). Default.\n",
    "  --full             Full generator x n x scenario grid at --reps reps.\n",
    "  --reps N           Replications per cell (full). Default: 1000.\n",
    "  --n LIST           Comma-separated per-group sample sizes. Default: 200,500.\n",
    "  --generators LIST  Subset of normal,ig. Default: both.\n",
    "  --scenarios LIST   Subset of invariant,metric_viol,scalar_viol. Default: all.\n",
    "  --alpha X          Rejection level. Default: 0.05.\n",
    "  --no-ml            Skip the ML likelihood-ratio reference arm.\n",
    "  --cores N          mclapply cores. Default: detectCores()-1.\n",
    "  --seed-base N      Base seed. Default: 20260704.\n",
    "  --results-dir PATH Output directory. Default: results.\n",
    "  --help             Show this help.\n",
    sep = ""
  )
}

opts <- list(
  mode = "smoke", reps = 1000L,
  n = c(200L, 500L),
  generators = c("normal", "ig"),
  scenarios = c("invariant", "metric_viol", "scalar_viol"),
  alpha = 0.05, with_ml = TRUE,
  cores = max(1L, parallel::detectCores() - 1L),
  seed_base = 20260704L, results_dir = NULL, explicit = character()
)

args <- commandArgs(trailingOnly = TRUE)
i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--full") opts$mode <- "full"
  else if (a == "--reps") { opts$reps <- as.integer(take()); opts$explicit <- c(opts$explicit, "reps") }
  else if (a == "--n") { opts$n <- as.integer(parse_csv_numeric(take())); opts$explicit <- c(opts$explicit, "n") }
  else if (a == "--generators") { opts$generators <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "generators") }
  else if (a == "--scenarios") { opts$scenarios <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "scenarios") }
  else if (a == "--alpha") opts$alpha <- as.numeric(take())
  else if (a == "--no-ml") opts$with_ml <- FALSE
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
bad <- setdiff(opts$generators, c("normal", "ig"))
if (length(bad)) stop("unknown generators: ", paste(bad, collapse = ","), call. = FALSE)
bad <- setdiff(opts$scenarios, c("invariant", "metric_viol", "scalar_viol"))
if (length(bad)) stop("unknown scenarios: ", paste(bad, collapse = ","), call. = FALSE)

# Smoke defaults keep a single cheap slice unless the axis was set explicitly.
if (opts$mode == "smoke") {
  if (!"reps" %in% opts$explicit) opts$reps <- 40L
  if (!"n" %in% opts$explicit) opts$n <- 250L
  if (!"generators" %in% opts$explicit) opts$generators <- "normal"
}
z_crit <- stats::qnorm(1 - opts$alpha / 2)
results_path <- opts$results_dir %||% ensure_results_dir()
dir.create(results_path, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
p <- 6L; m <- 2L
ov <- paste0("x", 1:6)
model <- "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf1 ~~ f2\n"

base_loadings <- c(1.0, 0.8, 1.2, 1.0, 0.7, 1.3)   # markers x1, x4
make_lambda <- function(lam) {
  L <- matrix(0, p, m)
  L[1:3, 1] <- lam[1:3]; L[4:6, 2] <- lam[4:6]
  L
}
Lambda <- make_lambda(base_loadings)
nu       <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)  # common intercepts
alpha2   <- c(0.5, -0.3)                      # group-2 latent means (group 1 = 0)
Phi1     <- matrix(c(1.0, 0.3, 0.3, 1.0), 2, 2)
Phi2     <- matrix(c(1.2, 0.25, 0.25, 1.2), 2, 2)  # factor (co)variances may differ
theta    <- rep(0.5, p)
metric_break <- 0.45   # added to the x2 loading in group 2 under metric_viol
scalar_break <- 0.6    # off-col(Lambda) intercept shift under scalar_viol

# A unit vector orthogonal to col(Lambda) inside the first factor's block, used
# to move a group-2 intercept off the loadings so scalar (not metric) breaks.
eps_off <- local({
  e <- c(base_loadings[3], 0, -base_loadings[1], 0, 0, 0)  # 1.2*x1 - 1.0*x3 in f1
  stopifnot(max(abs(crossprod(Lambda, e))) < 1e-12)
  e / sqrt(sum(e^2))
})

# Per-group (Lambda, mu, Sigma) for a scenario. Group 1 is always the reference
# (equal loadings, alpha = 0). Group 2 carries the manipulation.
group_pop <- function(scenario) {
  L1 <- Lambda; L2 <- Lambda
  a2 <- alpha2
  nu2 <- nu
  if (scenario == "metric_viol") {
    lam2 <- base_loadings; lam2[2] <- lam2[2] + metric_break
    L2 <- make_lambda(lam2)
  } else if (scenario == "scalar_viol") {
    nu2 <- nu + scalar_break * eps_off
  }
  mu1 <- nu
  mu2 <- nu2 + as.vector(L2 %*% a2)
  S1 <- L1 %*% Phi1 %*% t(L1) + diag(theta, p)
  S2 <- L2 %*% Phi2 %*% t(L2) + diag(theta, p)
  for (S in list(S1, S2)) {
    if (min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) <= 1e-8)
      stop("population covariance not PD for scenario ", scenario, call. = FALSE)
  }
  dimnames(S1) <- dimnames(S2) <- list(ov, ov)
  list(mu = list(mu1, mu2), Sigma = list(S1, S2), scenario = scenario)
}

# ---------------------------------------------------------------------------
# Partables (shared across cells): configural mean-structure, metric (loadings
# equal), scalar (loadings + intercepts equal, latent means free).
# ---------------------------------------------------------------------------
pt_config <- core$lavaan_lavaanify(model, meanstructure = TRUE, n_groups = 2L)
pt_metric <- core$lavaan_lavaanify(model, meanstructure = TRUE, n_groups = 2L,
                                   group_equal = "loadings")

# ---------------------------------------------------------------------------
# Generators. Normal draws per replicate; ig calibrates once per (scenario,
# group) then batches. Both add the group mean after drawing (ig is centered).
# ---------------------------------------------------------------------------
calibrate_group <- function(generator, Sigma) {
  if (generator == "normal") return(list(kind = "normal", chol = chol(Sigma)))
  cal <- core$sim_ig_calibrate(Sigma,
                               target_skewness = rep(2, p),
                               target_excess_kurtosis = rep(7, p),
                               root = "symmetric", generator_family = "pearson")
  list(kind = "ig", cal = cal)
}

# Batched sampler for one group: idx -> (n x p) matrix with the group mean added.
group_sampler <- function(gcal, mu, n, reps, seed) {
  if (gcal$kind == "normal") {
    ch <- gcal$chol
    return(function(idx) {
      set.seed(seed + idx)
      X <- matrix(stats::rnorm(n * p), n, p) %*% ch
      sweep(X, 2, mu, "+")
    })
  }
  batch <- core$sim_ig_draw(gcal$cal, n = n, reps = reps, seed_base = seed)
  function(idx) sweep(batch$draws[[idx]], 2, mu, "+")
}

# ---------------------------------------------------------------------------
# Per-replicate metrics.
# ---------------------------------------------------------------------------
# The ML reference covers metric only. True-scalar invariance frees the latent
# means, which changes the parameterization (npar 38 -> 40); magmaan's
# restriction-map nested LRT does not handle that pair, so there is no ML scalar
# comparison. The closed-form reference-group map has no such trouble, which is
# part of the point.
metric_names <- c(
  "rej_cf_metric_nt", "rej_cf_metric_emp",
  "rej_cf_scalar_nt", "rej_cf_scalar_emp",
  "rej_ml_metric_nt", "rej_ml_metric_rob",
  "W_cf_metric_nt", "W_cf_scalar_nt",
  "conv_cf", "conv_ml"
)

reject <- function(pval) if (is.null(pval) || is.na(pval)) NA_real_ else as.numeric(pval < opts$alpha)

worker <- function(idx, s1, s2, grp) {
  out <- stats::setNames(rep(NA_real_, length(metric_names)), metric_names)
  X1 <- tryCatch(s1(idx), error = function(e) NULL)
  X2 <- tryCatch(s2(idx), error = function(e) NULL)
  if (is.null(X1) || is.null(X2)) return(out)
  colnames(X1) <- colnames(X2) <- ov
  ss <- tryCatch(core$data_sample_stats_from_raw(list(X1, X2)), error = function(e) NULL)
  if (is.null(ss)) return(out)

  ## Closed-form arm -------------------------------------------------------
  gfit_c <- tryCatch(core$noniterative_cfa_fit_impl(pt_config, ss, "guttman"), error = function(e) NULL)
  gfit_m <- tryCatch(core$noniterative_cfa_fit_impl(pt_metric, ss, "guttman"), error = function(e) NULL)
  out["conv_cf"] <- as.numeric(!is.null(gfit_c) && !is.null(gfit_m))
  if (!is.null(gfit_m)) {
    con_nt  <- tryCatch(core$noniterative_cfa_constrained_impl(gfit_m, "guttman", "ntml", "nt", NULL), error = function(e) NULL)
    con_emp <- tryCatch(core$noniterative_cfa_constrained_impl(gfit_m, "guttman", "ntml", "empirical", list(X1, X2)), error = function(e) NULL)
    if (!is.null(con_nt))  { out["rej_cf_metric_nt"]  <- reject(con_nt$p_wald);  out["W_cf_metric_nt"] <- con_nt$W }
    if (!is.null(con_emp)) out["rej_cf_metric_emp"] <- reject(con_emp$p_wald)
  }
  if (!is.null(gfit_c)) {
    sc_nt  <- tryCatch(core$noniterative_cfa_scalar_impl(gfit_c, 1L, "guttman", "ntml", "nt", NULL), error = function(e) NULL)
    sc_emp <- tryCatch(core$noniterative_cfa_scalar_impl(gfit_c, 1L, "guttman", "ntml", "empirical", list(X1, X2)), error = function(e) NULL)
    if (!is.null(sc_nt))  { out["rej_cf_scalar_nt"]  <- reject(sc_nt$p_value);  out["W_cf_scalar_nt"] <- sc_nt$W }
    if (!is.null(sc_emp)) out["rej_cf_scalar_emp"] <- reject(sc_emp$p_value)
  }

  ## ML likelihood-ratio reference arm for metric invariance (guarded) -----
  if (opts$with_ml) {
    dat <- data.frame(rbind(X1, X2), group = rep(c("g1", "g2"), c(nrow(X1), nrow(X2))))
    ml <- tryCatch({
      fit_conf <- suppressMessages(magmaan::magmaan(model, dat, estimator = "ML",
                                                    groups = "group", meanstructure = TRUE))
      fit_metr <- suppressMessages(magmaan::magmaan(model, dat, estimator = "ML",
                                                    groups = "group", meanstructure = TRUE,
                                                    group_equal = "loadings"))
      list(conf = fit_conf, metr = fit_metr)
    }, error = function(e) NULL)
    out["conv_ml"] <- as.numeric(!is.null(ml))
    if (!is.null(ml)) {
      # Multi-group nested LRT takes per-group matrices, not the stacked frame.
      lm <- tryCatch(magmaan::robust_nested_lrt(ml$conf, ml$metr, data = list(X1, X2)),
                     error = function(e) NULL)
      if (!is.null(lm)) {
        out["rej_ml_metric_nt"]  <- reject(tryCatch(as.numeric(lm$p_unscaled), error = function(e) NA_real_))
        out["rej_ml_metric_rob"] <- reject(tryCatch(as.numeric(lm$p_scaled),   error = function(e) NA_real_))
      }
    }
  }
  out
}

# ---------------------------------------------------------------------------
# Startup validation: on the population the closed-form map recovers the truth
# and the scalar map returns the injected latent means.
# ---------------------------------------------------------------------------
local({
  pop <- group_pop("invariant")
  ss <- list(S = pop$Sigma, mean = pop$mu, nobs = c(100000L, 100000L))
  gfit <- core$noniterative_cfa_fit_impl(pt_config, ss, "guttman")
  sc <- core$noniterative_cfa_scalar_impl(gfit, 1L, "guttman", "ntml", "nt", NULL)
  if (max(abs(sc$alpha[[1]] - alpha2)) > 1e-3)
    stop("scalar map does not recover the population latent means (max err ",
         signif(max(abs(sc$alpha[[1]] - alpha2)), 3), ")", call. = FALSE)
  if (sc$W > 1e-4)
    stop("scalar W should vanish on the invariant population; got ", signif(sc$W, 3), call. = FALSE)
  con <- core$noniterative_cfa_constrained_impl(
    core$noniterative_cfa_fit_impl(pt_metric, ss, "guttman"), "guttman", "ntml", "nt", NULL)
  if (con$W > 1e-4)
    stop("metric W should vanish on the invariant population; got ", signif(con$W, 3), call. = FALSE)
})
message("Startup validation OK (closed-form map + scalar/metric W vanish on the invariant population).")

# ---------------------------------------------------------------------------
# Run the grid.
# ---------------------------------------------------------------------------
design <- expand.grid(generator = opts$generators, scenario = opts$scenarios,
                      n = opts$n, stringsAsFactors = FALSE)
design <- design[order(design$generator, design$scenario, design$n), ]
message(sprintf("Grid: %d cells (gen %s x scenario %s x n %s); reps=%d cores=%d ml=%s",
                nrow(design), paste(opts$generators, collapse = ","),
                paste(opts$scenarios, collapse = ","), paste(opts$n, collapse = ","),
                opts$reps, opts$cores, opts$with_ml))

rows <- vector("list", nrow(design))
t0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(design))) {
  generator <- design$generator[ci]; scenario <- design$scenario[ci]; n <- design$n[ci]
  pop <- group_pop(scenario)
  seed_cell <- opts$seed_base + ci * 100003L
  g1 <- calibrate_group(generator, pop$Sigma[[1]])
  g2 <- calibrate_group(generator, pop$Sigma[[2]])
  s1 <- group_sampler(g1, pop$mu[[1]], n, opts$reps, seed_cell + 1L)
  s2 <- group_sampler(g2, pop$mu[[2]], n, opts$reps, seed_cell + 2L)

  res <- parallel::mclapply(seq_len(opts$reps), function(idx) worker(idx, s1, s2, NULL),
                            mc.cores = opts$cores, mc.preschedule = TRUE)
  ok <- vapply(res, function(r) is.numeric(r) && length(r) == length(metric_names), logical(1))
  mat <- do.call(rbind, res[ok])
  agg <- colMeans(mat, na.rm = TRUE)
  rows[[ci]] <- data.frame(
    generator = generator, scenario = scenario, n = n,
    reps = sum(ok), as.list(round(agg, 4)),
    stringsAsFactors = FALSE, check.names = FALSE)
  el <- proc.time()[["elapsed"]] - t0
  message(sprintf("  [%d/%d] %s/%s/n=%d done (%.0fs, %.0fs elapsed)",
                  ci, nrow(design), generator, scenario, n,
                  el / ci, el))
}
results <- do.call(rbind, rows)

results_csv <- file.path(results_path, "invariance_rejection.csv")
write_csv(results, results_csv)
write_metadata(
  file.path(results_path, "metadata.csv"),
  values = list(
    experiment = "54-noniterative-invariance", mode = opts$mode, reps = opts$reps,
    n = paste(opts$n, collapse = ","), generators = paste(opts$generators, collapse = ","),
    scenarios = paste(opts$scenarios, collapse = ","), alpha = opts$alpha,
    with_ml = opts$with_ml, seed_base = opts$seed_base, cores = opts$cores),
  packages = c("magmaan"))

message("\nWrote:\n  ", results_csv, "\n  ", file.path(results_path, "metadata.csv"))
print(results, row.names = FALSE)
