#!/usr/bin/env Rscript
# Robust (generalized / Satorra-Bentler-scaled) modification indices and score
# tests: do they change which omitted path looks significant, and do they reduce
# to the naive statistic where theory says they must?
#
# This is the first consumer of magmaan's robust score-test R surface
# (`modification_indices_robust()` / `score_tests_robust()`), which wraps the
# `inference::frontier` (continuous ML/ULS/GLS/WLS) and `estimate::frontier`
# (ordinal/mixed) robust entry points. lavaan has no robust score test, so there
# is no oracle here; the experiment is a behavioural consumer, not a parity check.
#
# Each candidate row carries the ordinary `mi` plus the robust
# `mi_scaled = mi / scaling_factor` with `scaling_factor = g'B1g / g'A1g`. Two
# regimes make the scaling bite:
#   * ordinal DWLS/ULS -- intrinsic: the diagonal/identity weight is not the
#     NACOV inverse, so c != 1 even on normal latent data.
#   * continuous GLS under heavy tails -- the empirical (ADF) meat departs from
#     the normal-theory bread, so c != 1; on normal data, or with model-implied
#     meat, or under full WLS (W = Gamma^-1), the correction collapses (c -> 1).
#
# Parts:
#   1. demo       -- one misspecified dataset per regime: naive mi vs robust
#                    mi_scaled + scaling_factor for the omitted residual cov.
#   2. reductions -- the c -> 1 identities the surface must honour (correctness
#                    gates): ordinal full-WLS, continuous model-implied meat,
#                    continuous empirical meat on normal data.
#   3. simulation -- detection rate of the omitted-path score test, naive vs
#                    robust, across {ordinal DWLS, continuous GLS normal,
#                    continuous GLS heavy-tailed}. Point rates only.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n INT] [--seed-base S]
#                            [--no-sim] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 200L, n = 600L, seed_base = 20260614L, do_sim = TRUE,
              smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n INT]",
          "[--seed-base S] [--no-sim] [--smoke]\n")
      cat("  --reps       replications per simulation cell (default 200)\n")
      cat("  --n          sample size per dataset (default 600)\n")
      cat("  --seed-base  base seed (default 20260614)\n")
      cat("  --no-sim     run demo + reductions only\n")
      cat("  --smoke      fast path: tiny reps, smaller n\n")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (a == "--n") { i <- i + 1L; out$n <- as.integer(args[[i]])
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (a == "--no-sim") { out$do_sim <- FALSE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) { out$reps <- 12L; out$n <- 400L }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
suppressMessages(library(magmaan))
set_single_threaded_math()
res_dir <- ensure_results_dir()

# ---- model + data generators -------------------------------------------------

# Six-indicator one-factor CFA. The TRUE data carry a residual covariance
# x1 ~~ x2 (a shared nuisance component) that the fitted model omits, so the
# omitted-path candidate x1 ~~ x2 is the headline modification index / score.
loadings <- c(0.75, 0.70, 0.70, 0.65, 0.65, 0.60)
ov <- paste0("x", seq_along(loadings))
cfa_model <- paste("f =~", paste(ov, collapse = " + "))

# Continuous data: x_j = lambda_j * eta + gamma*(j in 1:2)*u + e_j, all standard
# normal. `heavy` rescales each observation by sqrt(df / chi2_df) -> multivariate
# t with `df` d.o.f.: same covariance (estimates stay sane) but excess kurtosis,
# which is exactly what the normal-theory bread mis-states and the empirical meat
# corrects.
gen_continuous <- function(n, gamma = 0.30, heavy = FALSE, df = 6) {
  eta <- rnorm(n)
  u   <- rnorm(n)
  X <- sapply(seq_along(loadings), function(j) {
    extra <- if (j <= 2L) gamma * u else 0
    psi <- 1 - loadings[j]^2          # unit-variance indicators (before nuisance)
    loadings[j] * eta + extra + rnorm(n, sd = sqrt(psi))
  })
  if (heavy) {
    s <- sqrt(df / rchisq(n, df))
    X <- X * s
  }
  colnames(X) <- ov
  as.data.frame(X)
}

# Ordinal data: threshold the same continuous structure into 3 categories. The
# latent variables are normal; the robust scaling is intrinsic to DWLS/ULS.
gen_ordinal <- function(n, gamma = 0.30) {
  df <- gen_continuous(n, gamma = gamma, heavy = FALSE)
  thr <- c(-0.5, 0.6)
  for (nm in ov) {
    df[[nm]] <- ordered(cut(df[[nm]], c(-Inf, thr, Inf), labels = FALSE))
  }
  df
}

# Pull the x1 ~~ x2 row (order-free) out of a score/MI table.
omitted_row <- function(tab, a = "x1", b = "x2") {
  hit <- tab$op == "~~" &
    ((tab$lhs == a & tab$rhs == b) | (tab$lhs == b & tab$rhs == a))
  if (!any(hit)) return(NULL)
  tab[which(hit)[1L], , drop = FALSE]
}

# Rcpp DataFrame turns `mi_scaled`/`scaling_factor` into `mi.scaled`/
# `scaling.factor`; accept either spelling.
col <- function(row, nm) {
  cand <- c(nm, gsub("_", ".", nm))
  hit <- cand[cand %in% names(row)]
  if (!length(hit)) return(NA_real_)
  row[[hit[1L]]]
}

fit_continuous <- function(df, estimator = "GLS") {
  magmaan(cfa_model, df, estimator = estimator, se = "none", test = "none")
}
fit_ordinal <- function(df, estimator = "DWLS") {
  magmaan(cfa_model, df, estimator = estimator, ordered = ov,
          se = "none", test = "none")
}

# ---- part 1: demo ------------------------------------------------------------

cat("[1/3] demo: naive vs robust modification index for the omitted x1 ~~ x2\n")
set.seed(cfg$seed_base)

demo_rows <- list()

# Ordinal DWLS -- intrinsic correction.
df_ord <- gen_ordinal(cfg$n)
fit_ord <- fit_ordinal(df_ord, "DWLS")
mi_ord_naive  <- omitted_row(modification_indices(fit_ord, candidates = "all"))
mi_ord_robust <- omitted_row(modification_indices_robust(fit_ord, candidates = "all"))
demo_rows[["ordinal_dwls"]] <- data.frame(
  regime = "ordinal_DWLS", meat = "polychoric_NACOV",
  mi = mi_ord_naive$mi, mi_scaled = col(mi_ord_robust, "mi_scaled"),
  scaling_factor = col(mi_ord_robust, "scaling_factor"),
  p_naive = mi_ord_naive$pvalue, p_robust = mi_ord_robust$pvalue)

# Continuous GLS, heavy-tailed -- nonnormality correction (empirical meat).
df_t <- gen_continuous(cfg$n, heavy = TRUE)
fit_t <- fit_continuous(df_t, "GLS")
mi_t_naive  <- omitted_row(modification_indices(fit_t, candidates = "all"))
# Unstructured moments => bread uses Gamma_NT(S) = W^-1, so c-1 reflects only the
# empirical-vs-normal-theory gap (excess kurtosis), not model misspecification.
mi_t_robust <- omitted_row(modification_indices_robust(
  fit_t, data = df_t, cov = "empirical", moments = "unstructured",
  candidates = "all"))
demo_rows[["continuous_gls_t"]] <- data.frame(
  regime = "continuous_GLS_t", meat = "empirical_ADF",
  mi = mi_t_naive$mi, mi_scaled = col(mi_t_robust, "mi_scaled"),
  scaling_factor = col(mi_t_robust, "scaling_factor"),
  p_naive = mi_t_naive$pvalue, p_robust = mi_t_robust$pvalue)

demo <- do.call(rbind, demo_rows)
write_csv(demo, file.path(res_dir, "demo.csv"))
for (r in seq_len(nrow(demo))) {
  cat(sprintf("  %-18s mi=%.2f  mi_scaled=%.2f  c=%.3f  (p: %.3g -> %.3g)\n",
              demo$regime[r], demo$mi[r], demo$mi_scaled[r],
              demo$scaling_factor[r], demo$p_naive[r], demo$p_robust[r]))
}

# score_tests_robust(): release a (true) loading-equality constraint on an
# ordinal fit. Under full WLS the released-constraint score test must reduce
# (c ~ 1); under DWLS it scales against the NACOV meat.
cfa_eq <- "f =~ x1 + L*x2 + L*x3 + x4 + x5 + x6"
df_sc <- gen_ordinal(cfg$n)
score_rows <- lapply(c("WLS", "DWLS"), function(est) {
  f <- magmaan(cfa_eq, df_sc, estimator = est, ordered = ov,
               se = "none", test = "none")
  sn <- score_tests(f)[1L, ]
  sr <- score_tests_robust(f)[1L, ]
  data.frame(estimator = est, mi = sn$mi, mi_scaled = col(sr, "mi_scaled"),
             scaling_factor = col(sr, "scaling_factor"))
})
score_demo <- do.call(rbind, score_rows)
write_csv(score_demo, file.path(res_dir, "score_demo.csv"))
cat("  score test (release L: x2 = x3 loading):\n")
for (r in seq_len(nrow(score_demo))) {
  cat(sprintf("    ordinal %-4s  mi=%.3f  mi_scaled=%.3f  c=%.3f\n",
              score_demo$estimator[r], score_demo$mi[r],
              score_demo$mi_scaled[r], score_demo$scaling_factor[r]))
}

# ---- part 2: reductions (c -> 1 correctness gates) ---------------------------

cat("[2/3] reductions: the scaling must collapse where theory says c = 1\n")
set.seed(cfg$seed_base + 1L)

red_rows <- list()
add_red <- function(label, naive_mi, robust_scaled, c) {
  red_rows[[label]] <<- data.frame(
    case = label, mi = naive_mi, mi_scaled = robust_scaled,
    scaling_factor = c, abs_c_minus_1 = abs(c - 1))
}

# (a) Ordinal full-WLS: W = NACOV^-1, so the meat collapses onto the bread.
df_wls <- gen_ordinal(cfg$n)
fit_wls <- fit_ordinal(df_wls, "WLS")
r_naive  <- omitted_row(modification_indices(fit_wls, candidates = "all"))
r_robust <- omitted_row(modification_indices_robust(fit_wls, candidates = "all"))
add_red("ordinal_WLS", r_naive$mi, col(r_robust, "mi_scaled"),
        col(r_robust, "scaling_factor"))

# (b) Continuous GLS, model-implied meat with unstructured moments: the meat is
# Gamma_NT(S) = W^-1, so the sandwich collapses exactly (c = 1) regardless of fit.
df_mi <- gen_continuous(cfg$n, heavy = FALSE)
fit_mi <- fit_continuous(df_mi, "GLS")
r_naive  <- omitted_row(modification_indices(fit_mi, candidates = "all"))
r_robust <- omitted_row(modification_indices_robust(
  fit_mi, cov = "model_implied", moments = "unstructured", candidates = "all"))
add_red("continuous_GLS_unstructured_NT", r_naive$mi, col(r_robust, "mi_scaled"),
        col(r_robust, "scaling_factor"))

# (c) Continuous GLS, empirical meat on NORMAL data: c -> 1 up to sampling noise
# (empirical Gamma_hat estimates the same Gamma_NT(S) when the data are normal).
r_robust_emp <- omitted_row(modification_indices_robust(
  fit_mi, data = df_mi, cov = "empirical", moments = "unstructured",
  candidates = "all"))
add_red("continuous_GLS_empirical_normal", r_naive$mi,
        col(r_robust_emp, "mi_scaled"), col(r_robust_emp, "scaling_factor"))

reductions <- do.call(rbind, red_rows)
write_csv(reductions, file.path(res_dir, "reductions.csv"))
for (r in seq_len(nrow(reductions))) {
  cat(sprintf("  %-34s c=%.4f  |c-1|=%.2e\n", reductions$case[r],
              reductions$scaling_factor[r], reductions$abs_c_minus_1[r]))
}

# ---- part 3: simulation ------------------------------------------------------

if (cfg$do_sim) {
  cat(sprintf("[3/3] simulation: %d reps x 3 regimes, detection of x1 ~~ x2\n",
              cfg$reps))
  regimes <- c("ordinal_DWLS", "continuous_GLS_normal", "continuous_GLS_t")
  acc <- lapply(regimes, function(rg) {
    rej_naive <- rej_robust <- 0L; ok <- 0L
    for (b in seq_len(cfg$reps)) {
      set.seed(cfg$seed_base + 1000L * match(rg, regimes) + b)
      out <- tryCatch({
        if (rg == "ordinal_DWLS") {
          d <- gen_ordinal(cfg$n); f <- fit_ordinal(d, "DWLS")
          rn <- omitted_row(modification_indices(f, candidates = "all"))
          rr <- omitted_row(modification_indices_robust(f, candidates = "all"))
        } else {
          heavy <- (rg == "continuous_GLS_t")
          d <- gen_continuous(cfg$n, heavy = heavy); f <- fit_continuous(d, "GLS")
          rn <- omitted_row(modification_indices(f, candidates = "all"))
          rr <- omitted_row(modification_indices_robust(
            f, data = d, cov = "empirical", moments = "unstructured",
            candidates = "all"))
        }
        list(pn = rn$pvalue, pr = rr$pvalue)
      }, error = function(e) NULL)
      if (is.null(out) || !is.finite(out$pn) || !is.finite(out$pr)) next
      ok <- ok + 1L
      rej_naive  <- rej_naive  + (out$pn < 0.05)
      rej_robust <- rej_robust + (out$pr < 0.05)
    }
    data.frame(regime = rg, n = cfg$n, reps_ok = ok,
               detect_naive = rej_naive / max(ok, 1L),
               detect_robust = rej_robust / max(ok, 1L))
  })
  sim <- do.call(rbind, acc)
  write_csv(sim, file.path(res_dir, "simulation_summary.csv"))
  for (r in seq_len(nrow(sim))) {
    cat(sprintf("  %-22s detect: naive=%.2f  robust=%.2f  (%d/%d ok)\n",
                sim$regime[r], sim$detect_naive[r], sim$detect_robust[r],
                sim$reps_ok[r], cfg$reps))
  }
} else {
  cat("[3/3] simulation skipped (--no-sim)\n")
}

# ---- metadata ----------------------------------------------------------------

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(reps = cfg$reps, n = cfg$n, seed_base = cfg$seed_base,
                do_sim = cfg$do_sim, smoke = cfg$smoke),
  packages = "magmaan")

cat("done. results in", res_dir, "\n")
