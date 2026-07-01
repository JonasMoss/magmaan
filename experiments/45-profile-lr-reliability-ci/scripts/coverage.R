#!/usr/bin/env Rscript
# Small-sample coverage of the profile-LR reliability interval, and how well the
# Bartlett correction repairs it. The df-1 profile-LR is asymptotically chi^2_1,
# but at the sample sizes reliability is reported at E[T] departs from 1, so the
# chi^2 threshold miscalibrates the interval. The departure is MILD for simple
# omega (near-nominal already) but SEVERE for maximal-reliability-type
# coefficients (a steep, near-ceiling functional): uncorrected coverage collapses.
# Rescaling the threshold by c = E[T] (Bartlett) is the fix; the open question is
# how to estimate c from one sample.
#
# Coverage of g_pop by {g0 : T(g0) <= threshold} is exactly {T(g_pop) <= threshold},
# so per replication we need only T(g_pop) (one constrained fit). Thresholds:
#   NT              q = qchisq(.95, 1)
#   Bartlett-oracle q * cbar,   cbar = cell-mean of T(g_pop)          (infeasible ceiling)
#   Bartlett-null   q * c_null, c_null = param-bootstrap mean of T from the CONSTRAINED
#                   (null) model theta_tilde(g_pop), testing g_pop    (feasible)
#   Bartlett-unc    q * c_unc,  c_unc = param-bootstrap mean of T from the fitted
#                   model theta_hat, testing ghat                     (feasible, weaker)
# Wald / logit-Wald (lavaan delta-method SE) are the estimator-centric baselines,
# available where the coefficient has a := form (all 1-factor; omega on bifactor).
#
# Usage:
#   Rscript scripts/coverage.R [--reps 300] [--ns 30,50,100,200]
#            [--pops high,mod,bf] [--func omega|H] [--B 60]
#            [--seed-base S] [--cores K] [--smoke]

.here <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  if (length(fa)) dirname(dirname(normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)))
  else normalizePath(".", mustWork = TRUE)
}
exp_dir <- .here()
source(file.path(dirname(exp_dir), "_support", "R", "helpers.R"))
source(file.path(exp_dir, "R", "engine.R"))
source(file.path(exp_dir, "R", "functionals.R"))

parse_args <- function(a) {
  o <- list(reps = 300L, ns = c(30L, 50L, 100L, 200L), pops = c("high", "mod", "bf"),
            func = "omega", B = 60L, boot = c("null"), scalings = FALSE,
            seed_base = 20260701L,
            cores = max(1L, min(6L, parallel::detectCores() - 1L)), smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat(readLines(file.path(exp_dir, "scripts", "coverage.R"))[2:30], sep = "\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (x == "--ns")   { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--pops") { i <- i + 1L; o$pops <- parse_csv_arg(a[[i]]) }
    else if (x == "--func") { i <- i + 1L; o$func <- a[[i]] }
    else if (x == "--B")    { i <- i + 1L; o$B <- as.integer(a[[i]]) }
    else if (x == "--boot") { i <- i + 1L; o$boot <- parse_csv_arg(a[[i]]) }  # unc,null,bs
    else if (x == "--scalings") o$scalings <- TRUE                            # robust/misspec c
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--cores") { i <- i + 1L; o$cores <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  bad <- setdiff(o$boot, c("unc", "null", "bs"))
  if (length(bad)) stop("unknown --boot variant(s): ", paste(bad, collapse = ", "), call. = FALSE)
  if (o$smoke) { o$reps <- 60L; o$ns <- c(50L); o$B <- 40L; o$pops <- c("high", "bf")
                 o$boot <- c("null", "bs"); o$scalings <- TRUE }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("lavaan"); require_pkg("nloptr"); require_pkg("MASS"); require_pkg("parallel")
suppressPackageStartupMessages(library(lavaan))
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
q1 <- qchisq(0.95, 1); z975 <- qnorm(0.975)

## ---- populations --------------------------------------------------------------
make_pop_1f <- function(lam) {
  p <- length(lam); psi <- 1 - lam^2
  list(kind = "1f", p = p, cols = list(seq_len(p)), Lam = matrix(lam, p, 1),
       psi = psi, Sigma = tcrossprod(lam) + diag(psi), ov = paste0("x", seq_len(p)),
       orthogonal = FALSE)
}
make_pop_bf <- function(per, k, gen, grp) {
  p <- per * k; items <- split(seq_len(p), rep(seq_len(k), each = per))
  Lam <- matrix(0, p, k + 1L); Lam[, 1] <- gen
  for (j in seq_len(k)) Lam[items[[j]], j + 1L] <- grp[j]
  psi <- 1 - rowSums(Lam^2)
  list(kind = "bf", p = p, cols = c(list(seq_len(p)), items), Lam = Lam, psi = psi,
       Sigma = tcrossprod(Lam) + diag(psi), ov = paste0("x", seq_len(p)),
       orthogonal = TRUE)
}
pops <- list(
  high = make_pop_1f(seq(0.60, 0.85, length.out = 6L)),          # 1-factor, omega ~ .87
  mod  = make_pop_1f(seq(0.40, 0.60, length.out = 6L)),          # 1-factor, omega ~ .67
  bf   = make_pop_bf(5L, 3L, 0.60, c(0.50, 0.40, 0.45)))         # bifactor p=15

func_spec <- switch(cfg$func,
  omega = list(mk = make_omega_total, lav = lav_omega_total),
  H     = list(mk = make_H,           lav = lav_H_diag),         # 1-factor: H; bifactor: rho*_gen
  stop("unknown --func: ", cfg$func))

# a valid := baseline exists for omega on any pop, and for H only on one factor
lav_available <- function(pop) cfg$func == "omega" || pop$kind == "1f"

## ---- one replication ----------------------------------------------------------
one_rep <- function(popkey, n, seed) {
  pop <- pops[[popkey]]; p <- pop$p; ov <- pop$ov; cols <- pop$cols
  eng <- make_engine(cols, p); gfun <- func_spec$mk(eng)
  z_true <- eng$zeta_from(pop$Lam, pop$psi); g_pop <- eng$gval(gfun, z_true)

  set.seed(seed)
  X <- MASS::mvrnorm(n, rep(0, p), pop$Sigma); colnames(X) <- ov
  S <- crossprod(sweep(X, 2, colMeans(X))) / n
  logdetS <- tryCatch(as.numeric(determinant(S, logarithm = TRUE)$modulus),
                      error = function(e) NA_real_)
  if (is.na(logdetS)) return(NULL)

  unc <- tryCatch(eng$fit_unc(S, logdetS, z_true), error = function(e) NULL)
  if (is.null(unc) || any(eng$unpack(unc$z)$psi <= 0)) return(NULL)   # improper drop
  ghat <- eng$gval(gfun, unc$z)
  Tpop <- lr_profile(eng, S, logdetS, n, unc, gfun, g_pop)
  if (!is.finite(Tpop)) return(NULL)

  # feasible Bartlett factors (selected via --boot): unconstrained Gaussian DGP
  # (test ghat), null Gaussian DGP (test g_pop), Bollen-Stine null (distribution-
  # free, test g_pop). The null variants share the constrained fit theta_tilde(g0).
  c_unc <- NA_real_; c_null <- NA_real_; c_bs <- NA_real_
  if ("unc" %in% cfg$boot) {
    bu <- boot_lr_mean(eng, eng$sigma_of(unc$z), n, unc$z, gfun, ghat, cfg$B)
    if (!is.null(bu)) c_unc <- bu$c
  }
  if (any(c("null", "bs") %in% cfg$boot)) {
    con <- eng$fit_con(S, logdetS, unc$z, gfun, g_pop)
    if (!is.null(con)) {
      Sig0 <- eng$sigma_of(con$z)
      if ("null" %in% cfg$boot) {
        bn <- boot_lr_mean(eng, Sig0, n, con$z, gfun, g_pop, cfg$B)
        if (!is.null(bn)) c_null <- bn$c
      }
      if ("bs" %in% cfg$boot) {
        bb <- boot_lr_mean_bs(eng, X, n, con$z, gfun, g_pop, Sig0, cfg$B)
        if (!is.null(bb)) c_bs <- bb$c
      }
    }
  }
  # robust / misspec reference scalings (variance ratios); -> 1 under normal+correct
  c_robust <- NA_real_; c_misspec <- NA_real_
  if (cfg$scalings) {
    sc <- tryCatch(lr_scalings(eng, unc$z, X, gfun, n), error = function(e) NULL)
    if (!is.null(sc)) { c_robust <- sc$c_robust; c_misspec <- sc$c_misspec }
  }

  # baseline Wald / logit-Wald from lavaan delta-method SE, where a := form exists
  cover_wald <- NA; cover_logit <- NA; width_wald <- NA_real_; width_logit <- NA_real_
  if (lav_available(pop)) {
    dat <- as.data.frame(X)
    fit <- tryCatch(sem(func_spec$lav(ov, cols), data = dat, std.lv = TRUE,
                        meanstructure = FALSE, orthogonal = isTRUE(pop$orthogonal)),
                    error = function(e) NULL)
    if (!is.null(fit) && lavInspect(fit, "converged")) {
      pe <- parameterEstimates(fit); rr <- which(pe$lhs == "rel")[1]
      gh <- pe$est[rr]; se <- pe$se[rr]
      if (is.finite(se) && se > 0) {
        wl <- gh - z975 * se; wh <- gh + z975 * se
        cover_wald <- wl <= g_pop && g_pop <= wh; width_wald <- wh - wl
        gc <- min(1 - 1e-9, max(1e-9, gh)); se_e <- se / (gc * (1 - gc)); et <- qlogis(gc)
        ll <- plogis(et - z975 * se_e); lh <- plogis(et + z975 * se_e)
        cover_logit <- ll <= g_pop && g_pop <= lh; width_logit <- lh - ll
      }
    }
  }

  data.frame(pop = popkey, kind = pop$kind, n = n, seed = seed, g_pop = g_pop,
             ghat = ghat, Tpop = Tpop, c_unc = c_unc, c_null = c_null, c_bs = c_bs,
             c_robust = c_robust, c_misspec = c_misspec,
             cover_wald = cover_wald, cover_logit = cover_logit,
             width_wald = width_wald, width_logit = width_logit,
             stringsAsFactors = FALSE)
}

## ---- run grid -----------------------------------------------------------------
cells <- expand.grid(pop = cfg$pops, n = cfg$ns, stringsAsFactors = FALSE)
message(sprintf("coverage: func=%s reps=%d B=%d cores=%d | %d cells",
                cfg$func, cfg$reps, cfg$B, cfg$cores, nrow(cells)))
rep_rows <- list()
for (ci in seq_len(nrow(cells))) {
  pk <- cells$pop[ci]; n <- cells$n[ci]; t0 <- Sys.time()
  seeds <- cfg$seed_base + ci * 100000L + seq_len(cfg$reps)
  rr <- parallel::mclapply(seeds, function(s) one_rep(pk, n, s),
                           mc.cores = cfg$cores, mc.preschedule = TRUE)
  rr <- rr[vapply(rr, is.data.frame, logical(1))]
  rep_rows[[ci]] <- do.call(rbind, rr)
  message(sprintf("  [%d/%d] pop=%s n=%d: %d/%d proper, %.0fs",
                  ci, nrow(cells), pk, n, length(rr), cfg$reps,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
reps <- do.call(rbind, rep_rows)
write_csv(reps, file.path(res_dir, paste0("coverage_reps_", cfg$func, ".csv")))

## ---- aggregate ----------------------------------------------------------------
q_cov <- function(T, c, sel = TRUE) { ok <- is.finite(c) & sel
  if (!any(ok)) NA_real_ else mean(T[ok] <= c[ok] * q1) }
agg <- do.call(rbind, lapply(split(reps, list(reps$pop, reps$n), drop = TRUE),
  function(d) {
    cbar <- mean(d$Tpop)
    data.frame(pop = d$pop[1], kind = d$kind[1], n = d$n[1], g_pop = d$g_pop[1],
      reps = nrow(d), mean_T = cbar,
      mean_c_null = mean(d$c_null, na.rm = TRUE), mean_c_unc = mean(d$c_unc, na.rm = TRUE),
      mean_c_bs = mean(d$c_bs, na.rm = TRUE),
      mean_c_robust = mean(d$c_robust, na.rm = TRUE),
      mean_c_misspec = mean(d$c_misspec, na.rm = TRUE),
      cov_nt = mean(d$Tpop <= q1),
      cov_bart_oracle = mean(d$Tpop <= cbar * q1),
      cov_bart_null = q_cov(d$Tpop, d$c_null),
      cov_bart_unc = q_cov(d$Tpop, d$c_unc),
      cov_bart_bs = q_cov(d$Tpop, d$c_bs),
      cov_wald = mean(d$cover_wald), cov_logit = mean(d$cover_logit))
  }))
agg <- agg[order(agg$kind, agg$pop, agg$n), ]
write_csv(agg, file.path(res_dir, paste0("coverage_", cfg$func, ".csv")))
write_metadata(file.path(res_dir, paste0("coverage_metadata_", cfg$func, ".csv")),
  values = list(func = cfg$func, reps = cfg$reps, B = cfg$B,
    ns = paste(cfg$ns, collapse = ","), pops = paste(cfg$pops, collapse = ","),
    seed_base = cfg$seed_base,
    populations = "1f high lam .60-.85; 1f mod lam .40-.60; bf p15 gen .6 grp (.5,.4,.45)",
    correction = "oracle c=cell-mean T(g_pop); null c=bootstrap from theta_tilde(g_pop); unc c=bootstrap from theta_hat"),
  packages = "lavaan, nloptr, MASS")

options(width = 220)
cat("\nCoverage of g_pop (nominal 0.95):\n")
show <- c("pop", "n", "g_pop", "reps", "mean_T", "cov_nt", "cov_bart_oracle")
if ("null" %in% cfg$boot) show <- c(show, "cov_bart_null")
if ("bs"   %in% cfg$boot) show <- c(show, "cov_bart_bs")
if ("unc"  %in% cfg$boot) show <- c(show, "cov_bart_unc")
show <- c(show, "cov_wald", "cov_logit")
print(agg[, show], row.names = FALSE, digits = 3)
if (cfg$scalings) {
  cat("\nReference scalings (variance ratios; -> 1 under normal + correct model):\n")
  print(agg[, c("pop", "n", "mean_c_robust", "mean_c_misspec")], row.names = FALSE, digits = 3)
}
cat("\nWrote ", file.path(res_dir, paste0("coverage_", cfg$func, ".csv")), "\n", sep = "")
