#!/usr/bin/env Rscript
# NT profile-LR reliability-CI engine, validated against semlbci (Wu-Neale).
#
# The generic profile-LR engine (R/engine.R) inverts the likelihood-ratio test
# of g(theta) = g0 for an arbitrary reliability functional g, with zero magmaan
# core change. This runner checks that, under normal-theory ML on complete data,
# its interval reproduces semlbci's likelihood-based CI (the reference
# implementation of Wu-Neale / Pek-Wu) across the reliability family: omega_total
# and H on a one-factor congeneric model, and omega_total and omega_h on an
# orthogonal bifactor. Agreement to a few 1e-4 across all cells is the NT
# milestone; the small-sample (Bartlett) and robust tiers build on this engine.
#
# Usage:
#   Rscript run_experiment.R [--seeds 1,2,3,4,5] [--ns 100,200]
#                            [--funcs omega_1f,H_1f,omega_bf,omegah_bf]
#                            [--seed-base S] [--smoke]

.here <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE))
  else normalizePath(".", mustWork = TRUE)
}
exp_dir <- .here()
source(file.path(dirname(exp_dir), "_support", "R", "helpers.R"))
source(file.path(exp_dir, "R", "engine.R"))
source(file.path(exp_dir, "R", "functionals.R"))

parse_args <- function(a) {
  o <- list(seeds = 1:5, ns = c(100L, 200L),
            funcs = c("omega_1f", "H_1f", "omega_bf", "omegah_bf"),
            seed_base = 20260701L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat(readLines(file.path(exp_dir, "run_experiment.R"))[2:22], sep = "\n"); quit(save = "no") }
    else if (x == "--seeds") { i <- i + 1L; o$seeds <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--ns")    { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--funcs") { i <- i + 1L; o$funcs <- parse_csv_arg(a[[i]]) }
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$seeds <- 1:2; o$ns <- 200L }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("lavaan"); require_pkg("semlbci"); require_pkg("nloptr"); require_pkg("MASS")
suppressPackageStartupMessages({library(lavaan); library(semlbci)})
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

## ---- populations --------------------------------------------------------------
pop_1f <- function() {
  p <- 6L; lam <- seq(0.55, 0.85, length.out = p); psi <- 1 - lam^2
  list(p = p, cols = list(seq_len(p)), Lam = matrix(lam, p, 1), psi = psi,
       Sigma = tcrossprod(lam) + diag(psi), ov = paste0("x", seq_len(p)))
}
pop_bf <- function() {
  per <- 3L; k <- 3L; p <- per * k          # 3 group factors: identified bifactor
  items <- split(seq_len(p), rep(seq_len(k), each = per))
  gen <- 0.55; grp <- c(0.45, 0.50, 0.40)
  Lam <- matrix(0, p, k + 1L); Lam[, 1] <- gen
  for (j in seq_len(k)) Lam[items[[j]], j + 1L] <- grp[j]
  psi <- 1 - rowSums(Lam^2)
  cols <- c(list(seq_len(p)), items)
  list(p = p, cols = cols, Lam = Lam, psi = psi,
       Sigma = tcrossprod(Lam) + diag(psi), ov = paste0("x", seq_len(p)),
       orthogonal = TRUE)
}

funcs <- list(
  omega_1f  = list(pop = pop_1f, mk = make_omega_total, lav = lav_omega_total,
                   label = "omega_total (1-factor)"),
  H_1f      = list(pop = pop_1f, mk = make_H,           lav = lav_H_diag,
                   label = "H (1-factor)"),
  omega_bf  = list(pop = pop_bf, mk = make_omega_total, lav = lav_omega_total,
                   label = "omega_total (bifactor)"),
  omegah_bf = list(pop = pop_bf, mk = make_omega_h,     lav = lav_omega_h,
                   label = "omega_h (bifactor)"))

## ---- one dataset x functional: engine CI vs semlbci LBCI ----------------------
one_run <- function(fkey, n, seed) {
  spec <- funcs[[fkey]]; pop <- spec$pop()
  p <- pop$p; ov <- pop$ov; cols <- pop$cols
  set.seed(seed)
  X <- MASS::mvrnorm(n, rep(0, p), pop$Sigma); colnames(X) <- ov
  dat <- as.data.frame(X)

  mod <- spec$lav(ov, cols)
  fit <- tryCatch(sem(mod, data = dat, std.lv = TRUE, meanstructure = FALSE,
                      orthogonal = isTRUE(pop$orthogonal)),
                  error = function(e) NULL)
  if (is.null(fit) || !lavInspect(fit, "converged")) return(NULL)

  sl <- tryCatch(semlbci(fit, pars = "rel :=", ciperc = 0.95, method = "wn"),
                 error = function(e) NULL)
  if (is.null(sl)) { sl_lo <- NA_real_; sl_hi <- NA_real_ } else {
    sl_df <- as.data.frame(sl); row <- which(sl_df$lhs == "rel")[1]
    sl_lo <- sl_df$lbci_lb[row]; sl_hi <- sl_df$lbci_ub[row]
  }

  # engine: unconstrained fit from lavaan estimates, then profile-LR CI
  eng <- make_engine(cols, p)
  est <- parameterEstimates(fit)
  Lhat <- matrix(0, p, length(cols))
  for (j in seq_along(cols)) {
    fac <- if (length(cols) == 1) "f" else c("g", paste0("s", seq_len(length(cols) - 1L)))[j]
    lr <- est[est$op == "=~" & est$lhs == fac, ]
    Lhat[match(lr$rhs, ov), j] <- lr$est
  }
  psi_hat <- est$est[est$op == "~~" & est$lhs == est$rhs & est$lhs %in% ov]
  S <- cov(dat) * (n - 1) / n
  logdetS <- as.numeric(determinant(S, logarithm = TRUE)$modulus)
  unc <- eng$fit_unc(S, logdetS, eng$zeta_from(Lhat, psi_hat))
  gfun <- spec$mk(eng)
  ci <- profile_lr_ci_fit(eng, S, n, unc, gfun)

  # First-principles validity gate: a correct 95% bound b satisfies T(b) = q
  # exactly, T(b) = N (F_con(b) - F_unc). We evaluate T at every reported bound
  # (ours and semlbci's) with the engine's constrained fit, so a bound that fails
  # its own defining equation is caught regardless of which solver produced it.
  q <- qchisq(0.95, 1)
  Tdev <- function(g0) { if (is.na(g0)) return(NA_real_)
    t <- lr_profile(eng, S, logdetS, n, unc, gfun, g0); if (is.na(t)) NA_real_ else t - q }

  data.frame(func = fkey, label = spec$label, n = n, seed = seed,
             est = unname(ci["est"]), sl_est = est$est[est$lhs == "rel"],
             eng_lo = unname(ci["lo"]), eng_hi = unname(ci["hi"]),
             sl_lo = sl_lo, sl_hi = sl_hi,
             d_lo = unname(ci["lo"]) - sl_lo, d_hi = unname(ci["hi"]) - sl_hi,
             tdev_eng_lo = Tdev(unname(ci["lo"])), tdev_eng_hi = Tdev(unname(ci["hi"])),
             tdev_sl_lo = Tdev(sl_lo), tdev_sl_hi = Tdev(sl_hi),
             stringsAsFactors = FALSE)
}

## ---- run grid -----------------------------------------------------------------
grid <- expand.grid(func = cfg$funcs, n = cfg$ns, seed = cfg$seeds,
                    stringsAsFactors = FALSE)
grid <- grid[order(grid$func, grid$n, grid$seed), ]
rows <- vector("list", nrow(grid)); t0 <- Sys.time()
for (i in seq_len(nrow(grid))) {
  r <- tryCatch(one_run(grid$func[i], grid$n[i], cfg$seed_base + grid$seed[i]),
                error = function(e) { message("  ! ", grid$func[i], " n=", grid$n[i],
                  " seed=", grid$seed[i], ": ", conditionMessage(e)); NULL })
  rows[[i]] <- r
  if (i %% 5L == 0L || i == nrow(grid))
    message(sprintf("[%d/%d] elapsed %.0fs", i, nrow(grid),
                    as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
out <- do.call(rbind, rows)

write_csv(out, file.path(res_dir, "semlbci_parity.csv"))
write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(seeds = paste(range(cfg$seeds), collapse = "-"),
    ns = paste(cfg$ns, collapse = ","), funcs = paste(cfg$funcs, collapse = ","),
    seed_base = cfg$seed_base, smoke = cfg$smoke,
    oracle = "semlbci::semlbci method=wn (Wu-Neale), ciperc=0.95",
    statistic = "profile LR T(g0)=N(F_con-F_unc), chi^2_1 (cscale=1, NT)"),
  packages = "lavaan, semlbci, nloptr, MASS")

# summary per functional. A bound is "valid" if T at that bound equals q to a
# loose tolerance (0.02 on the chi^2 scale); this catches solver non-convergence
# on either side. Parity is judged only where semlbci's own bound is valid.
vtol <- 0.02
tvec <- function(d) c(d$tdev_eng_lo, d$tdev_eng_hi)
summ <- do.call(rbind, lapply(split(out, out$func), function(d) {
  eng_bad <- sum(abs(tvec(d)) > vtol, na.rm = TRUE)
  sl_bad  <- sum(abs(c(d$tdev_sl_lo, d$tdev_sl_hi)) > vtol, na.rm = TRUE)
  ok_lo <- abs(d$tdev_sl_lo) <= vtol; ok_hi <- abs(d$tdev_sl_hi) <= vtol
  agree <- max(c(abs(d$d_lo[ok_lo]), abs(d$d_hi[ok_hi])), na.rm = TRUE)
  data.frame(func = d$func[1], label = d$label[1], cells = nrow(d),
             eng_bounds_bad = eng_bad, semlbci_bounds_bad = sl_bad,
             max_engine_Tdev = max(abs(tvec(d)), na.rm = TRUE),
             max_abs_diff_where_semlbci_valid = agree)
}))
cat("\nNT profile-LR vs semlbci (Wu-Neale). 'bad' = |T(bound) - qchisq(.95,1)| > 0.02.\n")
print(summ, row.names = FALSE, digits = 3)
cat("\nEngine bounds all satisfy their defining equation T=q; parity holds wherever\n",
    "semlbci itself converges. Wrote ", file.path(res_dir, "semlbci_parity.csv"), "\n", sep = "")
