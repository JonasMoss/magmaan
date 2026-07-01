#!/usr/bin/env Rscript
# Investigating the analytic (Lawley) Bartlett factor for the profile-LR of a
# reliability functional. The profile-LR W testing g(theta)=g0 is invariant to
# how g is scaled, so its small-sample mean inflation E[W] != 1 is a property of
# the CONSTRAINT SURFACE {g(theta)=g_pop} inside the model manifold: its
# curvature. That is why omega (a gentle ratio-of-sums surface) sits at E[T]~1
# while maximal reliability rho* (a matrix-inverse, near-ceiling, sharply curved
# surface) blows up to E[T]~4. The Lawley factor is E[W] to O(1/N); for a scalar
# restriction it splits into
#   (a) a functional-curvature term  (grad g, hess g against the info metric A), and
#   (b) a generic model term         (the likelihood's own 3rd/4th derivatives;
#                                      the same for every functional on a model).
#
# We isolate the two pieces with a LOCAL-GAUSSIAN comparator that is exactly the
# analytic curvature factor evaluated by simulation:
#   oracle E[T]  : real data, real fits (ground truth).
#   local-Gauss  : draw theta_hat ~ N(theta_pop, A^-1/N); set
#                  W = N * min_{g(theta)=g_pop} (theta-theta_hat)' A (theta-theta_hat).
#                  Quadratic likelihood + Gaussian theta_hat, but the EXACT
#                  nonlinear constraint, so it carries functional curvature to all
#                  orders and nothing else.
# Reading:
#   - a LINEAR functional must give local-Gauss E[W] ~ 1 exactly (machinery check);
#   - if for rho* local-Gauss ~ oracle, curvature is the whole story and E[W_lg]
#     (computable from A_hat, g_hat at ONE sample) is the feasible analytic factor;
#   - if local-Gauss undershoots oracle, the generic model term matters too.
#
# Usage:
#   Rscript scripts/lawley.R [--pop bf|high] [--ns 50,100] [--reps 1500] [--draws 3000]
#            [--funcs rho,omega,omegah,lin] [--seed-base S] [--cores K] [--smoke]
#
# Besides oracle vs local-Gauss, the runner records the smallest fitted
# uniqueness min(psi_hat) per oracle replication and reports E[T] split by the
# bottom/top quartile of min(psi_hat): if the near-boundary quartile carries the
# inflation, the rho* blowup is a boundary-proximity effect (non-perturbative),
# not smooth curvature, so an O(1/N) analytic factor cannot reach it.

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
  o <- list(pop = "bf", ns = c(50L, 100L), reps = 1500L, draws = 3000L,
            funcs = c("rho", "omega", "omegah", "lin"),
            seed_base = 20260701L,
            cores = max(1L, min(5L, parallel::detectCores() - 1L)), smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat(readLines(file.path(exp_dir, "scripts", "lawley.R"))[2:44], sep = "\n"); quit(save = "no") }
    else if (x == "--pop")   { i <- i + 1L; o$pop <- a[[i]] }
    else if (x == "--ns")    { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--reps")  { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (x == "--draws") { i <- i + 1L; o$draws <- as.integer(a[[i]]) }
    else if (x == "--funcs") { i <- i + 1L; o$funcs <- parse_csv_arg(a[[i]]) }
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--cores") { i <- i + 1L; o$cores <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 300L; o$draws <- 800L; o$ns <- c(50L) }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("nloptr"); require_pkg("MASS"); require_pkg("parallel")
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

## ---- populations (kept local to this probe) -----------------------------------
make_pop_1f <- function(lam) {
  p <- length(lam); psi <- 1 - lam^2
  list(kind = "1f", p = p, cols = list(seq_len(p)), Lam = matrix(lam, p, 1),
       psi = psi, Sigma = tcrossprod(lam) + diag(psi))
}
make_pop_bf <- function(per, k, gen, grp) {
  p <- per * k; items <- split(seq_len(p), rep(seq_len(k), each = per))
  Lam <- matrix(0, p, k + 1L); Lam[, 1] <- gen
  for (j in seq_len(k)) Lam[items[[j]], j + 1L] <- grp[j]
  psi <- 1 - rowSums(Lam^2)
  list(kind = "bf", p = p, cols = c(list(seq_len(p)), items), Lam = Lam, psi = psi,
       Sigma = tcrossprod(Lam) + diag(psi))
}
pops <- list(high = make_pop_1f(seq(0.60, 0.85, length.out = 6L)),
             bf   = make_pop_bf(5L, 3L, 0.60, c(0.50, 0.40, 0.45)))

# a linear (zero-curvature) control functional: sum of the general/target-column
# loadings. Linear in the loading coordinates, so its constraint surface is flat
# and the local-Gaussian E[W] must be exactly 1.
make_gsum <- function(eng, col = 1L) function(z) sum(eng$unpack(z)$Lam[, col])

func_menu <- function(eng) list(
  rho    = make_H(eng, tcol = 1L),      # 1f: coefficient H; bf: maximal reliability rho*
  omega  = make_omega_total(eng),
  omegah = make_omega_h(eng, gcol = 1L),
  lin    = make_gsum(eng, col = 1L))

## ---- expected information A in z-space, at a given z ---------------------------
info_A <- function(eng, z) {
  Sig <- eng$sigma_of(z); Si <- solve(Sig); D <- sigma_jac(eng, z)
  Wm <- 0.5 * kronecker(Si, Si)
  t(D) %*% Wm %*% D
}

## ---- one local-Gaussian W: N * min_{g(z)=g0} (z-zhat)' A (z-zhat) --------------
# quadratic objective, exact nonlinear constraint g(z)=g0 (SLSQP). Returns NA if
# the constrained solve does not land on the constraint surface.
opts_qp <- list(algorithm = "NLOPT_LD_SLSQP", xtol_rel = 1e-10, maxeval = 800)
wlg_one <- function(eng, gfun, A, zhat, g0, n) {
  obj <- function(z) { d <- z - zhat
    list(objective = as.numeric(crossprod(d, A %*% d)), gradient = as.numeric(2 * (A %*% d))) }
  con <- function(z) list(constraints = eng$gval(gfun, z) - g0,
                          jacobian = matrix(g_grad_z(eng, gfun, z), nrow = 1))
  r <- tryCatch(nloptr::nloptr(zhat, eval_f = obj, eval_g_eq = con, opts = opts_qp),
                error = function(e) NULL)
  if (is.null(r) || abs(eng$gval(gfun, r$solution) - g0) > 1e-5) return(NA_real_)
  n * r$objective
}

## ---- ground-truth oracle E[T]: one data replication ---------------------------
# returns c(T, minpsi) or NULL; minpsi = smallest fitted uniqueness (boundary proxy)
oracle_T <- function(eng, gfun, Sigma, z_true, g_pop, n, seed) {
  p <- eng$p
  set.seed(seed)
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  S <- crossprod(sweep(X, 2, colMeans(X))) / n
  ld <- tryCatch(as.numeric(determinant(S, logarithm = TRUE)$modulus), error = function(e) NA_real_)
  if (is.na(ld)) return(NULL)
  unc <- tryCatch(eng$fit_unc(S, ld, z_true), error = function(e) NULL)
  if (is.null(unc)) return(NULL)
  psi_hat <- eng$unpack(unc$z)$psi
  if (any(psi_hat <= 0)) return(NULL)                                    # improper drop
  Tv <- lr_profile(eng, S, ld, n, unc, gfun, g_pop)
  if (!is.finite(Tv)) return(NULL)
  c(T = Tv, minpsi = min(psi_hat))
}

## ---- run ----------------------------------------------------------------------
pop <- pops[[cfg$pop]]; if (is.null(pop)) stop("unknown --pop: ", cfg$pop)
p <- pop$p; eng <- make_engine(pop$cols, p)
z_true <- eng$zeta_from(pop$Lam, pop$psi)
A <- info_A(eng, z_true)
Ai <- solve(A); Lchol <- chol(Ai)                 # Ai = t(Lchol) %*% Lchol
menu <- func_menu(eng)
sel <- intersect(cfg$funcs, names(menu))
message(sprintf("lawley: pop=%s ns=%s reps=%d draws=%d cores=%d | funcs: %s",
                cfg$pop, paste(cfg$ns, collapse = ","), cfg$reps, cfg$draws, cfg$cores,
                paste(sel, collapse = ",")))

rows <- list()
for (n in cfg$ns) for (fn in sel) {
  gfun <- menu[[fn]]; g_pop <- eng$gval(gfun, z_true); t0 <- Sys.time()

  # local-Gaussian E[W]: draw zhat = z_true + t(Lchol) %*% u / sqrt(n), u ~ N(0,I)
  seeds_d <- cfg$seed_base + 700000L + seq_len(cfg$draws)
  wl <- parallel::mclapply(seeds_d, function(s) {
    set.seed(s); u <- stats::rnorm(length(z_true))
    zhat <- z_true + as.numeric(crossprod(Lchol, u)) / sqrt(n)
    wlg_one(eng, gfun, A, zhat, g_pop, n)
  }, mc.cores = cfg$cores, mc.preschedule = TRUE)
  wl <- unlist(wl); wl_ok <- wl[is.finite(wl)]

  # oracle E[T], carrying min(psi_hat) as a boundary proxy
  seeds_o <- cfg$seed_base + 900000L + seq_len(cfg$reps)
  ot <- parallel::mclapply(seeds_o, function(s)
    oracle_T(eng, gfun, pop$Sigma, z_true, g_pop, n, s),
    mc.cores = cfg$cores, mc.preschedule = TRUE)
  ot <- ot[vapply(ot, function(x) is.numeric(x) && length(x) == 2L, logical(1))]
  Tv <- vapply(ot, `[[`, numeric(1), 1L); mp <- vapply(ot, `[[`, numeric(1), 2L)

  # boundary split: E[T] in the bottom vs top quartile of min(psi_hat)
  qs <- quantile(mp, c(0.25, 0.75)); near <- mp <= qs[1]; far <- mp >= qs[2]
  et_near <- mean(Tv[near]); et_far <- mean(Tv[far])
  sp <- suppressWarnings(cor(Tv, mp, method = "spearman"))

  rows[[length(rows) + 1L]] <- data.frame(
    pop = cfg$pop, kind = pop$kind, n = n, func = fn, g_pop = g_pop,
    oracle_ET = mean(Tv), oracle_reps = length(Tv),
    lg_ET = mean(wl_ok), lg_draws = length(wl_ok),
    gap = mean(Tv) - mean(wl_ok), recov = (mean(wl_ok) - 1) / (mean(Tv) - 1),
    et_near_bd = et_near, et_far_bd = et_far, sp_T_minpsi = sp,
    stringsAsFactors = FALSE)
  message(sprintf(paste0("  n=%-3d %-6s g_pop=%.3f  oracle=%.3f (%d)  local-Gauss=%.3f  gap=%.3f  ",
                         "recov=%.0f%%  E[T|near bd]=%.2f E[T|far]=%.2f  cor(T,minpsi)=%.2f  %.0fs"),
                  n, fn, g_pop, mean(Tv), length(Tv), mean(wl_ok), mean(Tv) - mean(wl_ok),
                  100 * (mean(wl_ok) - 1) / (mean(Tv) - 1), et_near, et_far, sp,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
out <- do.call(rbind, rows)
write_csv(out, file.path(res_dir, sprintf("lawley_%s.csv", cfg$pop)))
write_metadata(file.path(res_dir, sprintf("lawley_metadata_%s.csv", cfg$pop)),
  values = list(pop = cfg$pop, ns = paste(cfg$ns, collapse = ","), reps = cfg$reps,
    draws = cfg$draws, funcs = paste(sel, collapse = ","), seed_base = cfg$seed_base,
    note = "oracle=real data/fits; local-Gauss=quadratic F + N(theta_pop,A^-1/N) draws + exact nonlinear constraint; recov=(lg-1)/(oracle-1); boundary split by quartiles of min(psi_hat)"),
  packages = "nloptr, MASS")

options(width = 200)
cat("\nAnalytic-curvature decomposition (nominal E[W]=1; recov = share of inflation the analytic factor captures):\n")
print(out[, c("n", "func", "g_pop", "oracle_ET", "lg_ET", "recov", "et_near_bd", "et_far_bd", "sp_T_minpsi")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, sprintf("lawley_%s.csv", cfg$pop)), "\n", sep = "")
