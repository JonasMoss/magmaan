#!/usr/bin/env Rscript
# Companion oracle generator for the robust (generalized / Satorra-Bentler
# -scaled) equality-release score test in inference::frontier.
#
# lavaan does NOT implement this statistic: lavTestScore() warns
#   "se is not `standard'; not implemented yet; falling back to ordinary score test"
# and returns the ordinary number. So there is no lavaan function to diff
# against. Instead we assemble the robust statistic from lavaan's exposed
# internals and let magmaan (which derives its own delta/weight/Γ̂ from the same
# raw data) reproduce it.
#
# Geometry. For releasing one equality constraint, with NT release score
# X2_nt (= lavTestScore), model Jacobian Δ, NT weight V, empirical NACOV Γ̂, and
# constraint Jacobian R:
#   A1 = Δ'VΔ,  B1 = Δ'VΓ̂VΔ              (parameter-space bread/meat)
#   d  = released direction (∝ Rᵀ),  K = null(R)   (nuisance subspace)
#   g  = d − K(K'A1K)⁻¹K'A1 d           (efficient-score direction)
#   c  = gᵀB1g / gᵀA1g,  mi_scaled = X2_nt / c
# c is a ratio of θ-space quadratic forms, hence invariant to lavaan's vech
# convention and to the scale of d / basis of K — so magmaan's own
# delta/weight/Γ̂ must give the same c.
#
# NOTE ON VERSION: this is assembled at the *installed* lavaan, which may differ
# from the pinned fixture version. delta/gamma/wls.v/gradient at a fixed θ̂ are
# version-stable, so the number is not sensitive to that; the recorded
# `lavaan_version` is informational. This script is standalone (it does NOT run
# the full regen_oracle.R, which would rewrite the pinned corpus).
#
# Usage:  Rscript tests/tools/regen_robust_score.R
# CI never runs R; it consumes the checked-in JSON.

suppressMessages({
  library(lavaan)
  library(jsonlite)
  library(MASS)
})

fixtures <- file.path("tests", "fixtures", "score")
dir.create(fixtures, showWarnings = FALSE, recursive = TRUE)
installed <- as.character(packageVersion("lavaan"))

# Robust release-score assembled from lavaan internals (see header).
robust_score_assemble <- function(fit) {
  X2_nt <- suppressWarnings(lavTestScore(fit)$uni$X2[1])
  Delta <- lavInspect(fit, "delta")
  Vmat  <- lavInspect(fit, "wls.v")
  Gamma <- lavInspect(fit, "gamma")
  A1 <- t(Delta) %*% Vmat %*% Delta
  B1 <- t(Delta) %*% Vmat %*% Gamma %*% Vmat %*% Delta
  R  <- fit@Model@ceq.JAC
  d  <- as.numeric(R[1, ]); d <- d / sqrt(sum(d * d))
  K  <- MASS::Null(t(R))
  g  <- d - K %*% solve(t(K) %*% A1 %*% K, t(K) %*% (A1 %*% d))
  c_fac <- as.numeric((t(g) %*% B1 %*% g) / (t(g) %*% A1 %*% g))
  list(X2_nt = X2_nt, scaling = c_fac, mi_scaled = X2_nt / c_fac)
}

# Heavy-tailed multivariate-t data (df=6) from an exact 1-factor model with two
# ~equal loadings, so the a == b constraint is sensible and Γ̂ ≠ Γ_NT (c ≠ 1).
set.seed(20260602L)
n <- 600L
lambda <- c(1.0, 0.8, 0.8, 0.9)
theta  <- c(0.6, 0.7, 0.7, 0.5)
Sigma  <- lambda %*% t(lambda) + diag(theta)
L      <- t(chol(Sigma))
df_t   <- 6
Z      <- matrix(rnorm(n * 4L), n, 4L)
w      <- rchisq(n, df_t) / df_t
X      <- sqrt((df_t - 2) / df_t) * (Z %*% t(L)) / sqrt(w)
colnames(X) <- paste0("x", 1:4)
dat    <- as.data.frame(X)

model <- "f =~ x1 + a*x2 + b*x3 + x4\na == b"
fit   <- sem(model, data = dat, estimator = "MLM")
if (!lavInspect(fit, "converged")) stop("robust score oracle fit did not converge")
rs <- robust_score_assemble(fit)

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "score_robust",
    corpus_id = "0006_robust_release_mlm",
    tool = "lavaan internals (R-assembled): delta/wls.v/gamma/ceq.JAC",
    lavaan_version = installed,
    note = paste("lavaan has no robust lavTestScore (it falls back to NT);",
                 "mi_scaled assembled from internals, version-stable")),
  input = model,
  estimator = "MLM",
  bread = "expected",
  n_obs = n,
  raw = unname(X),
  score_tests_robust = list(rows = list(list(
    lhs = "a", op = "==", rhs = "b", df = 1L,
    mi = rs$X2_nt, scaling_factor = rs$scaling, mi_scaled = rs$mi_scaled))))

out_path <- file.path(fixtures, "0006_robust_release_mlm.score_robust.json")
write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
           null = "null", na = "null", digits = NA)
cat(sprintf("wrote %s\n  X2_nt=%.6f  c=%.6f  mi_scaled=%.6f  (lavaan %s)\n",
            out_path, rs$X2_nt, rs$scaling, rs$mi_scaled, installed))

# ── 0007: continuous DWLS (diagonal-ADF weight, moment-metric sandwich) ───────
# Same assembly, but wls.v is now the DIAGONAL ADF weight and gamma the full
# ADF NACOV, so c ≠ 1 already measures the diagonal-vs-full weight gap (the
# data is the same heavy-tailed t). Exercises magmaan's continuous-LS robust
# tier: A1 = Δ'WΔ, B1 = Δ'WΓ̂WΔ with W the estimation weight.
set.seed(20260612L)
Z2 <- matrix(rnorm(n * 4L), n, 4L)
w2 <- rchisq(n, df_t) / df_t
X2 <- sqrt((df_t - 2) / df_t) * (Z2 %*% t(L)) / sqrt(w2)
colnames(X2) <- paste0("x", 1:4)
dat2 <- as.data.frame(X2)

# `se = "robust.sem"` makes lavaan compute the ADF NACOV, so both the DWLS
# estimation weight (wls.v = 1/diag(Γ̂)) and the assembled meat use the
# empirical Γ̂ — the same convention magmaan's diag(empirical_gamma)⁻¹ weight
# and raw-data meat reproduce. The default (NT-gamma) DWLS would weight by
# 1/diag(Γ_NT(S)) instead.
fit_dwls <- sem(model, data = dat2, estimator = "DWLS", ordered = FALSE,
                se = "robust.sem")
if (!lavInspect(fit_dwls, "converged")) stop("DWLS robust score oracle fit did not converge")
rs_dwls <- robust_score_assemble(fit_dwls)

payload_dwls <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "score_robust",
    corpus_id = "0007_robust_release_dwls_continuous",
    tool = "lavaan internals (R-assembled): delta/wls.v/gamma/ceq.JAC",
    lavaan_version = installed,
    note = paste("continuous DWLS (diagonal-ADF weight); c measures the",
                 "diagonal-vs-full weight gap in the moment metric")),
  input = model,
  estimator = "DWLS",
  bread = "expected",
  n_obs = n,
  raw = unname(X2),
  score_tests_robust = list(rows = list(list(
    lhs = "a", op = "==", rhs = "b", df = 1L,
    mi = rs_dwls$X2_nt, scaling_factor = rs_dwls$scaling,
    mi_scaled = rs_dwls$mi_scaled))))

out_dwls <- file.path(fixtures,
                      "0007_robust_release_dwls_continuous.score_robust.json")
write_json(payload_dwls, out_dwls, pretty = TRUE, auto_unbox = TRUE,
           null = "null", na = "null", digits = NA)
cat(sprintf("wrote %s\n  X2_nt=%.6f  c=%.6f  mi_scaled=%.6f  (lavaan %s)\n",
            out_dwls, rs_dwls$X2_nt, rs_dwls$scaling, rs_dwls$mi_scaled,
            installed))

# ── 0008: ordinal DWLS / WLSMV (polychoric NACOV meat) ────────────────────────
# All-ordinal 3-category CFA fit by WLSMV (DWLS weight); delta now carries the
# threshold rows and gamma is the polychoric NACOV. Exercises magmaan's
# ordinal robust tier over the [thresholds ; correlations] moment metric.
set.seed(20260613L)
n_ord  <- 800L
lam_o  <- c(0.88, 0.80, 0.80, 0.64)
eta_o  <- rnorm(n_ord)
Y      <- sapply(seq_along(lam_o), function(j)
  lam_o[j] * eta_o + sqrt(1 - lam_o[j]^2) * rnorm(n_ord))
Xo     <- 1L + (Y > -0.5) + (Y > 0.45)
colnames(Xo) <- paste0("x", 1:4)
dato   <- as.data.frame(lapply(as.data.frame(Xo), ordered))

model_ord <- paste(
  "f =~ x1 + a*x2 + b*x3 + x4",
  "x1 | t1 + t2",
  "x2 | t1 + t2",
  "x3 | t1 + t2",
  "x4 | t1 + t2",
  "x1 ~*~ 1*x1",
  "x2 ~*~ 1*x2",
  "x3 ~*~ 1*x3",
  "x4 ~*~ 1*x4",
  "a == b",
  sep = "\n")

fit_ord <- sem(model_ord, data = dato, estimator = "WLSMV",
               ordered = colnames(Xo))
if (!lavInspect(fit_ord, "converged")) stop("WLSMV robust score oracle fit did not converge")
rs_ord <- robust_score_assemble(fit_ord)

payload_ord <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "score_robust",
    corpus_id = "0008_robust_release_wlsmv_ordinal",
    tool = "lavaan internals (R-assembled): delta/wls.v/gamma/ceq.JAC",
    lavaan_version = installed,
    note = paste("all-ordinal WLSMV (DWLS weight, polychoric NACOV meat);",
                 "delta carries threshold rows")),
  input = model_ord,
  estimator = "WLSMV",
  bread = "expected",
  n_obs = n_ord,
  raw = unname(Xo),
  score_tests_robust = list(rows = list(list(
    lhs = "a", op = "==", rhs = "b", df = 1L,
    mi = rs_ord$X2_nt, scaling_factor = rs_ord$scaling,
    mi_scaled = rs_ord$mi_scaled))))

out_ord <- file.path(fixtures,
                     "0008_robust_release_wlsmv_ordinal.score_robust.json")
write_json(payload_ord, out_ord, pretty = TRUE, auto_unbox = TRUE,
           null = "null", na = "null", digits = NA)
cat(sprintf("wrote %s\n  X2_nt=%.6f  c=%.6f  mi_scaled=%.6f  (lavaan %s)\n",
            out_ord, rs_ord$X2_nt, rs_ord$scaling, rs_ord$mi_scaled,
            installed))
