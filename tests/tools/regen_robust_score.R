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
