# Population, invariance-ladder syntax, and violation injector for the
# measurement-invariance FMG/pEBA difference-test study.
#
# SMOKE MODEL (implemented here): one factor, six indicators, two groups, mean
# structure; the null population is scalar-invariant (equal loadings AND equal
# intercepts) with a group-2 latent mean + variance shift, so configural /
# metric / scalar are all correctly specified under the null. Identical
# structural base to experiments 21/23.
#
# BUILD-OUT (TODO, gated in run_experiment.R): the Brace & Savalei two-factor
# 4/8/15-indicator models (p in {8,16,30}, phi = 0.5) for the high-dimensional
# pEBA cell. Not implemented yet; `build_population(p)` errors for p != 6 so the
# smoke cannot silently run the wrong model.

`%||%` <- function(a, b) if (is.null(a)) b else a

build_population <- function(p = 6L) {
  if (!identical(as.integer(p), 6L)) {
    stop("only the p=6 one-factor smoke model is implemented; the two-factor ",
         "p in {8,16,30} Brace-Savalei models are the build-out (see report.qmd)",
         call. = FALSE)
  }
  list(
    p       = 6L,
    ov      = paste0("x", 1:6),
    lambda  = c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85),  # x1 is the marker
    nu      = c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35),  # equal across groups (null)
    theta   = c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60),  # equal across groups (null)
    psi     = c(1.00, 1.30),                          # per-group factor variance
    alpha   = c(0.00, 0.30)                           # per-group latent mean
  )
}

# Implied per-group mean/covariance, optionally with a planted violation in
# group 2. `violate` is NULL (null population) or a list(rung, delta, k):
#   rung = "weak"   inflate k group-2 loadings by (1 + delta)
#   rung = "strong" shift k group-2 intercepts by +delta
#   rung = "strict" inflate k group-2 residual variances by (1 + delta)
# Violations are planted on the LAST k indicators (x6, x5, ...) so the marker
# stays invariant. NOTE: `delta` is a raw magnitude knob; calibrating it to a
# target Delta-RMSEA per rung is a TODO (see report.qmd open question 3).
group_moments <- function(pop, g, violate = NULL) {
  lambda <- pop$lambda
  nu     <- pop$nu
  theta  <- pop$theta
  if (!is.null(violate) && g == 2L) {
    k   <- violate$k %||% 1L
    idx <- rev(seq_len(pop$p))[seq_len(k)]
    switch(violate$rung,
      weak   = lambda[idx] <- lambda[idx] * (1 + violate$delta),
      strong = nu[idx]     <- nu[idx] + violate$delta,
      strict = theta[idx]  <- theta[idx] * (1 + violate$delta),
      stop("unknown rung: ", violate$rung, call. = FALSE))
  }
  Sigma <- tcrossprod(lambda) * pop$psi[g] + diag(theta)
  mu    <- nu + lambda * pop$alpha[g]
  list(Sigma = Sigma, mu = mu)
}

# Invariance-ladder syntax. Shared labels tie a parameter across groups (magmaan
# and lavaan agree), so the same string defines the same model in both.
invariance_syntax <- function(level, ov = paste0("x", 1:6)) {
  marker  <- ov[1L]
  rest    <- ov[-1L]
  free    <- sprintf("f =~ %s", paste(ov, collapse = " + "))
  tied    <- sprintf("f =~ %s + %s", marker,
                     paste(sprintf("L%d*%s", seq_along(rest) + 1L, rest),
                           collapse = " + "))
  int_t   <- paste(sprintf("%s ~ t%d*1", ov, seq_along(ov)), collapse = "\n")
  res_t   <- paste(sprintf("%s ~~ e%d*%s", ov, seq_along(ov), ov), collapse = "\n")
  switch(level,
    configural = free,
    metric     = tied,
    scalar     = paste(tied, int_t, "f ~ c(0, NA)*1", sep = "\n"),
    strict     = paste(tied, int_t, res_t, "f ~ c(0, NA)*1", sep = "\n"),
    stop("unknown invariance level: ", level, call. = FALSE))
}

# Adjacent ladder pairs for the difference tests: (less restricted, more
# restricted, label). The difference test of each step is H1 (less) vs H0 (more).
ladder_pairs <- function() {
  list(
    list(h1 = "configural", h0 = "metric", step = "weak"),
    list(h1 = "metric",     h0 = "scalar", step = "strong"),
    list(h1 = "scalar",     h0 = "strict", step = "strict")
  )
}
