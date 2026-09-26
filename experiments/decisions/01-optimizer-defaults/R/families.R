# Simulated populations. Each population carries its covariance (and mean),
# the models fitted to its draws, and its role:
#   test     never used while developing the layered start, the information
#            coordinates or the PSD scaling;
#   control  used during development (Ernst: the 2026-09-22 start and scaling
#            work), reported separately and never pooled with test families.
# `unit_invariant` marks models whose fit is invariant under a separate change
# of units per variable (no equality constraints across variables). Every
# model here is invariant under a common change of units.

lisrel_moments <- function(Lambda, B, Psi, Theta) {
  A <- solve(diag(nrow(B)) - B)
  Sigma <- Lambda %*% A %*% Psi %*% t(A) %*% t(Lambda) + Theta
  (Sigma + t(Sigma)) / 2
}

standardized_moments <- function(Lambda, B, Psi) {
  Sigma <- lisrel_moments(Lambda, B, Psi, diag(0, nrow(Lambda)))
  Sigma + diag(1 - diag(Sigma), nrow(Sigma))
}

named <- function(Sigma, names) {
  dimnames(Sigma) <- list(names, names)
  Sigma
}

fitted_model <- function(key, syntax, unit_invariant = TRUE, std_lv = FALSE,
                         meanstructure = FALSE) {
  list(key = key, syntax = syntax, unit_invariant = unit_invariant,
       std_lv = std_lv, meanstructure = meanstructure)
}

population <- function(family, key, role, source, Sigma, models, mu = NULL) {
  list(family = family, key = key, role = role, source = source,
       Sigma = Sigma, mu = mu, models = models, p = nrow(Sigma))
}

cfa_syntax <- function(blocks, names) {
  paste(vapply(seq_along(blocks), function(k)
    paste0("f", k, " =~ ", paste(names[blocks[[k]]], collapse = " + ")), ""),
    collapse = "\n")
}

# research/47: standardized one-factor, two-factor and three-latent path
# populations with interior, face, improper and misfit truths.
family_research47 <- function() {
  x <- function(p) paste0("x", seq_len(p))
  one <- function(key, lambda) {
    p <- length(lambda)
    population("research47", key, "test", "research/47 multiinfo designs",
      named(standardized_moments(matrix(lambda, p, 1), matrix(0, 1, 1), diag(1)), x(p)),
      list(fitted_model("one_factor", paste0("f =~ ", paste(x(p), collapse = " + ")))))
  }
  two <- function(key, r, resid_cov = 0) {
    Lambda <- cbind(c(rep(.7, 3), rep(0, 3)), c(rep(0, 3), rep(.7, 3)))
    Sigma <- standardized_moments(Lambda, matrix(0, 2, 2), matrix(c(1, r, r, 1), 2))
    for (k in 1:3) Sigma[k, k + 3] <- Sigma[k + 3, k] <- Sigma[k, k + 3] + resid_cov
    population("research47", key, "test", "research/47 multiinfo designs",
      named(Sigma, x(6)),
      list(fitted_model("two_factor", "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6")))
  }
  path <- function(key, r2) {
    b21 <- sqrt(.5); b31 <- .3
    b32 <- (-2 * b31 * b21 + sqrt((2 * b31 * b21)^2 - 4 * (b31^2 - r2))) / 2
    B <- matrix(0, 3, 3); B[2, 1] <- b21; B[3, 1] <- b31; B[3, 2] <- b32
    population("research47", key, "test", "research/47 multiinfo designs",
      named(standardized_moments(kronecker(diag(3), matrix(.7, 3, 1)), B,
                                 diag(c(1, .5, 1 - r2))), x(9)),
      list(fitted_model("path", paste(
        "f1 =~ x1 + x2 + x3", "f2 =~ x4 + x5 + x6", "f3 =~ x7 + x8 + x9",
        "f2 ~ f1", "f3 ~ f1 + f2", sep = "\n"))))
  }
  list(
    one("r47_f1_p3", c(.9, .3, .9)),
    one("r47_f1_p5", c(.9, .3, .9, .3, .9)),
    one("r47_f1_p3_face", c(1, .3, .9)),
    one("r47_f1_p5_face", c(1, .3, .9, .3, .9)),
    one("r47_imp_f1_p5", c(1.03, .3, .9, .3, .9)),
    two("r47_f2_r90", .90), two("r47_f2_r97", .97), two("r47_f2_r99", .99),
    two("r47_f2_r100", 1), two("r47_imp_f2_r103", 1.03),
    two("r47_mis_f2_resid", .95, resid_cov = .10),
    path("r47_path_r2_90", .9), path("r47_path_r2_100", 1))
}

# engineering/15 and the Ernst et al. design (also De Jonckere & Rosseel 2022,
# Study 1): X -> Y, three indicators each, unit residual variances, var(X) = 1.
family_ernst <- function() {
  nm <- c("x1", "x2", "x3", "y1", "y2", "y3")
  one <- function(key, role, source, l, beta, psi) {
    Lambda <- matrix(0, 6, 2); Lambda[1:3, 1] <- l; Lambda[4:6, 2] <- l
    B <- matrix(c(0, beta, 0, 0), 2)
    population("ernst", key, role, source,
      named(lisrel_moments(Lambda, B, diag(c(1, psi)), diag(6)), nm),
      list(fitted_model("sem", "X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nY ~ X")))
  }
  list(
    one("ernst", "control", "Ernst et al.; used for the 2026-09-22 start work",
        c(1, .8, .6), .25, 1),
    one("weak_marker", "test", "engineering/15: marker loading .1",
        c(.1, .8, .6), .25, 1),
    one("high_r2", "test", "engineering/15: R2(Y) = .98",
        c(1, .8, .6), sqrt(.98), .02))
}

# Boomsma (1985, Psychometrika 50, 229-242): two correlated or uncorrelated
# factors, 3 or 4 indicators, small/medium/large heterogeneous loadings.
family_boomsma <- function() {
  base <- list(S = c(.4, .4, .6, .6), M = c(.6, .6, .8, .8), L = c(.8, .8, .9, .9))
  out <- list()
  for (k in c(3L, 4L)) for (kind in c("U", "C")) for (size in names(base)) {
    l <- base[[size]][seq_len(k)]; p <- 2L * k; nm <- paste0("x", seq_len(p))
    Lambda <- matrix(0, p, 2); Lambda[seq_len(k), 1] <- l; Lambda[k + seq_len(k), 2] <- l
    r <- if (kind == "U") 0 else .3
    key <- paste0("boomsma_", k, kind, size)
    out[[key]] <- population("boomsma", key, "test", "Boomsma (1985) design",
      named(standardized_moments(Lambda, matrix(0, 2, 2), matrix(c(1, r, r, 1), 2)), nm),
      list(fitted_model("cfa", cfa_syntax(list(seq_len(k), k + seq_len(k)), nm))))
  }
  unname(out)
}

# Wolf, Harrington, Clark & Miller (2013): one to three factors, equal
# standardized loadings, factor correlation .3.
family_wolf <- function() {
  cells <- rbind(data.frame(m = 1L, k = c(4L, 6L, 8L)),
                 data.frame(m = 2L, k = c(3L, 6L, 8L)),
                 data.frame(m = 3L, k = c(3L, 6L, 8L)))
  out <- list()
  for (i in seq_len(nrow(cells))) for (l in c(.5, .8)) {
    m <- cells$m[i]; k <- cells$k[i]; p <- m * k; nm <- paste0("x", seq_len(p))
    Phi <- matrix(.3, m, m); diag(Phi) <- 1
    blocks <- lapply(seq_len(m), function(j) (j - 1L) * k + seq_len(k))
    key <- sprintf("wolf_%df%di_l%02d", m, k, round(100 * l))
    out[[key]] <- population("wolf", key, "test", "Wolf et al. (2013) design",
      named(standardized_moments(kronecker(diag(m), matrix(l, k, 1)),
                                 matrix(0, m, m), Phi), nm),
      list(fitted_model("cfa", cfa_syntax(blocks, nm))))
  }
  unname(out)
}

# Chen, Bollen, Paxton, Curran & Kirby (2001), as used by De Jonckere &
# Rosseel (2022, Study 2): eta1 -> eta2 -> eta3 (beta .6), unit primary
# loadings, cross-loadings .3 of eta1 on Y4, eta2 on Y7 and eta3 on Y6.
# m0 fits every cross-loading; m1 omits eta2 -> Y7; m2 also eta3 -> Y6;
# m3 also eta1 -> Y4 (their M4).
family_chen <- function() {
  nm <- paste0("Y", 1:9)
  Lambda <- matrix(0, 9, 3)
  Lambda[1:4, 1] <- c(1, 1, 1, .3); Lambda[4:7, 2] <- c(1, 1, 1, .3)
  Lambda[7:9, 3] <- 1; Lambda[6, 3] <- .3
  B <- matrix(0, 3, 3); B[2, 1] <- .6; B[3, 2] <- .6
  Theta <- diag(ifelse(nm %in% c("Y4", "Y6", "Y7"), .2895, .51))
  sem <- function(e1, e2, e3) paste(e1, e2, e3, "eta2 ~ eta1", "eta3 ~ eta2", sep = "\n")
  models <- list(
    fitted_model("m0", sem("eta1 =~ Y1 + Y2 + Y3 + Y4", "eta2 =~ Y4 + Y5 + Y6 + Y7",
                           "eta3 =~ Y7 + Y8 + Y9 + Y6")),
    fitted_model("m1", sem("eta1 =~ Y1 + Y2 + Y3 + Y4", "eta2 =~ Y4 + Y5 + Y6",
                           "eta3 =~ Y7 + Y8 + Y9 + Y6")),
    fitted_model("m2", sem("eta1 =~ Y1 + Y2 + Y3 + Y4", "eta2 =~ Y4 + Y5 + Y6",
                           "eta3 =~ Y7 + Y8 + Y9")),
    fitted_model("m3", sem("eta1 =~ Y1 + Y2 + Y3", "eta2 =~ Y4 + Y5 + Y6",
                           "eta3 =~ Y7 + Y8 + Y9")))
  list(population("chen", "chen_2001", "test",
    "Chen et al. (2001); De Jonckere & Rosseel (2022) Study 2",
    named(lisrel_moments(Lambda, B, diag(3), Theta), nm), models))
}

# Models with equality constraints across variables (correct in the
# population): equal loadings and residual variances under std.lv, equal
# regression paths and disturbances in a four-factor chain, and equal
# intercepts with a mean structure. Not invariant under separate units.
family_constrained <- function() {
  out <- list()
  nm9 <- paste0("x", 1:9); nm12 <- paste0("x", 1:12)
  for (l in c(.4, .6)) {
    Phi <- matrix(.3, 3, 3); diag(Phi) <- 1
    Sigma <- named(standardized_moments(kronecker(diag(3), matrix(l, 3, 1)),
                                        matrix(0, 3, 3), Phi), nm9)
    eq_l <- paste(sprintf("f%d =~ a%d*x%d + a%d*x%d + a%d*x%d", 1:3, 1:3,
                          3 * (0:2) + 1, 1:3, 3 * (0:2) + 2, 1:3, 3 * (0:2) + 3),
                  collapse = "\n")
    eq_e <- paste(sprintf("x%d ~~ e%d*x%d", 1:9, rep(1:3, each = 3), 1:9), collapse = "\n")
    key <- sprintf("eqcfa_l%02d", round(100 * l))
    out[[key]] <- population("constrained", key, "test", "equality-constrained CFA",
      Sigma, list(
        fitted_model("equal_loadings", eq_l, FALSE, std_lv = TRUE),
        fitted_model("equal_loadings_residuals", paste(eq_l, eq_e, sep = "\n"),
                     FALSE, std_lv = TRUE)))
  }
  for (beta in c(.2, .5)) {
    B <- matrix(0, 4, 4); B[2, 1] <- B[3, 2] <- B[4, 3] <- beta
    Psi <- diag(c(1, rep(1 - beta^2, 3)))
    Sigma <- named(standardized_moments(kronecker(diag(4), matrix(.7, 3, 1)), B, Psi), nm12)
    meas <- cfa_syntax(lapply(1:4, function(j) (j - 1L) * 3L + 1:3), nm12)
    chain <- "f2 ~ b*f1\nf3 ~ b*f2\nf4 ~ b*f3"
    key <- sprintf("eqchain_b%02d", round(100 * beta))
    out[[key]] <- population("constrained", key, "test", "equality-constrained chain SEM",
      Sigma, list(
        fitted_model("equal_paths", paste(meas, chain, sep = "\n"), FALSE),
        fitted_model("equal_paths_disturbances", paste(meas, chain,
          "f2 ~~ d*f2\nf3 ~~ d*f3\nf4 ~~ d*f4", sep = "\n"), FALSE)))
  }
  Phi <- matrix(.3, 3, 3); diag(Phi) <- 1
  Sigma <- named(standardized_moments(kronecker(diag(3), matrix(.5, 3, 1)),
                                      matrix(0, 3, 3), Phi), nm9)
  mu <- setNames(rep(2, 9), nm9)
  meas <- cfa_syntax(lapply(1:3, function(j) (j - 1L) * 3L + 1:3), nm9)
  eq_i <- paste(sprintf("x%d ~ i%d*1", 1:9, rep(1:3, each = 3)), collapse = "\n")
  eq_e <- paste(sprintf("x%d ~~ e%d*x%d", 1:9, rep(1:3, each = 3), 1:9), collapse = "\n")
  out$eqmean <- population("constrained", "eqmean", "test",
    "mean-structure CFA with equal intercepts", Sigma, list(
      fitted_model("equal_intercepts", paste(meas, eq_i, sep = "\n"), FALSE,
                   meanstructure = TRUE),
      fitted_model("equal_intercepts_residuals", paste(meas, eq_i, eq_e, sep = "\n"),
                   FALSE, meanstructure = TRUE)), mu = mu)
  unname(out)
}

all_populations <- function() {
  pops <- c(family_research47(), family_ernst(), family_boomsma(),
            family_wolf(), family_chen(), family_constrained())
  names(pops) <- vapply(pops, `[[`, "", "key")
  pops
}

# Two sample sizes per population: a small one (25, or p + 10 when larger)
# and a moderate one.
population_ns <- function(pop) c(max(25L, pop$p + 10L), 100L)
