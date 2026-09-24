synthetic_case <- function(id, stress = FALSE) {
  q <- if (id == "higher_order") 4L else 3L
  p <- 9L
  L <- matrix(0, p, q)
  for (j in 1:3) L[(3*j-2):(3*j), j] <- c(0.8, 0.7, 0.9)
  B <- matrix(0, q, q); P <- diag(q); T <- diag(rep(0.5, p))
  if (id == "mediation") {
    B[2, 1] <- if (stress) 0.97 else 0.5
    B[3, 2] <- 0.5; B[3, 1] <- 0.2
    P[2, 2] <- if (stress) 0.06 else 0.75
  } else if (id == "correlated_predictors") {
    P[1, 2] <- P[2, 1] <- if (stress) 0.95 else 0.4
    B[3, 1:2] <- c(0.4, 0.3)
  } else if (id == "correlated_disturbances") {
    B[2:3, 1] <- c(0.6, 0.5)
    P[2, 3] <- P[3, 2] <- if (stress) 0.8 else 0.3
    T[2, 5] <- T[5, 2] <- 0.1
  } else if (id == "higher_order") {
    B[1:3, 4] <- if (stress) c(0.95, 0.9, 0.9) else c(0.7, 0.8, 0.6)
    diag(P)[1:3] <- 1 - B[1:3, 4]^2
  } else stop("Unknown synthetic case")
  m <- list(lambda = L, beta = B, psi = P, theta = T)
  # Put total variances at one to make the starting population comparable.
  m <- rescale(m, sqrt(diag(implied(m)$C)))
  list(id = id, origin = "controlled", m = m,
       mask = lapply(m, function(z) z != 0), S = implied(m)$Sigma, n = 400L,
       oracle = NA_real_)
}
empirical_case <- function(id, repo) {
  vars <- if (id == "hs_3factor_cfa") paste0("x", 1:9) else
    c(paste0("y", 1:8), paste0("x", 1:3))
  dataset <- if (id == "hs_3factor_cfa") "HolzingerSwineford1939" else "PoliticalDemocracy"
  env <- new.env(); utils::data(list = dataset, package = "lavaan", envir = env)
  X <- stats::na.omit(get(dataset, env)[, vars])
  syntax <- paste(readLines(file.path(repo, "benchmarks", "cases", id, "model.lav")), collapse = "\n")
  f <- lavaan::sem(syntax, data = X, meanstructure = FALSE, fixed.x = FALSE)
  if (!lavaan::lavInspect(f, "converged")) stop("Reference did not converge: ", id)
  m <- lavaan::lavInspect(f, "est")
  free <- lavaan::lavInspect(f, "free")
  if (is.null(m$beta)) { m$beta <- m$psi * 0; free$beta <- m$beta }
  m <- m[c("lambda", "beta", "psi", "theta")]
  mask <- lapply(free[names(m)], function(z) z > 0)
  # Check that the template has no shared/equality-constrained free entries.
  for (nm in names(mask)) {
    z <- free[[nm]]
    if (nm %in% c("psi", "theta")) z[upper.tri(z)] <- 0
    ids <- z[z > 0]
    if (anyDuplicated(ids)) stop("Shared parameter unsupported in pilot")
  }
  order <- rownames(m$lambda)
  S <- stats::cov(X[, order]) * (nrow(X) - 1) / nrow(X)
  list(id = id, origin = paste0("lavaan::", dataset), m = m, mask = mask,
       S = S, n = nrow(X), oracle = 2 * as.numeric(lavaan::fitMeasures(f, "fmin")))
}
# Starts are deliberately reference-assisted, not proposed production heuristics.
common_start <- function(case, seed) {
  set.seed(seed)
  m <- case$m
  for (nm in c("lambda", "beta")) {
    nz <- m[[nm]] != 0
    m[[nm]][nz] <- m[[nm]][nz] * runif(sum(nz), 0.8, 1.2)
  }
  m$psi <- 1.15 * m$psi
  m$theta <- 1.15 * m$theta
  m
}
