# Freeze an information-based affine map at one common starting model.
# This changes coordinates, never the covariance restrictions or objective.
precondition_map <- function(map, start, method = c("none", "diagonal", "full")) {
  method <- match.arg(method)
  if (method == "none") return(map)
  x0 <- map$pack(start)
  I <- geometry(map, x0)$information
  n <- length(x0)
  T <- if (method == "diagonal") diag(1 / sqrt(diag(I)), n) else
    backsolve(chol(I), diag(n))
  list(pack = function(m) as.vector(solve(T, map$pack(m) - x0)),
       unpack = function(z) map$unpack(x0 + as.vector(T %*% z)),
       derivatives = function(u) {
         ds <- map$derivatives(u)
         p <- nrow(ds[[1]])
         Z <- vapply(ds, as.vector, numeric(p * p)) %*% T
         lapply(seq_len(n), function(k) matrix(Z[, k], p, p))
       })
}
# Diagonal congruence invariants; raw eigenvalues of Psi cannot be compared
# between marker, disturbance-unit, and total-unit coordinates.
boundary_diagnostics <- function(u) {
  correlation_min <- function(V) {
    if (any(diag(V) <= 0)) return(-Inf)
    min_eigen(V / sqrt(outer(diag(V), diag(V))))
  }
  c(disturbance_fraction = min(diag(u$m$psi) / diag(u$C)),
    measurement_fraction = min(diag(u$m$theta) / diag(u$Sigma)),
    disturbance_correlation = correlation_min(u$m$psi),
    measurement_correlation = correlation_min(u$m$theta))
}

# Prototype selector: compare directional coupling after removing parameter units,
# then freeze full information whitening in the chosen chart. The score is a
# local heuristic, not a forecast of nonlinear curvature or boundary activity.
select_preconditioner <- function(base, mask, start) {
  t0 <- proc.time()[[3]]
  charts <- c('marker', 'disturbance', 'total')
  candidates <- lapply(charts, function(ch) chart_map(base, mask, ch))
  scores <- vapply(candidates, function(map) {
    I <- geometry(map, map$pack(start))$information
    d <- sqrt(diag(I)); E <- I / outer(d, d)
    ee <- eigen(E, symmetric = TRUE, only.values = TRUE)$values
    if (min(ee) <= max(ee)*1e-10) Inf else max(ee)/min(ee)
  }, 0.0)
  if (!any(is.finite(scores))) stop('No locally full-rank candidate; diagnose identification')
  best <- which.min(scores)
  chosen <- precondition_map(candidates[[best]], start, 'full')
  list(chart = charts[best], map = chosen,
       scores = data.frame(chart = charts, score = scores, selected = seq_along(charts) == best),
       seconds = proc.time()[[3]] - t0)
}
