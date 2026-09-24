# Opt-in local accuracy diagnostic for a returned complete-data ML fit.
#
# frontier_newton_accuracy(fit) computes the Newton distance
#   d = sqrt(G' I^{-1} G),
# with G the total score and I the total observed information, both reduced by
# the model's linear equality constraints. Under the local quadratic
# approximation, d is the largest predicted remaining correction of any linear
# contrast of the parameters, in units of its information-based standard error,
# and d^2 / 2 (`predicted_gain`) is the predicted remaining improvement of the
# total negative log likelihood. `passed` is TRUE when d <= `budget`; the
# default .01 is one hundredth of a standard error.
#
# This is an approximation, not a bound on the distance to an optimum, and it
# says nothing about global optimality. It applies at regular interior points.
# Ordinary fits with improper estimates are eligible (they are interior to the
# ambient domain). For a PSD fit whose solution is on the covariance boundary
# (`covariance_interior` FALSE), the Newton step is infeasible and the cone
# stationarity audit in fit$diagnostics$geometric_stationarity applies instead.
#
# `status` is "available", or explains why no distance is reported:
# "nonpositive_curvature", "ill_conditioned" (equilibrated condition number
# above 1e12), "solve_unreliable", "unsupported" (nonlinear equality
# constraints) or "unavailable". Works on ordinary ML fits and on
# frontier_fit_ml_psd() fits; FIML, least-squares and ordinal fits are refused.
frontier_newton_accuracy <- function(fit, budget = 0.01) {
  if (!is.list(fit) || is.null(fit$theta)) {
    stop("frontier_newton_accuracy(): `fit` must be a fitted magmaan model")
  }
  if (!is.numeric(budget) || length(budget) != 1L || !is.finite(budget) ||
      budget <= 0) {
    stop("frontier_newton_accuracy(): `budget` must be one positive number")
  }
  est <- fit$options$estimator
  if (!is.null(est) && !identical(toupper(as.character(est)[1L]), "ML")) {
    stop("frontier_newton_accuracy(): only complete-data ML fits are supported")
  }
  miss <- fit$options$missing
  if (!is.null(miss) && tolower(as.character(miss)[1L]) %in% c("ml", "fiml")) {
    stop("frontier_newton_accuracy(): FIML fits are not supported")
  }
  if (length(fit$ordered)) {
    stop("frontier_newton_accuracy(): ordinal fits are not supported")
  }
  frontier_newton_accuracy_impl(fit, budget)
}
