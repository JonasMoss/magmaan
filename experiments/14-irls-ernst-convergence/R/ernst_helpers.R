# Helpers local to the Ernst IRLS convergence experiment. These mirror the
# paper-pilot design without depending on the paper helper package.

ernst_model <- function() {
  "f1 =~ x1 + x2 + x3
f2 =~ x4 + x5 + x6
f2 ~ f1"
}

ernst_design_grid <- function(n_values = 10L:100L, reps = 200L,
                              beta = 0.25) {
  grid <- data.frame(
    design = "ernst_convergence",
    n = as.integer(n_values),
    n_factors = 2L,
    indicators_per_factor = 3L,
    beta = beta,
    stringsAsFactors = FALSE
  )
  grid$cell <- seq_len(nrow(grid))
  grid$reps <- as.integer(reps)
  grid
}

ernst_data <- function(n, beta = 0.25, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  loadings <- c(1, 0.8, 0.6, 1, 0.8, 0.6)
  eta1 <- stats::rnorm(n)
  eta2 <- beta * eta1 + stats::rnorm(n)
  eta <- cbind(eta1, eta2)
  lambda <- matrix(0, nrow = 6L, ncol = 2L)
  lambda[1:3, 1L] <- loadings[1:3]
  lambda[4:6, 2L] <- loadings[4:6]
  eps <- matrix(stats::rnorm(n * 6L), ncol = 6L)
  out <- as.data.frame(eta %*% t(lambda) + eps)
  names(out) <- paste0("x", seq_len(6L))
  out
}

solution_diagnostics <- function(fit, tol = -1e-8) {
  pt <- fit$partable
  if (is.null(pt) || !all(c("lhs", "op", "rhs", "est") %in% names(pt))) {
    return(list(improper = NA, n_neg_var = NA_integer_, min_var = NA_real_))
  }
  var_rows <- pt$op == "~~" & pt$lhs == pt$rhs
  vars <- pt$est[var_rows]
  if (!length(vars)) {
    return(list(improper = FALSE, n_neg_var = 0L, min_var = NA_real_))
  }
  list(
    improper = any(vars < tol, na.rm = TRUE),
    n_neg_var = sum(vars < tol, na.rm = TRUE),
    min_var = min(vars, na.rm = TRUE)
  )
}

audit_fields <- function(fit) {
  a <- if (is.list(fit)) fit$audit else NULL
  d <- if (is.list(fit)) fit$diagnostics else NULL
  pick_lgl <- function(x, k) if (is.null(x) || is.null(x[[k]])) NA else
    isTRUE(x[[k]])
  pick_num <- function(x, k) if (is.null(x) || !length(x[[k]]))
    NA_real_ else as.numeric(x[[k]][1L])
  pick_chr <- function(x, k) if (is.null(x) || is.null(x[[k]]))
    NA_character_ else as.character(x[[k]])
  list(
    audit_stationary = pick_lgl(a, "stationary"),
    audit_grad_inf = pick_num(a, "grad_inf_norm"),
    audit_grad_scaled = pick_num(a, "grad_scaled_inf"),
    audit_grad_rhs = pick_num(a, "stationarity_rhs"),
    audit_f_consistent = pick_lgl(a, "f_consistent"),
    audit_advisory = pick_chr(a, "advisory_status"),
    diag_sigma_pd = pick_lgl(d, "sigma_pd_all"),
    diag_lin_eq_ok = pick_lgl(d, "lin_eq_satisfied"),
    diag_nl_eq_ok = pick_lgl(d, "nl_eq_satisfied"),
    diag_snlls_fallback = pick_lgl(d, "snlls_profile_fallback")
  )
}

safe_condition <- function(S) {
  ev <- eigen((S + t(S)) / 2, symmetric = TRUE, only.values = TRUE)$values
  if (!length(ev)) return(c(min = NA_real_, max = NA_real_,
                            condition = NA_real_))
  c(min = min(ev), max = max(ev),
    condition = if (min(ev) > 0) max(ev) / min(ev) else Inf)
}

converged_by_audit <- function(x) {
  !is.na(x$audit_advisory) & x$audit_advisory == "converged"
}

failure_summary <- function(x) {
  converged <- converged_by_audit(x)
  solver_ok <- is.na(x$error) | !nzchar(x$error)
  data.frame(
    reps = nrow(x),
    n_converged = sum(converged),
    convergence_rate = mean(converged),
    n_error = sum(!solver_ok),
    solver_error_rate = mean(!solver_ok),
    failure_rate = mean(!converged),
    median_iter = stats::median(x$iter[solver_ok], na.rm = TRUE),
    median_fevals = stats::median(x$f_evals[solver_ok], na.rm = TRUE),
    median_wall_usec = stats::median(x$wall_usec[solver_ok], na.rm = TRUE),
    kappa_med = stats::median(x$kappa_s, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

summarize_by <- function(raw, group_cols) {
  key <- do.call(paste, c(raw[group_cols], sep = "\r"))
  parts <- lapply(split(raw, key), function(x) {
    cbind(x[1L, group_cols, drop = FALSE], failure_summary(x),
          row.names = NULL)
  })
  out <- do.call(rbind, parts)
  row.names(out) <- NULL
  out
}
