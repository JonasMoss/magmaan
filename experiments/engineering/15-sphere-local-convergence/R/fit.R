# One fit and its outcome. Success is a certified local optimum: the fit is
# returned with converged = TRUE (magmaan's stationarity verdict), or, on the
# sphere route, the sphere run passes its own stationarity audit and the
# optimum lies outside the identification (magmaan_user_chart_singular).
# Global optimality is not assessed.

fit_route <- function(route, ident, data, optimizer = NULL, psd = FALSE) {
  if (identical(route, "sphere")) {
    args <- c(list(model_syntax, data), identifications[[ident]])
    if (!is.null(optimizer)) args$optimizer <- optimizer
    if (psd) args$psd <- TRUE
    return(do.call(frontier_fit_sphere, args))
  }
  if (psd) {
    spec <- do.call(model_spec, c(list(model_syntax), identifications[[ident]]))
    return(frontier_fit_ml_psd(spec, data, optimizer = optimizer %||% "nlopt-slsqp"))
  }
  args <- c(list(model_syntax, data), identifications[[ident]])
  if (!is.null(optimizer)) args$optimizer <- optimizer
  do.call(magmaan, args)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# A stop far out on a ridge (where the likelihood keeps improving toward
# infinity and no estimate exists) can still pass a stationarity test. The
# guard is identification-free: the standardized solution. `runaway_measure`
# is the largest of |standardized loading|, |residual variance / implied
# indicator variance| and |factor correlation|. Proper solutions keep it
# near or below 1; beyond `runaway_bound` the stop counts as a runaway, not a
# local optimum. Raw parameters cannot serve: a weak marker legitimately puts
# loadings near 50 in the marker chart.
runaway_bound <- 10

runaway_measure <- function(pt) {
  e <- stats::setNames(pt$est, paste0(pt$lhs, pt$op, pt$rhs))
  get <- function(k) if (k %in% names(e)) e[[k]] else NA_real_
  vx <- get("X~~X")
  b <- get("Y~X")
  vy <- b^2 * vx + get("Y~~Y")
  cxy <- b * vx
  r <- cxy / sqrt(abs(vx * vy))
  vals <- abs(r)
  for (f in c("X", "Y")) {
    v <- if (f == "X") vx else vy
    for (j in 1:3) {
      ind <- paste0(tolower(f), j)
      l <- get(paste0(f, "=~", ind))
      th <- get(paste0(ind, "~~", ind))
      sj <- l^2 * v + th
      vals <- c(vals, abs(l) * sqrt(abs(v)) / sqrt(abs(sj)), abs(th / sj))
    }
  }
  if (any(!is.finite(vals))) return(Inf)
  max(vals)
}

# status: certified, flagged (sphere optimum outside the identification, the
# sphere run passing its own stationarity audit), *_runaway (the same with a
# parameter beyond `runaway_bound`), flagged_uncertified, uncertified
# (returned, converged = FALSE), error. `f` is the objective at the returned
# or failing point. `clean`: the optimizer stopped on its convergence test.
outcome <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  f <- suppressWarnings(tryCatch(expr,
    magmaan_user_chart_singular = function(c) c,
    error = function(c) c))
  el <- proc.time()[["elapsed"]] - t0
  clean_status <- c("converged", "line_search_salvaged")
  if (inherits(f, "magmaan_user_chart_singular")) {
    g <- f$gauge
    mx <- runaway_measure(g$sphere_partable)
    st <- if (!isTRUE(g$driven_stationary)) "flagged_uncertified"
          else if (mx > runaway_bound) "flagged_runaway" else "flagged"
    return(list(status = st, f = g$fmin_sphere, max_abs = NA_real_, runaway = mx, admissible = NA,
                error = "", clean = g$optimizer_status %in% clean_status, time = el))
  }
  if (inherits(f, "condition")) {
    msg <- conditionMessage(f)
    fv <- suppressWarnings(as.numeric(sub(".*f=([-+0-9.eE]+).*", "\\1", msg)))
    kind <- if (grepl("\\[([A-Za-z]+)\\]", msg)) sub(".*\\[([A-Za-z]+)\\].*", "\\1", msg) else "other"
    return(list(status = "error", f = fv, max_abs = NA_real_, runaway = NA_real_, admissible = NA,
                error = kind, clean = FALSE, time = el))
  }
  adm <- f$diagnostics$admissibility$admissible
  mx <- runaway_measure(f$partable)
  st <- if (!isTRUE(f$converged)) "uncertified"
        else if (mx > runaway_bound) "certified_runaway" else "certified"
  list(status = st, f = f$fmin, max_abs = max(abs(f$theta)), runaway = mx,
       admissible = if (is.null(adm)) NA else isTRUE(adm), error = "",
       clean = isTRUE(f$optimizer_status %in% clean_status), time = el)
}

success <- function(status) status %in% c("certified", "flagged")

# Why a fit failed. "optimizer": PORT or SLSQP, from the same start policy,
# stops cleanly at a certified local optimum that is not a runaway, so one
# exists. "no_estimate": neither
# does, and the failing point fits better than the best admissible (PSD-ML)
# solution, so the likelihood keeps improving into the improper region
# (a nonexistence ridge). Otherwise "unclassified".
classify_failure <- function(route, ident, data, f_fail, f_psd) {
  for (opt in c("port", "nlopt-slsqp")) {
    alt <- outcome(fit_route(route, ident, data, optimizer = opt))
    if (success(alt$status) && alt$clean) return(list(type = "optimizer", fixed_by = opt))
  }
  if (is.finite(f_fail) && is.finite(f_psd) && f_fail < f_psd - 1e-8)
    return(list(type = "no_estimate", fixed_by = ""))
  list(type = "unclassified", fixed_by = "")
}

# The fits of one draw: ML and PSD-ML, ordinary and sphere, marker and
# std.lv. On the sphere route the identification only sets the reporting
# scale (the fit itself is the same). ML failures are labelled by
# classify_failure. PSD-ML has a closed domain, so an estimate exists; a
# PSD-ML failure is labelled "optimizer" when another PSD-ML route reaches a
# certified local optimum on the same draw.
run_draw <- function(design, n, rep, seed) {
  d <- designs_all()[[design]]
  data <- draw_data(design_sigma(d), n, seed)
  row <- function(estimator, ident, route, o, failure = "", fixed_by = "", f_psd = NA_real_)
    data.frame(design = design, n = n, rep = rep, seed = seed, estimator = estimator,
               ident = ident, route = route, status = o$status,
               success = success(o$status), error = o$error, f = o$f, f_psd = f_psd,
               failure = failure, fixed_by = fixed_by, max_abs = o$max_abs,
               runaway = o$runaway, admissible = o$admissible, time = o$time,
               stringsAsFactors = FALSE)
  rows <- list()
  psd_out <- list()
  for (ident in names(identifications)) for (route in c("ordinary", "sphere")) {
    psd_out[[paste(ident, route)]] <- outcome(fit_route(route, ident, data, psd = TRUE))
  }
  psd_ok <- vapply(psd_out, function(o) success(o$status), logical(1))
  sph <- psd_out[["marker sphere"]]
  f_psd <- if (success(sph$status)) sph$f else NA_real_
  for (k in names(psd_out)) {
    o <- psd_out[[k]]
    parts <- strsplit(k, " ")[[1]]
    fl <- if (success(o$status)) "" else if (any(psd_ok)) "optimizer" else "unclassified"
    rows[[length(rows) + 1L]] <- row("PSD-ML", parts[1], parts[2], o, failure = fl)
  }
  for (ident in names(identifications)) for (route in c("ordinary", "sphere")) {
    o <- outcome(fit_route(route, ident, data))
    ft <- list(type = "", fixed_by = "")
    if (!success(o$status)) ft <- classify_failure(route, ident, data, o$f, f_psd)
    rows[[length(rows) + 1L]] <- row("ML", ident, route, o, ft$type, ft$fixed_by,
                                     if (success(o$status)) NA_real_ else f_psd)
  }
  out <- do.call(rbind, rows)
  # Descriptive only: how far a successful fit sits above the lowest local
  # optimum any fit of the same estimator reached on the same data.
  out$gap_to_best <- NA_real_
  for (est in unique(out$estimator)) {
    i <- out$estimator == est
    ok <- i & out$success & is.finite(out$f)
    if (any(ok)) out$gap_to_best[ok] <- out$f[ok] - min(out$f[ok])
  }
  out
}
