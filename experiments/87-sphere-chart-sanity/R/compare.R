# Fitting wrappers and the comparisons for the recovery section.

# Evaluate `expr`, keeping the value or the condition and the elapsed time.
# Warnings (admissibility notes and the like) are muffled but counted.
timed <- function(expr) {
  n_warn <- 0L
  t0 <- proc.time()[["elapsed"]]
  value <- withCallingHandlers(
    tryCatch(expr, magmaan_user_chart_singular = function(e) e,
             error = function(e) e),
    warning = function(w) {
      n_warn <<- n_warn + 1L
      invokeRestart("muffleWarning")
    })
  list(value = value, time = proc.time()[["elapsed"]] - t0, warnings = n_warn)
}

# The frontier PSD-ML entry point returns a plain fit list, not a classed fit.
is_fit <- function(x) {
  inherits(x, "magmaan_fit") ||
    (is.list(x) && !inherits(x, "condition") && !is.null(x$partable) && !is.null(x$theta))
}
err_msg <- function(x) if (inherits(x, "condition")) conditionMessage(x) else NA_character_

data_for <- function(case, estimator, data) {
  data[[if (identical(estimator, "FIML")) paste0(case$data, "_miss") else case$data]]
}

case_spec <- function(case, d) {
  opts <- case$opts
  opts$bounds <- NULL
  if (!is.null(case$groups)) {
    opts$group <- case$groups
    opts$group_labels <- unique(as.character(d[[case$groups]]))
  }
  do.call(model_spec, c(list(syntax = case$syntax), opts))
}

# Model and data arguments for a case. Least-squares estimators get the
# n - 1 covariance (lavaan's convention for them); PSD gets a prepared data
# object because its entry point takes a model spec.
fit_input <- function(case, estimator, d) {
  if (estimator %in% c("ULS", "GLS", "WLS", "PSD")) {
    spec <- case_spec(case, d)
    scaling <- if (identical(estimator, "PSD")) "n" else "n-1"
    return(list(model = spec, groups = NULL, opts = list(),
                data = df_to_data(d, spec, group = case$groups, scaling = scaling)))
  }
  list(model = case$syntax, data = d, groups = case$groups,
       opts = case$opts[setdiff(names(case$opts), "bounds")])
}

# Ordinary magmaan fit for a case and estimator. PSD is the frontier PSD-ML
# fit without preconditioning, which is what the sphere PSD route polishes to.
fit_ordinary <- function(case, estimator, inp, W = NULL) {
  if (identical(estimator, "PSD")) {
    return(frontier_fit_ml_psd(inp$model, inp$data, preconditioning = "none"))
  }
  args <- c(list(model = inp$model, data = inp$data, estimator = estimator,
                 groups = inp$groups, optimizer = case$optimizer,
                 bounds = case$opts$bounds, W = W), inp$opts)
  do.call(magmaan, args)
}

fit_sphere <- function(case, estimator, inp, W = NULL) {
  psd <- identical(estimator, "PSD")
  args <- c(list(model = inp$model, data = inp$data,
                 estimator = if (psd) "ML" else estimator,
                 groups = inp$groups, psd = psd, optimizer = case$optimizer,
                 bounds = case$opts$bounds, W = W), inp$opts)
  do.call(frontier_fit_sphere, args)
}

lavaan_args <- function(case) {
  o <- case$opts
  out <- list()
  if (isTRUE(o$std_lv)) out$std.lv <- TRUE
  # lavaan's effect.coding = TRUE also effect-codes the intercepts; magmaan's
  # effect_coding covers the loadings only (see the report).
  if (isTRUE(o$effect_coding)) out$effect.coding <- "loadings"
  if (isTRUE(o$orthogonal)) out$orthogonal <- TRUE
  if (isTRUE(o$meanstructure)) out$meanstructure <- TRUE
  if (!is.null(o$group_equal)) out$group.equal <- o$group_equal
  if (!is.null(o$group_partial)) out$group.partial <- o$group_partial
  if (!is.null(o$bounds)) out$bounds <- o$bounds
  if (!is.null(case$groups)) out$group <- case$groups
  out
}

fit_lavaan <- function(case, estimator, d) {
  if (identical(estimator, "PSD")) return(NULL)
  args <- c(list(model = case$syntax, data = d), lavaan_args(case))
  if (identical(estimator, "FIML")) {
    args$estimator <- "ML"
    args$missing <- "ml"
  } else {
    args$estimator <- estimator
  }
  fun <- if (identical(case$opts$model_type, "growth")) lavaan::growth else lavaan::sem
  suppressWarnings(do.call(fun, args))
}

free_idx <- function(pt) which(pt$free > 0 & !pt$op %in% c("==", "<", ">", ":="))

max_abs <- function(a, b) {
  if (is.null(a) || is.null(b) || length(a) != length(b)) return(NA_real_)
  d <- abs(as.numeric(a) - as.numeric(b))
  if (!length(d) || all(is.na(d))) NA_real_ else max(d, na.rm = TRUE)
}

max_rel <- function(a, b, floor = 1e-3) {
  if (is.null(a) || is.null(b) || length(a) != length(b)) return(NA_real_)
  d <- abs(as.numeric(a) - as.numeric(b)) / pmax(abs(as.numeric(b)), floor)
  if (!length(d) || all(is.na(d))) NA_real_ else max(d, na.rm = TRUE)
}

row_key <- function(pt) {
  g <- if (!is.null(pt$group)) pt$group else 1L
  paste(pt$lhs, pt$op, pt$rhs, g, sep = "|")
}

# Largest |estimate difference| against lavaan over magmaan's free
# parameters, matched by (lhs, op, rhs, group). lavaan adds saturated
# observed intercepts to multi-group models; those rows are not compared.
lavaan_diff <- function(fit, lav) {
  if (!is_fit(fit) || is.null(lav) || inherits(lav, "condition")) return(NA_real_)
  lp <- lavaan::parTable(lav)
  mp <- fit$partable[free_idx(fit$partable), ]
  k <- match(row_key(mp), row_key(lp))
  if (anyNA(k)) return(NA_real_)
  max_abs(mp$est, lp$est[k])
}

lavaan_chisq <- function(lav) {
  if (is.null(lav) || inherits(lav, "condition")) return(NA_real_)
  unname(lavaan::fitMeasures(lav, "chisq"))
}

safe <- function(expr) tryCatch(expr, error = function(e) e)

# Post-fit outputs a user would compute on either fit. Each entry is a
# numeric vector, or the condition when that output errored.
model_vcov <- function(fit, estimator) {
  if (identical(estimator, "FIML")) return(vcov(fit))
  info <- magmaan_core$inference_information_expected(fit)
  magmaan_core$inference_vcov(info, fit)
}

# Sandwich with the empirical meat (FIML: the MLR sandwich).
robust_vcov <- function(fit, estimator, d) {
  if (identical(estimator, "FIML")) return(vcov(fit, regime = "robust"))
  raw <- if (!is.null(fit$raw_data$X)) fit$raw_data$X else d
  vcov(fit, regime = "model", data = raw)
}

post_fit <- function(fit, case, estimator, d) {
  out <- list()
  vc <- safe(model_vcov(fit, estimator))
  out$se <- if (inherits(vc, "condition")) vc else safe(parameter_table(fit, vc)$se)
  # The ordinary PSD entry point returns a bare fit list without the raw data
  # the empirical sandwich needs, so robust SEs are compared for the other
  # estimators only.
  if (!identical(estimator, "PSD")) {
    vr <- safe(robust_vcov(fit, estimator, d))
    out$se_robust <- if (inherits(vr, "condition")) vr else safe(parameter_table(fit, vr)$se)
  }
  out$std <- if (inherits(vc, "condition")) vc else safe(standardized(fit, vc)$theta)
  fm <- safe(fit_measures(fit))
  keep <- c("chisq", "df", "cfi", "tli", "rmsea", "srmr")
  out$fit_measures <- if (inherits(fm, "condition")) fm else unlist(fm[intersect(keep, names(fm))])
  if (identical(estimator, "ML")) {
    mi <- safe(modification_indices(fit))
    out$mod_indices <- if (inherits(mi, "condition")) mi else mi$mi
  }
  if (grepl(":=", case$syntax, fixed = TRUE) && !inherits(vc, "condition")) {
    df <- safe(compute_defined(case$syntax, fit, vc))
    out$defined <- if (inherits(df, "condition")) df else c(df$est, df$se)
  }
  out
}

# Compare two post-fit lists: max relative difference per output, or a flag
# when only one side errored.
compare_post <- function(a, b) {
  keys <- union(names(a), names(b))
  vapply(keys, function(k) {
    x <- a[[k]]
    y <- b[[k]]
    ex <- inherits(x, "condition")
    ey <- inherits(y, "condition")
    if (ex && ey) return(NA_real_)
    if (ex != ey) return(Inf)
    max_rel(x, y)
  }, numeric(1))
}

gauge_sets <- function(fit) {
  g <- fit$gauge
  list(units = sort(unique(g$units$latent)),
       pass = sort(unique(g$passthrough$latent)))
}

run_recovery_case <- function(case, estimator, data) {
  d <- data_for(case, estimator, data)
  l <- timed(fit_lavaan(case, estimator, d))
  lav <- l$value
  W <- if (identical(estimator, "WLS") && !inherits(lav, "condition")) {
    lavaan::lavInspect(lav, "wls.v")
  }
  inp <- fit_input(case, estimator, d)
  o <- timed(fit_ordinary(case, estimator, inp, W))
  s <- timed(fit_sphere(case, estimator, inp, W))
  ord <- o$value
  sph <- s$value
  row <- data.frame(
    case = case$id, family = case$family, label = case$label,
    estimator = estimator,
    ord_ok = is_fit(ord) && isTRUE(ord$converged),
    sph_ok = is_fit(sph) && isTRUE(sph$converged),
    sph_singular = inherits(sph, "magmaan_user_chart_singular"),
    ord_error = err_msg(ord), sph_error = err_msg(sph),
    lav_ok = !is.null(lav) && !inherits(lav, "condition") &&
      isTRUE(lavaan::lavInspect(lav, "converged")),
    stringsAsFactors = FALSE)
  row$gauge_ok <- NA
  row$n_units <- NA_integer_
  row$n_pass <- NA_integer_
  if (is_fit(sph) || inherits(sph, "magmaan_user_chart_singular")) {
    gs <- gauge_sets(sph)
    row$gauge_ok <- identical(gs$units, sort(unique(case$units))) &&
      identical(gs$pass, sort(unique(case$pass)))
    row$n_units <- nrow(sph$gauge$units)
    row$n_pass <- nrow(sph$gauge$passthrough)
    row$pass_reasons <- paste(unique(sph$gauge$passthrough$reason), collapse = "; ")
  } else {
    row$pass_reasons <- NA_character_
  }
  both <- is_fit(ord) && is_fit(sph)
  row$d_fmin <- if (both) abs(sph$fmin - ord$fmin) else NA_real_
  row$d_est <- if (both) max_abs(sph$partable$est[free_idx(sph$partable)],
                                 ord$partable$est[free_idx(ord$partable)]) else NA_real_
  row$lav_d_ord <- lavaan_diff(ord, lav)
  row$lav_d_sph <- lavaan_diff(sph, lav)
  row$polish_shift <- if (is_fit(sph)) sph$gauge$polish$shift else NA_real_
  row$pin_residual <- if (is_fit(sph)) sph$gauge$pin_residual else NA_real_
  row$time_ord <- o$time
  row$time_sph <- s$time
  row$time_lav <- l$time
  post_keys <- c("se", "se_robust", "std", "fit_measures", "mod_indices", "defined")
  for (k in post_keys) row[[paste0("d_", k)]] <- NA_real_
  if (both) {
    cmp <- compare_post(post_fit(sph, case, estimator, d),
                        post_fit(ord, case, estimator, d))
    for (k in names(cmp)) row[[paste0("d_", k)]] <- cmp[[k]]
  }
  row
}
