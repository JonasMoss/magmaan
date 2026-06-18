#!/usr/bin/env Rscript
# Pairwise-deletion ordinal stage-two estimator comparison.
#
# This experiment reuses one pairwise OrdinalStats object per simulated data set,
# builds each stage-two weight once, then refits H1/H0 and computes several
# p-value transforms from one robust spectrum per outcome.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N[,N]] [--missing-rate P[,P]]
#       [--models m8_2f,m12_3f] [--estimators uls,dwls,wls,nt,dls25,dls50,dls75]
#       [--pd-gamma overlap[,nominal]] [--truths fake,real]
#       [--seed-base S] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

`%||%` <- function(a, b) if (is.null(a)) b else a

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (nm in miss) x[[nm]] <- NA
    x[, cols, drop = FALSE]
  })
  out <- do.call(rbind, xs)
  rownames(out) <- NULL
  out
}

elapsed_ms <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, elapsed_ms = 1000 * (proc.time()[["elapsed"]] - t0))
}

parse_args <- function(args) {
  out <- list(
    reps = 60L,
    n = c(800L),
    missing_rate = c(.15, .30),
    models = c("m8_2f", "m12_3f"),
    estimators = c("uls", "dwls", "wls", "nt", "dls25", "dls50", "dls75"),
    pd_gamma = c("overlap"),
    truths = c("fake", "real"),
    seed_base = 20260618L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N[,N]]\n",
          "       [--missing-rate P[,P]] [--models m8_2f,m12_3f]\n",
          "       [--estimators uls,dwls,wls,nt,dls25,dls50,dls75]\n",
          "       [--pd-gamma overlap[,nominal]] [--truths fake,real]\n",
          "       [--seed-base S] [--smoke]\n\n",
          "Compares all-ordinal pairwise-deletion stage-two weights while\n",
          "reusing the same threshold/polychoric/Gamma statistics per data set.\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") {
      i <- i + 1L; out$n <- as.integer(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--n=")) {
      out$n <- as.integer(parse_csv_arg(sub("^--n=", "", a)))
    } else if (a == "--missing-rate") {
      i <- i + 1L; out$missing_rate <- as.numeric(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--missing-rate=")) {
      out$missing_rate <- as.numeric(parse_csv_arg(sub("^--missing-rate=", "", a)))
    } else if (a == "--models") {
      i <- i + 1L; out$models <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--models=")) {
      out$models <- parse_csv_arg(sub("^--models=", "", a))
    } else if (a == "--estimators") {
      i <- i + 1L; out$estimators <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--estimators=")) {
      out$estimators <- parse_csv_arg(sub("^--estimators=", "", a))
    } else if (a == "--pd-gamma") {
      i <- i + 1L; out$pd_gamma <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--pd-gamma=")) {
      out$pd_gamma <- parse_csv_arg(sub("^--pd-gamma=", "", a))
    } else if (a == "--truths") {
      i <- i + 1L; out$truths <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--truths=")) {
      out$truths <- parse_csv_arg(sub("^--truths=", "", a))
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 2L
    out$n <- 320L
    out$missing_rate <- .15
    out$models <- "m8_2f"
    out$pd_gamma <- "overlap"
    out$truths <- c("fake", "real")
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n)) || any(out$n < 80L)) stop("--n must be >= 80")
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .95)) {
    stop("--missing-rate must be in [0, .95)", call. = FALSE)
  }
  bad_models <- setdiff(out$models, c("m8_2f", "m12_3f"))
  if (length(bad_models)) stop("unknown model: ", paste(bad_models, collapse = ", "))
  bad_truth <- setdiff(out$truths, c("fake", "real"))
  if (length(bad_truth)) stop("unknown truth: ", paste(bad_truth, collapse = ", "))
  bad_gamma <- setdiff(out$pd_gamma, c("overlap", "nominal"))
  if (length(bad_gamma)) stop("unknown pd-gamma: ", paste(bad_gamma, collapse = ", "))
  out
}

estimator_specs <- function(tokens) {
  parse_one <- function(tok) {
    key <- tolower(tok)
    if (key == "uls") return(data.frame(estimator = "ULS", stage2 = "uls", dls_a = NA_real_))
    if (key == "dwls") return(data.frame(estimator = "DWLS", stage2 = "dwls", dls_a = NA_real_))
    if (key %in% c("wls", "adf")) return(data.frame(estimator = "WLS", stage2 = "wls", dls_a = NA_real_))
    if (key %in% c("nt", "gls")) return(data.frame(estimator = "NT", stage2 = "nt", dls_a = NA_real_))
    if (grepl("^dls[0-9]+$", key)) {
      a <- as.numeric(sub("^dls", "", key)) / 100
      return(data.frame(estimator = sprintf("DLS%.2f", a), stage2 = "dls", dls_a = a))
    }
    if (grepl("^dls[0-9.]+$", key)) {
      a <- as.numeric(sub("^dls", "", key))
      return(data.frame(estimator = sprintf("DLS%.2f", a), stage2 = "dls", dls_a = a))
    }
    stop("unknown estimator: ", tok, call. = FALSE)
  }
  out <- do.call(rbind, lapply(tokens, parse_one))
  if (any(out$stage2 == "dls" & (out$dls_a < 0 | out$dls_a > 1))) {
    stop("DLS estimator suffix must resolve to a in [0, 1]", call. = FALSE)
  }
  rownames(out) <- NULL
  out
}

model_definition <- function(id) {
  thresholds <- c(-0.80, -0.15, 0.55, 1.15)
  if (id == "m8_2f") {
    return(list(
      id = id,
      n_factor = 2L,
      indicators = list(f1 = paste0("y", 1:4), f2 = paste0("y", 5:8)),
      loadings = c(.82, .76, .70, .64, .80, .74, .68, .62),
      factor_cor = matrix(c(1, .35, .35, 1), 2, 2),
      shifts = data.frame(item = c("y4", "y8"), mult = c(1.35, .70)),
      thresholds = thresholds,
      se_rhs = c("y2", "y4", "y6", "y8")
    ))
  }
  list(
    id = id,
    n_factor = 3L,
    indicators = list(f1 = paste0("y", 1:4),
                      f2 = paste0("y", 5:8),
                      f3 = paste0("y", 9:12)),
    loadings = c(.82, .76, .70, .64, .80, .74, .68, .62, .78, .72, .66, .60),
    factor_cor = matrix(c(1, .35, .25, .35, 1, .30, .25, .30, 1), 3, 3),
    shifts = data.frame(item = c("y4", "y8", "y12"), mult = c(1.30, .72, 1.25)),
    thresholds = thresholds,
    se_rhs = c("y2", "y4", "y6", "y8", "y10", "y12")
  )
}

model_syntax <- function(def) {
  load_rows <- vapply(names(def$indicators), function(f) {
    paste(f, "=~", paste(def$indicators[[f]], collapse = " + "))
  }, character(1))
  cov_rows <- character()
  fs <- names(def$indicators)
  if (length(fs) > 1L) {
    for (j in seq_len(length(fs) - 1L)) {
      for (i in (j + 1L):length(fs)) {
        cov_rows <- c(cov_rows, paste(fs[[j]], "~~", fs[[i]]))
      }
    }
  }
  paste(c(load_rows, cov_rows), collapse = "\n")
}

specs_for <- function(def) {
  ov <- unlist(def$indicators, use.names = FALSE)
  syntax <- model_syntax(def)
  list(
    h1 = model_spec(syntax, ordered = ov, group = "group",
                    group_labels = c("A", "B"), parameterization = "delta"),
    h0 = model_spec(syntax, ordered = ov, group = "group",
                    group_labels = c("A", "B"), parameterization = "delta",
                    group_equal = "loadings"),
    ov = ov
  )
}

draw_group <- function(def, n, truth, group_label) {
  ov <- unlist(def$indicators, use.names = FALSE)
  zf <- matrix(rnorm(n * def$n_factor), n, def$n_factor) %*% chol(def$factor_cor)
  colnames(zf) <- names(def$indicators)
  lambda <- stats::setNames(def$loadings, ov)
  if (identical(truth, "real") && identical(group_label, "B")) {
    for (r in seq_len(nrow(def$shifts))) {
      item <- def$shifts$item[[r]]
      lambda[[item]] <- max(.35, min(.94, lambda[[item]] * def$shifts$mult[[r]]))
    }
  }
  factor_for <- rep(names(def$indicators), lengths(def$indicators))
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    lam <- lambda[[ov[[j]]]]
    z[, j] <- lam * zf[, factor_for[[j]]] + rnorm(n, sd = sqrt(1 - lam^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, def$thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$group <- group_label
  out
}

draw_data <- function(def, n_per_group, truth, missing_rate) {
  dat <- rbind(draw_group(def, n_per_group, truth, "A"),
               draw_group(def, n_per_group, truth, "B"))
  ov <- unlist(def$indicators, use.names = FALSE)
  if (missing_rate > 0) {
    for (nm in ov) {
      miss <- runif(nrow(dat)) < missing_rate
      dat[[nm]][miss] <- NA
    }
  }
  dat
}

stats_with_weight <- function(stats, W, estimator) {
  out <- stats
  out$W_wls <- W
  out$stage2_estimator <- estimator
  out
}

timing_row <- function(tag, estimator, stage, elapsed_ms, ok = TRUE, error = "") {
  data.frame(tag, estimator = estimator, stage = stage,
             elapsed_ms = elapsed_ms, ok = ok, error = error,
             stringsAsFactors = FALSE)
}

fmg_pvalue <- function(chisq, df, eigvals, method, param = 4) {
  core <- magmaan::magmaan_core
  out <- core$robust_fmg_test(chisq, as.integer(df), as.numeric(eigvals),
                              method = method, param = param)
  out$p_value
}

gof_p_rows <- function(tag, estimator, spectrum) {
  chisq <- spectrum$chisq_standard
  df <- spectrum$df
  eigvals <- as.numeric(spectrum$eigvals)
  methods <- c("naive", "SB", "SS", "pEBA4", "pOLS", "all")
  p <- c(
    stats::pchisq(chisq, df, lower.tail = FALSE),
    fmg_pvalue(chisq, df, eigvals, "sb"),
    fmg_pvalue(chisq, df, eigvals, "scaled_shifted"),
    fmg_pvalue(chisq, df, eigvals, "peba", 4),
    fmg_pvalue(chisq, df, eigvals, "pols", 2),
    fmg_pvalue(chisq, df, eigvals, "all")
  )
  data.frame(
    tag,
    estimator = estimator,
    outcome = "gof",
    method = methods,
    p_value = p,
    reject_05 = is.finite(p) & p < .05,
    base_stat = chisq,
    df = df,
    trace = sum(eigvals),
    scaling_factor = sum(eigvals) / df,
    stringsAsFactors = FALSE
  )
}

nested_p_rows <- function(tag, estimator, nested) {
  eigvals <- as.numeric(nested$eigenvalues)
  methods <- c("naive", "SB", "adjusted", "mixture", "pEBA4", "pOLS", "all")
  p <- c(
    nested$p_unscaled,
    nested$p_scaled,
    nested$p_adjusted,
    nested$p_mixture,
    fmg_pvalue(nested$T_diff, nested$df_diff, eigvals, "peba", 4),
    fmg_pvalue(nested$T_diff, nested$df_diff, eigvals, "pols", 2),
    fmg_pvalue(nested$T_diff, nested$df_diff, eigvals, "all")
  )
  data.frame(
    tag,
    estimator = estimator,
    outcome = "nested",
    method = methods,
    p_value = p,
    reject_05 = is.finite(p) & p < .05,
    base_stat = nested$T_diff,
    df = nested$df_diff,
    trace = sum(eigvals),
    scaling_factor = nested$scale_c %||% NA_real_,
    stringsAsFactors = FALSE
  )
}

se_rows <- function(tag, estimator, fit, spectrum, rhs_keep) {
  pt <- fit$partable
  if (is.null(pt) || !all(c("op", "rhs", "lhs", "free") %in% names(pt))) {
    return(data.frame())
  }
  idx <- which(pt$op == "=~" & pt$rhs %in% rhs_keep & pt$free > 0)
  if (!length(idx)) return(data.frame())
  free <- as.integer(pt$free[idx])
  theta <- as.numeric(fit$theta)
  se <- as.numeric(spectrum$se)
  data.frame(
    tag,
    estimator = estimator,
    param = paste(pt$lhs[idx], "=~", pt$rhs[idx]),
    rhs = pt$rhs[idx],
    group = pt$group[idx] %||% NA_integer_,
    free = free,
    estimate = theta[free],
    se = se[free],
    finite_se = is.finite(se[free]),
    se_source = "h0_gof_spectrum",
    stringsAsFactors = FALSE
  )
}

failure_p_rows <- function(tag, estimator, error) {
  base <- expand.grid(
    estimator = estimator,
    outcome = c("gof", "nested"),
    method = "fit_failed",
    stringsAsFactors = FALSE
  )
  cbind(
    tag[rep(1L, nrow(base)), , drop = FALSE],
    base,
    p_value = NA_real_,
    reject_05 = FALSE,
    base_stat = NA_real_,
    df = NA_integer_,
    trace = NA_real_,
    scaling_factor = NA_real_,
    error = error,
    stringsAsFactors = FALSE
  )
}

run_estimator <- function(tag, def, specs, stats, est_spec, control) {
  core <- magmaan::magmaan_core
  estimator <- est_spec$estimator
  timings <- list()

  dls_a <- if (is.finite(est_spec$dls_a)) est_spec$dls_a else 0.5
  wt <- elapsed_ms(core$ordinal_stage2_weight_blocks(
    stats, stage2_weight = est_spec$stage2,
    dls_a = dls_a))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "weight", wt$elapsed_ms)
  wstats <- stats_with_weight(stats, wt$value, estimator)

  h1 <- elapsed_ms(core$fit_wls_ordinal(specs$h1, wstats, control = control))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "fit_h1", h1$elapsed_ms)
  h0 <- elapsed_ms(core$fit_wls_ordinal(specs$h0, wstats, control = control))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "fit_h0", h0$elapsed_ms)
  if (!isTRUE(h1$value$converged) || !isTRUE(h0$value$converged)) {
    stop("stage-two fit did not converge", call. = FALSE)
  }

  gof <- elapsed_ms(core$infer_ordinal_robust(h0$value, wstats, weight = "WLS"))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "gof_spectrum", gof$elapsed_ms)
  gof_rows <- gof_p_rows(tag, estimator, gof$value)
  se <- se_rows(tag, estimator, h0$value, gof$value, def$se_rhs)

  nested <- elapsed_ms(robust_nested_lrt(
    h1$value, h0$value, data = wstats, method = "restriction_map",
    A.method = "delta", weight = "WLS"))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "nested_spectrum", nested$elapsed_ms)
  nested_rows <- nested_p_rows(tag, estimator, nested$value)

  total <- sum(vapply(timings, function(x) x$elapsed_ms, numeric(1)))
  timings[[length(timings) + 1L]] <- timing_row(tag, estimator, "total_estimator", total)
  list(p_values = rbind(gof_rows, nested_rows),
       se = se,
       timings = rbind_fill(timings))
}

summarize_rejection <- function(p_values) {
  if (!nrow(p_values)) return(data.frame())
  key <- interaction(p_values$model, p_values$n_per_group,
                     p_values$missing_rate, p_values$truth, p_values$pd_gamma,
                     p_values$estimator, p_values$outcome, p_values$method,
                     drop = TRUE)
  rows <- lapply(split(p_values, key), function(d) {
    ok <- is.finite(d$p_value)
    rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
    data.frame(
      model = d$model[1L],
      n_per_group = d$n_per_group[1L],
      missing_rate = d$missing_rate[1L],
      realized_missing = mean(d$realized_missing, na.rm = TRUE),
      truth = d$truth[1L],
      pd_gamma = d$pd_gamma[1L],
      estimator = d$estimator[1L],
      outcome = d$outcome[1L],
      method = d$method[1L],
      reps = nrow(d),
      usable = sum(ok),
      failures = sum(!ok),
      rejection_rate = rate,
      mc_se = if (sum(ok) > 0) sqrt(rate * (1 - rate) / sum(ok)) else NA_real_,
      mean_base = mean(d$base_stat, na.rm = TRUE),
      mean_trace = mean(d$trace, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$model, out$n_per_group, out$missing_rate, out$truth,
            out$pd_gamma, out$estimator, out$outcome, out$method), ]
}

summarize_timings <- function(timings) {
  if (!nrow(timings)) return(data.frame())
  ok <- timings[timings$ok %in% TRUE, , drop = FALSE]
  key <- interaction(ok$model, ok$n_per_group, ok$missing_rate, ok$pd_gamma,
                     ok$estimator, ok$stage, drop = TRUE)
  rows <- lapply(split(ok, key), function(d) data.frame(
    model = d$model[1L],
    n_per_group = d$n_per_group[1L],
    missing_rate = d$missing_rate[1L],
    pd_gamma = d$pd_gamma[1L],
    estimator = d$estimator[1L],
    stage = d$stage[1L],
    n = nrow(d),
    median_elapsed_ms = stats::median(d$elapsed_ms, na.rm = TRUE),
    mean_elapsed_ms = mean(d$elapsed_ms, na.rm = TRUE),
    stringsAsFactors = FALSE
  ))
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$model, out$n_per_group, out$missing_rate,
            out$pd_gamma, out$estimator, out$stage), ]
}

summarize_se <- function(se) {
  if (!nrow(se)) return(data.frame())
  key <- interaction(se$model, se$n_per_group, se$missing_rate, se$pd_gamma,
                     se$estimator, se$param, se$group, drop = TRUE)
  rows <- lapply(split(se, key), function(d) data.frame(
    model = d$model[1L],
    n_per_group = d$n_per_group[1L],
    missing_rate = d$missing_rate[1L],
    pd_gamma = d$pd_gamma[1L],
    estimator = d$estimator[1L],
    param = d$param[1L],
    group = d$group[1L],
    n = nrow(d),
    finite_rate = mean(d$finite_se),
    median_se = stats::median(d$se, na.rm = TRUE),
    median_estimate = stats::median(d$estimate, na.rm = TRUE),
    stringsAsFactors = FALSE
  ))
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$model, out$n_per_group, out$missing_rate,
            out$pd_gamma, out$estimator, out$param, out$group), ]
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
est_grid <- estimator_specs(cfg$estimators)
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- ensure_results_dir()
control <- list(max_iter = 2500L, ftol = 1e-10, gtol = 1e-8)

cat(sprintf("ordinal stage-two pairwise: %d model(s), %d reps, estimators={%s}\n",
            length(cfg$models), cfg$reps, paste(est_grid$estimator, collapse = ",")))

p_rows <- list(); timing_rows <- list(); se_out <- list()
ip <- it <- is <- 0L

for (model_i in seq_along(cfg$models)) {
  def <- model_definition(cfg$models[[model_i]])
  specs <- specs_for(def)
  for (n in cfg$n) {
    for (mr in cfg$missing_rate) {
      for (truth in cfg$truths) {
        for (rep in seq_len(cfg$reps)) {
          seed <- cfg$seed_base + model_i * 10000000L + n * 1000L +
            round(mr * 1000) * 10L + rep +
            if (identical(truth, "real")) 500000L else 0L
          set.seed(seed)
          dat <- draw_data(def, n, truth, mr)
          realized_missing <- mean(is.na(dat[, specs$ov]))
          for (pd in cfg$pd_gamma) {
            tag <- data.frame(
              model = def$id,
              n_per_group = n,
              missing_rate = mr,
              realized_missing = realized_missing,
              truth = truth,
              rep = rep,
              pd_gamma = pd,
              seed = seed,
              stringsAsFactors = FALSE
            )
            stats_result <- tryCatch({
              elapsed_ms(core$data_ordinal_stats_from_df(
                dat, specs$h1, ordered = specs$ov, group = "group",
                missing = "pairwise", pd_gamma = pd,
                full_wls_weight = TRUE))
            }, error = function(e) e)
            if (inherits(stats_result, "error")) {
              for (est in est_grid$estimator) {
                ip <- ip + 1L
                p_rows[[ip]] <- failure_p_rows(tag, est, conditionMessage(stats_result))
                it <- it + 1L
                timing_rows[[it]] <- timing_row(tag, est, "stats", NA_real_,
                                                ok = FALSE,
                                                error = conditionMessage(stats_result))
              }
              next
            }
            it <- it + 1L
            timing_rows[[it]] <- timing_row(tag, "", "stats", stats_result$elapsed_ms)

            for (ei in seq_len(nrow(est_grid))) {
              est_spec <- est_grid[ei, , drop = FALSE]
              ans <- tryCatch(
                run_estimator(tag, def, specs, stats_result$value, est_spec, control),
                error = function(e) e
              )
              if (inherits(ans, "error")) {
                ip <- ip + 1L
                p_rows[[ip]] <- failure_p_rows(tag, est_spec$estimator,
                                               conditionMessage(ans))
                it <- it + 1L
                timing_rows[[it]] <- timing_row(tag, est_spec$estimator,
                                                "total_estimator", NA_real_,
                                                ok = FALSE,
                                                error = conditionMessage(ans))
              } else {
                ip <- ip + 1L; p_rows[[ip]] <- ans$p_values
                it <- it + 1L; timing_rows[[it]] <- ans$timings
                if (nrow(ans$se)) {
                  is <- is + 1L; se_out[[is]] <- ans$se
                }
              }
            }
          }
          cat(sprintf("model=%s n=%d missing=%.2f truth=%s rep=%d/%d\n",
                      def$id, n, mr, truth, rep, cfg$reps))
        }
      }
    }
  }
}

p_values <- rbind_fill(p_rows)
timings <- rbind_fill(timing_rows)
se_diag <- rbind_fill(se_out)

front_p <- c("model", "n_per_group", "missing_rate", "realized_missing",
             "truth", "rep", "pd_gamma", "estimator", "outcome", "method",
             "p_value", "reject_05", "base_stat", "df", "trace",
             "scaling_factor", "seed")
p_values <- p_values[, c(front_p, setdiff(names(p_values), front_p)), drop = FALSE]
write_csv(p_values, file.path(res_dir, "p_values.csv"))

front_t <- c("model", "n_per_group", "missing_rate", "truth", "rep",
             "pd_gamma", "estimator", "stage", "elapsed_ms", "ok", "error")
timings <- timings[, c(front_t, setdiff(names(timings), front_t)), drop = FALSE]
write_csv(timings, file.path(res_dir, "timings.csv"))

if (nrow(se_diag)) {
  front_se <- c("model", "n_per_group", "missing_rate", "truth", "rep",
                "pd_gamma", "estimator", "param", "rhs", "group", "free",
                "estimate", "se", "finite_se")
  se_diag <- se_diag[, c(front_se, setdiff(names(se_diag), front_se)), drop = FALSE]
}
write_csv(se_diag, file.path(res_dir, "se_diagnostics.csv"))

summary <- summarize_rejection(p_values)
write_csv(summary, file.path(res_dir, "summary.csv"))

timing_summary <- summarize_timings(timings)
write_csv(timing_summary, file.path(res_dir, "timing_summary.csv"))

se_summary <- summarize_se(se_diag)
write_csv(se_summary, file.path(res_dir, "se_summary.csv"))

metadata <- metadata_frame(
  list(
    reps = cfg$reps,
    n = cfg$n,
    missing_rate = cfg$missing_rate,
    models = cfg$models,
    estimators = est_grid$estimator,
    pd_gamma = cfg$pd_gamma,
    truths = cfg$truths,
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    question = "ordinal pairwise stage-two estimator p-values and timing"
  ),
  packages = c("magmaan")
)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

cat("wrote:\n")
for (nm in c("p_values.csv", "timings.csv", "se_diagnostics.csv",
             "summary.csv", "timing_summary.csv", "se_summary.csv",
             "metadata.csv")) {
  cat("  ", file.path(res_dir, nm), "\n", sep = "")
}
