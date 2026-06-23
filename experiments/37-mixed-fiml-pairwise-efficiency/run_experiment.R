#!/usr/bin/env Rscript
# Efficiency of mixed continuous/ordinal first-stage statistics under item
# missingness: fully pairwise (pairwise x pairwise) vs. the FIML-continuous
# hybrid (pairwise x FIML).
#
# Both first stages keep ordinal thresholds, polychorics, and polyserials on
# observed pairs. They differ only in the continuous mean/covariance block:
# the pairwise stage uses pairwise-complete moments; the hybrid replaces that
# block with saturated continuous FIML. We fit the same mixed DWLS model from
# each and compare the sampling behaviour (bias, SD, RMSE) of the structural
# parameters, plus robust IJ standard-error calibration.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N[,N]] [--missing-rate P[,P]]
#       [--mechanisms mcar,mar] [--ref-n N] [--seed-base S] [--smoke]

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
  base <- dirname(dirname(script))
  file.path(base, "_support", "R")
}
.support_r <- .support_helpers()
source(file.path(.support_r, "helpers.R"))
source(file.path(.support_r, "missingness.R"))
rm(.support_helpers, .support_r)

`%||%` <- function(a, b) if (is.null(a)) b else a

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    for (nm in setdiff(cols, names(x))) x[[nm]] <- NA
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

# ---- configuration ----------------------------------------------------------

parse_args <- function(args) {
  out <- list(
    reps = 300L,
    n = c(400L, 800L),
    missing_rate = c(.15, .35),
    mechanisms = c("mcar", "mar"),
    ref_n = 200000L,
    seed_base = 20260623L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    take <- function() { i <<- i + 1L; args[[i]] }
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N[,N]]\n",
          "       [--missing-rate P[,P]] [--mechanisms mcar,mar]\n",
          "       [--ref-n N] [--seed-base S] [--smoke]\n\n",
          "Compares pairwise-x-pairwise and pairwise-x-FIML mixed first-stage\n",
          "statistics by the sampling efficiency of the mixed DWLS estimates.\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { out$reps <- as.integer(take())
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") { out$n <- as.integer(parse_csv_arg(take()))
    } else if (startsWith(a, "--n=")) { out$n <- as.integer(parse_csv_arg(sub("^--n=", "", a)))
    } else if (a == "--missing-rate") { out$missing_rate <- as.numeric(parse_csv_arg(take()))
    } else if (startsWith(a, "--missing-rate=")) { out$missing_rate <- as.numeric(parse_csv_arg(sub("^--missing-rate=", "", a)))
    } else if (a == "--mechanisms") { out$mechanisms <- parse_csv_arg(take())
    } else if (startsWith(a, "--mechanisms=")) { out$mechanisms <- parse_csv_arg(sub("^--mechanisms=", "", a))
    } else if (a == "--ref-n") { out$ref_n <- as.integer(take())
    } else if (startsWith(a, "--ref-n=")) { out$ref_n <- as.integer(sub("^--ref-n=", "", a))
    } else if (a == "--seed-base") { out$seed_base <- as.integer(take())
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 4L
    out$n <- 400L
    out$missing_rate <- .30
    out$mechanisms <- c("mcar", "mar")
    out$ref_n <- 20000L
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n)) || any(out$n < 100L)) stop("--n must be >= 100")
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .9)) {
    stop("--missing-rate must be in [0, .9)", call. = FALSE)
  }
  bad_mech <- setdiff(out$mechanisms, c("mcar", "mar"))
  if (length(bad_mech)) stop("unknown mechanism: ", paste(bad_mech, collapse = ", "))
  out
}

# ---- population model -------------------------------------------------------

# Two correlated factors. f1 has four continuous indicators (the block the
# hybrid first stage touches), f2 has four ordinal indicators (3 categories,
# untouched by the hybrid). Marker-variable identification, delta param.
population <- function() {
  list(
    cont = paste0("x", 1:4),
    ord = paste0("y", 1:4),
    lambda_cont = c(.80, .75, .70, .65),
    lambda_ord = c(.80, .75, .70, .65),
    factor_cor = 0.40,
    thresholds = c(-0.5, 0.6)   # 3 categories
  )
}

model_syntax <- function(pop) {
  paste(
    paste("f1 =~", paste(pop$cont, collapse = " + ")),
    paste("f2 =~", paste(pop$ord, collapse = " + ")),
    "f1 ~~ f2",
    sep = "\n"
  )
}

model_spec_for <- function(pop) {
  magmaan::model_spec(model_syntax(pop), ordered = pop$ord,
                      parameterization = "delta", meanstructure = TRUE)
}

# Draw a complete mixed data frame. Continuous columns first so the SB-2005
# masks can use x1, x2 as fully observed anchors.
draw_data <- function(pop, n) {
  L <- chol(matrix(c(1, pop$factor_cor, pop$factor_cor, 1), 2, 2))
  f <- matrix(rnorm(n * 2), n, 2) %*% L
  cont <- sapply(seq_along(pop$cont), function(j) {
    lam <- pop$lambda_cont[j]
    lam * f[, 1] + sqrt(1 - lam^2) * rnorm(n)
  })
  ord <- sapply(seq_along(pop$ord), function(j) {
    lam <- pop$lambda_ord[j]
    z <- lam * f[, 2] + sqrt(1 - lam^2) * rnorm(n)
    cut(z, c(-Inf, pop$thresholds, Inf), labels = FALSE)
  })
  df <- data.frame(cont, ord)
  names(df) <- c(pop$cont, pop$ord)
  df
}

as_ordered <- function(df, pop) {
  for (nm in pop$ord) df[[nm]] <- ordered(df[[nm]], levels = seq_len(length(pop$thresholds) + 1L))
  df
}

impose_missing <- function(df, pop, mechanism, rate, seed) {
  anchors <- pop$cont[1:2]
  if (rate <= 0) return(df)
  if (mechanism == "mcar") {
    res <- sb2005_mcar(df, rate = rate, intact = anchors, seed = seed)
  } else {
    res <- sb2005_mar(df, rate = rate, predictors = anchors, seed = seed)
  }
  res$data
}

# ---- per-parameter bookkeeping ----------------------------------------------

is_cont_var <- function(v) grepl("^x", v)

classify_param <- function(op, lhs, rhs) {
  if (op == "=~") return(if (is_cont_var(rhs)) "cont_loading" else "ord_loading")
  if (op == "~~") {
    if (lhs == rhs && startsWith(lhs, "f")) {
      return(if (lhs == "f1") "cont_factor_var" else "ord_factor_var")
    }
    if (startsWith(lhs, "f") && startsWith(rhs, "f")) return("factor_cov")
    if (is_cont_var(lhs) && is_cont_var(rhs)) return("cont_resid")
    return("ord_resid")
  }
  if (op == "|") return("ord_threshold")
  if (op == "~1") return(if (is_cont_var(lhs)) "cont_mean" else "ord_mean")
  "other"
}

family_block <- c(
  cont_loading = "continuous", cont_resid = "continuous",
  cont_mean = "continuous", cont_factor_var = "continuous",
  ord_loading = "ordinal", ord_threshold = "ordinal",
  ord_mean = "ordinal", ord_factor_var = "ordinal",
  factor_cov = "cross", ord_resid = "ordinal", other = "other"
)

free_param_table <- function(pt) {
  idx <- which(pt$free > 0L)
  data.frame(
    free = as.integer(pt$free[idx]),
    key = paste(pt$lhs[idx], pt$op[idx], pt$rhs[idx],
                "g", pt$group[idx] %||% 1L),
    family = mapply(classify_param, pt$op[idx], pt$lhs[idx], pt$rhs[idx]),
    stringsAsFactors = FALSE
  )
}

# Reference ("pseudo-true") parameter vector from one large complete sample,
# cached so repeated runs reuse it.
reference_theta <- function(pop, spec, ref_n, seed_base) {
  key <- calibration_cache_key(
    population = "mixed_2f_4cont_4ord",
    generator = "draw_data",
    options = list(ref_n = ref_n, factor_cor = pop$factor_cor,
                   lambda_cont = pop$lambda_cont, lambda_ord = pop$lambda_ord,
                   thresholds = pop$thresholds)
  )
  if (calibration_cache_exists(key)) return(calibration_cache_read(key))
  set.seed(seed_base + 99L)
  df <- as_ordered(draw_data(pop, ref_n), pop)
  stats <- core$data_mixed_ordinal_stats_observed_from_df(df, spec, ordered = pop$ord)
  fit <- core$fit_dwls_mixed_ordinal(spec, stats, control = ctrl)
  if (!isTRUE(fit$converged)) stop("reference fit did not converge")
  ft <- free_param_table(fit$partable)
  ref <- stats::setNames(fit$theta[ft$free], ft$key)
  calibration_cache_write(key, ref, metadata = list(ref_n = ref_n))
  ref
}

# ---- one estimator on one data set ------------------------------------------

stats_builder <- list(
  pp = function(df) core$data_mixed_ordinal_stats_observed_from_df(df, spec, ordered = pop$ord),
  pf = function(df) core$data_mixed_ordinal_stats_hybrid_fiml_from_df(df, spec, ordered = pop$ord)
)

run_estimator <- function(df, estimator) {
  built <- elapsed_ms(stats_builder[[estimator]](df))
  fit <- core$fit_dwls_mixed_ordinal(spec, built$value, control = ctrl)
  converged <- isTRUE(fit$converged) && all(is.finite(fit$theta))
  se <- rep(NA_real_, length(fit$theta))
  if (converged) {
    ij <- tryCatch(core$robust_mixed_ordinal_ij(fit, built$value, weight = "DWLS"),
                   error = function(e) NULL)
    if (!is.null(ij) && length(ij$se) == length(fit$theta)) se <- as.numeric(ij$se)
  }
  list(theta = fit$theta, se = se, converged = converged,
       elapsed_ms = built$elapsed_ms, partable = fit$partable)
}

# ---- summaries --------------------------------------------------------------

summarize_params <- function(est_long, conv) {
  if (!nrow(est_long)) return(data.frame())
  cells <- unique(conv[, c("mechanism", "n", "missing_rate")])
  out <- list()
  for (ci in seq_len(nrow(cells))) {
    cell <- cells[ci, ]
    csub <- conv[conv$mechanism == cell$mechanism & conv$n == cell$n &
                 conv$missing_rate == cell$missing_rate, ]
    wide <- reshape(csub[, c("rep", "estimator", "converged")],
                    timevar = "estimator", idvar = "rep", direction = "wide")
    paired_reps <- wide$rep[rowSums(is.na(wide)) == 0 &
                            wide$converged.pp & wide$converged.pf]
    esub <- est_long[est_long$mechanism == cell$mechanism & est_long$n == cell$n &
                     est_long$missing_rate == cell$missing_rate &
                     est_long$rep %in% paired_reps, ]
    if (!nrow(esub)) next
    key <- interaction(esub$estimator, esub$key, drop = TRUE)
    rows <- lapply(split(esub, key), function(d) {
      true <- d$true[1L]
      err <- d$est - true
      data.frame(
        mechanism = cell$mechanism, n = cell$n, missing_rate = cell$missing_rate,
        estimator = d$estimator[1L], key = d$key[1L], family = d$family[1L],
        block = family_block[d$family[1L]],
        true = true, reps = nrow(d), n_paired = length(paired_reps),
        mean_est = mean(d$est), bias = mean(err),
        emp_sd = stats::sd(d$est),
        rmse = sqrt(mean(err^2)),
        mean_se = mean(d$se, na.rm = TRUE),
        se_ratio = mean(d$se, na.rm = TRUE) / stats::sd(d$est),
        stringsAsFactors = FALSE
      )
    })
    out[[ci]] <- do.call(rbind, rows)
  }
  res <- rbind_fill(out)
  rownames(res) <- NULL
  res
}

# Headline: relative RMSE (hybrid / pairwise) by block, mechanism, n, rate.
summarize_efficiency <- function(per_param) {
  if (!nrow(per_param)) return(data.frame())
  wide <- reshape(
    per_param[, c("mechanism", "n", "missing_rate", "block", "family", "key",
                  "rmse", "emp_sd", "bias", "estimator")],
    timevar = "estimator",
    idvar = c("mechanism", "n", "missing_rate", "block", "family", "key"),
    direction = "wide")
  wide$rel_rmse <- wide$rmse.pf / wide$rmse.pp
  wide$rel_var <- (wide$emp_sd.pf / wide$emp_sd.pp)^2
  key <- interaction(wide$mechanism, wide$n, wide$missing_rate, wide$block,
                     drop = TRUE)
  rows <- lapply(split(wide, key), function(d) data.frame(
    mechanism = d$mechanism[1L], n = d$n[1L], missing_rate = d$missing_rate[1L],
    block = d$block[1L], n_params = nrow(d),
    median_rel_rmse = stats::median(d$rel_rmse, na.rm = TRUE),
    median_rel_var = stats::median(d$rel_var, na.rm = TRUE),
    max_abs_bias_pp = max(abs(d$bias.pp), na.rm = TRUE),
    max_abs_bias_pf = max(abs(d$bias.pf), na.rm = TRUE),
    stringsAsFactors = FALSE))
  out <- do.call(rbind, rows)
  out <- out[order(out$mechanism, out$block, out$n, out$missing_rate), ]
  rownames(out) <- NULL
  out
}

# ---- driver -----------------------------------------------------------------

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- ensure_results_dir()
ctrl <- list(max_iter = 4000L, ftol = 1e-12, gtol = 1e-8)

pop <- population()
spec <- model_spec_for(pop)
ref <- reference_theta(pop, spec, cfg$ref_n, cfg$seed_base)

cat(sprintf("mixed FIML/pairwise efficiency: %d reps, n={%s}, miss={%s}, mech={%s}\n",
            cfg$reps, paste(cfg$n, collapse = ","),
            paste(cfg$missing_rate, collapse = ","),
            paste(cfg$mechanisms, collapse = ",")))

est_rows <- list(); conv_rows <- list(); ie <- 0L; ic <- 0L

for (mech in cfg$mechanisms) {
  for (n in cfg$n) {
    for (mr in cfg$missing_rate) {
      for (rep in seq_len(cfg$reps)) {
        seed <- cfg$seed_base +
          match(mech, c("mcar", "mar")) * 100000000L +
          n * 1000L + round(mr * 1000) * 17L + rep
        set.seed(seed)
        df0 <- draw_data(pop, n)
        df <- impose_missing(df0, pop, mech, mr, seed = seed + 7L)
        df <- as_ordered(df, pop)
        for (estimator in c("pp", "pf")) {
          ans <- tryCatch(run_estimator(df, estimator), error = function(e) e)
          ic <- ic + 1L
          if (inherits(ans, "error")) {
            conv_rows[[ic]] <- data.frame(
              mechanism = mech, n = n, missing_rate = mr, rep = rep,
              estimator = estimator, converged = FALSE, elapsed_ms = NA_real_,
              error = conditionMessage(ans), stringsAsFactors = FALSE)
            next
          }
          conv_rows[[ic]] <- data.frame(
            mechanism = mech, n = n, missing_rate = mr, rep = rep,
            estimator = estimator, converged = ans$converged,
            elapsed_ms = ans$elapsed_ms, error = "", stringsAsFactors = FALSE)
          if (!ans$converged) next
          ft <- free_param_table(ans$partable)
          ie <- ie + 1L
          est_rows[[ie]] <- data.frame(
            mechanism = mech, n = n, missing_rate = mr, rep = rep,
            estimator = estimator, key = ft$key, family = ft$family,
            est = ans$theta[ft$free], true = unname(ref[ft$key]),
            se = ans$se[ft$free], stringsAsFactors = FALSE)
        }
      }
      cat(sprintf("mech=%s n=%d miss=%.2f done (%d reps)\n", mech, n, mr, cfg$reps))
    }
  }
}

est_long <- rbind_fill(est_rows)
conv <- rbind_fill(conv_rows)

per_param <- summarize_params(est_long, conv)
efficiency <- summarize_efficiency(per_param)

conv_summary <- do.call(rbind, lapply(
  split(conv, interaction(conv$mechanism, conv$n, conv$missing_rate,
                          conv$estimator, drop = TRUE)),
  function(d) data.frame(
    mechanism = d$mechanism[1L], n = d$n[1L], missing_rate = d$missing_rate[1L],
    estimator = d$estimator[1L], reps = nrow(d),
    converged = sum(d$converged), conv_rate = mean(d$converged),
    median_ms = stats::median(d$elapsed_ms, na.rm = TRUE),
    stringsAsFactors = FALSE)))
rownames(conv_summary) <- NULL

write_csv(per_param, file.path(res_dir, "per_param.csv"))
write_csv(efficiency, file.path(res_dir, "efficiency.csv"))
write_csv(conv_summary, file.path(res_dir, "convergence.csv"))

metadata <- metadata_frame(
  list(
    reps = cfg$reps, n = cfg$n, missing_rate = cfg$missing_rate,
    mechanisms = cfg$mechanisms, ref_n = cfg$ref_n, seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    estimators = "pp=pairwise x pairwise; pf=pairwise x FIML continuous",
    question = "mixed first-stage efficiency: pairwise continuous vs FIML continuous"
  ),
  packages = c("magmaan")
)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

cat("wrote:\n")
for (nm in c("per_param.csv", "efficiency.csv", "convergence.csv", "metadata.csv")) {
  cat("  ", file.path(res_dir, nm), "\n", sep = "")
}
