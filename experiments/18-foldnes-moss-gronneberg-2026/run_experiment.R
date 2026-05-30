#!/usr/bin/env Rscript
# Foldnes, Moss & Gronneberg (2026) — Study 1 goodness-of-fit, magmaan vs lavaan.
#
# "Penalized eigenvalue block averaging: Extension to nested model comparison
# and Monte Carlo evaluations", Behavior Research Methods 58:107.
#
# This experiment asks a parity question, not a methods question: can magmaan
# reproduce the goodness-of-fit statistics that the paper's Study 1 is built on
# — the normal-theory T_ML, Browne's reweighted-least-squares T_RLS, and the
# Satorra-Bentler (SB) and scaled-shifted (SS) robust corrections — on the
# paper's correctly-specified five-factor design, and does it match lavaan where
# lavaan exposes the same statistic?
#
# magmaan additionally computes the UGamma eigenvalue spectrum and the four
# "versions" the paper crosses (biased vs unbiased Gamma) x (T_ML vs T_RLS) for
# SB and SS. lavaan parity anchors on the four it exposes: T_ML (standard),
# T_RLS (browne.residual.nt.model), SB (satorra.bentler), SS (scaled.shifted).
# The unbiased-Gamma and RLS-based robust variants are magmaan-only here; in the
# paper they are produced by semTests, not lavaan.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--cells FILTER] [--seed-base S]
#                            [--lavaan-parity] [--smoke]
#
# `--cells FILTER` is a comma-separated list of key=value pairs restricting the
# crossed grid, e.g. `p=15,N=400,dist=vm1`. `--smoke` runs one tiny cell once.

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
source(experiment_path("R", "population.R"))

parse_args <- function(args) {
  out <- list(reps = 5L, cells_filter = NULL, seed_base = 20260530L,
              lavaan_parity = FALSE, smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--cells FILTER] ",
          "[--seed-base S] [--lavaan-parity] [--smoke]\n", sep = "")
      cat("  --cells: comma-separated key=value over {p, N, dist}, e.g.\n",
          "           p=15,N=400,dist=vm1\n", sep = "")
      cat("  dist in {norm, vm1, vm2}; vm1=(skew 2,kurt 7), vm2=(skew 3,kurt 21)\n")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cells") {
      i <- i + 1L; out$cells_filter <- args[[i]]
    } else if (startsWith(a, "--cells=")) {
      out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--lavaan-parity") {
      out$lavaan_parity <- TRUE
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

results_dir <- ensure_results_dir()
set_single_threaded_math()
require_pkg("magmaan")
core <- magmaan::magmaan_core

# ── Distribution targets (paper: VM1 = moderate, VM2 = severe) ───────────────
dist_moments <- list(
  norm = c(0, 0),
  vm1  = c(2, 7),
  vm2  = c(3, 21)
)

# ── Per-cell context (constant across reps) ─────────────────────────────────
.spec_cache <- new.env(parent = emptyenv())
get_spec <- function(p, per_factor) {
  key <- as.character(p)
  if (!is.null(.spec_cache[[key]])) return(.spec_cache[[key]])
  spec <- magmaan::model_spec(build_5factor_syntax(p, per_factor))
  .spec_cache[[key]] <- spec
  spec
}

sample_data_bundle <- function(X, varnames) {
  colnames(X) <- varnames
  storage.mode(X) <- "double"
  ss <- core$data_sample_stats_from_raw(list(X))
  dimnames(ss$S[[1L]]) <- list(varnames, varnames)
  if (!is.null(ss$mean) && length(ss$mean)) names(ss$mean[[1L]]) <- varnames
  ss$X <- list(X)
  ss$ov_names <- list(varnames)
  ss$group_var <- ""
  ss$group_labels <- character()
  ss$scaling <- "n"
  class(ss) <- c("magmaan_data", "list")
  ss
}

# ── Per-replicate fit and statistics ────────────────────────────────────────
fit_one_rep <- function(ctx, N, dist, seed, keep_parity = FALSE) {
  pop <- ctx$pop
  set.seed(seed)
  X <- sample_population(pop, N, dist, fl = ctx$fl)
  colnames(X) <- ctx$varnames
  storage.mode(X) <- "double"
  dat <- sample_data_bundle(X, ctx$varnames)

  fit <- tryCatch(core$fit_ml(ctx$spec, dat), error = function(e) e)
  if (inherits(fit, "error")) return(list(ok = FALSE, error = conditionMessage(fit)))
  if (!isTRUE(fit$converged)) {
    return(list(ok = FALSE,
                error = paste0("not converged: ", as.character(fit$optimizer_status %||% ""))))
  }

  samp <- core$fit_sample_stats(fit)
  T_ML <- core$inference_chi2_stat(samp, fit$fmin)
  df   <- core$inference_df_stat(fit$partable, samp)
  implied <- core$model_implied(fit)
  T_RLS <- tryCatch(core$inference_rls_chi2(fit, implied)$statistic,
                    error = function(e) NA_real_)

  # UGamma eigenvalue spectrum: biased (sample) and unbiased (Browne) Gamma.
  uf <- core$robust_build_u_factor_parts(fit$partable, samp, fit$theta)
  Zc <- core$robust_casewise_contributions(fit$partable, list(X))
  M_s  <- core$robust_reduced_gamma_sample(uf, Zc, fit$nobs)
  M_nt <- core$robust_reduced_gamma_nt(uf)
  n_scalar <- as.numeric(unlist(fit$nobs, use.names = FALSE))[[1L]]
  M_ub <- tryCatch(core$robust_reduced_gamma_unbiased(uf, n_scalar, M_s, M_nt),
                   error = function(e) e)
  ev_b  <- core$robust_ugamma_eigenvalues(M_s)
  ev_ub <- if (inherits(M_ub, "error")) NULL else core$robust_ugamma_eigenvalues(M_ub)

  # SB / SS on each base statistic and each Gamma version.
  robust_row <- function(name, base_T, ev) {
    if (is.null(ev) || !is.finite(base_T)) {
      return(data.frame(stat_name = name, statistic = NA_real_, df = NA_real_,
                        scaling = NA_real_, stringsAsFactors = FALSE))
    }
    sb <- core$robust_satorra_bentler(base_T, df, ev)
    ss <- core$robust_scaled_shifted(base_T, df, ev)
    rbind(
      data.frame(stat_name = paste0("SB_", name), statistic = sb$chi2_scaled,
                 df = sb$df, scaling = sb$scale_c, stringsAsFactors = FALSE),
      data.frame(stat_name = paste0("SS_", name), statistic = ss$chi2_adj,
                 df = ss$df, scaling = NA_real_, stringsAsFactors = FALSE)
    )
  }

  chi2_rows <- rbind(
    data.frame(stat_name = "ML",  statistic = T_ML,  df = df, scaling = NA_real_,
               stringsAsFactors = FALSE),
    data.frame(stat_name = "RLS", statistic = T_RLS, df = df, scaling = NA_real_,
               stringsAsFactors = FALSE),
    robust_row("ML_biased",   T_ML,  ev_b),
    robust_row("RLS_biased",  T_RLS, ev_b),
    robust_row("ML_unbiased", T_ML,  ev_ub),
    robust_row("RLS_unbiased", T_RLS, ev_ub)
  )
  chi2_rows$p_value <- stats::pchisq(chi2_rows$statistic, df = chi2_rows$df,
                                     lower.tail = FALSE)

  parity <- if (isTRUE(keep_parity)) {
    list(data = as.data.frame(X), chi2_rows = chi2_rows,
         eig_mean_biased = mean(ev_b))
  } else NULL

  list(ok = TRUE, chi2_rows = chi2_rows, fmin = as.numeric(fit$fmin),
       eig_min = min(ev_b), eig_max = max(ev_b), eig_mean = mean(ev_b),
       parity = parity)
}

# ── Cell grid ───────────────────────────────────────────────────────────────
parse_cells_filter <- function(s) {
  if (is.null(s) || !nzchar(s)) return(list())
  pairs <- strsplit(s, ",", fixed = TRUE)[[1L]]
  out <- list()
  for (p in pairs) {
    kv <- strsplit(p, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("bad --cells entry: ", p, call. = FALSE)
    out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}

cell_grid <- expand.grid(
  p    = c(15L, 30L),
  N    = c(400L, 800L),
  dist = c("norm", "vm1", "vm2"),
  stringsAsFactors = FALSE
)
if (isTRUE(args$smoke)) {
  cell_grid <- data.frame(p = 15L, N = 400L, dist = "vm1",
                          stringsAsFactors = FALSE)
  args$reps <- 1L
}

filter <- parse_cells_filter(args$cells_filter)
for (key in names(filter)) {
  if (!(key %in% names(cell_grid))) stop("--cells: unknown key '", key, "'",
                                         call. = FALSE)
  target <- if (is.numeric(cell_grid[[key]])) as.numeric(filter[[key]]) else filter[[key]]
  cell_grid <- cell_grid[cell_grid[[key]] == target, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)

# Fleishman coefficients per non-normal distribution, and an achieved-moment
# table for the report (verified on one large draw).
fl_cache <- list()
moment_rows <- list()
for (d in setdiff(unique(cell_grid$dist), "norm")) {
  tgt <- dist_moments[[d]]
  fl <- fleishman_coef(tgt[[1L]], tgt[[2L]])
  fl_cache[[d]] <- fl
  set.seed(0xF1E15)
  Z <- stats::rnorm(2e5L)
  Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3
  m3 <- mean((Y - mean(Y))^3) / stats::sd(Y)^3
  m4 <- mean((Y - mean(Y))^4) / stats::var(Y)^2 - 3
  moment_rows[[d]] <- data.frame(
    dist = d, skew_target = tgt[[1L]], kurt_target = tgt[[2L]],
    skew_achieved = m3, kurt_achieved = m4,
    fleishman_resid = fl$resid, stringsAsFactors = FALSE)
}
if (length(moment_rows)) {
  utils::write.csv(do.call(rbind, moment_rows),
                   file.path(results_dir, "moments.csv"), row.names = FALSE)
}
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"),
                 row.names = FALSE)

cat(sprintf("foldnes-moss-gronneberg-2026 Study 1: magmaan %s, %s\n",
            as.character(utils::packageVersion("magmaan")), R.version.string))
cat(sprintf("  reps=%d, cells=%d%s\n", args$reps, nrow(cell_grid),
            if (isTRUE(args$lavaan_parity)) ", lavaan parity ON" else ""))

# ── Run ─────────────────────────────────────────────────────────────────────
all_chi2 <- list(); all_meta <- list(); all_parity <- vector("list", nrow(cell_grid))
t0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  pop <- build_population_5factor(cell$p)
  ctx <- list(pop = pop, spec = get_spec(cell$p, pop$per_factor),
              varnames = paste0("x", seq_len(cell$p)),
              fl = if (cell$dist == "norm") NULL else fl_cache[[cell$dist]])
  cat(sprintf("  cell %2d/%2d: p=%3d N=%4d dist=%-4s ", ci, nrow(cell_grid),
              cell$p, cell$N, cell$dist))
  ok <- 0L; fail <- 0L; tc0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    seed <- args$seed_base + ci * 10000L + rep_idx
    keep <- isTRUE(args$lavaan_parity) && rep_idx == 1L
    out <- fit_one_rep(ctx, cell$N, cell$dist, seed, keep_parity = keep)
    if (!isTRUE(out$ok)) { fail <- fail + 1L; next }
    ok <- ok + 1L
    all_chi2[[length(all_chi2) + 1L]] <- cbind(
      out$chi2_rows,
      data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                 p = cell$p, N = cell$N, dist = cell$dist,
                 eig_mean = out$eig_mean, stringsAsFactors = FALSE))
    if (keep) all_parity[[ci]] <- out$parity
  }
  tc1 <- proc.time()[["elapsed"]]
  cat(sprintf("ok=%d fail=%d (%.1fs)\n", ok, fail, tc1 - tc0))
  all_meta[[ci]] <- data.frame(cell_idx = ci, p = cell$p, N = cell$N,
                               dist = cell$dist, rep_ok = ok, rep_fail = fail,
                               seconds = tc1 - tc0, stringsAsFactors = FALSE)
}
t1 <- proc.time()[["elapsed"]]

write_rows(all_chi2, file.path(results_dir, "fits_chi2.csv"))
write_rows(all_meta, file.path(results_dir, "cell_meta.csv"))

# ── Rejection-rate summary (paper's RR criterion, alpha = .05) ───────────────
chi2_long <- if (length(all_chi2)) do.call(rbind, all_chi2) else NULL
if (!is.null(chi2_long) && nrow(chi2_long)) {
  by_grp <- split(chi2_long, list(chi2_long$cell_idx, chi2_long$stat_name),
                  drop = TRUE)
  summary_chi2 <- do.call(rbind, lapply(by_grp, function(g) {
    data.frame(cell_idx = g$cell_idx[[1L]], p = g$p[[1L]], N = g$N[[1L]],
               dist = g$dist[[1L]], stat_name = g$stat_name[[1L]],
               n_reps = nrow(g), mean_chi2 = mean(g$statistic, na.rm = TRUE),
               mean_df = mean(g$df, na.rm = TRUE),
               reject_05 = mean(g$p_value < 0.05, na.rm = TRUE),
               mean_scaling = mean(g$eig_mean, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }))
  utils::write.csv(summary_chi2, file.path(results_dir, "summary_chi2.csv"),
                   row.names = FALSE)
}

meta <- metadata_frame(
  values = list(reps = args$reps, seed_base = args$seed_base,
                cells_filter = args$cells_filter,
                lavaan_parity = isTRUE(args$lavaan_parity),
                n_cells = nrow(cell_grid),
                total_seconds = sprintf("%.2f", t1 - t0)),
  packages = c("magmaan", "lavaan"))
write_csv(meta, file.path(results_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1fs — results in %s\n", t1 - t0, results_dir))

# ── lavaan parity (rep 1 per cell) ──────────────────────────────────────────
if (isTRUE(args$lavaan_parity)) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    cat("--lavaan-parity requested but lavaan not installed; skipping.\n")
  } else {
    cat("running lavaan parity (one fit per cell)...\n")
    rows <- list()
    add <- function(ci, cell, metric, mag, lav) {
      rows[[length(rows) + 1L]] <<- data.frame(
        cell_idx = ci, p = cell$p, N = cell$N, dist = cell$dist,
        metric = metric, magmaan = mag, lavaan = lav,
        abs_diff = abs(mag - lav), stringsAsFactors = FALSE)
    }
    for (ci in seq_len(nrow(cell_grid))) {
      cell <- as.list(cell_grid[ci, , drop = FALSE])
      cached <- all_parity[[ci]]
      if (is.null(cached)) next
      syntax <- build_5factor_syntax(cell$p)
      lv <- tryCatch(lavaan::cfa(
        syntax, data = cached$data, estimator = "ML", std.lv = FALSE,
        test = c("standard", "satorra.bentler", "scaled.shifted",
                 "browne.residual.nt.model")),
        error = function(e) e)
      if (inherits(lv, "error")) {
        add(ci, cell, "lavaan_error", NA_real_, NA_real_); next
      }
      ts <- lavaan::lavInspect(lv, "test")
      mg <- cached$chi2_rows
      mg_stat <- function(nm) mg$statistic[match(nm, mg$stat_name)]
      lv_stat <- function(nm) ts[[nm]]$stat %||% NA_real_
      add(ci, cell, "T_ML",  mg_stat("ML"),  lv_stat("standard"))
      add(ci, cell, "T_RLS", mg_stat("RLS"), lv_stat("browne.residual.nt.model"))
      add(ci, cell, "SB",    mg_stat("SB_ML_biased"), lv_stat("satorra.bentler"))
      add(ci, cell, "SS",    mg_stat("SS_ML_biased"), lv_stat("scaled.shifted"))
      # SB scaling factor.
      add(ci, cell, "SB_scaling", cached$eig_mean_biased,
          ts[["satorra.bentler"]]$scaling.factor %||% NA_real_)
    }
    parity_df <- do.call(rbind, rows)
    utils::write.csv(parity_df, file.path(results_dir, "lavaan_parity.csv"),
                     row.names = FALSE)
    cat(sprintf("lavaan parity: %d rows; max |Δ| = %.3g\n",
                nrow(parity_df), max(parity_df$abs_diff, na.rm = TRUE)))
  }
}
