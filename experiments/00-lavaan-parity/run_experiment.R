#!/usr/bin/env Rscript

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

parse_args <- function(args) {
  out <- list(
    audit_run = "audit-parity-v7",
    oracle_run = "latest",
    refit_csv = "/tmp/refit_disagreements_v8.csv",
    from_oracle = FALSE,
    collect_lavaan = FALSE,
    install_latest_lavaan = FALSE,
    run_refit = FALSE,
    books = NULL,
    weights = NULL,
    limit = NA_integer_
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--audit-run RUN_ID] ",
        "[--refit-csv PATH] [--from-oracle] [--collect-lavaan] ",
        "[--oracle-run RUN_ID] [--install-latest-lavaan] ",
        "[--books IDS] [--weights IDS] [--limit N] [--run-refit]\n",
        "\n",
        "Self-contained: audits magmaan against a cached lavaan oracle under ",
        "results/lavaan-oracle/<run-id>. Run --collect-lavaan once (needs the ",
        "lavaan package and the textbook-corpus submodule) to build that ",
        "cache; subsequent runs replay it. With no cache present the run ",
        "stops and asks you to --collect-lavaan first.\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    }
    if (grepl("=", arg, fixed = TRUE)) {
      parts <- strsplit(arg, "=", fixed = TRUE)[[1L]]
      arg <- parts[[1L]]
      value <- paste(parts[-1L], collapse = "=")
    }
    need_value <- function() {
      if (!is.null(value)) return(value)
      i <<- i + 1L
      if (i > length(args)) stop(arg, " requires a value", call. = FALSE)
      args[[i]]
    }
    if (arg == "--audit-run") out$audit_run <- need_value()
    else if (arg == "--oracle-run") out$oracle_run <- need_value()
    else if (arg == "--refit-csv") out$refit_csv <- need_value()
    else if (arg == "--from-oracle") out$from_oracle <- TRUE
    else if (arg == "--collect-lavaan") out$collect_lavaan <- TRUE
    else if (arg == "--install-latest-lavaan") out$install_latest_lavaan <- TRUE
    else if (arg == "--run-refit") out$run_refit <- TRUE
    else if (arg == "--books") out$books <- parse_csv_arg(need_value())
    else if (arg == "--weights") out$weights <- parse_csv_arg(need_value())
    else if (arg == "--limit") out$limit <- as.integer(need_value())
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  out
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths) | dir.exists(paths)]
  if (length(hit)) hit[[1L]] else paths[[1L]]
}

# Experiment-local harness (corpus loading, magmaan problem construction, lavaan
# oracle). Experiments are sinks: this used to reach into the snlls-continuous
# paper package; the needed pieces now live under this experiment's R/.
source(experiment_path("R", "corpus.R"))
source(experiment_path("R", "problem.R"))
source(experiment_path("R", "lavaan_oracle.R"))

copy_optional <- function(from, to) {
  if (!file.exists(from)) return(FALSE)
  dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
  file.copy(from, to, overwrite = TRUE)
  TRUE
}

run_script <- function(path, args = character()) {
  if (!file.exists(path)) stop("missing script: ", path, call. = FALSE)
  status <- system2("Rscript", c(path, args))
  if (!identical(status, 0L)) {
    stop("script failed: ", path, call. = FALSE)
  }
}


read_csv <- function(path) {
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                  na.strings = c("", "NA"))
}

read_csv_keep_empty <- function(path) {
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                  na.strings = "NA")
}

boolish <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes")
}

finite_ratio <- function(num, den) {
  out <- rep(NA_real_, length(num))
  ok <- is.finite(num) & is.finite(den) & abs(den) > 0
  out[ok] <- num[ok] / den[ok]
  out
}

empty_conditioned_audit <- function() {
  list(
    adf_gamma_dim = NA_integer_,
    adf_gamma_rank = NA_integer_,
    adf_gamma_rcond = NA_real_,
    adf_gamma_tol = NA_real_,
    adf_gamma_min_eigen = NA_real_,
    adf_gamma_max_eigen = NA_real_,
    adf_conditioning_warning = NA,
    adf_conditioned_fmin = NA_real_,
    adf_conditioned_grad_inf = NA_real_,
    adf_conditioned_stationary = NA
  )
}

conditioned_adf_weight <- function(data, include_means) {
  X <- as.matrix(data)
  Gamma <- if (isTRUE(include_means)) {
    magmaan::magmaan_core$robust_empirical_gamma_with_means(X)
  } else {
    magmaan::magmaan_core$robust_empirical_gamma(X)
  }
  Gamma <- 0.5 * (Gamma + t(Gamma))
  eig <- eigen(Gamma, symmetric = TRUE)
  values <- eig$values
  scale <- max(abs(values), 1)
  tol <- nrow(Gamma) * .Machine$double.eps * scale
  keep <- values > tol
  if (!any(keep)) {
    stop("conditioned ADF weight has numerical rank zero", call. = FALSE)
  }

  Q <- eig$vectors[, keep, drop = FALSE]
  A <- sweep(Q, 2L, 1 / sqrt(values[keep]), `*`)
  W <- tcrossprod(A)
  W <- 0.5 * (W + t(W))
  list(
    W = W,
    dim = nrow(Gamma),
    rank = sum(keep),
    rcond = rcond(Gamma),
    tol = tol,
    min_eigen = min(values),
    max_eigen = max(values),
    warning = any(!keep) || any(values < -tol)
  )
}

conditioned_adf_audit <- function(case, prob, theta_full, estimator) {
  out <- empty_conditioned_audit()
  if (!toupper(case$estimator) %in% c("ADF", "WLS")) return(out)
  if (!identical(estimator, "WLS") || is.null(case$data)) return(out)

  diag <- tryCatch(
    conditioned_adf_weight(
      case$data,
      include_means = isTRUE(case$model_spec_args$meanstructure)
    ),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_conditioned_error"))
  if (inherits(diag, "audit_parity_conditioned_error")) {
    out$adf_conditioning_warning <- TRUE
    return(out)
  }

  ev <- tryCatch(
    magmaan::magmaan_core$evaluate_at(
      prob$spec, prob$dat, as.numeric(theta_full),
      estimator = "WLS", W = diag$W),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_conditioned_error"))

  out$adf_gamma_dim <- diag$dim
  out$adf_gamma_rank <- diag$rank
  out$adf_gamma_rcond <- diag$rcond
  out$adf_gamma_tol <- diag$tol
  out$adf_gamma_min_eigen <- diag$min_eigen
  out$adf_gamma_max_eigen <- diag$max_eigen
  out$adf_conditioning_warning <- isTRUE(diag$warning)
  if (!inherits(ev, "audit_parity_conditioned_error")) {
    out$adf_conditioned_fmin <- ev$fmin
    out$adf_conditioned_grad_inf <-
      as.numeric(ev$audit$grad_inf_norm %||% NA_real_)
    out$adf_conditioned_stationary <- isTRUE(ev$audit$stationary)
  }
  out
}

oracle_root <- function() file.path(results_dir(create = TRUE), "lavaan-oracle")

resolve_oracle_run <- function(run_id) {
  root <- oracle_root()
  if (identical(run_id, "latest")) {
    latest <- file.path(root, "LATEST")
    if (!file.exists(latest)) {
      stop("No lavaan oracle cache exists yet. Run `Rscript ",
           "experiments/00-lavaan-parity/run_experiment.R --collect-lavaan` ",
           "first, or pass --audit-run to use a paper-side combined sweep.",
           call. = FALSE)
    }
    run_id <- trimws(readLines(latest, warn = FALSE)[[1L]])
  }
  path <- file.path(root, run_id)
  if (!dir.exists(path)) {
    stop("missing lavaan oracle cache: ", path, call. = FALSE)
  }
  list(run_id = run_id, path = path)
}

read_meta_value <- function(meta, key, default = "") {
  if (!nrow(meta) || !"key" %in% names(meta) || !"value" %in% names(meta)) {
    return(default)
  }
  hit <- meta$value[meta$key == key]
  if (length(hit)) hit[[1L]] else default
}

canon_keys <- function(df) {
  swap <- df$op == "~~" & df$lhs > df$rhs
  if (any(swap)) {
    lhs <- df$lhs[swap]
    df$lhs[swap] <- df$rhs[swap]
    df$rhs[swap] <- lhs
  }
  df
}

empty_audit_row <- function(case, lav = NULL, error = NA_character_) {
  data.frame(
    case_id = case$id, book = case$corpus, family = case$family,
    weight = case$estimator,
    n_free_mag = NA_integer_,
    n_free_lav = if (!is.null(lav)) lav$n_free_lav else NA_integer_,
    n_matched = NA_integer_,
    lavaan_converged = if (!is.null(lav)) lav$lavaan_converged else NA,
    lavaan_fx = if (!is.null(lav)) lav$lavaan_fx else NA_real_,
    lavaan_fx_bare = NA_real_,
    magmaan_fmin = NA_real_, magmaan_iter = NA_integer_,
    audit_advisory = NA_character_, audit_grad_inf = NA_real_,
    audit_stationary = NA,
    adf_gamma_dim = NA_integer_, adf_gamma_rank = NA_integer_,
    adf_gamma_rcond = NA_real_, adf_gamma_tol = NA_real_,
    adf_gamma_min_eigen = NA_real_, adf_gamma_max_eigen = NA_real_,
    adf_conditioning_warning = NA,
    adf_conditioned_fmin = NA_real_,
    adf_conditioned_grad_inf = NA_real_,
    adf_conditioned_stationary = NA,
    error = error, stringsAsFactors = FALSE
  )
}

audit_bessel <- function(case, n_total) {
  if (is.null(n_total) || !is.finite(n_total) || n_total <= 1) return(1)
  if (toupper(case$estimator) %in% c("GLS", "WLS", "ADF", "ULS", "DWLS")) {
    (n_total - 1) / n_total
  } else {
    1
  }
}

audit_cached_lavaan <- function(case, lav, estimates) {
  if (!is.na(lav$error)) {
    return(empty_audit_row(case, lav, paste0("lavaan-cache: ", lav$error)))
  }

  prob <- tryCatch(make_problem(case),
                   error = function(e) structure(
                     list(error = conditionMessage(e)),
                     class = "audit_parity_magmaan_error"))
  if (inherits(prob, "audit_parity_magmaan_error")) {
    return(empty_audit_row(case, lav, paste0("magmaan-prep: ", prob$error)))
  }

  pt <- as.data.frame(prob$spec$partable)
  pt$row_idx <- seq_len(nrow(pt))
  free_pt <- pt[pt$free > 0L, , drop = FALSE]
  if (!"group" %in% names(free_pt)) free_pt$group <- 1L
  free_pt <- canon_keys(free_pt)
  lav_est <- canon_keys(estimates)
  matched <- merge(free_pt[, c("row_idx", "lhs", "op", "rhs", "group")],
                   lav_est[, c("lhs", "op", "rhs", "group", "est"),
                           drop = FALSE],
                   by = c("lhs", "op", "rhs", "group"), sort = FALSE)
  prob$spec$partable$ustart[matched$row_idx] <- matched$est

  ss <- magmaan:::sample_stats_arg(prob$dat)
  pt_for_starts <- magmaan:::partable_arg(prob$spec)
  theta_full <- tryCatch(
    magmaan::magmaan_core$fit_start_values(pt_for_starts, ss),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_magmaan_error"))
  if (inherits(theta_full, "audit_parity_magmaan_error")) {
    out <- empty_audit_row(case, lav,
                           paste0("magmaan-starts: ", theta_full$error))
    out$n_free_mag <- nrow(free_pt)
    out$n_matched <- nrow(matched)
    return(out)
  }

  estimator <- toupper(case$estimator)
  if (identical(estimator, "ADF")) estimator <- "WLS"
  ev <- tryCatch(
    magmaan::magmaan_core$evaluate_at(
      prob$spec, prob$dat, as.numeric(theta_full),
      estimator = estimator,
      W = if (estimator == "WLS") prob$W else NULL),
    error = function(e) structure(list(error = conditionMessage(e)),
                                  class = "audit_parity_magmaan_error"))
  if (inherits(ev, "audit_parity_magmaan_error")) {
    out <- empty_audit_row(case, lav, paste0("magmaan-eval: ", ev$error))
    out$n_free_mag <- nrow(free_pt)
    out$n_matched <- nrow(matched)
    return(out)
  }

  n_total <- sum(unlist(prob$dat$nobs))
  bessel <- audit_bessel(case, n_total)
  conditioned <- as.data.frame(
    conditioned_adf_audit(case, prob, theta_full, estimator),
    stringsAsFactors = FALSE
  )
  cbind(data.frame(
    case_id = case$id, book = case$corpus, family = case$family,
    weight = case$estimator,
    n_free_mag = nrow(free_pt), n_free_lav = lav$n_free_lav,
    n_matched = nrow(matched),
    lavaan_converged = isTRUE(lav$lavaan_converged),
    lavaan_fx = lav$lavaan_fx,
    lavaan_fx_bare = lav$lavaan_fx / bessel,
    magmaan_fmin = ev$fmin, magmaan_iter = 0L,
    audit_advisory = as.character(ev$audit$advisory_status %||% NA),
    audit_grad_inf = as.numeric(ev$audit$grad_inf_norm %||% NA_real_),
    audit_stationary = isTRUE(ev$audit$stationary),
    stringsAsFactors = FALSE
  ), conditioned, data.frame(error = NA_character_, stringsAsFactors = FALSE))
}

audit_lavaan_oracle <- function(run_id) {
  oracle <- resolve_oracle_run(run_id)
  fits_rds <- file.path(oracle$path, "lavaan_fits.rds")
  estimates_rds <- file.path(oracle$path, "lavaan_estimates.rds")
  fits <- if (file.exists(fits_rds)) {
    readRDS(fits_rds)
  } else {
    read_csv(file.path(oracle$path, "lavaan_fits.csv"))
  }
  estimates <- if (file.exists(estimates_rds)) {
    readRDS(estimates_rds)
  } else {
    read_csv_keep_empty(file.path(oracle$path, "lavaan_estimates.csv"))
  }
  oracle_meta <- read_csv(file.path(oracle$path, "metadata.csv"))

  corpus_root_dir <- corpus_root()
  need <- unique(fits[, c("book", "weight"), drop = FALSE])
  all_cases <- list()
  for (i in seq_len(nrow(need))) {
    all_cases <- c(all_cases, corpus_cases(corpus_root_dir,
                                           weights = need$weight[[i]],
                                           books = need$book[[i]]))
  }

  rows <- vector("list", nrow(fits))
  started <- Sys.time()
  for (i in seq_len(nrow(fits))) {
    key <- fits$case_id[[i]]
    if (i %% 25L == 1L || i == nrow(fits)) {
      elapsed <- round(as.numeric(difftime(Sys.time(), started,
                                           units = "secs")))
      message(sprintf("[%3d/%3d  %3ds] audit %s", i, nrow(fits), elapsed,
                      key))
    }
    if (!key %in% names(all_cases)) {
      rows[[i]] <- data.frame(
        case_id = key, book = fits$book[[i]], family = fits$family[[i]],
        weight = fits$weight[[i]], n_free_mag = NA_integer_,
        n_free_lav = fits$n_free_lav[[i]], n_matched = NA_integer_,
        lavaan_converged = fits$lavaan_converged[[i]],
        lavaan_fx = fits$lavaan_fx[[i]], lavaan_fx_bare = NA_real_,
        magmaan_fmin = NA_real_, magmaan_iter = NA_integer_,
        audit_advisory = NA_character_, audit_grad_inf = NA_real_,
        audit_stationary = NA,
        adf_gamma_dim = NA_integer_, adf_gamma_rank = NA_integer_,
        adf_gamma_rcond = NA_real_, adf_gamma_tol = NA_real_,
        adf_gamma_min_eigen = NA_real_, adf_gamma_max_eigen = NA_real_,
        adf_conditioning_warning = NA,
        adf_conditioned_fmin = NA_real_,
        adf_conditioned_grad_inf = NA_real_,
        adf_conditioned_stationary = NA,
        error = "case missing from corpus_cases", stringsAsFactors = FALSE)
      next
    }
    rows[[i]] <- audit_cached_lavaan(
      all_cases[[key]], fits[i, , drop = FALSE],
      estimates[estimates$case_id == key, , drop = FALSE])
  }
  parity <- do.call(rbind, rows)
  ok <- parity[is.na(parity$error), , drop = FALSE]
  disagree <- ok[ok$lavaan_converged != ok$audit_stationary, , drop = FALSE]
  meta <- data.frame(
    run_id = paste0("oracle-", oracle$run_id),
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    oracle_run = oracle$run_id,
    oracle_path = oracle$path,
    oracle_generated = read_meta_value(oracle_meta, "generated"),
    corpus_root = corpus_root_dir,
    books = read_meta_value(oracle_meta, "books", "from oracle"),
    weights = read_meta_value(oracle_meta, "weights", "from oracle"),
    lavaan_version = read_meta_value(oracle_meta, "lavaan_version"),
    n_cells = nrow(parity),
    n_with_both_fits = nrow(ok),
    n_lavaan_only_error = sum(grepl("^lavaan-cache:", parity$error)),
    n_magmaan_only_error = sum(grepl("^magmaan", parity$error)),
    n_agree_converged = sum(ok$lavaan_converged & ok$audit_stationary),
    n_agree_failed = sum(!ok$lavaan_converged & !ok$audit_stationary),
    n_lavaan_yes_audit_no = sum(ok$lavaan_converged & !ok$audit_stationary),
    n_lavaan_no_audit_yes = sum(!ok$lavaan_converged & ok$audit_stationary),
    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    stringsAsFactors = FALSE
  )
  list(parity = parity, disagree = disagree, meta = meta,
       source = oracle$path, oracle_run = oracle$run_id,
       lavaan_version = meta$lavaan_version[[1L]])
}

classify_disagreement <- function(x) {
  ratio <- finite_ratio(x$magmaan_fmin, x$lavaan_fx_bare)
  out <- rep("gradient verdict split", nrow(x))
  out[grepl("little_2013_ch3_fig_3_6_1indicator", x$case_id, fixed = TRUE)] <-
    "auto.fix.single parity gap"
  out[is.finite(ratio) & ratio > 2] <- "LCS / phantom-latent objective gap"
  out[grepl("muthen_2017_ch2_ex2_1__adf", x$case_id, fixed = TRUE)] <-
    "ADF conditioning side finding"
  out[is.finite(x$audit_grad_inf) & x$audit_grad_inf < 1e-2] <-
    "tolerance-boundary gradient verdict"
  out
}

summarize_audit <- function(parity, meta) {
  ok <- parity[is.na(parity$error), , drop = FALSE]
  lavaan_converged <- boolish(ok$lavaan_converged)
  audit_stationary <- boolish(ok$audit_stationary)
  data.frame(
    audit_run = if ("run_id" %in% names(meta)) meta$run_id[[1L]] else "",
    n_cells = nrow(parity),
    n_with_both_fits = nrow(ok),
    n_errors = sum(!is.na(parity$error)),
    n_agree = sum(lavaan_converged == audit_stationary, na.rm = TRUE),
    n_lavaan_yes_audit_no = sum(lavaan_converged & !audit_stationary,
                                na.rm = TRUE),
    n_lavaan_no_audit_yes = sum(!lavaan_converged & audit_stationary,
                                na.rm = TRUE),
    agreement_rate = if (nrow(ok)) {
      mean(lavaan_converged == audit_stationary, na.rm = TRUE)
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

summarize_by <- function(parity, var) {
  ok <- parity[is.na(parity$error), , drop = FALSE]
  split_rows <- split(ok, ok[[var]], drop = TRUE)
  do.call(rbind, lapply(split_rows, function(x) {
    lavaan_converged <- boolish(x$lavaan_converged)
    audit_stationary <- boolish(x$audit_stationary)
    data.frame(
      block = var,
      level = as.character(x[[var]][[1L]]),
      n = nrow(x),
      agree = sum(lavaan_converged == audit_stationary, na.rm = TRUE),
      lavaan_yes_audit_no = sum(lavaan_converged & !audit_stationary,
                                na.rm = TRUE),
      lavaan_no_audit_yes = sum(!lavaan_converged & audit_stationary,
                                na.rm = TRUE),
      agreement_rate = mean(lavaan_converged == audit_stationary,
                            na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}

refit_summary <- function(refit) {
  if (!nrow(refit)) {
    return(data.frame(metric = character(), value = character(),
                      stringsAsFactors = FALSE))
  }
  agrees <- boolish(refit$v8_agrees)
  errors <- !is.na(refit$v8_err)
  data.frame(
    metric = c("v7 disagreement cells refit",
               "v8 agreements among v7 disagreements",
               "v8 remaining disagreements",
               "v8 errors"),
    value = c(nrow(refit),
              sum(agrees, na.rm = TRUE),
              sum(!agrees & !errors, na.rm = TRUE),
              sum(errors, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

if (isTRUE(args$collect_lavaan)) {
  collect_args <- character()
  if (!identical(args$oracle_run, "latest")) {
    collect_args <- c(collect_args, "--run-id", args$oracle_run)
  }
  if (isTRUE(args$install_latest_lavaan)) {
    collect_args <- c(collect_args, "--install-latest-lavaan")
  }
  if (!is.null(args$books)) {
    collect_args <- c(collect_args, "--books", paste(args$books, collapse = ","))
  }
  if (!is.null(args$weights)) {
    collect_args <- c(collect_args, "--weights",
                      paste(args$weights, collapse = ","))
  }
  if (is.finite(args$limit) && args$limit > 0L) {
    collect_args <- c(collect_args, "--limit", as.character(args$limit))
  }
  run_script(experiment_path("scripts", "collect_lavaan_oracle.R"),
             collect_args)
  args$from_oracle <- TRUE
}

# The self-contained cached-oracle audit is the only audit path: replay each
# cached lavaan theta_hat through magmaan's evaluate_at terminal audit. Populate
# the oracle once with --collect-lavaan. (The former default copied a paper's
# curated CSVs; experiments are sinks and own their results.)
args$from_oracle <- TRUE

if (isTRUE(args$run_refit)) {
  run_script(experiment_path("scripts", "refit_disagreements.R"))
  args$refit_csv <- file.path(results_dir(create = TRUE), "refit-disagreements",
                              "refit_disagreements_v8.csv")
}

results <- ensure_results_dir()
audit_dst <- file.path(results, "audit-parity")
refit_dst <- file.path(results, "refit-disagreements")
audit_mode <- "cached lavaan oracle"

oracle_audit <- audit_lavaan_oracle(args$oracle_run)
parity <- oracle_audit$parity
disagree <- oracle_audit$disagree
meta <- oracle_audit$meta
oracle_run <- oracle_audit$oracle_run
oracle_path <- oracle_audit$source
lavaan_version <- oracle_audit$lavaan_version
write_csv(parity, file.path(audit_dst, "lavaan_audit_parity.csv"))
write_csv(disagree, file.path(audit_dst, "lavaan_audit_parity_disagree.csv"))
write_csv(meta, file.path(audit_dst, "lavaan_audit_parity_meta.csv"))

has_refit <- copy_optional(args$refit_csv,
                           file.path(refit_dst, "refit_disagreements_v8.csv"))
refit <- if (has_refit) {
  read_csv(file.path(refit_dst, "refit_disagreements_v8.csv"))
} else {
  data.frame()
}

disagree$fmin_ratio <- finite_ratio(disagree$magmaan_fmin,
                                    disagree$lavaan_fx_bare)
disagree$classification <- classify_disagreement(disagree)
write_csv(disagree, file.path(results, "audit_disagreements_enriched.csv"))

current_failures <- disagree

write_csv(summarize_audit(parity, meta), file.path(results, "audit_summary.csv"))
write_csv(summarize_by(parity, "book"), file.path(results, "audit_by_book.csv"))
write_csv(summarize_by(parity, "weight"), file.path(results, "audit_by_weight.csv"))
write_csv(refit_summary(refit), file.path(results, "refit_summary.csv"))
write_csv(current_failures, file.path(results, "current_failures.csv"))

drivers <- data.frame(
  driver = c("lavaan oracle collector", "cached-oracle magmaan audit",
             "disagreement refit", "ADF conditioning probe"),
  path = c(
    experiment_path("scripts", "collect_lavaan_oracle.R"),
    experiment_path("run_experiment.R"),
    experiment_path("scripts", "refit_disagreements.R"),
    experiment_path("scripts", "diagnose_adf_conditioning.R")
  ),
  role = c(
    "slow lavaan-only theta_hat/objective/convergence cache",
    "fresh magmaan evaluate_at audit against cached lavaan theta_hat",
    "targeted refit of the audit disagreement cells",
    "eigendecomposition of the saturated Muthen ADF conditioning case"
  ),
  expected_runtime = c("slow, about lavaan side of full sweep",
                       "fast relative to lavaan collection",
                       "about 3 s", "about 1 s"),
  stringsAsFactors = FALSE
)
write_csv(drivers, file.path(results, "driver_inventory.csv"))

write_metadata(
  file.path(results, "metadata.csv"),
  values = list(
    audit_mode = audit_mode,
    audit_run = args$audit_run,
    audit_source = oracle_path,
    oracle_run = oracle_run,
    oracle_path = oracle_path,
    lavaan_version = lavaan_version,
    refit_source = if (has_refit) args$refit_csv else "",
    has_refit = has_refit,
    n_audit_cells = nrow(parity),
    n_disagreements = nrow(disagree),
    n_current_failures = nrow(current_failures)
  )
)

cat(sprintf(
  "%s: %d audit cells, %d disagreements, %d current/projected failures\n",
  audit_mode, nrow(parity), nrow(disagree), nrow(current_failures)
))
if (nzchar(oracle_run)) {
  cat(sprintf("used lavaan oracle '%s' from %s\n", oracle_run, oracle_path))
}
cat(sprintf("wrote normalized tables to %s\n", results))
