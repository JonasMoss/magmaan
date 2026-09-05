#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--replication] [options]\n\n",
  "Identify the matrix-estimation conventions in the Savalei-Falk (2014)\n",
  "robust FIML and two-stage tests. One fit of each method per generated\n",
  "sample is reused for all canonical FIML residual/sandwich choices and\n",
  "all ML2S Stage-1 bread/meat by Stage-2 H choices.\n\n",
  "Profiles:\n",
  "  --smoke        2 kurtosis cells x 4 reps (default)\n",
  "  --pilot        2 kurtosis cells x 200 reps\n",
  "  --replication  2 kurtosis cells x 1,000 reps\n\n",
  "Options:\n",
  "  --reps N --cores N --seed-base N --results-dir PATH --resume\n",
  sep = "")

parse_args <- function(args) {
  out <- list(profile = "smoke", reps = NULL,
              cores = max(1L, min(8L, parallel::detectCores() - 1L)),
              seed_base = 20140280, results_dir = NULL, resume = FALSE)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing option value", call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--pilot") out$profile <- "pilot"
    else if (a == "--replication") out$profile <- "replication"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--resume") out$resume <- TRUE
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (is.null(out$reps)) {
    out$reps <- switch(out$profile, smoke = 4L, pilot = 200L,
                       replication = 1000L)
  }
  if (is.na(out$reps) || out$reps < 1L ||
      is.na(out$cores) || out$cores < 1L ||
      !is.finite(out$seed_base) || out$seed_base < 1) {
    stop("reps, cores, and seed-base must be positive", call. = FALSE)
  }
  out
}

model_2_population <- function() {
  lambda <- c(
    0.743, 0.252, 0.604, 0.540, 0.201, 0.578, 0.729,
    0.536, 0.528, 0.692, 0.694, 0.509, 0.698, 0.562,
    0.426, 0.530, 0.637, 0.755, 0.564, 0.647, 0.825)
  phi <- matrix(c(
    1, 0.446, -0.124,
    0.446, 1, -0.085,
    -0.124, -0.085, 1), 3L, 3L, byrow = TRUE)
  Lambda <- matrix(0, 21L, 3L)
  Lambda[cbind(seq_len(21L), rep(seq_len(3L), each = 7L))] <- lambda
  Sigma <- Lambda %*% phi %*% t(Lambda) + diag(1 - lambda^2)
  dimnames(Sigma) <- list(paste0("V", seq_len(21L)),
                          paste0("V", seq_len(21L)))
  list(
    Sigma = Sigma,
    syntax = paste(
      "F1 =~ V1 + V2 + V3 + V4 + V5 + V6 + V7",
      "F2 =~ V8 + V9 + V10 + V11 + V12 + V13 + V14",
      "F3 =~ V15 + V16 + V17 + V18 + V19 + V20 + V21",
      sep = "\n"))
}

apply_mp_mar_l <- function(X, proportion, seed) {
  sets <- list(
    list(condition = 1L, delete = c(2L, 8L, 9L)),
    list(condition = 3L, delete = 4L),
    list(condition = 5L, delete = c(6L, 7L)),
    list(condition = 15L, delete = c(13L, 14L, 16L)),
    list(condition = 17L, delete = 18L),
    list(condition = 19L, delete = c(20L, 21L)))
  target <- as.integer(round(nrow(X) * proportion))
  set.seed(seed)
  eligible_counts <- integer(length(sets))
  for (j in seq_along(sets)) {
    eligible <- which(X[, sets[[j]]$condition] > 0)
    eligible_counts[[j]] <- length(eligible)
    if (length(eligible) < target) {
      stop("fewer MAR-L eligible rows than the deletion target", call. = FALSE)
    }
    selected <- sample(eligible, target, replace = FALSE)
    X[selected, sets[[j]]$delete] <- NA_real_
  }
  list(
    data = X,
    eligible_min = min(eligible_counts),
    eligible_max = max(eligible_counts),
    patterns = nrow(unique(is.na(X))),
    missing_rate = mean(is.na(X)))
}

mp_mar_l_is_feasible <- function(X, proportion = 0.30) {
  conditioning <- c(1L, 3L, 5L, 15L, 17L, 19L)
  target <- as.integer(round(nrow(X) * proportion))
  all(colSums(X[, conditioning, drop = FALSE] > 0) >= target)
}

fit_one <- function(X, model, rep_id, kurtosis, seed_base) {
  failure <- function(method, message, missing_rate = NA_real_,
                      patterns = NA_integer_, eligible_min = NA_integer_,
                      eligible_max = NA_integer_, converged = FALSE) data.frame(
    rep = rep_id, kurtosis = kurtosis, method = method,
    information = NA_character_, residual_information = NA_character_,
    omega_bread_point = NA_character_, omega_bread_kind = NA_character_,
    omega_meat_point = NA_character_, stage2_information = NA_character_,
    chisq = NA_real_, chisq_scaled = NA_real_, df = NA_integer_,
    scaling_factor = NA_real_, p_value = NA_real_, reject = NA,
    min_omega_bread_eigenvalue = NA_real_,
    omega_bread_negative_eigenvalues = NA_integer_,
    min_stage2_information_eigenvalue = NA_real_,
    stage2_information_negative_eigenvalues = NA_integer_,
    min_projection_eigenvalue = NA_real_,
    projection_negative_eigenvalues = NA_integer_,
    projection_rank = NA_integer_, delta_rank = NA_integer_,
    saturated_identity_error = NA_real_, stage1_bread_difference = NA_real_,
    default_scale_error = NA_real_, missing_rate = missing_rate,
    patterns = patterns, eligible_min = eligible_min,
    eligible_max = eligible_max, converged = converged,
    admissible = NA, error = message, stringsAsFactors = FALSE)

  miss <- tryCatch(
    apply_mp_mar_l(X, 0.30, seed_base + kurtosis * 100000 + rep_id),
    error = identity)
  if (inherits(miss, "error")) {
    return(do.call(rbind, lapply(c("ml2s", "fiml"), failure,
                                message = conditionMessage(miss))))
  }
  dat <- as.data.frame(miss$data)
  colnames(dat) <- paste0("V", seq_len(ncol(dat)))
  fits <- lapply(c(ml2s = "ML2S", fiml = "FIML"), function(estimator)
    tryCatch(magmaan::magmaan(
      model, dat, estimator = estimator, se = "none", test = "none"),
      error = identity))
  if (any(vapply(fits, inherits, logical(1), what = "error"))) {
    return(do.call(rbind, lapply(names(fits), function(method) {
      fit <- fits[[method]]
      failure(method,
              if (inherits(fit, "error")) conditionMessage(fit)
              else "the companion estimator failed",
              miss$missing_rate, miss$patterns, miss$eligible_min,
              miss$eligible_max, !inherits(fit, "error") &&
                isTRUE(fit$converged))
    })))
  }
  audits <- list(
    ml2s = tryCatch(
      magmaan::magmaan_core$frontier_ml2s_information_choices(fits$ml2s),
      error = identity),
    fiml = tryCatch(
      magmaan::magmaan_core$frontier_fiml_information_choices(fits$fiml),
      error = identity))
  if (any(vapply(audits, inherits, logical(1), what = "error"))) {
    return(do.call(rbind, lapply(names(audits), function(method)
      failure(method,
              if (inherits(audits[[method]], "error"))
                conditionMessage(audits[[method]])
              else "the companion matrix audit failed",
              miss$missing_rate, miss$patterns, miss$eligible_min,
              miss$eligible_max, isTRUE(fits[[method]]$converged)))))
  }

  ml2s <- audits$ml2s$choices
  ml2s$method <- "ml2s"
  ml2s$residual_information <- ml2s$stage2_information
  ml2s$omega_bread_point <- ml2s$stage1_bread_point
  ml2s$omega_bread_kind <- ml2s$stage1_bread_kind
  ml2s$omega_meat_point <- ml2s$stage1_meat_point
  ml2s$min_omega_bread_eigenvalue <-
    ml2s$min_stage1_information_eigenvalue
  ml2s$omega_bread_negative_eigenvalues <-
    ml2s$stage1_information_negative_eigenvalues
  ml2s$min_stage2_information_eigenvalue <-
    ml2s$min_information_eigenvalue
  ml2s$stage2_information_negative_eigenvalues <-
    ml2s$information_negative_eigenvalues
  ml2s$min_projection_eigenvalue <- ml2s$min_projector_eigenvalue
  ml2s$projection_negative_eigenvalues <-
    ml2s$projector_negative_eigenvalues
  ml2s$projection_rank <- ml2s$projector_rank
  ml2s$saturated_identity_error <-
    audits$ml2s$saturated_expected_observed_max_abs
  ml2s$stage1_bread_difference <-
    audits$ml2s$stage1_expected_observed_max_abs
  default_row <- ml2s$omega_bread_point == "saturated" &
    ml2s$omega_bread_kind == "observed" &
    ml2s$omega_meat_point == "saturated" &
    ml2s$stage2_information == "saturated_expected"
  ml2s$default_scale_error <- if (sum(default_row) == 1L) {
    abs(ml2s$scaling_factor[default_row] - fits$ml2s$scaling_factor)
  } else NA_real_

  fiml <- audits$fiml$choices
  fiml$method <- "fiml"
  fiml$stage2_information <- NA_character_
  fiml$min_projection_eigenvalue <-
    fiml$min_residual_information_eigenvalue
  fiml$projection_negative_eigenvalues <-
    fiml$residual_information_negative_eigenvalues
  fiml$projection_rank <- fiml$residual_rank
  fiml$min_stage2_information_eigenvalue <- NA_real_
  fiml$stage2_information_negative_eigenvalues <- NA_integer_
  fiml$saturated_identity_error <- NA_real_
  fiml$stage1_bread_difference <- NA_real_
  fiml$default_scale_error <- NA_real_

  matrix_cols <- c(
    "method", "information", "residual_information",
    "omega_bread_point", "omega_bread_kind", "omega_meat_point",
    "stage2_information", "trace_ugamma", "scaling_factor", "chisq_scaled",
    "min_omega_bread_eigenvalue", "omega_bread_negative_eigenvalues",
    "min_stage2_information_eigenvalue",
    "stage2_information_negative_eigenvalues", "min_projection_eigenvalue",
    "projection_negative_eigenvalues", "projection_rank",
    "saturated_identity_error", "stage1_bread_difference",
    "default_scale_error")
  rows <- rbind(ml2s[, matrix_cols], fiml[, matrix_cols])
  method_audit <- list(ml2s = audits$ml2s, fiml = audits$fiml)
  rows$rep <- rep_id
  rows$kurtosis <- kurtosis
  rows$chisq <- vapply(rows$method, function(m) method_audit[[m]]$chisq,
                       numeric(1))
  rows$df <- vapply(rows$method,
                    function(m) as.integer(method_audit[[m]]$df), integer(1))
  rows$p_value <- stats::pchisq(rows$chisq_scaled, rows$df,
                                lower.tail = FALSE)
  rows$reject <- rows$p_value < 0.05
  rows$delta_rank <- vapply(rows$method,
                            function(m) as.integer(method_audit[[m]]$delta_rank),
                            integer(1))
  rows$missing_rate <- miss$missing_rate
  rows$patterns <- miss$patterns
  rows$eligible_min <- miss$eligible_min
  rows$eligible_max <- miss$eligible_max
  rows$converged <- vapply(rows$method,
                           function(m) isTRUE(fits[[m]]$converged), logical(1))
  rows$admissible <- vapply(rows$method, function(m)
    fits[[m]]$diagnostics$admissibility$admissible %||% NA, logical(1))
  rows$error <- ""
  rows[, names(failure("ml2s", ""))]
}

summarize_results <- function(raw) {
  ok <- !is.na(raw$information) & is.finite(raw$p_value)
  x <- raw[ok, , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  keys <- interaction(x$kurtosis, x$method, x$information, drop = TRUE)
  parts <- split(x, keys)
  do.call(rbind, lapply(parts, function(z) data.frame(
    kurtosis = z$kurtosis[[1L]],
    method = z$method[[1L]],
    information = z$information[[1L]],
    residual_information = z$residual_information[[1L]],
    omega_bread_point = z$omega_bread_point[[1L]],
    omega_bread_kind = z$omega_bread_kind[[1L]],
    omega_meat_point = z$omega_meat_point[[1L]],
    stage2_information = z$stage2_information[[1L]],
    replications = nrow(z),
    rejection_rate = mean(z$reject),
    rejection_se = sqrt(mean(z$reject) * (1 - mean(z$reject)) / nrow(z)),
    mean_scaling_factor = mean(z$scaling_factor),
    median_scaling_factor = stats::median(z$scaling_factor),
    mean_chisq_scaled = mean(z$chisq_scaled),
    omega_bread_indefinite_rate =
      mean(z$omega_bread_negative_eigenvalues > 0),
    stage2_information_indefinite_rate = if (all(is.na(
      z$stage2_information_negative_eigenvalues))) NA_real_ else mean(
        z$stage2_information_negative_eigenvalues > 0, na.rm = TRUE),
    projection_indefinite_rate =
      mean(z$projection_negative_eigenvalues > 0),
    median_min_omega_bread_eigenvalue =
      stats::median(z$min_omega_bread_eigenvalue),
    median_min_stage2_information_eigenvalue = if (all(is.na(
      z$min_stage2_information_eigenvalue))) NA_real_ else stats::median(
        z$min_stage2_information_eigenvalue, na.rm = TRUE),
    median_min_projection_eigenvalue =
      stats::median(z$min_projection_eigenvalue),
    stringsAsFactors = FALSE)))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
out_dir <- args$results_dir %||% experiment_path("results", args$profile)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
pop <- model_2_population()
model <- magmaan::model_spec(pop$syntax, meanstructure = TRUE)
design <- data.frame(
  kurtosis = c(7L, 15L), skewness = 2, n = 200L,
  missing_proportion = 0.30, patterns = "MP",
  mechanism = "MAR-L", model = "Savalei-Falk Model 2",
  paper_ts_rejection = c(0.100, NA_real_),
  paper_fiml_rejection = c(0.633, NA_real_),
  fleishman_a = NA_real_, fleishman_b = NA_real_,
  fleishman_c = NA_real_, fleishman_d = NA_real_,
  transform_derivative_minimum = NA_real_,
  theoretical_eligible_proportion = NA_real_,
  intermediate_min_eigenvalue = NA_real_,
  candidate_draws_generated = NA_integer_,
  candidate_draws_consumed = NA_integer_,
  eligibility_rejections = NA_integer_,
  stringsAsFactors = FALSE)

all_raw <- list()
reused_cells <- 0L
wall_start <- proc.time()[["elapsed"]]
for (cell in seq_len(nrow(design))) {
  k <- design$kurtosis[[cell]]
  cat(sprintf("Calibrating VM skewness 2, excess kurtosis %d...\n", k))
  cal <- magmaan::magmaan_core$sim_vm_calibrate(
    pop$Sigma, rep(2, 21L), rep(k, 21L))
  coef <- unname(cal$coefficients[1L, ])
  design[cell, c("fleishman_a", "fleishman_b", "fleishman_c",
                 "fleishman_d")] <- coef
  design$transform_derivative_minimum[[cell]] <-
    coef[[2L]] - coef[[3L]]^2 / (3 * coef[[4L]])
  zero <- stats::uniroot(
    function(z) coef[[1L]] + coef[[2L]] * z +
      coef[[3L]] * z^2 + coef[[4L]] * z^3,
    c(-10, 10))$root
  design$theoretical_eligible_proportion[[cell]] <-
    stats::pnorm(zero, lower.tail = FALSE)
  design$intermediate_min_eigenvalue[[cell]] <-
    min(eigen(cal$intermediate_corr, symmetric = TRUE,
              only.values = TRUE)$values)
  candidate_n <- as.integer(ceiling(1.15 * args$reps) + 12L)
  candidate_draws <- magmaan::magmaan_core$sim_vm_draw(
    cal, n = 200L, reps = candidate_n,
    seed_base = args$seed_base + k * 1000000)$draws
  feasible <- vapply(candidate_draws, mp_mar_l_is_feasible, logical(1))
  chosen <- which(feasible)
  if (length(chosen) < args$reps) {
    stop("candidate batch did not contain enough MAR-L-feasible samples",
         call. = FALSE)
  }
  chosen <- chosen[seq_len(args$reps)]
  draws <- candidate_draws[chosen]
  design$candidate_draws_generated[[cell]] <- candidate_n
  design$candidate_draws_consumed[[cell]] <- max(chosen)
  design$eligibility_rejections[[cell]] <- max(chosen) - args$reps
  checkpoint <- file.path(out_dir, sprintf(
    "replications-k%d-n%d-seed%.0f-checkpoint.csv",
    k, args$reps, args$seed_base))
  if (args$resume && file.exists(checkpoint)) {
    cached <- read.csv(checkpoint, stringsAsFactors = FALSE,
                       check.names = FALSE)
    cached$error[is.na(cached$error)] <- ""
    cached_counts <- table(cached$rep)
    cached_methods <- with(cached, table(rep, method))
    if (nrow(cached) == 80L * args$reps &&
        length(cached_counts) == args$reps && all(cached_counts == 80L) &&
        all(cached_methods[, "fiml"] == 48L) &&
        all(cached_methods[, "ml2s"] == 32L)) {
      all_raw[[cell]] <- cached
      reused_cells <- reused_cells + 1L
      cat(sprintf("[k=%d] reused complete checkpoint (%d/%d)\n",
                  k, args$reps, args$reps))
      next
    }
  }
  chunks <- split(seq_len(args$reps),
                  ceiling(seq_len(args$reps) / max(1L, args$cores)))
  cell_rows <- list()
  for (chunk_id in seq_along(chunks)) {
    ids <- chunks[[chunk_id]]
    worker <- function(i) fit_one(
      draws[[i]], model, i, k, args$seed_base)
    rows <- if (args$cores > 1L && length(ids) > 1L) {
      parallel::mclapply(ids, worker, mc.cores = min(args$cores, length(ids)),
                         mc.preschedule = TRUE, mc.set.seed = FALSE)
    } else lapply(ids, worker)
    cell_rows[[chunk_id]] <- do.call(rbind, rows)
    cat(sprintf(
      "[k=%d] %d/%d replications complete (%.1f%%)\n",
      k, max(ids), args$reps, 100 * max(ids) / args$reps))
  }
  all_raw[[cell]] <- do.call(rbind, cell_rows)
  write_csv(all_raw[[cell]], checkpoint)
}
raw <- do.call(rbind, all_raw)
summary <- summarize_results(raw)
failure_rows <- !is.na(raw[["error"]]) & nzchar(as.character(raw[["error"]]))
failure_columns <- c("rep", "kurtosis", "method", "converged", "error")
failures <- raw[failure_rows, failure_columns, drop = FALSE]
failures <- failures[!duplicated(failures), , drop = FALSE]
valid <- raw[!is.na(raw$information), , drop = FALSE]
replication_key <- interaction(valid$kurtosis, valid$rep, drop = TRUE)
method_counts <- with(valid, table(kurtosis, rep, method))
audit <- data.frame(
  key = c("saturated_identity_max_abs", "stage1_bread_difference_max_abs",
          "default_scale_max_abs",
          "delta_rank_min", "delta_rank_max", "df_min", "df_max"),
  value = c(max(valid$saturated_identity_error, na.rm = TRUE),
            max(valid$stage1_bread_difference, na.rm = TRUE),
            max(valid$default_scale_error, na.rm = TRUE),
            min(valid$delta_rank), max(valid$delta_rank),
            min(valid$df), max(valid$df)),
  stringsAsFactors = FALSE)
write_csv(raw, file.path(out_dir, "replications.csv"))
write_csv(summary, file.path(out_dir, "summary.csv"))
write_csv(failures, file.path(out_dir, "failures.csv"))
write_csv(design, file.path(out_dir, "design.csv"))
write_csv(audit, file.path(out_dir, "audit.csv"))

stopifnot(
  nrow(valid) > 0,
  all(table(replication_key) == 80L),
  all(method_counts[, , "fiml"] == 48L),
  all(method_counts[, , "ml2s"] == 32L),
  max(valid$saturated_identity_error, na.rm = TRUE) < 1e-7,
  max(valid$default_scale_error, na.rm = TRUE) < 1e-7,
  all(valid$delta_rank == 66L),
  all(valid$df == 186L))

write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(
    experiment = "80-savalei-falk-2014-ml2s-information",
    profile = args$profile,
    reps_per_cell = args$reps,
    cells = nrow(design),
    seed_base = args$seed_base,
    cores = args$cores,
    resume = args$resume,
    checkpoint_cells_reused = reused_cells,
    elapsed_seconds = proc.time()[["elapsed"]] - wall_start,
    paper = "Savalei and Falk (2014), doi:10.1080/10705511.2014.882692",
    paper_target = paste(
      "Model 2, N=200, MAR-L, MP, 30% missing, s2k7:",
      "TS rejection 0.100; FIML rejection 0.633"),
    kurtosis_convention = "Fleishman excess kurtosis, matching the EQS/SEM convention",
    git_head = git_scalar(c("rev-parse", "HEAD")),
    git_dirty = git_dirty()),
  packages = "magmaan")
cat("Wrote results to ", out_dir, "\n", sep = "")
