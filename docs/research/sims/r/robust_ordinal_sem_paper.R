# Paper-local robust ordinal SEM simulation runner.
#
# First milestone: Designs 1--2 from
# docs/research/notes/robust_ordinal_sem_paper_plan.md. The runner emits tidy
# CSV rows for robust polychoric moment builders without fitting SEMs yet.

ros_find_repo_root <- function(start = getwd()) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    marker <- file.path(path, "docs", "research", "sims", "r", "common.R")
    if (file.exists(marker) && file.exists(file.path(path, "AGENTS.md"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("Could not locate magmaan repository root", call. = FALSE)
    }
    path <- parent
  }
}

ros_repo_root <- ros_find_repo_root()
source(file.path(ros_repo_root, "docs", "research", "sims", "r", "common.R"))

ros_require_magmaan <- function() {
  sim_require("magmaan")
  magmaan::magmaan_core
}

ros_gumbel <- function(n, location = 0, scale = 1) {
  location - scale * log(-log(stats::runif(n)))
}

ros_welz_thresholds <- function() {
  stats::qnorm(c(.10, .30, .70, .90))
}

ros_welz_bivariate_thresholds <- function() {
  list(
    x1 = c(-1.5, 0, 0.5, 1),
    x2 = c(-1, 1, 1.5, 2)
  )
}

ros_welz_matrix_r <- function() {
  loadings <- c(.8, .7, .6, .5, .4)
  R <- tcrossprod(loadings)
  diag(R) <- 1
  dimnames(R) <- list(paste0("x", seq_along(loadings)),
                      paste0("x", seq_along(loadings)))
  R
}

ros_ordinalize_matrix <- function(z, thresholds) {
  p <- ncol(z)
  out <- matrix(NA_real_, nrow(z), p)
  for (j in seq_len(p)) {
    th <- thresholds[[j]]
    out[, j] <- cut(z[, j], c(-Inf, th, Inf), labels = FALSE)
  }
  colnames(out) <- colnames(z) %||% paste0("x", seq_len(p))
  out
}

ros_sim_welz_bivariate <- function(n, contamination, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  rho <- 0.5
  truth <- matrix(c(1, rho, rho, 1), 2, 2,
                  dimnames = list(c("x1", "x2"), c("x1", "x2")))
  clean <- sim_mvn(n, truth)
  contam <- cbind(stats::rnorm(n, mean = 2.5, sd = 0.5),
                  stats::rnorm(n, mean = -2.5, sd = 0.5))
  use_contam <- stats::runif(n) < contamination
  z <- clean
  z[use_contam, ] <- contam[use_contam, ]
  colnames(z) <- c("x1", "x2")
  list(
    design = "welz_bivariate",
    X = ros_ordinalize_matrix(z, ros_welz_bivariate_thresholds()),
    truth_R = truth,
    n_contaminated = sum(use_contam)
  )
}

ros_sim_welz_matrix <- function(n, contamination, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  truth <- ros_welz_matrix_r()
  p <- ncol(truth)
  clean <- sim_mvn(n, truth)
  contam <- matrix(ros_gumbel(n * p, location = 0, scale = 3), n, p)
  use_contam <- stats::runif(n) < contamination
  z <- clean
  z[use_contam, ] <- contam[use_contam, ]
  colnames(z) <- colnames(truth)
  list(
    design = "welz_matrix",
    X = ros_ordinalize_matrix(z, rep(list(ros_welz_thresholds()), p)),
    truth_R = truth,
    n_contaminated = sum(use_contam)
  )
}

ros_method_specs <- function() {
  list(
    list(method = "ml", tuning = "ml",
         args = list(robust = "ml")),
    list(method = "wma_hard_cap", tuning = "h_k=1.6",
         args = list(robust = "h_weighted", h_kind = "wma_hard_cap",
                     h_k = 1.6)),
    list(method = "smooth_cap", tuning = "h_a=1.6;h_b=2.2",
         args = list(robust = "h_weighted", h_kind = "smooth_cap",
                     h_a = 1.6, h_b = 2.2, h_lambda = 0.2)),
    list(method = "exp_cap", tuning = "h_k=1.6;lambda=0.2",
         args = list(robust = "h_weighted", h_kind = "exp_cap",
                     h_k = 1.6, h_lambda = 0.2)),
    list(method = "dpd", tuning = "alpha=0.3",
         args = list(robust = "dpd", alpha = 0.3)),
    list(method = "huber_residual", tuning = "hard_huber;k=1.345",
         args = list(robust = "huber_residual", clip = "hard_huber",
                     k = 1.345)),
    list(method = "pseudo_huber", tuning = "pseudo_huber;k=1.345",
         args = list(robust = "huber_residual", clip = "pseudo_huber",
                     k = 1.345)),
    list(method = "tukey_biweight", tuning = "tukey_biweight;k=4.685",
         args = list(robust = "huber_residual", clip = "tukey_biweight",
                     k = 4.685))
  )
}

ros_diag_value <- function(stats, name, default = NA) {
  diag <- stats$diagnostics
  if (!is.list(diag) || !length(diag) || is.null(diag[[1]][[name]])) {
    return(default)
  }
  diag[[1]][[name]]
}

ros_block_nobs <- function(stats, block = 1L) {
  nobs <- stats$n_obs %||% stats$nobs
  if (is.null(nobs) || length(nobs) < block) {
    stop("ordinal stats did not include block sample sizes", call. = FALSE)
  }
  nobs[[block]]
}

ros_pair_rows <- function(stats, truth_R, meta, elapsed_ms) {
  R <- stats$R[[1]]
  n <- ros_block_nobs(stats)
  pairs <- which(lower.tri(R), arr.ind = TRUE)
  threshold_dim <- length(stats$thresholds[[1]])
  nacov_diag <- diag(stats$NACOV[[1]])
  min_eigen <- suppressWarnings(min(eigen(R, symmetric = TRUE,
                                          only.values = TRUE)$values))
  diag_min <- ros_diag_value(stats, "min_eigen_r", min_eigen)
  repair <- ros_diag_value(stats, "r_repair_applied", FALSE)

  rows <- vector("list", nrow(pairs))
  for (k in seq_len(nrow(pairs))) {
    i <- pairs[k, "row"]
    j <- pairs[k, "col"]
    moment_index <- threshold_dim + k
    se <- if (moment_index <= length(nacov_diag)) {
      sqrt(max(0, nacov_diag[[moment_index]]) / n)
    } else {
      NA_real_
    }
    est <- R[i, j]
    rows[[k]] <- data.frame(
      design = meta$design,
      rep = meta$rep,
      n = meta$n,
      contamination = meta$contamination,
      n_contaminated = meta$n_contaminated,
      method = meta$method,
      tuning = meta$tuning,
      converged = TRUE,
      elapsed_ms = elapsed_ms,
      iterations = NA_integer_,
      min_eigen_r = diag_min,
      r_repair = isTRUE(repair),
      objective = NA_real_,
      pair = paste0(colnames(R)[i] %||% paste0("x", i), ":",
                    colnames(R)[j] %||% paste0("x", j)),
      parameter = "rho",
      truth = truth_R[i, j],
      estimate = est,
      se = se,
      ci_low = est - stats::qnorm(.975) * se,
      ci_high = est + stats::qnorm(.975) * se,
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

ros_failure_row <- function(meta, elapsed_ms, err) {
  data.frame(
    design = meta$design,
    rep = meta$rep,
    n = meta$n,
    contamination = meta$contamination,
    n_contaminated = meta$n_contaminated,
    method = meta$method,
    tuning = meta$tuning,
    converged = FALSE,
    elapsed_ms = elapsed_ms,
    iterations = NA_integer_,
    min_eigen_r = NA_real_,
    r_repair = NA,
    objective = NA_real_,
    pair = NA_character_,
    parameter = "__moment_builder__",
    truth = NA_real_,
    estimate = NA_real_,
    se = NA_real_,
    ci_low = NA_real_,
    ci_high = NA_real_,
    error = conditionMessage(err),
    stringsAsFactors = FALSE
  )
}

ros_eval_method <- function(sim, method_spec, rep, n, contamination) {
  core <- ros_require_magmaan()
  meta <- list(
    design = sim$design,
    rep = rep,
    n = n,
    contamination = contamination,
    n_contaminated = sim$n_contaminated,
    method = method_spec$method,
    tuning = method_spec$tuning
  )

  started <- proc.time()[["elapsed"]]
  result <- tryCatch({
    args <- c(list(X = list(sim$X)), method_spec$args)
    stats <- do.call(core$data_ordinal_stats_from_raw, args)
    elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - started)
    ros_pair_rows(stats, sim$truth_R, meta, elapsed_ms)
  }, error = function(err) {
    elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - started)
    ros_failure_row(meta, elapsed_ms, err)
  })
  result
}

robust_ordinal_paper_run <- function(
    designs = c("welz_bivariate", "welz_matrix"),
    reps = 10L,
    n = 1000L,
    contaminations = c(0, .01, .05, .10, .15, .20, .30),
    seed = 20260520L,
    methods = ros_method_specs()) {
  designs <- match.arg(designs, c("welz_bivariate", "welz_matrix"),
                       several.ok = TRUE)
  rows <- list()
  cursor <- 0L
  for (design in designs) {
    for (contamination in contaminations) {
      for (rep in seq_len(reps)) {
        rep_seed <- seed + rep +
          1000L * match(design, c("welz_bivariate", "welz_matrix")) +
          100000L * match(contamination, contaminations)
        sim <- switch(
          design,
          welz_bivariate = ros_sim_welz_bivariate(n, contamination, rep_seed),
          welz_matrix = ros_sim_welz_matrix(n, contamination, rep_seed)
        )
        for (method in methods) {
          cursor <- cursor + 1L
          rows[[cursor]] <- ros_eval_method(sim, method, rep, n, contamination)
        }
      }
    }
  }
  do.call(rbind, rows)
}

ros_parse_arg <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

ros_parse_csv <- function(x, default) {
  if (is.null(x) || !nzchar(x)) return(default)
  strsplit(x, ",", fixed = TRUE)[[1]]
}

ros_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  reps <- as.integer(ros_parse_arg(args, "reps", "1"))
  n <- as.integer(ros_parse_arg(args, "n", "250"))
  seed <- as.integer(ros_parse_arg(args, "seed", "20260520"))
  designs <- ros_parse_csv(ros_parse_arg(args, "designs", NULL),
                           c("welz_bivariate", "welz_matrix"))
  contaminations <- as.numeric(ros_parse_csv(
    ros_parse_arg(args, "contaminations", NULL),
    c("0", ".15")
  ))
  out <- ros_parse_arg(
    args, "out",
    file.path(ros_repo_root, "docs", "research", "sims", "results",
              "robust_ordinal_sem_paper.csv")
  )

  result <- robust_ordinal_paper_run(
    designs = designs,
    reps = reps,
    n = n,
    contaminations = contaminations,
    seed = seed
  )
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(result, out, row.names = FALSE)
  message("wrote ", nrow(result), " rows to ", out)
  invisible(result)
}

if (sys.nframe() == 0L) {
  ros_main()
}
