#!/usr/bin/env Rscript

.profile_script <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(
      sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE
    ))
  }
  normalizePath("run_profiles.R", mustWork = TRUE)
}
source(file.path(dirname(.profile_script()), "run_experiment.R"))
rm(.profile_script)

profile_usage <- function() cat(
  "Usage: Rscript run_profiles.R [options]\n\n",
  "Trace confirmed PSD-ML basins over a fixed-beta grid.\n\n",
  "Options:\n",
  "  --results-dir PATH --cases-per-n N --grid-points N --max-iter N\n",
  sep = "")

parse_profile_args <- function(args) {
  out <- list(
    results_dir = experiment_path("results", "pilot"),
    cases_per_n = 5L,
    grid_points = 41L,
    max_iter = 10000L
  )
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
    args[[i]]
  }
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      profile_usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--results-dir") {
      out$results_dir <- take()
    } else if (arg == "--cases-per-n") {
      out$cases_per_n <- as.integer(take())
    } else if (arg == "--grid-points") {
      out$grid_points <- as.integer(take())
    } else if (arg == "--max-iter") {
      out$max_iter <- as.integer(take())
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  out$results_dir <- normalizePath(out$results_dir, mustWork = TRUE)
  if (anyNA(c(out$cases_per_n, out$grid_points, out$max_iter)) ||
      out$cases_per_n < 1L || out$grid_points < 11L ||
      out$max_iter < 1L) {
    stop("invalid cases-per-n, grid-points, or max-iter", call. = FALSE)
  }
  out
}

read_profile_result <- function(root, name) {
  utils::read.csv(
    file.path(root, name),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fixed_beta_model <- function(beta) {
  model_spec(paste(
    "Y =~ y1 + y2 + y3",
    "X =~ x1 + x2 + x3",
    "Y ~ b*X",
    sprintf("b == %.17g", beta),
    sep = "\n"
  ))
}

fit_profile_point <- function(beta, theta, data, control,
                              n_theta, n_implied) {
  model <- fixed_beta_model(beta)
  beta_row <- which(
    model$partable$lhs == "Y" &
      model$partable$op == "~" &
      model$partable$rhs == "X"
  )
  beta_index <- model$partable$free[[beta_row]]
  theta[[beta_index]] <- beta
  attempt <- fit_attempt(with_start(model, theta), data, control)
  payload <- fit_payload(attempt, n_theta, n_implied)
  list(payload = payload, theta = if (attempt$returned) {
    attempt$fit$theta
  } else {
    theta
  })
}

trace_branch <- function(grid, source, data, control,
                         theta_columns, n_implied) {
  n_theta <- length(theta_columns)
  source_theta <- as.numeric(source[1L, theta_columns])
  anchor <- which.min(abs(grid - source$beta_hat))
  rows <- vector("list", length(grid))

  anchor_fit <- fit_profile_point(
    grid[[anchor]], source_theta, data, control, n_theta, n_implied
  )
  rows[[anchor]] <- list(
    payload = anchor_fit$payload,
    direction = "anchor"
  )

  theta <- anchor_fit$theta
  if (anchor < length(grid)) {
    for (index in seq.int(anchor + 1L, length(grid))) {
      fit <- fit_profile_point(
        grid[[index]], theta, data, control, n_theta, n_implied
      )
      rows[[index]] <- list(payload = fit$payload, direction = "up")
      if (fit$payload$returned) theta <- fit$theta
    }
  }
  theta <- anchor_fit$theta
  if (anchor > 1L) {
    for (index in seq.int(anchor - 1L, 1L)) {
      fit <- fit_profile_point(
        grid[[index]], theta, data, control, n_theta, n_implied
      )
      rows[[index]] <- list(payload = fit$payload, direction = "down")
      if (fit$payload$returned) theta <- fit$theta
    }
  }

  do.call(rbind, lapply(seq_along(rows), function(index) {
    payload <- rows[[index]]$payload
    data.frame(
      target_beta = grid[[index]],
      direction = rows[[index]]$direction,
      returned = payload$returned,
      audit_converged = payload$audit_converged,
      admissible = payload$admissible,
      eligible = payload$eligible,
      fmin = payload$fmin,
      beta_hat = payload$beta_hat,
      stationarity_inf = payload$stationarity_inf,
      constraint_violation_inf = payload$constraint_violation_inf,
      min_covariance_eigenvalue = payload$min_covariance_eigenvalue,
      rank_signature = payload$rank_signature,
      f_evals = payload$f_evals,
      elapsed_ms = payload$elapsed_ms,
      optimizer_status = payload$optimizer_status,
      error = payload$error,
      stringsAsFactors = FALSE
    )
  }))
}

select_profile_cases <- function(summary, cases_per_n) {
  eligible <- summary[
    summary$random_core & summary$persistent_competing,
    ,
    drop = FALSE
  ]
  rows <- lapply(split(eligible, eligible$n), function(x) {
    default_gap <- ifelse(
      is.finite(x$default_objective_gap),
      x$default_objective_gap,
      -Inf
    )
    x <- x[order(!x$default_best, default_gap, decreasing = TRUE), ]
    head(x, cases_per_n)
  })
  do.call(rbind, rows)
}

local_minima_count <- function(beta, f) {
  ok <- is.finite(beta) & is.finite(f)
  beta <- beta[ok]
  f <- f[ok]
  if (length(f) < 3L) return(NA_integer_)
  order_beta <- order(beta)
  f <- f[order_beta]
  sum(f[2:(length(f) - 1L)] <= f[1:(length(f) - 2L)] &
        f[2:(length(f) - 1L)] <= f[3:length(f)])
}

profile_main <- function() {
  cfg <- parse_profile_args(commandArgs(trailingOnly = TRUE))
  confirmation <- read_profile_result(
    cfg$results_dir, "confirmation_summary.csv"
  )
  confirmed_raw <- read_profile_result(
    cfg$results_dir, "confirmation_raw.csv"
  )
  metadata_frame <- read_profile_result(cfg$results_dir, "metadata.csv")
  metadata <- setNames(metadata_frame$value, metadata_frame$key)
  seed_base <- as.integer(metadata[["seed_base"]])
  cases <- select_profile_cases(confirmation, cfg$cases_per_n)
  if (!nrow(cases)) {
    message("No persistent random-core competing basins to profile.")
    return(invisible(NULL))
  }
  theta_columns <- grep(
    "^theta_[0-9]+$", names(confirmed_raw), value = TRUE
  )
  n_implied <- length(grep(
    "^implied_[0-9]+$", names(confirmed_raw), value = TRUE
  ))
  control <- list(
    max_iter = cfg$max_iter,
    gtol = 1e-9,
    ftol = 1e-14
  )
  rows <- list()
  selected_rows <- list()
  started <- Sys.time()
  for (i in seq_len(nrow(cases))) {
    key <- cases[i, ]
    representatives <- confirmed_raw[
      confirmed_raw$n == key$n &
        confirmed_raw$rep == key$rep &
        confirmed_raw$cluster_representative &
        confirmed_raw$eligible,
      ,
      drop = FALSE
    ]
    representatives <- representatives[
      order(representatives$objective_cluster), ,
      drop = FALSE
    ]
    representatives <- head(representatives, 2L)
    transformed <- asinh(c(representatives$beta_hat, 0.25))
    span <- range(transformed)
    if (diff(span) < 0.5) span <- mean(span) + c(-0.25, 0.25)
    span <- span + c(-0.25, 0.25)
    grid <- sinh(seq(span[[1L]], span[[2L]], length.out = cfg$grid_points))
    grid <- sort(unique(c(grid, representatives$beta_hat, 0.25)))
    x <- simulate_data(
      key$n, seed = dataset_seed(seed_base, key$n, key$rep)
    )
    base_model <- model_spec(model_syntax())
    data <- df_to_data(x, base_model, scaling = "n-1")
    selected_rows[[length(selected_rows) + 1L]] <- data.frame(
      n = key$n,
      rep = key$rep,
      default_best = key$default_best,
      default_objective_gap = key$default_objective_gap,
      confirmed_second_objective_gap = key$confirmed_second_objective_gap,
      branches = nrow(representatives),
      grid_points = length(grid),
      stringsAsFactors = FALSE
    )
    for (branch in seq_len(nrow(representatives))) {
      source <- representatives[branch, ]
      traced <- trace_branch(
        grid, source, data, control, theta_columns, n_implied
      )
      traced$n <- key$n
      traced$rep <- key$rep
      traced$branch <- branch
      traced$source_cluster <- source$objective_cluster
      traced$source_beta <- source$beta_hat
      traced$source_fmin <- source$fmin
      rows[[length(rows) + 1L]] <- traced
    }
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    eta <- elapsed * (nrow(cases) - i) / i
    message(sprintf(
      "  profiles %d/%d, elapsed %.1fs, ETA %.1fs",
      i, nrow(cases), elapsed, eta
    ))
  }
  profiles <- do.call(rbind, rows)
  selected <- do.call(rbind, selected_rows)
  envelope_rows <- lapply(
    split(
      profiles,
      interaction(
        profiles$n, profiles$rep, profiles$target_beta,
        drop = TRUE, lex.order = TRUE
      )
    ),
    function(x) {
      eligible <- x[x$eligible, , drop = FALSE]
      data.frame(
        n = x$n[[1L]],
        rep = x$rep[[1L]],
        target_beta = x$target_beta[[1L]],
        branches = nrow(x),
        eligible_branches = nrow(eligible),
        envelope_fmin =
          if (nrow(eligible)) min(eligible$fmin) else NA_real_,
        stringsAsFactors = FALSE
      )
    }
  )
  envelope <- do.call(rbind, envelope_rows)
  summary_rows <- lapply(
    split(envelope, interaction(envelope$n, envelope$rep, drop = TRUE)),
    function(x) {
      selected_row <- selected[
        selected$n == x$n[[1L]] & selected$rep == x$rep[[1L]], ]
      data.frame(
        n = x$n[[1L]],
        rep = x$rep[[1L]],
        default_best = selected_row$default_best,
        default_objective_gap = selected_row$default_objective_gap,
        confirmed_second_objective_gap =
          selected_row$confirmed_second_objective_gap,
        grid_points = nrow(x),
        eligible_grid_rate = mean(x$eligible_branches > 0L),
        envelope_min = min(x$envelope_fmin, na.rm = TRUE),
        local_envelope_minima =
          local_minima_count(x$target_beta, x$envelope_fmin),
        stringsAsFactors = FALSE
      )
    }
  )
  summary <- do.call(rbind, summary_rows)
  summary <- summary[order(summary$n, summary$rep), ]

  write_csv(profiles, file.path(cfg$results_dir, "profile_raw.csv"))
  write_csv(envelope, file.path(cfg$results_dir, "profile_envelope.csv"))
  write_csv(summary, file.path(cfg$results_dir, "profile_summary.csv"))
  cat("\nProfile summary:\n")
  print(summary, row.names = FALSE, digits = 4)
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  message(
    "Wrote ", nrow(profiles), " branch-profile fits for ",
    nrow(summary), " datasets in ", sprintf("%.1fs", elapsed)
  )
}

profile_main()
