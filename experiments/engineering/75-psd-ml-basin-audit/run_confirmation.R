#!/usr/bin/env Rscript

.confirmation_script <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(
      sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE
    ))
  }
  normalizePath("run_confirmation.R", mustWork = TRUE)
}
source(file.path(dirname(.confirmation_script()), "run_experiment.R"))
rm(.confirmation_script)

confirmation_usage <- function() cat(
  "Usage: Rscript run_confirmation.R [options]\n\n",
  "Restart one representative from every candidate objective cluster under\n",
  "a tighter, 20,000-iteration PSD-SLSQP solve.\n\n",
  "Options:\n",
  "  --results-dir PATH --max-cases N --max-iter N\n",
  sep = "")

parse_confirmation_args <- function(args) {
  out <- list(
    results_dir = experiment_path("results", "pilot"),
    max_cases = 0L,
    max_iter = 20000L
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
      confirmation_usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--results-dir") {
      out$results_dir <- take()
    } else if (arg == "--max-cases") {
      out$max_cases <- as.integer(take())
    } else if (arg == "--max-iter") {
      out$max_iter <- as.integer(take())
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  out$results_dir <- normalizePath(out$results_dir, mustWork = TRUE)
  if (anyNA(c(out$max_cases, out$max_iter)) ||
      out$max_cases < 0L || out$max_iter < 1L) {
    stop("max-cases must be nonnegative and max-iter positive",
         call. = FALSE)
  }
  out
}

read_result <- function(root, name) {
  utils::read.csv(
    file.path(root, name),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

confirmation_main <- function() {
  cfg <- parse_confirmation_args(commandArgs(trailingOnly = TRUE))
  raw <- read_result(cfg$results_dir, "raw.csv")
  datasets <- read_result(cfg$results_dir, "dataset_summary.csv")
  candidates <- datasets[
    datasets$objective_basin_class == "competing" |
      datasets$best_equivalence_class ==
        "equal-objective-different-moments",
    ,
    drop = FALSE
  ]
  candidates <- candidates[order(candidates$n, candidates$rep), ]
  if (cfg$max_cases > 0L) {
    candidates <- head(candidates, cfg$max_cases)
  }
  if (!nrow(candidates)) {
    message("No confirmation candidates.")
    write_csv(
      candidates, file.path(cfg$results_dir, "confirmation_summary.csv")
    )
    return(invisible(NULL))
  }

  candidate_keys <- paste(candidates$n, candidates$rep, sep = ":")
  raw_keys <- paste(raw$n, raw$rep, sep = ":")
  representatives <- raw[
    raw_keys %in% candidate_keys &
      raw$cluster_representative & raw$eligible,
    ,
    drop = FALSE
  ]
  moment_candidates <- candidates[
    candidates$best_equivalence_class ==
      "equal-objective-different-moments",
    c("n", "rep"),
    drop = FALSE
  ]
  if (nrow(moment_candidates)) {
    moment_keys <- paste(
      moment_candidates$n, moment_candidates$rep, sep = ":"
    )
    moment_rows <- raw[
      raw_keys %in% moment_keys & raw$best_hit & raw$eligible,
      ,
      drop = FALSE
    ]
    farthest <- do.call(
      rbind,
      lapply(
        split(
          moment_rows,
          interaction(
            moment_rows$n, moment_rows$rep,
            drop = TRUE, lex.order = TRUE
          )
        ),
        function(x) x[which.max(x$implied_distance_to_best), , drop = FALSE]
      )
    )
    representatives <- rbind(representatives, farthest)
    representatives <- representatives[
      !duplicated(representatives[c("n", "rep", "start_id")]),
      ,
      drop = FALSE
    ]
  }
  representatives <- representatives[
    order(
      representatives$n, representatives$rep,
      representatives$objective_cluster
    ),
  ]
  theta_columns <- grep("^theta_[0-9]+$", names(raw), value = TRUE)
  n_theta <- length(theta_columns)
  implied_columns <- grep("^implied_[0-9]+$", names(raw), value = TRUE)
  n_implied <- length(implied_columns)
  model <- model_spec(model_syntax())
  control <- list(
    max_iter = cfg$max_iter,
    gtol = 1e-9,
    ftol = 1e-14
  )

  rows <- vector("list", nrow(representatives))
  started <- Sys.time()
  for (i in seq_len(nrow(representatives))) {
    source <- representatives[i, ]
    x <- simulate_data(source$n, seed = source$seed)
    data <- df_to_data(x, model, scaling = "n-1")
    theta <- as.numeric(source[1L, theta_columns])
    attempt <- fit_attempt(with_start(model, theta), data, control)
    payload <- fit_payload(attempt, n_theta, n_implied)
    row <- payload_row(
      payload, "", source$n, source$rep, source$seed,
      source$random_core, source$failure_enrichment,
      source$extreme_enrichment,
      paste0("confirm_cluster_", source$objective_cluster),
      "high_budget_restart", source$start_id, 0,
      source$parameter_distance_to_best
    )
    row$source_objective_cluster <- source$objective_cluster
    row$source_start_id <- source$start_id
    row$source_fmin <- source$fmin
    row$source_beta <- source$beta_hat
    row$source_objective_gap <- source$objective_gap
    rows[[i]] <- row
    if (i %% 25L == 0L || i == nrow(representatives)) {
      elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      eta <- elapsed * (nrow(representatives) - i) / i
      message(sprintf(
        "  confirmation %d/%d, elapsed %.1fs, ETA %.1fs",
        i, nrow(representatives), elapsed, eta
      ))
    }
  }
  confirmed_raw <- do.call(rbind, rows)
  confirmed_datasets <- classify_all(confirmed_raw)
  confirmed_raw <- attach_cluster_telemetry(
    confirmed_raw, confirmed_datasets
  )
  original <- candidates[, c(
    "n", "rep", "random_core", "failure_enrichment",
    "extreme_enrichment", "n_objective_clusters",
    "objective_basin_class", "best_equivalence_class",
    "default_best", "default_objective_gap",
    "second_objective_gap"
  )]
  names(original)[names(original) == "n_objective_clusters"] <-
    "original_objective_clusters"
  names(original)[names(original) == "objective_basin_class"] <-
    "original_basin_class"
  names(original)[names(original) == "best_equivalence_class"] <-
    "original_equivalence_class"
  confirmed <- confirmed_datasets[, c(
    "n", "rep", "n_starts", "n_eligible", "n_objective_clusters",
    "objective_basin_class", "best_equivalence_class",
    "second_objective_gap", "best_fmin", "best_beta",
    "best_boundary", "best_rank_signature"
  )]
  names(confirmed)[names(confirmed) == "n_starts"] <-
    "representatives_restarted"
  names(confirmed)[names(confirmed) == "n_eligible"] <-
    "confirmed_eligible"
  names(confirmed)[names(confirmed) == "n_objective_clusters"] <-
    "confirmed_objective_clusters"
  names(confirmed)[names(confirmed) == "objective_basin_class"] <-
    "confirmed_basin_class"
  names(confirmed)[names(confirmed) == "best_equivalence_class"] <-
    "confirmed_equivalence_class"
  names(confirmed)[names(confirmed) == "second_objective_gap"] <-
    "confirmed_second_objective_gap"
  summary <- merge(original, confirmed, by = c("n", "rep"), all.x = TRUE)
  summary$persistent_competing <-
    summary$confirmed_basin_class == "competing"
  summary$all_representatives_eligible <-
    summary$confirmed_eligible == summary$representatives_restarted

  write_csv(
    confirmed_raw, file.path(cfg$results_dir, "confirmation_raw.csv")
  )
  write_csv(
    summary, file.path(cfg$results_dir, "confirmation_summary.csv")
  )
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  cat("\nHigh-budget confirmation by N:\n")
  print(
    aggregate(
      cbind(
        persistent_competing,
        all_representatives_eligible
      ) ~ n,
      summary,
      mean
    ),
    row.names = FALSE,
    digits = 4
  )
  message(
    "Wrote ", nrow(confirmed_raw), " representative restarts for ",
    nrow(summary), " datasets in ", sprintf("%.1fs", elapsed)
  )
}

confirmation_main()
