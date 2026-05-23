#!/usr/bin/env Rscript

source("pilot/scripts/pilot-lib.R")

pilot_ensure_dirs()
pilot_require_magmaan()
pilot_require("jsonlite")

limit <- pilot_env_int("PILOT_TEXTBOOK_N", 24L)
max_iter <- pilot_env_int("PILOT_MAX_ITER", 5000L)
optimizers <- pilot_env_list(
  "PILOT_OPTIMIZERS",
  c("lbfgs", "nlopt-lbfgs", "nlopt-var2", "port")
)
control <- list(max_iter = max_iter, ftol = 1e-10, gtol = 1e-7)

manifest_cases <- pilot_case_index()
strict_ids <- vapply(manifest_cases, function(x) x$id, character(1))
strict_ids <- strict_ids[grepl("^(little|newsom)::", strict_ids)]

load_source_cases <- function(source, file) {
  payload <- pilot_read_json(file.path(pilot_repo_root(), "tests", "fixtures",
                                       source, file))
  out <- payload$cases
  for (i in seq_along(out)) {
    out[[i]]$source <- source
    out[[i]]$full_id <- paste(source, out[[i]]$id, sep = "::")
  }
  out
}

cases <- c(
  load_source_cases("little", "continuous_reference.json"),
  load_source_cases("newsom", "continuous_reference.json")
)
cases <- Filter(function(x) x$full_id %in% strict_ids, cases)
cases <- cases[seq_len(min(length(cases), limit))]

rows <- list()
k <- 0L
for (case in cases) {
  ss <- pilot_sample_stats_from_case(case)
  spec <- pilot_spec_from_case(case)
  estimator <- as.character(case$estimator %||% "ML")
  lavaan_f <- as.numeric((case$lavaan %||% list())$fmin %||% NA_real_)
  lavaan_target <- if (is.finite(lavaan_f)) 2 * lavaan_f else NA_real_

  for (optimizer in optimizers) {
    message("textbook case=", case$full_id, " optimizer=", optimizer)
    fit <- pilot_fit_once(spec, ss, optimizer, control, estimator = estimator)
    k <- k + 1L
    row <- pilot_result_row(
      data.frame(
        source = case$source,
        design = case$full_id,
        reference = case$name %||% case$label %||% case$id,
        replicate = NA_integer_,
        seed = NA_integer_,
        n = as.integer(case$n_obs),
        p = length(unlist(case$ov_names, use.names = FALSE)),
        estimator = estimator,
        start_policy = "magmaan_r_default",
        optimizer = optimizer,
        lavaan_fmin_target = lavaan_target,
        stringsAsFactors = FALSE
      ),
      fit
    )
    row$fmin_abs_diff_lavaan <- abs(row$fmin - row$lavaan_fmin_target)
    rows[[k]] <- row
  }
}

raw <- pilot_bind(rows)
pilot_write_csv(raw, pilot_path("results", "textbook_subset_raw.csv"))
