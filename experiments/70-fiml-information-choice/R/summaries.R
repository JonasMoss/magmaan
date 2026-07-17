fiml_information_summaries <- function(raw, grid) {
  raw <- merge(raw, grid, by = "cell_id", all.x = TRUE, sort = FALSE)
  raw <- raw[order(raw$cell_id, raw$rep, raw$parameter), ]

  summaries <- list()
  k <- 0L
  for (cell_id in unique(raw$cell_id)) {
    for (parameter in fiml_information_parameters$parameter) {
      block <- raw[
        raw$cell_id == cell_id & raw$parameter == parameter, , drop = FALSE]
      if (!nrow(block)) next
      for (m in seq_len(nrow(fiml_information_methods))) {
        method <- fiml_information_methods[m, ]
        se_name <- paste0("se_", method$method)
        available <- block$fit_ok & is.finite(block$estimate) &
          is.finite(block[[se_name]]) & block[[se_name]] >= 0
        used <- block[available, , drop = FALSE]
        n_method <- nrow(used)
        empirical_sd <- if (n_method >= 2L) sd(used$estimate) else NA_real_
        mean_estimate <- if (n_method) mean(used$estimate) else NA_real_
        mean_se <- if (n_method) mean(used[[se_name]]) else NA_real_
        centered_coverage <- NA_real_
        if (n_method >= 2L) {
          center_loo <- (n_method * mean_estimate - used$estimate) /
            (n_method - 1L)
          centered_coverage <- mean(
            abs(used$estimate - center_loo) <= 1.96 * used[[se_name]])
        }
        correct <- identical(block$specification[[1L]], "correct")
        true_coverage <- if (correct && n_method) mean(
          abs(used$estimate - used$target) <= 1.96 * used[[se_name]]
        ) else NA_real_
        left_miss <- if (correct && n_method) mean(
          used$estimate + 1.96 * used[[se_name]] < used$target
        ) else NA_real_
        right_miss <- if (correct && n_method) mean(
          used$estimate - 1.96 * used[[se_name]] > used$target
        ) else NA_real_

        k <- k + 1L
        summaries[[k]] <- data.frame(
          cell_id = cell_id,
          distribution = block$distribution[[1L]],
          specification = block$specification[[1L]],
          missingness = block$missingness[[1L]],
          n = block$n[[1L]],
          parameter = parameter,
          target_type = if (correct) "known_population" else "leave_one_out_center",
          method = method$method,
          information = method$information,
          covariance = method$covariance,
          reps_requested = length(unique(block$rep)),
          reps_fit = sum(block$fit_ok),
          reps_method = n_method,
          fit_failure_rate = mean(!block$fit_ok),
          method_failure_rate = 1 - n_method / nrow(block),
          mean_estimate = mean_estimate,
          target = if (correct) block$target[[1L]] else NA_real_,
          bias = if (correct && n_method)
            mean_estimate - block$target[[1L]] else NA_real_,
          empirical_sd = empirical_sd,
          mean_se = mean_se,
          se_sd_ratio = mean_se / empirical_sd,
          coverage_true = true_coverage,
          coverage_centered = centered_coverage,
          left_miss_true = left_miss,
          right_miss_true = right_miss,
          coverage_mcse = if (n_method && is.finite(true_coverage))
            sqrt(true_coverage * (1 - true_coverage) / n_method) else NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  cell_method <- do.call(rbind, summaries)

  replication <- raw[!duplicated(raw[c("cell_id", "rep")]), , drop = FALSE]
  replication$error[is.na(replication$error)] <- ""
  replication$warning[is.na(replication$warning)] <- ""
  timing <- aggregate(
    cbind(fit_seconds, inference_seconds) ~
      cell_id + distribution + specification + missingness + n,
    data = replication,
    FUN = function(x) c(
      mean = mean(x, na.rm = TRUE),
      median = median(x, na.rm = TRUE),
      p90 = unname(quantile(x, 0.90, na.rm = TRUE)))
  )
  timing <- do.call(data.frame, timing)
  names(timing) <- gsub(".", "_", names(timing), fixed = TRUE)

  failures <- replication[
    !replication$fit_ok | !replication$comparison_ok |
      (!is.na(replication$error) & nzchar(replication$error)),
    c("cell_id", "rep", "seed", "stage", "error"), drop = FALSE]
  warnings <- replication[
    !is.na(replication$warning) & nzchar(replication$warning),
    c("cell_id", "rep", "seed", "warning"), drop = FALSE]

  information_failures <- list()
  j <- 0L
  for (key in c("expected", "observed_h1", "observed_hessian")) {
    ok_name <- paste0(key, "_ok")
    for (cell_id in unique(replication$cell_id)) {
      block <- replication[
        replication$cell_id == cell_id, , drop = FALSE]
      j <- j + 1L
      information_failures[[j]] <- data.frame(
        cell_id = cell_id,
        distribution = block$distribution[[1L]],
        specification = block$specification[[1L]],
        missingness = block$missingness[[1L]],
        n = block$n[[1L]],
        information = key,
        reps = nrow(block),
        inversion_failure_rate = mean(!block[[ok_name]]),
        stringsAsFactors = FALSE
      )
    }
  }

  list(
    cell_method = cell_method,
    timing = timing,
    failures = failures,
    warnings = warnings,
    information_failures = do.call(rbind, information_failures)
  )
}
