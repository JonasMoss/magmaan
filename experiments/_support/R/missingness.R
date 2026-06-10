# SB-2005 missingness mechanism (MCAR / calibrated MAR), shared across
# experiments. Pure data-masking mechanism, no SEM logic — the sanctioned
# `experiments/_support` home (see experiments/AGENTS.md). Mirrors the same
# mechanism the pairwise-robust-sem paper uses; kept here so experiments do not
# source a paper. Functions: `sb2005_mar`, `sb2005_mcar`, plus internal helpers.

validate_rate <- function(rate) {
  if (!is.numeric(rate) || length(rate) != 1L || is.na(rate) ||
      rate < 0 || rate >= 1) {
    stop("`rate` must be a single number in [0, 1).", call. = FALSE)
  }
  rate
}

as_data_container <- function(data) {
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop("`data` must be a data frame or matrix.", call. = FALSE)
  }
  if (ncol(data) < 1L) {
    stop("`data` must have at least one column.", call. = FALSE)
  }
  data
}

resolve_cols <- function(data, cols, name) {
  p <- ncol(data)
  if (is.character(cols)) {
    matched <- match(cols, colnames(data))
    if (anyNA(matched)) {
      stop("Unknown column in `", name, "`: ",
           paste(cols[is.na(matched)], collapse = ", "), call. = FALSE)
    }
    cols <- matched
  }
  if (!is.numeric(cols) || anyNA(cols)) {
    stop("`", name, "` must be numeric indices or column names.", call. = FALSE)
  }
  cols <- unique(as.integer(cols))
  if (any(cols < 1L | cols > p)) {
    stop("`", name, "` contains columns outside `data`.", call. = FALSE)
  }
  cols
}

require_numeric_cols <- function(data, cols, name) {
  bad <- cols[!vapply(data[, cols, drop = FALSE], is.numeric, logical(1))]
  if (length(bad)) {
    labels <- colnames(data)
    if (is.null(labels)) {
      labels <- paste0("V", seq_len(ncol(data)))
    }
    stop("`", name, "` must reference numeric columns; nonnumeric: ",
         paste(labels[bad], collapse = ", "), call. = FALSE)
  }
}

col_values <- function(data, col) {
  data[, col, drop = TRUE]
}

set_seed_if_requested <- function(seed) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
}

apply_mask <- function(data, mask) {
  out <- data
  for (j in seq_len(ncol(out))) {
    out[mask[, j], j] <- NA
  }
  out
}

pattern_labels <- function(mask) {
  apply(mask, 1L, paste0, collapse = "")
}

missingness_summary <- function(x, target_cols = NULL) {
  if (inherits(x, "pwsem_missingness")) {
    mask <- x$mask
  } else if (is.logical(x)) {
    mask <- x
  } else {
    mask <- is.na(x)
  }
  mask <- as.matrix(mask)
  if (is.null(target_cols)) {
    target_cols <- seq_len(ncol(mask))
  } else {
    target_cols <- resolve_cols(as.data.frame(mask), target_cols, "target_cols")
  }
  target_mask <- mask[, target_cols, drop = FALSE]
  list(
    n = nrow(mask),
    p = ncol(mask),
    target_cols = target_cols,
    overall_rate = mean(target_mask),
    variable_rates = colMeans(mask),
    target_variable_rates = colMeans(target_mask),
    pattern_count = length(unique(pattern_labels(target_mask))),
    pattern_table = sort(table(pattern_labels(target_mask)), decreasing = TRUE)
  )
}

pattern_count <- function(x, target_cols = NULL) {
  missingness_summary(x, target_cols = target_cols)$pattern_count
}

new_missingness_result <- function(data, mask, mechanism, target_cols, details) {
  structure(
    list(
      data = data,
      mask = mask,
      mechanism = mechanism,
      target_cols = target_cols,
      details = details,
      summary = missingness_summary(mask, target_cols = target_cols)
    ),
    class = "pwsem_missingness"
  )
}

print.pwsem_missingness <- function(x, ...) {
  cat("<pwsem_missingness>\n")
  cat("  mechanism: ", x$mechanism, "\n", sep = "")
  cat("  n: ", x$summary$n, ", p: ", x$summary$p, "\n", sep = "")
  cat("  target columns: ", paste(x$target_cols, collapse = ", "), "\n", sep = "")
  cat("  target missing rate: ",
      sprintf("%.3f", x$summary$overall_rate), "\n", sep = "")
  cat("  target patterns: ", x$summary$pattern_count, "\n", sep = "")
  invisible(x)
}

sb2005_mcar <- function(data, rate = 0.15, intact = 1:2, seed = NULL) {
  data <- as_data_container(data)
  rate <- validate_rate(rate)
  intact <- resolve_cols(data, intact, "intact")
  target_cols <- setdiff(seq_len(ncol(data)), intact)
  if (!length(target_cols)) {
    stop("No columns remain after removing `intact` columns.", call. = FALSE)
  }
  set_seed_if_requested(seed)
  mask <- matrix(FALSE, nrow(data), ncol(data), dimnames = dimnames(data))
  mask[, target_cols] <- matrix(
    runif(nrow(data) * length(target_cols)) < rate,
    nrow(data),
    length(target_cols)
  )
  new_missingness_result(
    data = apply_mask(data, mask),
    mask = mask,
    mechanism = "Savalei-Bentler 2005 MCAR",
    target_cols = target_cols,
    details = list(rate = rate, intact = intact)
  )
}

rule_expected_rate <- function(rules, conditional_probs, n, p, target_cols) {
  keep <- matrix(1, n, p)
  for (k in seq_along(rules)) {
    cols <- rules[[k]]$cols
    rows <- rules[[k]]$rows
    if (!length(cols) || !any(rows)) next
    keep[rows, cols] <- keep[rows, cols, drop = FALSE] * (1 - conditional_probs[k])
  }
  mean(1 - keep[, target_cols, drop = FALSE])
}

calibrate_rule_probs <- function(rules, target_rate, target_cols, n, p,
                                 base_overall, calibrate) {
  eligible_rates <- vapply(rules, function(rule) mean(rule$rows), numeric(1))
  raw <- ifelse(eligible_rates > 0, base_overall / eligible_rates, 0)
  raw[!is.finite(raw)] <- 0
  if (!calibrate) {
    return(pmin(1, raw))
  }

  expected_at <- function(scale) {
    rule_expected_rate(rules, pmin(1, scale * raw), n, p, target_cols)
  }
  hi <- 1
  while (expected_at(hi) < target_rate && hi < 1024) {
    hi <- hi * 2
  }
  lo <- 0
  for (i in seq_len(60)) {
    mid <- (lo + hi) / 2
    if (expected_at(mid) < target_rate) lo <- mid else hi <- mid
  }
  pmin(1, hi * raw)
}

apply_row_rules <- function(rules, conditional_probs, n, p) {
  mask <- matrix(FALSE, n, p)
  for (k in seq_along(rules)) {
    rows <- rules[[k]]$rows
    cols <- rules[[k]]$cols
    if (!length(cols) || !any(rows) || conditional_probs[k] <= 0) next
    selected <- rows & (runif(n) < conditional_probs[k])
    mask[selected, cols] <- TRUE
  }
  mask
}

sb2005_mar <- function(data, rate = 0.15, predictors = 1:2, seed = NULL,
                       calibrate = TRUE) {
  data <- as_data_container(data)
  rate <- validate_rate(rate)
  predictors <- resolve_cols(data, predictors, "predictors")
  if (length(predictors) != 2L) {
    stop("`predictors` must contain exactly two columns.", call. = FALSE)
  }
  require_numeric_cols(data, predictors, "predictors")
  p <- ncol(data)
  if (p < 3L) {
    stop("Savalei-Bentler 2005 MAR needs at least three columns.", call. = FALSE)
  }

  x1 <- col_values(data, predictors[1L])
  x2 <- col_values(data, predictors[2L])
  target_cols <- setdiff(seq_len(p), predictors)
  early_cols <- intersect(seq.int(3L, min(7L, p)), target_cols)
  late_cols <- if (p >= 8L) intersect(seq.int(8L, p), target_cols) else integer()
  rules <- list(
    list(rows = x1 + x2 > 0, cols = target_cols),
    list(rows = x1 < 0, cols = early_cols),
    list(rows = x2 < 0, cols = late_cols)
  )
  base_overall <- rate * c(2 / 3, 1 / 3, 1 / 3)
  probs <- calibrate_rule_probs(
    rules = rules,
    target_rate = rate,
    target_cols = target_cols,
    n = nrow(data),
    p = p,
    base_overall = base_overall,
    calibrate = calibrate
  )

  set_seed_if_requested(seed)
  mask <- apply_row_rules(rules, probs, nrow(data), p)
  new_missingness_result(
    data = apply_mask(data, mask),
    mask = mask,
    mechanism = "Savalei-Bentler 2005 MAR",
    target_cols = target_cols,
    details = list(
      rate = rate,
      predictors = predictors,
      conditional_probs = probs,
      calibrated = calibrate,
      rules = c("x1 + x2 > 0", "x1 < 0", "x2 < 0")
    )
  )
}

default_sb2009_targets <- function(data) {
  targets <- c(1L, 4L, 6L, 10L, 13L, 16L)
  targets[targets <= ncol(data)]
}

sb2009_mcar <- function(data, rate = 0.15, target_cols = NULL, seed = NULL) {
  data <- as_data_container(data)
  rate <- validate_rate(rate)
  if (is.null(target_cols)) {
    target_cols <- default_sb2009_targets(data)
  }
  target_cols <- resolve_cols(data, target_cols, "target_cols")
  if (!length(target_cols)) {
    stop("`target_cols` must select at least one column.", call. = FALSE)
  }

  set_seed_if_requested(seed)
  mask <- matrix(FALSE, nrow(data), ncol(data), dimnames = dimnames(data))
  mask[, target_cols] <- matrix(
    runif(nrow(data) * length(target_cols)) < rate,
    nrow(data),
    length(target_cols)
  )
  new_missingness_result(
    data = apply_mask(data, mask),
    mask = mask,
    mechanism = "Savalei-Bentler 2009 MCAR",
    target_cols = target_cols,
    details = list(rate = rate, target_cols = target_cols)
  )
}

default_sb2009_rule_rows <- function(data, predictor_cols, target_cols) {
  rule_id <- seq_along(target_cols)
  predictor <- predictor_cols[((rule_id - 1L) %% length(predictor_cols)) + 1L]
  lapply(seq_along(target_cols), function(i) {
    z <- col_values(data, predictor[i])
    qs <- stats::quantile(z, probs = c(.25, .40, .50, .60, .75),
                          names = FALSE, type = 8, na.rm = TRUE)
    rows <- switch(
      ((i - 1L) %% 5L) + 1L,
      z > qs[3L],
      z < qs[3L],
      z > qs[2L],
      z < qs[4L],
      z >= qs[1L] & z <= qs[5L]
    )
    list(rows = rows, cols = target_cols[i], predictor = predictor[i])
  })
}

sb2009_mar <- function(data, rate = 0.15, target_cols = NULL,
                       predictor_cols = NULL, seed = NULL) {
  data <- as_data_container(data)
  rate <- validate_rate(rate)
  if (is.null(target_cols)) {
    target_cols <- default_sb2009_targets(data)
  }
  target_cols <- resolve_cols(data, target_cols, "target_cols")
  if (!length(target_cols)) {
    stop("`target_cols` must select at least one column.", call. = FALSE)
  }
  if (is.null(predictor_cols)) {
    predictor_cols <- setdiff(seq_len(ncol(data)), target_cols)
  }
  predictor_cols <- resolve_cols(data, predictor_cols, "predictor_cols")
  if (!length(predictor_cols)) {
    stop("`predictor_cols` must include at least one observed predictor.",
         call. = FALSE)
  }
  require_numeric_cols(data, predictor_cols, "predictor_cols")

  rules <- default_sb2009_rule_rows(data, predictor_cols, target_cols)
  eligible_rates <- vapply(rules, function(rule) mean(rule$rows), numeric(1))
  probs <- ifelse(eligible_rates > 0, rate / eligible_rates, 0)
  probs <- pmin(1, probs)

  set_seed_if_requested(seed)
  mask <- apply_row_rules(rules, probs, nrow(data), ncol(data))
  new_missingness_result(
    data = apply_mask(data, mask),
    mask = mask,
    mechanism = "Savalei-Bentler 2009-style MAR",
    target_cols = target_cols,
    details = list(
      rate = rate,
      target_cols = target_cols,
      predictor_cols = vapply(rules, `[[`, integer(1), "predictor"),
      conditional_probs = probs,
      note = paste(
        "The 2009 paper reports five independent MAR rules but does not print",
        "the exact thresholds; this uses five threshold rules assigned across",
        "the target variables to recreate the high-pattern-count design."
      )
    )
  )
}
