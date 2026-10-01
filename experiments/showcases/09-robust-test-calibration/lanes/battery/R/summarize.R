# Per-cell rejection rates. A replicate counts for a test when its fits
# converged and every arm returned a p-value (common valid draws), so arms are
# compared on the same samples.
summarize_cells <- function(x) {
  pcols <- grep("^p_", names(x), value = TRUE)
  x$valid <- !nzchar(x$status) & stats::complete.cases(x[, pcols])
  subsets <- list(converged = x$valid, admissible = x$valid & x$admissible %in% TRUE)
  out <- list()
  for (sn in names(subsets)) {
    y <- x[subsets[[sn]], ]
    if (!nrow(y)) next
    key <- interaction(y$case, y$dgp, y$n, y$test, drop = TRUE)
    for (k in split(seq_len(nrow(y)), key)) {
      z <- y[k, ]
      for (col in pcols) {
        p <- z[[col]]
        out[[length(out) + 1L]] <- data.frame(
          subset = sn, case = z$case[1], dgp = z$dgp[1], n = z$n[1], test = z$test[1],
          df = stats::median(z$df), arm = sub("^p_", "", col), valid = length(p),
          reject_01 = mean(p < .01), reject_05 = mean(p < .05), reject_10 = mean(p < .10),
          stringsAsFactors = FALSE)
      }
    }
  }
  s <- do.call(rbind, out)
  parts <- do.call(rbind, strsplit(s$arm, "_", fixed = TRUE))
  s$base <- parts[, 1]; s$spectrum <- parts[, 2]; s$method <- parts[, 3]
  s
}

summarize_failures <- function(x) {
  pcols <- grep("^p_", names(x), value = TRUE)
  x$outcome <- ifelse(nzchar(x$status), x$status,
                      ifelse(stats::complete.cases(x[, pcols]), "valid", "some arm unavailable"))
  x$outcome <- sub(":.*$", "", x$outcome)
  stats::aggregate(rep ~ case + dgp + n + test + outcome, data = x, FUN = length)
}

# Per arm, generator and test: rejection at 5% summarized over cases and n.
summarize_arms <- function(s) {
  key <- interaction(s$subset, s$test, s$dgp, s$arm, drop = TRUE)
  do.call(rbind, lapply(split(s, key), function(z) data.frame(
    subset = z$subset[1], test = z$test[1], dgp = z$dgp[1], arm = z$arm[1],
    base = z$base[1], spectrum = z$spectrum[1], method = z$method[1], cells = nrow(z),
    mean_abs_dev_05 = mean(abs(z$reject_05 - .05)), median_reject_05 = stats::median(z$reject_05),
    min_reject_05 = min(z$reject_05), max_reject_05 = max(z$reject_05),
    stringsAsFactors = FALSE)))
}
