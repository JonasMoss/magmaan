arm_grid <- function() {
  points <- expand.grid(index = c("rmsea", "cfi", "tli", "srmr"),
    arm = c("naive", "lavaan_robust", "misspec_robust"), stringsAsFactors = FALSE)
  intervals <- expand.grid(index = c("rmsea", "cfi", "tli"),
    arm = c("ncc_naive", "ncc_scaled", "wald_f", "wald_sqrt"), stringsAsFactors = FALSE)
  rbind(transform(points, kind = "point"), transform(intervals, kind = "interval"))
}
arm_metadata <- function(estimator) {
  convention <- if (estimator == "FIML") "MLR" else "MLM"
  data.frame(arm = c("naive", "lavaan_robust", "misspec_robust", "ncc_naive", "ncc_scaled", "wald_f", "wald_sqrt"),
    function_call = c("magmaanlab::convention_fit_measures(lab, 'ML')",
      paste0("magmaanlab::convention_fit_measures(lab, '",convention,"'); *.robust"),
      "magmaan::fit_measures(fit)", "magmaanlab::convention_fit_measures(lab, 'ML'); rmsea.ci.*",
      paste0("magmaanlab::convention_fit_measures(lab, '",convention,"'); rmsea.ci.*.scaled"),
      "magmaanlab::fit_index_intervals_misspec(lab, level=.90); variant=wald_f",
      "magmaanlab::fit_index_intervals_misspec(lab, level=.90); variant=wald_sqrt"))
}
run_arms <- function(data, population, cell) {
  out <- arm_grid(); out$estimate <- out$lower <- out$upper <- NA_real_
  out$reason <- "not_attempted"; out$converged <- FALSE
  attempt <- function(code) tryCatch(suppressWarnings(force(code)), error = function(e) e)
  model <- population_spec(population$record, means = TRUE)
  fit <- attempt(magmaan::magmaan(model, data, estimator = cell$estimator))
  if (inherits(fit, "error")) { out$reason <- conditionMessage(fit); return(out) }
  lab <- magmaan::as_lab_fit(fit)
  out$converged <- isTRUE(lab$converged)
  if (!isTRUE(lab$converged)) {
    # The verdict is a ten-field list; row reasons require its scalar status.
    out$reason <- paste0("not_converged: ", lab$verdict$status %||% "library_verdict_false")
    return(out)
  }
  naive <- attempt(magmaanlab::convention_fit_measures(lab, "ML"))
  robust <- attempt(magmaanlab::convention_fit_measures(lab, if (cell$estimator == "FIML") "MLR" else "MLM"))
  policy <- attempt(magmaan::fit_measures(fit))
  wald <- attempt(magmaanlab::fit_index_intervals_misspec(lab, level = .90))
  extract <- function(result, name) {
    if (inherits(result, "error")) return(list(value = NA_real_, reason = conditionMessage(result)))
    row <- result[match(name, result$index), ]
    if (!nrow(row) || is.na(row$index)) return(list(value = NA_real_, reason = "component_not_provided"))
    list(value = row$estimate, reason = if (is.na(row$reason) && !is.finite(row$estimate))
      "nonfinite_component" else row$reason)
  }
  for (j in seq_len(nrow(out))) {
    arm <- out$arm[j]; index <- out$index[j]
    if (out$kind[j] == "point") {
      source <- switch(arm, naive = naive, lavaan_robust = robust, misspec_robust = policy)
      field <- if (arm == "lavaan_robust" && index != "srmr") paste0(index, ".robust") else index
      if (arm == "lavaan_robust" && index == "srmr") {
        out$reason[j] <- "lavaan_has_no_robust_SRMR_correction"; next
      }
      value <- extract(source, field); out$estimate[j] <- value$value; out$reason[j] <- value$reason
    } else if (arm %in% c("ncc_naive", "ncc_scaled")) {
      if (index != "rmsea") { out$reason[j] <- "library_noncentral_intervals_only_for_RMSEA"; next }
      source <- if (arm == "ncc_naive") naive else robust
      suffix <- if (arm == "ncc_naive") "" else ".scaled"
      lo <- extract(source, paste0("rmsea.ci.lower", suffix))
      hi <- extract(source, paste0("rmsea.ci.upper", suffix))
      value <- extract(source, paste0("rmsea",suffix))
      out$estimate[j] <- value$value; out$lower[j] <- lo$value; out$upper[j] <- hi$value
      reasons <- c(lo$reason, hi$reason); reasons <- reasons[!is.na(reasons)]
      out$reason[j] <- if (length(reasons)) paste(unique(reasons),collapse="; ") else NA_character_
    } else {
      if (inherits(wald, "error")) { out$reason[j] <- conditionMessage(wald); next }
      row <- wald[wald$variant == arm & wald$index == index, ]
      if (nrow(row) != 1) { out$reason[j] <- "component_not_provided"; next }
      out$estimate[j] <- row$estimate; out$lower[j] <- row$lower; out$upper[j] <- row$upper
      out$reason[j] <- row$reason
    }
  }
  out
}
