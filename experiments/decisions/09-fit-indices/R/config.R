# Pricing controls only; production configuration belongs to the future criteria.
study_config <- function(mode = "pilot") {
  if (mode == "smoke") {
    cells <- data.frame(model = c("cfa_ptsd8", "growth6", "cfa_ptsd8", "cfa_mtmm9", "cfa_ptsd8"),
      delta = c(0, .1, -.1, -.1, -.1), generator = c("normal", "vm", "normal", "independent", "discretized"),
      N = 200L, estimator = c("ML", "ML", "FIML", "ML", "ML"))
  } else {
    cells <- expand.grid(model = c("cfa_ptsd8", "cfa_mtmm9", "growth6"),
      target_rmsea = c(0, .05), N = c(200L, 1000L), generator = c("normal", "vm"),
      estimator = c("ML", "FIML"), stringsAsFactors = FALSE)
  }
  if (!"delta" %in% names(cells)) cells$delta <- NA_real_
  if (!"target_rmsea" %in% names(cells)) cells$target_rmsea <- NA_real_
  cells$missing_rate <- ifelse(cells$estimator == "FIML", .15, 0)
  cells$skewness <- 1; cells$kurtosis <- 2; cells$categories <- 5L
  cells$cause <- ""; cells$missing_targets <- ""
  cells$id <- sprintf("cell_%03d", seq_len(nrow(cells)))
  cells
}
validate_config <- function(cells) {
  required <- names(study_config("smoke"))
  if (!all(required %in% names(cells)) || anyDuplicated(cells$id) ||
      any(!grepl("^[A-Za-z0-9_]+$", cells$id)) ||
      any(!cells$model %in% basis_ids()) ||
      any(!cells$generator %in% c("normal", "vm", "independent", "discretized")) ||
      any(!cells$estimator %in% c("ML", "FIML")) || any(cells$N < 20) ||
      any(cells$missing_rate < 0 | cells$missing_rate >= 1) ||
      any(cells$estimator == "ML" & cells$missing_rate > 0) ||
      any(!is.finite(cells$delta) & !is.finite(cells$target_rmsea))) stop("Invalid configuration")
  cells
}
