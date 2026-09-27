# Study-local starting points for the two-factor ML model only. A negative
# primitive variance is permitted by ordinary ML, never by the PSD domain.
# On standardized indicators, a unit vector v gives covariance
# I + sign * strength * (v v' - diag(v^2)). With strength=.5 this is PD
# for either sign. Independent factors give a block-diagonal PD full start.
spectral_start <- function(spec, sample, negative = character(),
                           minimum_direction = negative, strength = .5) {
  stopifnot(is.finite(strength), strength > 0, strength < 1,
            all(negative %in% c("X", "Y")), all(minimum_direction %in% c("X", "Y")))
  S <- sample$S[[1]]; sd <- sqrt(diag(S)); names(sd) <- ov_names
  pt <- spec$partable; values <- pt$ustart
  for (factor in c("X", "Y")) {
    rows <- which(pt$lhs == factor & pt$op == "=~")
    obs <- pt$rhs[rows]; idx <- match(obs, ov_names)
    R <- S[idx, idx] / outer(sd[obs], sd[obs]); diag(R) <- 0
    eig <- eigen(R, symmetric = TRUE)
    v <- eig$vectors[, if (factor %in% minimum_direction) length(obs) else 1L]
    if (v[which.max(abs(v))] < 0) v <- -v
    loading <- sd[obs] * v
    marker <- which(pt$free[rows] == 0)
    stopifnot(length(marker) == 1, pt$ustart[rows[marker]] == 1)
    anchor <- loading[marker]
    if (abs(anchor) < 1e-12 * max(sd[obs])) stop("spectral start outside requested marker chart")
    variance <- if (factor %in% negative) -strength else strength
    values[rows] <- loading / anchor
    values[pt$lhs == factor & pt$op == "~~" & pt$rhs == factor] <- variance * anchor^2
    for (j in seq_along(obs)) values[pt$lhs == obs[j] & pt$op == "~~" & pt$rhs == obs[j]] <-
      sd[obs[j]]^2 - variance * loading[j]^2
  }
  values[pt$op == "~"] <- 0
  theta <- numeric(max(pt$free)); theta[pt$free[pt$free > 0]] <- values[pt$free > 0]
  theta
}

start_recipes <- function(spec, sample, ablations = FALSE) {
  x <- list(
    spectral_positive = spectral_start(spec, sample),
    spectral_negative_x = spectral_start(spec, sample, "X"),
    spectral_negative_y = spectral_start(spec, sample, "Y"),
    spectral_negative_xy = spectral_start(spec, sample, c("X", "Y")))
  if (ablations) {
    x$negative_x_positive_direction <- spectral_start(spec, sample, "X", character())
    x$positive_x_negative_direction <- spectral_start(spec, sample, character(), "X")
  }
  x
}

# Attribute absence of a reference using separately retained raw flags. This
# taxonomy describes the screen; it is not a nonattainment classification.
unresolved_inventory <- function(fits, refs) {
  keys <- c("design", "n", "rep", "transform", "domain")
  missing <- refs[refs$domain == "ML" & refs$reference_label == "no_screened_reference", keys]
  rows <- lapply(seq_len(nrow(missing)), function(i) {
    key <- missing[i, ]; same <- rep(TRUE, nrow(fits))
    for (k in keys) same <- same & fits[[k]] == key[[k]]
    d <- fits[same & fits$route == "sphere", ]
    # Equality to screened_extreme means every preceding check passed.
    extreme_only <- sum(d$label == "screened_extreme")
    cbind(key, data.frame(attempts = nrow(d), extent_only_exclusions = extreme_only,
      inventory_label = if (extreme_only) "has_local_candidates_excluded_by_extent" else "no_attempt_passes_local_screen",
      min_finite_objective = if (any(is.finite(d$objective))) min(d$objective[is.finite(d$objective)]) else NA_real_,
      labels = paste(names(table(d$label)), as.integer(table(d$label)), sep = ":", collapse = ";")))
  })
  if (length(rows)) do.call(rbind, rows) else data.frame()
}
