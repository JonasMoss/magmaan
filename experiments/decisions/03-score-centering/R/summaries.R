wilson_interval <- function(success, total) {
  if (!total) return(c(NA_real_, NA_real_))
  z <- qnorm(.975); p <- success / total
  center <- (p + z^2 / (2 * total)) / (1 + z^2 / total)
  half <- z * sqrt(p * (1 - p) / total + z^2 / (4 * total^2)) / (1 + z^2 / total)
  c(center - half, center + half)
}

summary_groups <- function(x, columns) {
  split(seq_len(nrow(x)), do.call(paste, c(x[columns], sep = '|')))
}

centering_summaries <- function(raw, run_kind, bootstrap_reps = 400L) {
  columns <- c('family', 'lane', 'test', 'n', 'role', 'geometry', 'arm')
  rates <- covariance <- list()
  for (at in summary_groups(raw, columns)) {
    x <- raw[at, ]; key <- x[1, columns]
    for (method in c('sb', 'peba4')) {
      p <- x[[paste0('p_', method)]]; ok <- x$test_available & is.finite(p)
      ci <- wilson_interval(sum(p[ok] < .05), sum(ok))
      rates[[length(rates) + 1L]] <- cbind(key, method = method, draws = nrow(x),
        usable = sum(ok), unavailable = sum(!ok), rejection = if (any(ok)) mean(p[ok] < .05) else NA_real_,
        lower = ci[1], upper = ci[2])
    }
    ok <- x$covariance_available & is.finite(x$variance)
    known <- ok & !is.na(x$coverage)
    ci <- wilson_interval(sum(x$coverage[known]), sum(known))
    oracle <- ok & is.finite(x$oracle_variance)
    covariance[[length(covariance) + 1L]] <- cbind(key, draws = nrow(x), usable = sum(ok),
      unavailable = sum(!ok), known_targets = sum(known),
      coverage = if (any(known)) mean(x$coverage[known]) else NA_real_,
      lower = ci[1], upper = ci[2], mean_variance = if (any(ok)) mean(x$variance[ok]) else NA_real_,
      oracle_ratio = if (any(oracle)) mean(x$variance[oracle] / x$oracle_variance[oracle]) else NA_real_,
      empirical_mse_ratio = if (any(oracle)) mean((x$estimate[oracle] - x$target[oracle])^2 /
        x$oracle_variance[oracle]) else NA_real_,
      max_covariance_gap = if (any(ok)) max(x$covariance_gap[ok]) else NA_real_,
      max_score_mean = if (any(ok)) max(x$score_mean_norm[ok]) else NA_real_,
      max_group_mean = if (any(ok)) max(x$group_mean_norm[ok]) else NA_real_)
  }
  rates <- do.call(rbind, rates); covariance <- do.call(rbind, covariance)
  paired <- list()
  pair_columns <- c('family', 'lane', 'test', 'n', 'geometry')
  set.seed(173904L)
  for (at in summary_groups(raw, pair_columns)) {
    x <- raw[at, ]; key <- x[1, pair_columns]
    for (arm in setdiff(unique(x$arm), 'raw')) for (method in c('sb', 'peba4')) {
      pcol <- paste0('p_', method)
      pair <- function(role) {
        r <- x[x$role == role & x$arm == 'raw', c('rep', pcol, 'test_available')]
        c <- x[x$role == role & x$arm == arm, c('rep', pcol, 'test_available')]
        m <- merge(r, c, by = 'rep', suffixes = c('_raw', '_candidate'))
        ok <- m$test_available_raw & m$test_available_candidate &
          is.finite(m[[paste0(pcol, '_raw')]]) & is.finite(m[[paste0(pcol, '_candidate')]])
        m[ok, c(paste0(pcol, '_raw'), paste0(pcol, '_candidate')), drop = FALSE]
      }
      nul <- pair('null'); alt <- pair('power')
      nn <- nrow(nul); na <- nrow(alt)
      size_diff <- power_diff <- lower <- upper <- loss_lower <- NA_real_
      raw_power <- candidate_power <- raw_cut <- candidate_cut <- NA_real_
      if (nn) {
        size_diff <- abs(mean(nul[, 2] < .05) - .05) - abs(mean(nul[, 1] < .05) - .05)
        cuts <- apply(nul, 2, quantile, probs = .05, type = 1, names = FALSE)
        raw_cut <- cuts[1]; candidate_cut <- cuts[2]
        if (na) {
          raw_power <- mean(alt[, 1] <= cuts[1]); candidate_power <- mean(alt[, 2] <= cuts[2])
          power_diff <- candidate_power - raw_power
        }
        boot <- replicate(bootstrap_reps, {
          bnull <- nul[sample.int(nn, nn, replace = TRUE), , drop = FALSE]
          sd <- abs(mean(bnull[, 2] < .05) - .05) - abs(mean(bnull[, 1] < .05) - .05)
          pd <- NA_real_
          if (na) {
            balt <- alt[sample.int(na, na, replace = TRUE), , drop = FALSE]
            bc <- apply(bnull, 2, quantile, probs = .05, type = 1, names = FALSE)
            pd <- mean(balt[, 2] <= bc[2]) - mean(balt[, 1] <= bc[1])
          }
          c(sd, pd)
        })
        bounds <- quantile(boot[1, ], c(.025, .975), names = FALSE)
        lower <- bounds[1]; upper <- bounds[2]
        if (na) loss_lower <- quantile(boot[2, ], .025, names = FALSE)
      }
      nr <- x[x$role == 'null', ]; pr <- x[x$role == 'power', ]
      extra <- function(d) sum(!d$test_available[d$arm == arm]) - sum(!d$test_available[d$arm == 'raw'])
      paired[[length(paired) + 1L]] <- cbind(key, arm = arm, method = method,
        null_pairs = nn, power_pairs = na, extra_unavailable = extra(nr) + extra(pr),
        absolute_size_error_difference = size_diff, size_error_lower = lower, size_error_upper = upper,
        raw_null_cutoff = raw_cut, candidate_null_cutoff = candidate_cut,
        raw_matched_power = raw_power, candidate_matched_power = candidate_power,
        matched_power_difference = power_diff, power_difference_lower = loss_lower)
    }
  }
  paired <- do.call(rbind, paired)
  decisions <- list()
  decision_columns <- c('lane', 'geometry', 'arm')
  for (at in summary_groups(paired, decision_columns)) {
    x <- paired[at, ]; key <- x[1, decision_columns]
    r <- rates[rates$role == 'null' & rates$lane == key$lane & rates$geometry == key$geometry &
      rates$arm == key$arm, ]
    v <- covariance[covariance$lane == key$lane & covariance$geometry == key$geometry &
      covariance$arm == key$arm & covariance$known_targets > 0, ]
    enough <- all(x$null_pairs >= 1800 & x$power_pairs >= 1800)
    no_failures <- all(x$extra_unavailable <= 0)
    calibrated <- all(r$lower >= .03 & r$upper <= .07) &&
      all(is.finite(x$size_error_upper) & x$size_error_upper < .01)
    power_ok <- all(is.finite(x$power_difference_lower) & x$power_difference_lower > -.02)
    covered <- nrow(v) > 0 && all(v$lower >= .92 & v$upper <= .98)
    group_v <- v[v$family == 'grouped_mean' & v$n == 300, ]
    oracle_ok <- !nrow(group_v) || all(group_v$oracle_ratio >= .98 & group_v$oracle_ratio <= 1.02)
    covariance_benefit <- key$arm == 'group' && nrow(group_v) > 0
    score_benefit <- covariance_benefit || any(x$size_error_upper < -.01, na.rm = TRUE)
    for (component in c('parameter_covariance', 'score_calibration')) {
      eligible <- enough && no_failures && if (component == 'parameter_covariance')
        covered && oracle_ok && covariance_benefit else calibrated && power_ok && score_benefit
      status <- if (run_kind != 'confirmation') 'open_pilot_only' else if (!enough)
        'open_insufficient_pairs' else if (eligible) 'eligible_in_tested_scope' else 'not_eligible'
      decisions[[length(decisions) + 1L]] <- cbind(key, component = component,
        enough_pairs = enough, no_extra_failures = no_failures, calibrated = calibrated,
        coverage_ok = covered, oracle_ok = oracle_ok, matched_power_ok = power_ok,
        benefit = if (component == 'parameter_covariance') covariance_benefit else score_benefit,
        status = status)
    }
  }
  list(rates = rates, covariance = covariance, paired = paired, decisions = do.call(rbind, decisions))
}
