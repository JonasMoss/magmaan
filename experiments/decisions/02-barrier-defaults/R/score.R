# Scoring. A fit succeeds when the verdict certifies it and it is not a
# runaway: standardized extent above `std_bound` or marker-chart extent above
# `chart_bound` (criteria/barrier-ml.md). A problem is attainable when any arm
# or the population-start witness succeeds; only attainable problems score.
# Its best known objective is the lowest successful penalized objective;
# reaching it within 1e-6 (1 + |f|) is rule 2. The table helpers are adapted
# from decisions/01-optimizer-defaults. "estimator" holds the penalty weight.

problem_keys <- c("lane", "family", "role", "pop", "model", "n", "rep", "transform", "estimator")
group_keys <- setdiff(problem_keys, "transform")
candidate_arms <- c("layered_port", "default_lbfgs", "layered_lbfgs")

comparisons <- function(lane = "barrier-ml", candidate = candidate_arms)
  c(lapply(candidate_arms, function(a) c(a, "default")),
    list(c("layered_lbfgs", "layered_port"), c("layered_lbfgs", "default_lbfgs")))

primary_comparisons <- function(lane = "barrier-ml", candidate = candidate_arms)
  lapply(candidate, function(a) c(a, "default"))

key_of <- function(d, keys) do.call(paste, c(d[keys], sep = "|"))

score_rows <- function(raw, std_bound = 10, chart_bound = 1000) {
  raw$runaway_std <- raw$certified & !is.na(raw$std_extent) & raw$std_extent > std_bound
  raw$runaway_chart <- raw$certified & !is.na(raw$chart_extent) & raw$chart_extent > chart_bound
  raw$runaway <- raw$runaway_std | raw$runaway_chart
  raw$success <- raw$certified & !raw$runaway
  raw$pkey <- key_of(raw, problem_keys)
  ok <- raw[raw$success & is.finite(raw$fmin), ]
  best <- tapply(ok$fmin, ok$pkey, min)
  raw$attainable <- raw$pkey %in% names(best)
  raw$best_known <- unname(best[raw$pkey])
  raw$best <- raw$success & raw$attainable &
    raw$fmin <= raw$best_known + 1e-6 * (1 + abs(raw$best_known))
  raw
}

rate_table <- function(s, by) {
  arms <- s[s$arm != "witness", ]
  split_key <- key_of(arms, by)
  rows <- lapply(split(arms, split_key), function(d) {
    a <- d[d$attainable, ]
    cbind(d[1, by, drop = FALSE], data.frame(
      problems = nrow(d), attainable = nrow(a), success = sum(a$success),
      best = sum(a$best), success_unattainable = sum(d$success & !d$attainable),
      median_seconds = stats::median(d$seconds, na.rm = TRUE),
      total_seconds = sum(d$seconds, na.rm = TRUE),
      median_evals_success = stats::median(a$f_evals[a$success], na.rm = TRUE)))
  })
  out <- do.call(rbind, rows); rownames(out) <- NULL
  out[do.call(order, out[by]), ]
}

paired_table <- function(s, lane, by = c("lane", "role", "family", "estimator"),
                         candidate = candidate_arms) {
  out <- list()
  for (cmp in comparisons(lane, candidate)) {
    cand <- s[s$arm == cmp[1] & s$attainable, ]
    base <- s[s$arm == cmp[2] & s$attainable, ]
    m <- merge(cand[c(by, "pkey", "success", "best")], base[c("pkey", "success", "best")],
               by = "pkey", suffixes = c("_c", "_b"))
    for (d in split(m, key_of(m, by))) out[[length(out) + 1L]] <- cbind(d[1, by, drop = FALSE],
      data.frame(candidate = cmp[1], baseline = cmp[2], attainable = nrow(d),
                 wins = sum(d$success_c & !d$success_b), losses = sum(!d$success_c & d$success_b),
                 best_wins = sum(d$best_c & !d$best_b), best_losses = sum(!d$best_c & d$best_b)))
  }
  out <- do.call(rbind, out); rownames(out) <- NULL
  out
}

loss_table <- function(s, lane, candidate = candidate_arms) {
  out <- list()
  cols <- c(problem_keys, "seed", "fmin", "newton_status", "optimizer_status", "f_evals",
            "start_policy", "message")
  cols <- c(cols, "std_extent", "chart_extent", "runaway", "start_repaired")
  cmps <- primary_comparisons(lane, candidate)
  for (cmp in cmps) {
    cand <- s[s$arm == cmp[1] & s$attainable, ]
    base <- s[s$arm == cmp[2] & s$attainable, c("pkey", "success", "fmin")]
    m <- merge(cand, base, by = "pkey", suffixes = c("", "_baseline"))
    lost <- m[!m$success & m$success_baseline, ]
    if (nrow(lost)) out[[length(out) + 1L]] <- cbind(
      data.frame(candidate = cmp[1], baseline = cmp[2]), lost[cols],
      fmin_baseline = lost$fmin_baseline, best_known = lost$best_known)
  }
  if (!length(out)) return(data.frame(candidate = character(), baseline = character()))
  out <- do.call(rbind, out)
  # Attribution: which other arms (and the witness) solve the same problem.
  lost_keys <- key_of(out, problem_keys)
  others <- s[s$pkey %in% lost_keys, c("pkey", "arm", "success", "fmin")]
  # Solved (+) or not (-) by: d default, lp layered_port, dl default_lbfgs,
  # ll layered_lbfgs, w witness.
  short <- c(default = "d", layered_port = "lp", default_lbfgs = "dl", layered_lbfgs = "ll", witness = "w")
  tag <- tapply(seq_len(nrow(others)), others$pkey, function(i)
    paste0(short[others$arm[i]], ifelse(others$success[i], "+", "-"), collapse = " "))
  out$arms <- unname(tag[lost_keys])
  out$message <- substr(sub("^magmaan fit error ", "", out$message), 1, 60)
  # Every baseline is `default`; lane and start repairs are constant here.
  out$baseline <- NULL; out$lane <- NULL; out$start_repaired <- NULL
  # Rounded so the tracked file stays small; raw batches keep full precision.
  for (k in c("fmin", "fmin_baseline", "best_known", "std_extent", "chart_extent"))
    out[[k]] <- signif(out[[k]], 8)
  out
}

# Same draw, different units: the same success, and the same objective when
# every version succeeds. Common units (x100, x0.01) and separate units (mixed)
# are checked apart. `gap` is the relative objective spread when every version
# succeeds; at N = 25 the verdict's accuracy budget alone allows about 1e-6.
unit_groups <- function(s) {
  arms <- s[s$arm != "witness", ]
  out <- list()
  for (units in c("common", "separate")) {
    set <- if (units == "common") c("native", "x100", "x0.01") else c("native", "mixed")
    d <- arms[arms$transform %in% set, ]
    k <- key_of(d, c(group_keys, "arm"))
    complete <- tapply(d$transform, k, function(t) all(set %in% t))
    keep <- complete[k]
    d <- d[keep, ]; k <- k[keep]
    n_ok <- tapply(d$success, k, sum); n <- tapply(d$success, k, length)
    fs <- ifelse(d$success, d$fmin, NA_real_)
    lo <- suppressWarnings(tapply(fs, k, min, na.rm = TRUE))
    hi <- suppressWarnings(tapply(fs, k, max, na.rm = TRUE))
    first <- d[!duplicated(k), c("lane", "role", "family", "estimator", "arm")]
    kk <- k[!duplicated(k)]
    g <- cbind(first, units = units, n = as.integer(n[kk]), n_ok = as.integer(n_ok[kk]),
               gap = ifelse(n_ok[kk] == n[kk], (hi[kk] - lo[kk]) / (1 + abs(lo[kk])), NA_real_))
    pat <- tapply(paste0(d$transform, ":", ifelse(d$success, "ok", "fail"))[order(match(d$transform, set))],
                  k[order(match(d$transform, set))], paste, collapse = " ")
    g$pattern <- unname(pat[kk])
    g$consistent <- (g$n_ok == 0L) | (g$n_ok == g$n & g$gap <= 1e-6)
    out[[units]] <- g
  }
  do.call(rbind, out)
}

count_ok <- function(x, by, flag, names = c("groups", "consistent")) {
  a <- stats::aggregate(list(v = x[[flag]]), x[by], function(v) c(length(v), sum(v)))
  out <- cbind(a[by], as.data.frame(a$v))
  names(out)[length(by) + 1:2] <- names
  out
}

invariance_table <- function(s, g = unit_groups(s)) {
  count_ok(g, c("lane", "role", "family", "estimator", "arm", "units"), "consistent")
}

count_ok <- function(x, by, flag, names = c("groups", "consistent")) {
  a <- stats::aggregate(list(v = x[[flag]]), x[by], function(v) c(length(v), sum(v)))
  out <- cbind(a[by], as.data.frame(a$v))
  names(out)[length(by) + 1:2] <- names
  out
}

invariance_table <- function(s, g = unit_groups(s)) {
  count_ok(g, c("lane", "role", "family", "estimator", "arm", "units"), "consistent")
}

# Rules 1 to 3 of criteria/barrier-ml.md for one candidate, test families only.
decision_table <- function(s, lane, candidate, p = paired_table(s, lane),
                           inv = invariance_table(s)) {
  p <- p[p$role == "test" & p$candidate == candidate & p$baseline == "default", ]
  row <- function(rule, scope, value, pass) data.frame(rule = rule, scope = scope,
    value = value, pass = pass, stringsAsFactors = FALSE)
  r12 <- do.call(rbind, lapply(seq_len(nrow(p)), function(i) with(p[i, ], rbind(
    row("1 primary", paste(family, estimator), sprintf("wins %d losses %d", wins, losses),
        wins >= losses),
    row("2 best known", paste(family, estimator),
        sprintf("best wins %d losses %d", best_wins, best_losses), best_wins >= best_losses)))))
  inv <- inv[inv$role == "test" & inv$arm %in% c(candidate, "default"), ]
  inv$arm <- ifelse(inv$arm == candidate, "candidate", "baseline")
  w <- reshape(inv[c("family", "estimator", "units", "arm", "consistent")],
               idvar = c("family", "estimator", "units"), timevar = "arm", direction = "wide")
  r3 <- do.call(rbind, lapply(seq_len(nrow(w)), function(i) with(w[i, ], row("3 invariance",
    paste(family, estimator, units), sprintf("candidate %d baseline %d",
    consistent.candidate, consistent.baseline), consistent.candidate >= consistent.baseline))))
  rbind(r12, r3)
}

# The pre-registered choice: eligible candidates pass rules 1 to 3 everywhere;
# most pooled certified test problems wins, then most best known, then the
# smaller change (the order of candidate_arms).
choice_table <- function(s, lane, candidates = candidate_arms, p = paired_table(s, lane),
                         inv = invariance_table(s)) {
  rates <- rate_table(s, c("role", "arm"))
  rows <- lapply(seq_along(candidates), function(i) {
    cd <- candidates[i]; d <- decision_table(s, lane, cd, p, inv)
    data.frame(candidate = cd, eligible = all(d$pass), failed_rules = sum(!d$pass),
      success = sum(rates$success[rates$role == "test" & rates$arm == cd]),
      best = sum(rates$best[rates$role == "test" & rates$arm == cd]), order = i)
  })
  out <- do.call(rbind, rows)
  el <- out[out$eligible, ]
  pick <- if (nrow(el)) el$candidate[order(-el$success, -el$best, el$order)][1] else "none (author decides)"
  out$choice <- pick
  base <- rates[rates$role == "test" & rates$arm == "default", ]
  out$default_success <- sum(base$success); out$default_best <- sum(base$best)
  out$order <- NULL
  out
}

# Where the primary arms break invariance.
inconsistency_table <- function(s, lane, g = unit_groups(s), candidate = candidate_arms) {
  arms <- c("default", candidate)
  b <- g[g$arm %in% arms & !g$consistent, ]
  if (!nrow(b)) return(data.frame(role = character(), groups = integer()))
  b$pattern <- ifelse(b$n_ok == b$n, paste("objective differs,",
    as.character(cut(b$gap, c(1e-6, 1e-5, 1e-4, 1e-3, Inf),
                     c("gap 1e-6 to 1e-5", "gap 1e-5 to 1e-4", "gap 1e-4 to 1e-3", "gap over 1e-3")))),
    b$pattern)
  stats::aggregate(list(groups = b$pattern), b[c("role", "family", "estimator", "arm", "units", "pattern")], length)
}

failure_table <- function(s) {
  d <- s[s$arm != "witness" & !s$success, ]
  if (!nrow(d)) return(data.frame())
  d$cause <- ifelse(d$runaway_chart, "marker pole",
             ifelse(d$runaway_std, "standardized runaway",
             ifelse(grepl("barrier", d$message), "no start inside the barrier",
             ifelse(grepl("budget", d$message) | d$optimizer_status == "budget_exhausted", "budget exhausted",
             ifelse(nzchar(d$message), "other error", paste("verdict:", d$newton_status))))))
  stats::aggregate(list(fits = d$pkey), d[c("role", "family", "estimator", "arm", "transform", "cause")], length)
}

# Certified fits beyond several bounds of both extents, by arm (unit-free, so
# every transform counts).
runaway_table <- function(s) {
  d <- s[s$certified, ]
  if (!nrow(d)) return(data.frame())
  by <- c("lane", "role", "family", "estimator", "arm")
  out <- stats::aggregate(list(certified = d$certified), d[by], length)
  for (b in c(5, 10, 100)) out[[paste0("std_beyond_", b)]] <-
    stats::aggregate(list(x = !is.na(d$std_extent) & d$std_extent > b), d[by], sum)$x
  for (b in c(10, 100, 1000)) out[[paste0("chart_beyond_", b)]] <-
    stats::aggregate(list(x = !is.na(d$chart_extent) & d$chart_extent > b), d[by], sum)$x
  out
}

# Start repairs (the fitter pulls a start inside the barrier domain), by arm.
repair_table <- function(s) {
  d <- s[s$returned, ]
  if (!nrow(d)) return(data.frame())
  count_ok(transform(d, rep_flag = start_repaired %in% TRUE),
           c("role", "family", "estimator", "arm"), "rep_flag", c("fits", "repaired"))
}
