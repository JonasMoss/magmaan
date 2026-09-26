# Scoring. A fit succeeds when the verdict certifies it (and, for PSD fits,
# the estimate is admissible). A problem is attainable when any arm or the
# population-start witness succeeds; only attainable problems score. Its best
# known objective is the lowest successful objective; reaching it within
# 1e-6 (1 + |f|) is the secondary outcome.

problem_keys <- c("lane", "family", "role", "pop", "model", "n", "rep", "transform", "estimator")
group_keys <- setdiff(problem_keys, "transform")

comparisons <- function(lane) {
  switch(lane,
    "ml-gls" = list(c("layered_port", "default"), c("layered_lbfgs", "default"),
                    c("default_port", "default"), c("layered_port", "layered_lbfgs")),
    "psd-ml" = list(c("psd_layered", "psd_default"), c("psd_default_none", "psd_default"),
                    c("psd_layered_none", "psd_layered"),
                    c("twostage_default", "psd_default"), c("twostage_layered", "psd_default"),
                    c("twostage_layered", "twostage_default")))
}

primary_comparisons <- function(lane) comparisons(lane)[if (lane == "ml-gls") 1 else 1:5]

key_of <- function(d, keys) do.call(paste, c(d[keys], sep = "|"))

score_rows <- function(raw) {
  raw$success <- raw$certified & (raw$lane != "psd-ml" | raw$admissible %in% TRUE)
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

paired_table <- function(s, lane, by = c("lane", "role", "family", "estimator")) {
  out <- list()
  for (cmp in comparisons(lane)) {
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

loss_table <- function(s, lane) {
  out <- list()
  cols <- c(problem_keys, "seed", "fmin", "newton_status", "optimizer_status", "f_evals",
            "start_policy", "stage", "admissible", "message")
  for (cmp in primary_comparisons(lane)) {
    cand <- s[s$arm == cmp[1] & s$attainable, ]
    base <- s[s$arm == cmp[2] & s$attainable, c("pkey", "success", "fmin")]
    m <- merge(cand, base, by = "pkey", suffixes = c("", "_baseline"))
    lost <- m[!m$success & m$success_baseline, ]
    if (nrow(lost)) out[[length(out) + 1L]] <- cbind(
      data.frame(candidate = cmp[1], baseline = cmp[2]), lost[cols],
      fmin_baseline = lost$fmin_baseline, best_known = lost$best_known)
  }
  if (!length(out)) return(data.frame(candidate = character(), baseline = character()))
  do.call(rbind, out)
}

# Same draw, different units: the same success, and the same objective when
# every transform succeeds. Common units (x100, x0.01) and separate units
# (mixed) are checked apart.
invariance_table <- function(s) {
  arms <- s[s$arm != "witness", ]
  out <- list()
  for (set in list(common = c("native", "x100", "x0.01"), separate = c("native", "mixed"))) {
    d <- arms[arms$transform %in% set, ]
    g <- split(d, key_of(d, c(group_keys, "arm")))
    g <- g[vapply(g, function(x) all(set %in% x$transform), logical(1))]
    if (!length(g)) next
    rows <- lapply(g, function(x) {
      same <- length(unique(x$success)) == 1L
      f <- x$fmin[x$success]
      if (same && all(x$success)) same <- diff(range(f)) <= 1e-6 * (1 + abs(min(f)))
      cbind(x[1, c("lane", "role", "family", "estimator", "arm")], consistent = same)
    })
    r <- do.call(rbind, rows)
    agg <- stats::aggregate(consistent ~ lane + role + family + estimator + arm, r,
                            function(v) c(groups = length(v), consistent = sum(v)))
    agg <- cbind(agg[1:5], as.data.frame(agg$consistent))
    agg$units <- if (identical(set, c("native", "mixed"))) "separate" else "common"
    out[[length(out) + 1L]] <- agg
  }
  do.call(rbind, out)
}

# PSD tolerance check: fits that fail only in rescaled units, by cause.
tolerance_table <- function(s) {
  d <- s[s$lane == "psd-ml" & s$arm != "witness", ]
  if (!nrow(d)) return(data.frame())
  native <- d[d$transform == "native", c(group_keys, "arm", "success")]
  names(native)[names(native) == "success"] <- "success_native"
  m <- merge(d[d$transform != "native", ], native, by = c(group_keys, "arm"))
  m$cause <- ifelse(grepl("covariance-link residual", m$message), "link_feasibility",
             ifelse(m$certified & !(m$admissible %in% TRUE), "inadmissible_certified",
             ifelse(!m$returned, "error", "verdict")))
  lost <- m[m$success_native & !m$success, ]
  if (!nrow(lost)) return(data.frame(transform = character(), arm = character(), cause = character(), fits = integer()))
  stats::aggregate(list(fits = lost$pkey), lost[c("transform", "arm", "cause")], length)
}

# The rules of criteria/<lane>.md, evaluated on test families.
decision_table <- function(s, lane) {
  p <- paired_table(s, lane)
  p <- p[p$role == "test", ]
  row <- function(rule, scope, value, pass) data.frame(rule = rule, scope = scope,
    value = value, pass = pass, stringsAsFactors = FALSE)
  pair_rows <- function(rule, cand, base, best_gates) {
    d <- p[p$candidate == cand & p$baseline == base, ]
    do.call(rbind, lapply(seq_len(nrow(d)), function(i) with(d[i, ], row(rule,
      paste(family, estimator),
      sprintf("wins %d losses %d; best wins %d losses %d", wins, losses, best_wins, best_losses),
      wins >= losses && (!best_gates || best_wins >= best_losses)))))
  }
  if (lane == "ml-gls") {
    inv <- invariance_table(s)
    inv <- inv[inv$role == "test" & inv$arm %in% c("layered_port", "default"), ]
    w <- reshape(inv[c("family", "estimator", "units", "arm", "consistent")],
                 idvar = c("family", "estimator", "units"), timevar = "arm", direction = "wide")
    r2 <- do.call(rbind, lapply(seq_len(nrow(w)), function(i) with(w[i, ], row("2 invariance",
      paste(family, estimator, units), sprintf("candidate %d baseline %d",
      consistent.layered_port, consistent.default), consistent.layered_port >= consistent.default))))
    return(rbind(pair_rows("1 primary", "layered_port", "default", FALSE), r2))
  }
  rates <- rate_table(s, c("role", "arm"))
  secs <- function(a) rates$total_seconds[rates$role == "test" & rates$arm == a]
  tol <- tolerance_table(s)
  bites <- if (nrow(tol)) sum(tol$fits[tol$cause %in% c("link_feasibility", "inadmissible_certified")]) else 0L
  rbind(pair_rows("A layered cold start", "psd_layered", "psd_default", TRUE),
        pair_rows("B no preconditioning", "psd_default_none", "psd_default", TRUE),
        row("B no preconditioning", "total test seconds",
            sprintf("none %.0f diagonal %.0f", secs("psd_default_none"), secs("psd_default")),
            secs("psd_default_none") < secs("psd_default")),
        row("D tolerance bites", "rescaled-only failures from feasibility or admissibility",
            sprintf("%d fits", bites), bites == 0L))
}
