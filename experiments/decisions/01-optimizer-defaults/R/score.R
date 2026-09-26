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

# Comparisons whose losses are listed one by one: the candidates of the
# gating rules and the routes. Preconditioning (rule B) is decided by counts.
primary_comparisons <- function(lane) comparisons(lane)[if (lane == "ml-gls") 1 else c(1, 4, 5)]

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
  out <- do.call(rbind, out)
  # Attribution: which other arms (and the witness) solve the same problem.
  lost_keys <- key_of(out, problem_keys)
  others <- s[s$pkey %in% lost_keys, c("pkey", "arm", "success", "fmin")]
  tag <- tapply(seq_len(nrow(others)), others$pkey, function(i)
    paste(sprintf("%s:%s", others$arm[i], ifelse(others$success[i], "ok", "fail")), collapse = ";"))
  out$arms <- unname(tag[lost_keys])
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

# Where the primary arms break invariance: the success pattern across units,
# or the size of the objective spread when every version succeeds.
inconsistency_table <- function(s, lane, g = unit_groups(s)) {
  arms <- unique(unlist(primary_comparisons(lane)))
  b <- g[g$arm %in% arms & !g$consistent, ]
  if (!nrow(b)) return(data.frame(role = character(), groups = integer()))
  b$pattern <- ifelse(b$n_ok == b$n, paste("objective differs,",
    as.character(cut(b$gap, c(1e-6, 1e-5, 1e-4, 1e-3, Inf),
                     c("gap 1e-6 to 1e-5", "gap 1e-5 to 1e-4", "gap 1e-4 to 1e-3", "gap over 1e-3")))),
    b$pattern)
  stats::aggregate(list(groups = b$pattern), b[c("role", "family", "estimator", "arm", "units", "pattern")], length)
}

# Two-stage fits by the stage that returned them, and failures by cause and
# units (PSD lane).
stage_table <- function(s) {
  d <- s[grepl("^twostage", s$arm) & s$attainable, ]
  if (!nrow(d)) return(data.frame())
  count_ok(d, c("role", "family", "arm", "stage"), "success", c("fits", "success"))
}

failure_table <- function(s) {
  d <- s[s$arm != "witness" & !s$success, ]
  d$cause <- ifelse(grepl("round-trip", d$message), "link round-trip",
             ifelse(grepl("covariance-link residual", d$message), "link residual",
             ifelse(grepl("budget", d$message) | d$optimizer_status == "budget_exhausted", "budget exhausted",
             ifelse(nzchar(d$message), "other error",
             ifelse(d$certified, "inadmissible", paste("verdict:", d$newton_status))))))
  stats::aggregate(list(fits = d$pkey), d[c("role", "family", "estimator", "arm", "transform", "cause")], length)
}
