## magmaan R bindings -- Mplus-style WLSMV ordinal invariance helper.
##
## Run from the repo root after installing the package:
##     Rscript r-package/examples/mplus_wlsmv_invariance.R
##
## The all-ordinal case pins the helper against the small Mplus Demo DIFFTEST
## probe checked into tests/fixtures/mplus_wlsmv_invariance. The mixed case
## exercises the same helper and the mixed ordinal restriction-map bridge on a
## deterministic listwise mixed continuous/ordinal model.

suppressMessages(library(magmaan))
ctrl <- list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8)

fixture <- function(name) {
  read.csv(file.path("tests", "fixtures", "mplus_wlsmv_invariance", name),
           stringsAsFactors = FALSE)
}

assert_table <- function(got, ref, tol = 5e-6) {
  m <- merge(got, ref, by = "h0", suffixes = c(".got", ".ref"))
  stopifnot(nrow(m) == nrow(ref))
  for (nm in c("T_diff", "df_diff", "scale_c", "T_scaled", "p_scaled",
               "T_scaled_shifted", "df_scaled_shifted", "p_scaled_shifted")) {
    g <- m[[paste0(nm, ".got")]]
    r <- m[[paste0(nm, ".ref")]]
    stopifnot(max(abs(g - r)) < tol)
  }
  invisible(m)
}

make_delta_df <- function(n_total = 500L, seed = 20260619L,
                          missing_rate = .50) {
  set.seed(seed)
  ov <- paste0("y", 1:6)
  thresholds <- c(-1.3, -0.47, 0.47, 1.3)
  draw_group <- function(n, group_label) {
    eta <- rnorm(n)
    aux <- 0.5 * eta + sqrt(1 - 0.5^2) * rnorm(n)
    z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
    for (j in seq_along(ov)) {
      z[, j] <- 0.7 * eta + rnorm(n, sd = sqrt(1 - 0.7^2))
    }
    out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
      ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
    }))
    names(out) <- ov
    out$aux <- aux
    out$grp <- group_label
    out
  }
  n_group <- as.integer(n_total / 2L)
  dat <- rbind(draw_group(n_group, "A"), draw_group(n_group, "B"))
  b_rows <- which(dat$grp == "B")
  target <- as.integer(round(missing_rate * length(b_rows)))
  ranks <- rank(dat$aux[b_rows], ties.method = "random")
  weights <- pmax(length(b_rows) - ranks, 0)
  for (nm in ov[5:6]) {
    picked <- sample.int(length(b_rows), size = target, replace = FALSE,
                         prob = weights)
    dat[[nm]][b_rows[picked]] <- NA
  }
  dat$grp <- factor(dat$grp, levels = c("A", "B"))
  dat
}

ov <- paste0("y", 1:6)
model <- paste0("f =~ ", paste(ov, collapse = " + "))
all_ord <- mplus_wlsmv_invariance(
  model, make_delta_df(), ordered = ov, group = "grp",
  steps = c("metric", "scalar"), missing = "pairwise", control = ctrl)
all_ref <- fixture("all_ordinal_pairwise.csv")
assert_table(all_ord$tests, all_ref)
stopifnot(max(abs(all_ord$tests$T_scaled_shifted - all_ref$mplus_difftest)) < 1e-3)
stopifnot(max(abs(all_ord$tests$p_scaled_shifted - all_ref$mplus_p)) < 1e-3)

make_mixed_df <- function(n = 350L, seed = 44L) {
  set.seed(seed)
  draw <- function(n, grp, shift = 0, scale = 1) {
    eta <- rnorm(n, mean = shift, sd = scale)
    e <- matrix(rnorm(n * 3), n, 3)
    z1 <- 0.8 * eta + sqrt(1 - 0.8^2) * e[, 1]
    z2 <- 0.7 * eta + sqrt(1 - 0.7^2) * e[, 2]
    z3 <- 0.6 * eta + sqrt(1 - 0.6^2) * e[, 3]
    y4 <- 0.65 * eta + rnorm(n, sd = sqrt(1 - 0.65^2))
    data.frame(
      x1 = ordered(cut(z1, c(-Inf, -.5, .4, Inf), labels = FALSE)),
      x2 = ordered(cut(z2, c(-Inf, -.7, .2, Inf), labels = FALSE)),
      x3 = ordered(cut(z3, c(-Inf, -.4, .7, Inf), labels = FALSE)),
      y4 = y4,
      grp = grp)
  }
  out <- rbind(draw(n, "A"), draw(n, "B", shift = .25, scale = 1.15))
  out$grp <- factor(out$grp, levels = c("A", "B"))
  out
}

mixed <- mplus_wlsmv_invariance(
  "f =~ x1 + x2 + x3 + y4", make_mixed_df(),
  ordered = c("x1", "x2", "x3"), group = "grp",
  steps = c("metric", "scalar"), missing = "listwise", control = ctrl)
assert_table(mixed$tests, fixture("mixed_listwise.csv"))
stopifnot(!mixed$options$all_ordinal)

cat("Mplus-style WLSMV invariance helper: ok\n")
