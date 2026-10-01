# Harness validation on one draw per case and generator (n = 300): magmaan's
# statistics and p-values against lavaan, plus the internal identities the run
# relies on. Needs lavaan; the simulation itself does not.
lavaan_fit <- function(pop, syntax, d, ...) {
  fn <- if (identical(pop$lavaan_function, "growth")) lavaan::growth else lavaan::sem
  fn(syntax, data = d, meanstructure = pop$meanstructure, fixed.x = FALSE,
     group = if (length(pop$groups) > 1L) "group" else NULL, ...)
}

# Positions, among lavaan's equality rows, of the constraints H0 adds to H1, so
# lavTestScore releases exactly the tested restrictions (H1's own label and
# identification equalities stay imposed).
added_constraints <- function(pop, fit0) {
  added <- setdiff(trimws(strsplit(pop$h0, "\n")[[1]]), trimws(strsplit(pop$h1, "\n")[[1]]))
  added <- strsplit(added[grepl("==", added, fixed = TRUE)], "\\s*==\\s*")
  pt <- lavaan::parTable(fit0)
  eq <- pt[pt$op == "==", ]
  hit <- vapply(added, function(a) which(eq$lhs == a[1] & eq$rhs == a[2])[1], integer(1))
  if (anyNA(hit)) stop("could not locate the added constraints in the lavaan partable")
  hit
}

validate_case <- function(pop, dgp, seed) {
  cals <- calibrate_population(pop, dgp)
  d <- draw_sample(pop, cals, 300L, dgp, seed)
  X <- raw_blocks(pop, d)
  f1 <- fit_one(pop, pop$h1, d)
  sizes <- block_sizes(pop, d)
  g <- global_record(f1, d, X, magmaanlab::policy_inference(f1, d), sizes)
  tests <- c("standard", "browne.residual.nt.model", "satorra.bentler", "mean.var.adjusted")
  l1 <- lavaan_fit(pop, pop$h1, d, test = tests)
  lt <- lavaan::lavTest(l1, test = tests)
  names(lt) <- tests
  chk <- function(name, ours, ref, tol) data.frame(case = pop$id, dgp = dgp, check = name,
    magmaan = ours, reference = ref, abs_diff = abs(ours - ref), tol = tol,
    passed = is.finite(ours) && is.finite(ref) && abs(ours - ref) <= tol * max(1, abs(ref)),
    stringsAsFactors = FALSE)
  out <- list(
    chk("gof LR = lavaan standard", g[["stat_lr"]], lt$standard$stat, 1e-5),
    chk("gof RLS = lavaan browne.residual.nt.model", g[["stat_rls"]], lt$browne.residual.nt.model$stat, 1e-5),
    chk("gof p LR-SB = lavaan satorra.bentler", g[["p_lr_biased_sb"]], lt$satorra.bentler$pvalue, 1e-4),
    chk("gof p LR-MV = lavaan mean.var.adjusted", g[["p_lr_biased_mv"]], lt$mean.var.adjusted$pvalue, 1e-4),
    chk("gof own LR spectrum = shared biased spectrum", g[["check"]], 0, 1e-8),
    chk("gof score = squared norm of its row sums", g[["rows_check"]], 0, 1e-8))
  if (!pop$meanstructure || pop$id %in% c("cfa_long18"))
    out[[length(out) + 1L]] <- chk("gof score = RLS (means absent or saturated)", g[["stat_score"]], g[["stat_rls"]], 1e-6)
  if (!is.null(pop$h0)) {
    f0 <- fit_one(pop, pop$h0, d)
    v <- nested_record(f1, f0, d, X, sizes)
    l0 <- lavaan_fit(pop, pop$h0, d, test = tests)
    lt0 <- lavaan::lavTest(l0, test = tests)
    names(lt0) <- tests
    l0_plain <- lavaan_fit(pop, pop$h0, d)
    # magmaan's restriction map is the exact parameter nesting; lavaan's default
    # A.method = "delta" is a Jacobian column-space approximation of it.
    lrt <- lavaan::lavTestLRT(l1, l0, method = "satorra.2000", test = "satorra.bentler",
                              scaled_shifted = FALSE, A.method = "exact")
    out <- c(out, list(
      chk("nested LR = lavaan chisq difference", v[["stat_lr"]],
          lavaan::fitMeasures(l0, "chisq") - lavaan::fitMeasures(l1, "chisq"), 1e-5),
      chk("nested score = lavTestScore", v[["stat_score"]],
          lavaan::lavTestScore(l0_plain, univariate = FALSE,
                               release = added_constraints(pop, l0_plain))$test$X2, 1e-5),
      chk("nested RLS difference = lavaan difference", v[["stat_rls"]],
          lt0$browne.residual.nt.model$stat - lt$browne.residual.nt.model$stat, 1e-5),
      chk("nested p LR-SB = lavaan satorra.2000 (mean scaled)", v[["p_lr_biased_sb"]],
          lrt[["Pr(>Chisq)"]][2], 1e-3),
      chk("nested policy LR = restriction-map T_diff", v[["check"]], 0, 1e-8),
      chk("nested score = squared norm of its row sums", v[["rows_check"]], 0, 1e-8)))
  }
  do.call(rbind, out)
}

validate_all <- function(pops, seed_base) {
  out <- list()
  for (i in seq_along(pops)) for (dgp in c("normal", "disc")) {
    r <- tryCatch(validate_case(pops[[i]], dgp, seed_base + 100L * i),
                  error = function(e) data.frame(case = pops[[i]]$id, dgp = dgp, check = "validation error",
                    magmaan = NA, reference = NA, abs_diff = NA, tol = NA, passed = FALSE,
                    stringsAsFactors = FALSE, note = conditionMessage(e))[, 1:8])
    out[[length(out) + 1L]] <- r
  }
  do.call(rbind, out)
}
