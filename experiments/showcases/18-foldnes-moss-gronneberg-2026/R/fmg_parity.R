# semTests parity for the FMG eigenvalue-test family.
#
# semTests (Moss 2024) is the paper's own implementation of the penalized
# eigenvalue block-averaging tests. magmaan exposes the same single-model
# machinery through infer_fmg_test(chi2_source, df, eigvals, method, param):
# given a base statistic (T_ML or T_RLS), the model df, and the UGamma
# eigenvalue spectrum, it applies the chosen eigenvalue transform and returns
# the Imhof tail p-value. This file maps a semTests test name onto that call so
# the two implementations can be compared p-value for p-value.

# Accessor: prefer the core wrapper if the R package has wired it, else the
# exported Rcpp binding.
fmg_test_fn <- function() {
  core <- magmaan::magmaan_core
  if ("robust_fmg_test" %in% ls(core)) return(core$robust_fmg_test)
  get("infer_fmg_test", envir = asNamespace("magmaan"))
}

# Parse a semTests test name into magmaan's (method, param, chisq, unbiased),
# mirroring semTests:::split_input exactly (default chisq = "rls", default
# pEBA/pOLS param = 2, "ug" => unbiased Gamma).
fmg_spec_from_name <- function(name) {
  parts <- strsplit(tolower(name), "_", fixed = TRUE)[[1L]]
  type <- parts[[1L]]
  unbiased <- 1L
  chisq <- "rls"
  if (length(parts) == 3L) {
    unbiased <- if (parts[[2L]] == "ug") 2L else 1L
    chisq <- parts[[3L]]
  } else if (length(parts) == 2L) {
    if (parts[[2L]] %in% c("rls", "ml")) chisq <- parts[[2L]]
    else if (parts[[2L]] == "ug") unbiased <- 2L
  }
  param <- 2
  if (startsWith(type, "peba")) {
    method <- "peba"; v <- substring(type, 5L)
    if (nzchar(v)) param <- as.numeric(v)
  } else if (startsWith(type, "pols")) {
    method <- "pols"; v <- substring(type, 5L)
    if (nzchar(v)) param <- as.numeric(v)
  } else {
    method <- switch(type, sb = "sb", ss = "ss", sf = "scaled_f",
                     all = "all", pall = "penalized_all", std = "standard",
                     stop("unsupported FMG test name: ", name, call. = FALSE))
  }
  list(method = method, param = param, chisq = chisq, unbiased = unbiased)
}

# magmaan p-value for one semTests test name, using the cached spectrum/stats.
magmaan_fmg_pvalue <- function(name, cached, fmg = fmg_test_fn()) {
  sp <- fmg_spec_from_name(name)
  chi2 <- if (sp$chisq == "ml") cached$T_ML else cached$T_RLS
  ev <- if (sp$unbiased == 2L) cached$ev_unbiased else cached$ev_biased
  if (is.null(ev) || !is.finite(chi2)) return(NA_real_)
  res <- tryCatch(fmg(chi2, as.integer(cached$df), ev,
                      method = sp$method, param = sp$param),
                  error = function(e) NULL)
  if (is.null(res)) NA_real_ else res$p_value
}

# The single-model FMG family compared here: 8 methods x {T_ML, T_RLS} x
# {biased, unbiased Gamma}. Names follow semTests' vocabulary.
fmg_test_names <- function() {
  bases <- c("sb", "ss", "sf", "all", "pall", "peba2", "peba4", "peba6", "pols")
  grid <- expand.grid(base = bases, gamma = c("", "ug"), chisq = c("ml", "rls"),
                      stringsAsFactors = FALSE)
  apply(grid, 1L, function(r) {
    paste(c(r[["base"]], if (nzchar(r[["gamma"]])) r[["gamma"]], r[["chisq"]]),
          collapse = "_")
  })
}

# Compare magmaan vs semTests for one lavaan fit (rep-1 data of a cell).
# Returns a long data.frame: one row per test name.
fmg_parity_rows <- function(fit_lavaan, cached, test_names = fmg_test_names()) {
  st <- tryCatch(semTests::pvalues(fit_lavaan, tests = test_names),
                 error = function(e) e)
  if (inherits(st, "error")) {
    return(data.frame(test = test_names, magmaan = NA_real_, semtests = NA_real_,
                      abs_diff = NA_real_, error = conditionMessage(st),
                      stringsAsFactors = FALSE))
  }
  mg <- vapply(test_names, magmaan_fmg_pvalue, double(1), cached = cached)
  st <- as.numeric(st)                       # order-preserving with test_names
  data.frame(test = test_names, magmaan = mg, semtests = st,
             abs_diff = abs(mg - st), error = NA_character_,
             stringsAsFactors = FALSE)
}
