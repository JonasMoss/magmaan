#!/usr/bin/env Rscript
# =============================================================================
# Two-level (multilevel) ML oracle fixtures — Stream D of the multilevel-SEM
# plan. Emits one JSON per model to tests/fixtures/twolevel/, each carrying:
#   * the lavaan model syntax + the raw clustered data (X in ov order +
#     0-based cluster_id) so the C++ golden test can build a
#     data::ClusterSampleStats via Stream B and fit it via Stream C;
#   * lavaan's within/between sample statistics (an independent Stream-B check);
#   * the oracle: theta-hat (per parTable row), standard SEs, chisq, df, and
#     the deviance pieces (logl + unrestricted.logl) behind the LRT.
#
# v1 scope: random-intercept models over a SHARED observed variable set (the
# same observed variables decompose into a within and a between part; no
# between-level-only covariates). lavaan::sem(model, data, cluster="id") is the
# oracle; the UNBALANCED full-information cells are additionally cross-checked
# against Mplus (see regen_oracle_twolevel_mplus.R).
#
# Runnable standalone (Rscript tests/tools/regen_oracle_twolevel.R) or sourced
# from the tail of regen_oracle.R. Generation is the maintainer `just
# regen-oracle` step; CI never invokes R.
# =============================================================================

if (!exists("fixtures")) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
  repo_root <- normalizePath(file.path(script_dir, "..", ".."))
  fixtures <- file.path(repo_root, "tests", "fixtures")
}
suppressMessages({ library(lavaan); library(jsonlite) })

lav_ver <- as.character(packageVersion("lavaan"))
tl_dir <- file.path(fixtures, "twolevel")
dir.create(tl_dir, showWarnings = FALSE, recursive = TRUE)

# ---- model catalogue ------------------------------------------------------
# Each entry: a v1 shared-observed-set two-level model and the observed
# variables (in the order the fixture stores raw columns).
twolevel_models <- list(
  list(
    id = "twolevel_ri3",
    description = "Random-intercept 1-factor CFA, shared indicators y1-y3 (saturated, df=0); exercises theta-hat and SEs.",
    ov = c("y1", "y2", "y3"),
    model = "
      level: 1
        fw =~ y1 + y2 + y3
      level: 2
        fb =~ y1 + y2 + y3
    "
  ),
  list(
    id = "twolevel_1f4",
    description = "1-factor over 4 shared indicators y1-y4 at both levels (df=4).",
    ov = c("y1", "y2", "y3", "y4"),
    model = "
      level: 1
        fw =~ y1 + y2 + y3 + y4
      level: 2
        fb =~ y1 + y2 + y3 + y4
    "
  ),
  list(
    id = "twolevel_2f6",
    description = "Two-factor (y1-y3, y4-y6) over 6 shared indicators at both levels (df=16, well-fitting).",
    ov = paste0("y", 1:6),
    model = "
      level: 1
        fw1 =~ y1 + y2 + y3
        fw2 =~ y4 + y5 + y6
      level: 2
        fb1 =~ y1 + y2 + y3
        fb2 =~ y4 + y5 + y6
    "
  )
)

# ---- data: a compact, parity-stable subset of Demo.twolevel ---------------
data(Demo.twolevel, package = "lavaan")
keep_clusters <- sort(unique(Demo.twolevel$cluster))[1:80]
dat <- Demo.twolevel[Demo.twolevel$cluster %in% keep_clusters, ]

# ---- helpers --------------------------------------------------------------
mat_rows <- function(m) {            # matrix -> list of rows (row-major JSON)
  m <- unname(as.matrix(m))
  lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))
}

param_rows_json <- function(pt) {
  lapply(seq_len(nrow(pt)), function(i) {
    list(
      lhs   = pt$lhs[i],
      op    = pt$op[i],
      rhs   = pt$rhs[i],
      level = as.integer(pt$level[i]),
      block = as.integer(pt$block[i]),
      label = if (is.na(pt$label[i]) || pt$label[i] == "") NA else pt$label[i],
      free  = as.integer(pt$free[i]),
      est   = as.numeric(pt$est[i]),
      se    = as.numeric(pt$se[i])
    )
  })
}

emit_twolevel <- function(spec) {
  ov <- spec$ov
  # Order raw columns in ov order; build 0-based contiguous cluster ids.
  X <- as.matrix(dat[, ov, drop = FALSE])
  cl_raw <- dat$cluster
  cl <- match(cl_raw, unique(cl_raw)) - 1L     # 0-based, appearance order

  fit <- tryCatch(sem(spec$model, data = dat, cluster = "cluster"),
                  error = function(e) e)
  if (inherits(fit, "error")) {
    cat(sprintf("[%s] SKIP — lavaan error: %s\n", spec$id, conditionMessage(fit)))
    return(invisible(NULL))
  }
  converged <- isTRUE(lavInspect(fit, "converged"))
  pt <- parTable(fit)
  fm <- fitMeasures(fit, c("npar", "fmin", "chisq", "df", "pvalue",
                           "logl", "unrestricted.logl", "ntotal"))
  # Two-level sampstat is a list (within first, between/cluster second), each
  # with $cov and $mean; the second block is named after the cluster variable.
  ss <- lavInspect(fit, "sampstat")
  ss_within <- ss[[1]]
  ss_between <- ss[[2]]

  p <- length(ov)
  out <- list(
    id = spec$id,
    meta = list(
      tool = "lavaan::sem(model, data, cluster=)",
      lavaan_version = lav_ver,
      description = spec$description
    ),
    model = spec$model,
    cluster_var = "cluster",
    ov_names = ov,
    p = p,
    n_obs = nrow(X),
    n_clusters = length(unique(cl)),
    data = list(
      X = mat_rows(X),
      cluster_id = as.integer(cl)
    ),
    sampstat = list(
      within_cov   = mat_rows(ss_within$cov),
      between_cov  = mat_rows(ss_between$cov),
      between_mean = as.numeric(ss_between$mean)
    ),
    expected = list(
      converged = converged,
      npar = as.integer(fm[["npar"]]),
      saturated_moments = as.integer(p * (p + 1) + p),
      chisq = as.numeric(fm[["chisq"]]),
      df = as.integer(fm[["df"]]),
      pvalue = as.numeric(fm[["pvalue"]]),
      fmin = as.numeric(fm[["fmin"]]),
      logl = as.numeric(fm[["logl"]]),
      unrestricted_logl = as.numeric(fm[["unrestricted.logl"]]),
      ntotal = as.integer(fm[["ntotal"]]),
      params = param_rows_json(pt)
    )
  )

  out_path <- file.path(tl_dir, paste0(spec$id, ".json"))
  writeLines(toJSON(out, auto_unbox = TRUE, digits = 12, na = "null",
                    pretty = TRUE), out_path)
  cat(sprintf("[%s] wrote %s  (npar=%d chisq=%.3f df=%d p=%.3f, N=%d J=%d)\n",
              spec$id, basename(out_path), out$expected$npar,
              out$expected$chisq, out$expected$df, out$expected$pvalue,
              out$n_obs, out$n_clusters))
}

for (spec in twolevel_models) emit_twolevel(spec)
cat("two-level oracle fixtures ->", tl_dir, "\n")
