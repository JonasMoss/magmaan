#!/usr/bin/env Rscript
# regen_oracle.R --- regenerate fixture JSON from lavaan
#
# Reads tests/fixtures/corpus.json and emits one fixture per layer per
# corpus entry. Today this script handles the `flat` layer
# (lavaan::lavParseModelString output). `ptable`, `matrix_rep`, and `fit`
# layers are added in P3+ as the C++ side gains the matching consumers.
#
# Usage:
#   Rscript tools/regen_oracle.R                 # regenerate flat/*.flat.json
#   Rscript tools/regen_oracle.R --check         # diff against working tree (TODO)
#
# The script aborts loudly if the installed lavaan version disagrees with
# the version pinned in tests/fixtures/lavaan_version.txt.

suppressMessages({
  library(lavaan)
  library(jsonlite)
})

# --- pinned-version check --------------------------------------------------
script_dir <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])))
repo_root  <- normalizePath(file.path(script_dir, ".."))
fixtures   <- file.path(repo_root, "tests", "fixtures")

pinned    <- trimws(readLines(file.path(fixtures, "lavaan_version.txt"))[1])
installed <- as.character(packageVersion("lavaan"))
# R normalizes the package's hyphenated version (`0.6-22.2560`) into a
# dotted form (`0.6.22.2560`) when round-tripping through packageVersion().
# Compare on the dotted form so both spellings match.
norm_ver <- function(v) gsub("-", ".", v, fixed = TRUE)
if (norm_ver(pinned) != norm_ver(installed)) {
  stop(sprintf(
    "lavaan version mismatch: pinned=%s installed=%s\n  Update tests/fixtures/lavaan_version.txt OR install the pinned version.",
    pinned, installed))
}
cat("lavaan", installed, "(matches pinned ", pinned, ")\n", sep = "")

# --- corpus ----------------------------------------------------------------
corpus_path <- file.path(fixtures, "corpus.json")
corpus_json <- fromJSON(corpus_path, simplifyVector = FALSE)
models <- corpus_json$models
cat("corpus:", length(models), "models\n")

# --- modifier translator ---------------------------------------------------
# lavaan modifier list entry --> our Modifier JSON shape
#
#   list(fixed = 1)            --> {"kind":"fixed","value":1}
#   list(fixed = NA)           --> {"kind":"free"}
#   list(label = "a")          --> {"kind":"label","text":"a"}
#   list(start = 0.7)          --> {"kind":"start","value":0.7}
#   list(fixed = c(1, NA))     --> {"kind":"group","atoms":[{...},{...}]}
#   list(label = c("a", "b"))  --> {"kind":"group","atoms":[{...},{...}]}
#
# Multi-component modifier lists (e.g. fixed AND label set on the same row,
# from a heterogeneous c(...) ) are not supported in v0 fixtures yet; we'll
# extend this when we hit one.
atom_for_one <- function(field, val) {
  if (field == "fixed") {
    if (is.na(val)) return(list(kind = "free"))
    return(list(kind = "fixed", value = val))
  }
  if (field == "label") return(list(kind = "label", text = val))
  if (field == "start") return(list(kind = "start", value = val))
  stop(sprintf("unsupported modifier field '%s'", field))
}

modifier_to_json <- function(mod) {
  if (is.null(mod) || length(mod) == 0) return(NULL)
  if (length(mod) > 1) {
    stop(sprintf(
      "multi-field modifier not yet supported in this oracle (fields: %s)",
      paste(names(mod), collapse = ", ")))
  }
  field <- names(mod)[1]
  val   <- mod[[1]]
  if (length(val) == 1) return(atom_for_one(field, val))
  # Per-group vector
  atoms <- lapply(seq_along(val), function(i) atom_for_one(field, val[i]))
  list(kind = "group", atoms = atoms)
}

# --- row translator --------------------------------------------------------
flat_row_to_json <- function(row, modifier) {
  out <- list(
    lhs   = row$lhs,
    op    = row$op,
    rhs   = row$rhs,
    block = as.integer(row$block)
  )
  if (!is.null(modifier)) out$modifier <- modifier
  out
}

# --- constraint translator -------------------------------------------------
constraint_to_json <- function(c) {
  list(
    op  = c$op,
    lhs = c$lhs,
    rhs = c$rhs
  )
}

# --- per-model dump --------------------------------------------------------
flat_dir <- file.path(fixtures, "flat")
dir.create(flat_dir, showWarnings = FALSE, recursive = TRUE)

regenerated <- character(0)
for (m in models) {
  id    <- m$id
  model <- m$model

  flat        <- lavParseModelString(model, as.data.frame. = TRUE)
  modifiers   <- attr(flat, "modifiers")
  constraints <- attr(flat, "constraints")

  rows <- vector("list", nrow(flat))
  for (i in seq_len(nrow(flat))) {
    mod_idx  <- flat$mod.idx[i]
    modifier <- if (mod_idx > 0) modifier_to_json(modifiers[[mod_idx]]) else NULL
    rows[[i]] <- flat_row_to_json(flat[i, ], modifier)
  }
  cons <- if (length(constraints) > 0) lapply(constraints, constraint_to_json) else list()

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind   = "flat.partable",
      corpus_id      = id,
      tool           = "lavaan::lavParseModelString",
      lavaan_version = installed
    ),
    input       = model,
    rows        = rows,
    constraints = cons
  )

  out_path <- file.path(flat_dir, paste0(id, ".flat.json"))
  # auto_unbox=TRUE collapses every length-1 vector to a scalar so the JSON
  # reads naturally; jsonlite still keeps `rows` and `atoms` as arrays
  # because they're R lists, not atomic vectors.
  write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)
  regenerated <- c(regenerated, out_path)
}

cat("\nregenerated", length(regenerated), "flat fixtures under", flat_dir, "\n")

# === ptable layer ==========================================================
# parTable(lavaanify(model)) emits the full lavaanified partable. We
# translate to our model-description-only shape (no start/est/se).

ptable_dir <- file.path(fixtures, "ptable")
dir.create(ptable_dir, showWarnings = FALSE, recursive = TRUE)

# Translate one row of lavaan's parTable data.frame into our row JSON.
# lavaan columns: id lhs op rhs user block group free ustart exo label plabel
ptable_row_to_json <- function(row) {
  # NA handling — emit JSON null for NA values via jsonlite's na = "null".
  list(
    id     = as.integer(row$id),
    user   = as.integer(row$user),
    lhs    = row$lhs,
    op     = row$op,
    rhs    = row$rhs,
    block  = as.integer(row$block),
    group  = as.integer(row$group),
    free   = as.integer(row$free),
    exo    = as.integer(if (is.null(row$exo)) 0L else row$exo),
    ustart = as.numeric(row$ustart),
    label  = row$label,
    plabel = row$plabel
  )
}

regenerated_pt <- character(0)
for (m in models) {
  id    <- m$id
  model <- m$model

  # Use cfa()/sem() defaults so we exercise the same auto-* path that
  # practitioners hit. lavaanify's option names: auto.var, auto.cov.lv.x,
  # auto.cov.y, auto.fix.first, fixed.x.
  pt_or_err <- tryCatch(
    lavaanify(model,
              auto.var       = TRUE,
              auto.cov.lv.x  = TRUE,
              auto.cov.y     = FALSE,
              auto.fix.first = TRUE,
              fixed.x        = TRUE,
              ngroups        = 1),
    error = function(e) e
  )
  if (inherits(pt_or_err, "error")) {
    cat("  skip", id, "(lavaanify error: ", conditionMessage(pt_or_err), ")\n", sep = "")
    next
  }
  pt <- pt_or_err

  rows <- vector("list", nrow(pt))
  for (i in seq_len(nrow(pt))) rows[[i]] <- ptable_row_to_json(pt[i, ])

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind   = "ptable",
      corpus_id      = id,
      tool           = "lavaan::lavaanify",
      lavaan_version = installed
    ),
    input = model,
    rows  = rows
  )

  out_path <- file.path(ptable_dir, paste0(id, ".ptable.json"))
  write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)
  regenerated_pt <- c(regenerated_pt, out_path)
}

cat("regenerated", length(regenerated_pt), "ptable fixtures under", ptable_dir, "\n")

# === matrix_rep layer ======================================================
# lavMatrixRepresentation(pt) returns the partable with `mat`, `row`, `col`
# columns appended. Constraint rows have mat="" and row/col=0. We translate:
#   - mat: "lambda"/"theta"/"psi"/"beta"   → PascalCase
#          ""                              → emit Cell with used=false
#   - row/col: subtract 1 (lavaan is 1-based)
# Per-block variable orderings come from lav_partable_attributes(pt)$vnames.

matrix_rep_dir <- file.path(fixtures, "matrix_rep")
dir.create(matrix_rep_dir, showWarnings = FALSE, recursive = TRUE)

mat_to_pascal <- function(m) {
  if (is.null(m) || is.na(m) || nchar(m) == 0) return(NA_character_)
  switch(m,
         "lambda" = "Lambda",
         "theta"  = "Theta",
         "psi"    = "Psi",
         "beta"   = "Beta",
         "nu"     = "Nu",
         "alpha"  = "Alpha",
         NA_character_)
}

cell_to_json <- function(row) {
  mat <- mat_to_pascal(row$mat)
  used <- !is.na(mat)
  if (!used) {
    list(row_id = as.integer(row$id), used = FALSE)
  } else {
    list(
      row_id = as.integer(row$id),
      mat    = mat,
      row    = as.integer(row$row) - 1L,
      col    = as.integer(row$col) - 1L,
      block  = as.integer(row$block) - 1L,
      used   = TRUE
    )
  }
}

regenerated_mr <- character(0)
for (m in models) {
  id    <- m$id
  model <- m$model
  pt_or_err <- tryCatch(
    lavaanify(model,
              auto.var       = TRUE,
              auto.cov.lv.x  = TRUE,
              auto.cov.y     = FALSE,
              auto.fix.first = TRUE,
              fixed.x        = TRUE,
              ngroups        = 1),
    error = function(e) e
  )
  if (inherits(pt_or_err, "error")) {
    cat("  skip ", id, " (lavaanify error)\n", sep = "")
    next
  }
  pt <- pt_or_err
  rep_or_err <- tryCatch({
    attrs <- lav_partable_attributes(pt)$vnames
    rep <- lavMatrixRepresentation(pt)
    list(rep = rep, attrs = attrs)
  }, error = function(e) e)
  if (inherits(rep_or_err, "error")) {
    cat("  skip ", id, " (no matrix representation: ",
        conditionMessage(rep_or_err), ")\n", sep = "")
    next
  }
  rep   <- rep_or_err$rep
  attrs <- rep_or_err$attrs

  # Per-block variable orderings. For single-block v0 there's always one entry.
  # attrs$ov / attrs$lv are lists indexed by block.
  blocks <- list()
  for (b in seq_along(attrs$ov)) {
    blocks[[b]] <- list(
      block    = b - 1L,
      ov_names = unlist(attrs$ov[[b]]),
      lv_names = unlist(attrs$lv[[b]])
    )
  }

  cells <- lapply(seq_len(nrow(rep)), function(i) cell_to_json(rep[i, ]))

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind   = "matrix_rep",
      corpus_id      = id,
      tool           = "lavaan::lavMatrixRepresentation",
      lavaan_version = installed
    ),
    input  = model,
    blocks = blocks,
    cells  = cells
  )

  out_path <- file.path(matrix_rep_dir, paste0(id, ".mrep.json"))
  write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)
  regenerated_mr <- c(regenerated_mr, out_path)
}

cat("regenerated", length(regenerated_mr), "matrix_rep fixtures under", matrix_rep_dir, "\n")

# === fit layer =============================================================
# Fits each corpus model on HolzingerSwineford1939 (the only dataset our
# corpus references right now), dumps θ̂ in partable-free-index order plus
# the implied Σ from lavInspect(fit, "implied")$cov per block. Models that
# lavaan refuses to fit (under-identified, no formulas, etc.) are skipped
# with a logged reason.

fit_dir <- file.path(fixtures, "fit")
dir.create(fit_dir, showWarnings = FALSE, recursive = TRUE)

regenerated_fit <- character(0)
for (m in models) {
  id    <- m$id
  model <- m$model
  # Multi-group + mean-structure metadata (optional in corpus.json).
  n_groups      <- if (!is.null(m$n_groups))      m$n_groups      else 1L
  group_var     <- if (!is.null(m$group_var))     m$group_var     else NULL
  meanstructure <- if (!is.null(m$meanstructure)) m$meanstructure else FALSE

  cfa_args <- list(model = model, data = HolzingerSwineford1939,
                   std.lv = FALSE)
  if (!is.null(group_var) && nchar(group_var) > 0) {
    cfa_args$group <- group_var
  }
  if (meanstructure) {
    cfa_args$meanstructure <- TRUE
  }

  fit_or_err <- tryCatch(do.call(cfa, cfa_args),
                         error   = function(e) e,
                         warning = function(w) w)
  if (inherits(fit_or_err, "error")) {
    cat("  skip ", id, " (cfa error: ",
        conditionMessage(fit_or_err), ")\n", sep = "")
    next
  }
  if (inherits(fit_or_err, "warning")) {
    fit_or_err <- tryCatch(do.call(cfa, cfa_args), error = function(e) e)
    if (inherits(fit_or_err, "error")) {
      cat("  skip ", id, " (cfa error: ",
          conditionMessage(fit_or_err), ")\n", sep = "")
      next
    }
  }
  fit <- fit_or_err
  if (!lavInspect(fit, "converged")) {
    cat("  skip ", id, " (did not converge)\n", sep = "")
    next
  }

  # Free parameter estimates in partable-free-index order (1..n_free).
  pt_fitted <- parTable(fit)
  free_rows <- pt_fitted[pt_fitted$free > 0, ]
  free_rows <- free_rows[order(free_rows$free), ]
  theta_hat <- as.numeric(free_rows$est)

  # Inference: SEs in the same free-index order, plus chi² and df from
  # fitMeasures. Expected-info SEs are lavaan's `se = "standard"` default,
  # which is what our ExpectedInfoSE method computes.
  se_hat <- as.numeric(free_rows$se)
  fm     <- fitMeasures(fit)
  chi2_v <- as.numeric(fm["chisq"])
  df_v   <- as.integer(fm["df"])
  # Practical fit indices + log-likelihood-based information criteria + SRMR.
  # All come straight out of fitMeasures(); `npar` is lavaan's free-parameter
  # count (= our build_eq_constraints n_alpha for shared-label models).
  cfi_v   <- as.numeric(fm["cfi"])
  tli_v   <- as.numeric(fm["tli"])
  rmsea_v <- as.numeric(fm["rmsea"])
  srmr_v  <- as.numeric(fm["srmr"])
  logl_v  <- as.numeric(fm["logl"])
  ulogl_v <- as.numeric(fm["unrestricted.logl"])
  aic_v   <- as.numeric(fm["aic"])
  bic_v   <- as.numeric(fm["bic"])
  bic2_v  <- as.numeric(fm["bic2"])
  npar_v  <- as.integer(fm["npar"])

  # Observed-info SEs: re-fit with `information = "observed"` for
  # lavaan's H_obs⁻¹-based SEs. Reuse cfa_args so multi-group / mean
  # structure carry through.
  se_observed <- NULL
  obs_args    <- c(cfa_args, list(information = "observed"))
  fit_obs <- tryCatch(do.call(cfa, obs_args),
                      error = function(e) e, warning = function(w) NULL)
  if (!is.null(fit_obs) && !inherits(fit_obs, "error") &&
      lavInspect(fit_obs, "converged")) {
    pt_obs   <- parTable(fit_obs)
    free_obs <- pt_obs[pt_obs$free > 0, ]
    free_obs <- free_obs[order(free_obs$free), ]
    se_observed <- as.numeric(free_obs$se)
  }

  # Robust ("sandwich") SEs: re-fit with `estimator = "MLM"` (= robust.sem
  # SEs + Satorra-Bentler χ²) and with `estimator = "MLR"` (= robust.huber.
  # white SEs + Yuan-Bentler χ²). Also dump lavaan's empirical fourth-moment
  # ACOV Γ̂ (`lavInspect(fit, "gamma")`, per-unit) — the "meat" our
  # `robust_se` consumes. Single-group only (the robust-SE v1 surface is
  # single-block); skipped silently for multi-group fixtures.
  se_robust_sem        <- NULL
  se_robust_huberwhite <- NULL
  gamma_hat            <- NULL
  if (n_groups <= 1) {
    mlm_args <- c(cfa_args, list(estimator = "MLM"))
    fit_mlm  <- tryCatch(do.call(cfa, mlm_args),
                         error = function(e) e, warning = function(w) NULL)
    if (!is.null(fit_mlm) && !inherits(fit_mlm, "error") &&
        lavInspect(fit_mlm, "converged")) {
      pt_m   <- parTable(fit_mlm)
      free_m <- pt_m[pt_m$free > 0, ]; free_m <- free_m[order(free_m$free), ]
      se_robust_sem <- as.numeric(free_m$se)
      Gm <- tryCatch(lavInspect(fit_mlm, "gamma"), error = function(e) NULL)
      if (is.list(Gm)) Gm <- Gm[[1]]
      if (!is.null(Gm)) gamma_hat <- unname(as.matrix(Gm))
    }
    mlr_args <- c(cfa_args, list(estimator = "MLR"))
    fit_mlr  <- tryCatch(do.call(cfa, mlr_args),
                         error = function(e) e, warning = function(w) NULL)
    if (!is.null(fit_mlr) && !inherits(fit_mlr, "error") &&
        lavInspect(fit_mlr, "converged")) {
      pt_r   <- parTable(fit_mlr)
      free_r <- pt_r[pt_r$free > 0, ]; free_r <- free_r[order(free_r$free), ]
      se_robust_huberwhite <- as.numeric(free_r$se)
    }
  }

  # Browne residual NT (`browne.residual.nt`) and its model-based / RLS
  # variant (`browne.residual.nt.model`). Both are normal-theory and need
  # no extra fitting — they layer on top of the same θ̂.
  test_args <- c(cfa_args,
                 list(test = c("standard",
                               "browne.residual.nt",
                               "browne.residual.nt.model",
                               "satorra.bentler",
                               "mean.var.adjusted",
                               "scaled.shifted")))
  fit_tests <- tryCatch(do.call(cfa, test_args),
                        error = function(e) e, warning = function(w) NULL)
  browne_nt_chi2     <- NULL
  rls_chi2_value     <- NULL
  sb_chi2            <- NULL
  sb_scale           <- NULL
  mv_chi2            <- NULL
  mv_df_adj          <- NULL
  ss_chi2            <- NULL
  ss_a               <- NULL
  ss_b               <- NULL
  ugamma_eigvals_nt  <- NULL
  ugamma_eigvals_sum <- NULL
  ugamma_eigvals_sumsq <- NULL
  if (!is.null(fit_tests) && !inherits(fit_tests, "error") &&
      lavInspect(fit_tests, "converged")) {
    browne_nt_chi2 <- as.numeric(
        lavTest(fit_tests, "browne.residual.nt")$stat)
    rls_chi2_value <- as.numeric(
        lavTest(fit_tests, "browne.residual.nt.model")$stat)
    sb  <- lavTest(fit_tests, "satorra.bentler")
    mv  <- lavTest(fit_tests, "mean.var.adjusted")
    ss  <- lavTest(fit_tests, "scaled.shifted")
    sb_chi2   <- as.numeric(sb$stat)
    sb_scale  <- as.numeric(sb$scaling.factor)
    mv_chi2   <- as.numeric(mv$stat)
    mv_df_adj <- as.numeric(mv$df)
    ss_chi2   <- as.numeric(ss$stat)
    ss_a      <- as.numeric(ss$scaling.factor)
    ss_b      <- as.numeric(ss$shift.parameter)
    # Spectrum of UGamma — eigenvalues only. lavInspect returns the
    # UGamma matrix; nonzero eigenvalues equal our rank-df spectrum.
    UG <- tryCatch(lavInspect(fit_tests, "UGamma"),
                   error = function(e) NULL)
    if (!is.null(UG)) {
      ev_full  <- Re(eigen(UG, only.values = TRUE)$values)
      ev_nz    <- ev_full[abs(ev_full) > 1e-8]
      ev_nz    <- sort(ev_nz)        # ascending, matching our convention
      # I() preserves the array form even for a single eigenvalue
      # (jsonlite::toJSON with auto_unbox=TRUE would otherwise collapse
      # length-1 vectors to scalars, breaking the C++ side's
      # `j["ugamma_eigvals_nt"][k]` access pattern).
      ugamma_eigvals_nt    <- I(as.numeric(ev_nz))
      ugamma_eigvals_sum   <- sum(ev_nz)
      ugamma_eigvals_sumsq <- sum(ev_nz^2)
    }
  }

  # Per-block implied covariances + sample covariances + (optional) means.
  # For single-group, lavInspect returns a flat list with $cov (and $mean
  # if meanstructure). For multi-group, lavInspect returns a NAMED list
  # of per-group lists. We normalize both to the same per-block-array shape.
  implied  <- lavInspect(fit, "implied")
  sampstat <- lavInspect(fit, "sampstat")

  sigma_list       <- list()
  sample_cov_list  <- list()
  sample_mean_list <- list()
  if (n_groups <= 1) {
    sigma_list[[1]] <- list(block = 0L,
                            matrix = unname(as.matrix(implied$cov)))
    sample_cov_list[[1]] <- list(block = 0L,
                                 matrix = unname(as.matrix(sampstat$cov)))
    if (!is.null(sampstat$mean)) {
      sample_mean_list[[1]] <- list(block = 0L,
                                    vector = as.numeric(sampstat$mean))
    }
  } else {
    for (b in seq_len(n_groups)) {
      sigma_list[[b]] <- list(block = as.integer(b - 1L),
                              matrix = unname(as.matrix(implied[[b]]$cov)))
      sample_cov_list[[b]] <- list(block = as.integer(b - 1L),
                                   matrix = unname(as.matrix(sampstat[[b]]$cov)))
      if (!is.null(sampstat[[b]]$mean)) {
        sample_mean_list[[b]] <- list(block = as.integer(b - 1L),
                                      vector = as.numeric(sampstat[[b]]$mean))
      }
    }
  }
  if (length(sample_mean_list) == 0) sample_mean_list <- NULL

  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind   = "fit",
      corpus_id      = id,
      tool           = "lavaan::cfa",
      lavaan_version = installed
    ),
    input             = model,
    n_obs             = as.integer(lavInspect(fit, "ntotal")),
    # Per-block sizes — single-element array for single-group; one entry
    # per group for multi-group. C++ consumers can use whichever is
    # convenient.
    n_obs_per_block   = as.integer(lavInspect(fit, "nobs")),
    theta_hat         = theta_hat,
    se                = se_hat,
    se_observed       = se_observed,
    # Robust ("sandwich") SEs + the empirical fourth-moment ACOV Γ̂ (the
    # "meat"). Single-group only. `gamma_hat` is the p* × p* matrix
    # `lavInspect(fit, "gamma")` (per-unit divisor, matching our
    # `empirical_gamma`); jsonlite writes it row-major so the C++ side
    # reads `G[r][c]`.
    se_robust_sem        = se_robust_sem,
    se_robust_huberwhite = se_robust_huberwhite,
    gamma_hat            = gamma_hat,
    chi2              = chi2_v,
    df                = df_v,
    # Practical fit indices + log-likelihood-based information criteria +
    # SRMR (Bentler type). All from fitMeasures(fit); `npar` is lavaan's
    # free-parameter count.
    cfi               = cfi_v,
    tli               = tli_v,
    rmsea             = rmsea_v,
    srmr              = srmr_v,
    logl              = logl_v,
    unrestricted_logl = ulogl_v,
    aic               = aic_v,
    bic               = bic_v,
    bic2              = bic2_v,
    npar              = npar_v,
    # Browne residual NT family — both flavors. Layered on the same fit,
    # so they're cheap to dump alongside the standard chi².
    browne_residual_nt = browne_nt_chi2,
    rls_chi2           = rls_chi2_value,
    # Robust normal-theory tests + UΓ eigenvalue spectrum (used by the
    # robust_golden test). All come from the same fit, no extra solver
    # passes.
    sb_chi2              = sb_chi2,
    sb_scale             = sb_scale,
    mean_var_chi2        = mv_chi2,
    mean_var_df_adj      = mv_df_adj,
    scaled_shifted_chi2  = ss_chi2,
    scaled_shifted_a     = ss_a,
    scaled_shifted_b     = ss_b,
    ugamma_eigvals_nt    = ugamma_eigvals_nt,
    ugamma_eigvals_sum   = ugamma_eigvals_sum,
    ugamma_eigvals_sumsq = ugamma_eigvals_sumsq,
    implied_sigma     = sigma_list,
    sample_cov        = sample_cov_list,
    # Only present when the fit used mean structure (single or multi-group).
    sample_mean       = sample_mean_list
  )

  out_path <- file.path(fit_dir, paste0(id, ".fit.json"))
  write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
             null = "null", na = "null", digits = NA)
  regenerated_fit <- c(regenerated_fit, out_path)
}

cat("regenerated", length(regenerated_fit), "fit fixtures under", fit_dir, "\n")
