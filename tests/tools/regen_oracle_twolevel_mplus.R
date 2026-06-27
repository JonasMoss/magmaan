#!/usr/bin/env Rscript
# =============================================================================
# Two-level (multilevel) ML — Mplus cross-check tool. The companion to
# tests/tools/regen_oracle_twolevel.R, which writes the checked-in lavaan
# oracle fixtures (tests/fixtures/twolevel/*.json). lavaan is the oracle; this
# script is an INDEPENDENT second-opinion cross-check (it never overwrites the
# lavaan fixtures). For each two-level fixture it
#
#   * reads the lavaan model syntax + raw clustered data straight out of the
#     fixture JSON (so the two tools cannot drift),
#   * emits an Mplus TWOLEVEL input (TYPE=TWOLEVEL, CLUSTER=, %WITHIN% /
#     %BETWEEN% with `BY` measurement blocks) plus a free-format .dat,
#   * runs it through the Mplus 9.1 Demo CLI `mpdemo input.inp output.out`,
#   * regex-parses the .out for the model chi-square (Value + Degrees of
#     Freedom) and the MODEL RESULTS estimate/SE table (per Within/Between
#     level), and
#   * compares chi-square, df, and the free parameter estimates/SEs against the
#     lavaan oracle in the fixture, printing a PASS/FAIL report.
#
# Mplus 9.1 DEMO caps the model at 6 dependent + 2 independent observed
# variables, so fixtures with more than 6 observed variables are skipped; the
# three shipped fixtures (twolevel_ri3 = 3, twolevel_1f4 = 4, twolevel_2f6 = 6)
# all fit. The Demo binary lives at ~/mplusdemo/mpdemo (also on PATH as
# `mpdemo`); there is no MplusAutomation in this environment, hence the
# regex-parse convention.
#
# This is a maintainer-only tool (tests/tools/): it is NOT part of the CI suite,
# and CI invokes neither R nor Mplus. Run standalone:
#   Rscript tests/tools/regen_oracle_twolevel_mplus.R
# Optional first arg overrides the Mplus binary; --keep keeps the scratch dir.
# =============================================================================

suppressMessages({ library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
keep_work <- any(args == "--keep")
mp_override <- args[!startsWith(args, "--")]

# ---- locate repo / fixtures (mirror regen_oracle_twolevel.R) --------------
all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", all_args[grep("^--file=", all_args)])
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
repo_root <- normalizePath(file.path(script_dir, "..", ".."))
tl_dir <- file.path(repo_root, "tests", "fixtures", "twolevel")

# ---- locate the Mplus Demo CLI --------------------------------------------
find_mpdemo <- function() {
  if (length(mp_override)) return(mp_override[1])
  cand <- c(Sys.getenv("MPDEMO"),
            unname(Sys.which("mpdemo")),
            path.expand("~/.local/bin/mpdemo"),
            path.expand("~/mplusdemo/mpdemo"))
  cand <- cand[nzchar(cand)]
  hit <- cand[file.exists(cand)]
  if (length(hit)) hit[1] else NA_character_
}
mpdemo <- find_mpdemo()

# ---- tolerances -----------------------------------------------------------
TOL_CHISQ <- 5e-2   # Mplus prints chisq to 3 decimals; allow convergence slack
TOL_EST   <- 1e-2   # 3-decimal Mplus print + tiny optimizer differences
TOL_SE    <- 1e-2

MPLUS_DV_CAP <- 6L  # Mplus 9.1 Demo dependent-variable limit

# ---- model translation: lavaan `level:` syntax -> Mplus BY blocks ---------
# Returns list(within = list(list(factor=, ind=c(...)), ...), between = ...).
parse_level_model <- function(model_str) {
  lines <- trimws(strsplit(model_str, "\n", fixed = TRUE)[[1]])
  lines <- lines[nzchar(lines)]
  cur <- NA_integer_
  out <- list("1" = list(), "2" = list())
  for (ln in lines) {
    m <- regmatches(ln, regexec("^level:\\s*([0-9]+)\\s*$", ln))[[1]]
    if (length(m) == 2L) { cur <- m[2]; next }
    if (grepl("=~", ln, fixed = TRUE)) {
      sides <- strsplit(ln, "=~", fixed = TRUE)[[1]]
      fac <- trimws(sides[1])
      inds <- trimws(strsplit(trimws(sides[2]), "+", fixed = TRUE)[[1]])
      inds <- inds[nzchar(inds)]
      out[[cur]] <- c(out[[cur]], list(list(factor = fac, ind = inds)))
    }
  }
  list(within = out[["1"]], between = out[["2"]])
}

emit_model_block <- function(factors) {
  lines <- character(0)
  for (f in factors)
    lines <- c(lines, sprintf("  %s BY %s;", f$factor, paste(f$ind, collapse = " ")))
  # Explicit same-level latent covariances (lavaan auto-adds these; Mplus frees
  # them by default, so this is belt-and-suspenders for exact parity).
  facs <- vapply(factors, function(f) f$factor, character(1))
  if (length(facs) >= 2L) {
    cmb <- combn(facs, 2L)
    for (k in seq_len(ncol(cmb)))
      lines <- c(lines, sprintf("  %s WITH %s;", cmb[1, k], cmb[2, k]))
  }
  lines
}

build_inp <- function(id, ov, mdl) {
  names_line <- paste(c(ov, "clus"), collapse = " ")
  use_line <- paste(ov, collapse = " ")
  c(
    sprintf("TITLE: %s two-level ML cross-check;", id),
    sprintf("DATA: FILE = %s.dat;", id),
    "VARIABLE:",
    sprintf("  NAMES = %s;", names_line),
    sprintf("  USEVARIABLES = %s;", use_line),
    "  CLUSTER = clus;",
    "ANALYSIS:",
    "  TYPE = TWOLEVEL;",
    "  ESTIMATOR = ML;",
    "MODEL:",
    "  %WITHIN%",
    emit_model_block(mdl$within),
    "  %BETWEEN%",
    emit_model_block(mdl$between),
    "OUTPUT: ;"
  )
}

# ---- .out parsing ---------------------------------------------------------
parse_chisq <- function(out) {
  # The model fit block header is exactly "Chi-Square Test of Model Fit"; the
  # baseline block adds " for the Baseline Model", so anchor on end-of-line.
  idx <- grep("Chi-Square Test of Model Fit\\s*$", out)
  if (!length(idx)) return(list(value = NA_real_, df = NA_integer_))
  blk <- out[idx[1]:min(idx[1] + 8L, length(out))]
  vln <- grep("\\bValue\\b", blk, value = TRUE)
  dln <- grep("Degrees of Freedom", blk, value = TRUE)
  val <- if (length(vln))
    suppressWarnings(as.numeric(gsub("[^0-9.eE+-]", "", sub(".*Value", "", vln[1]))))
  else NA_real_
  df <- if (length(dln))
    suppressWarnings(as.integer(gsub("[^0-9]", "", dln[1]))) else NA_integer_
  list(value = val, df = df)
}

# Walk MODEL RESULTS into a data frame of (lhs, op, rhs, level, est, se), all
# names lower-cased to match the lavaan partable.
parse_model_results <- function(out) {
  i0 <- grep("^MODEL RESULTS\\s*$", out)
  i1 <- grep("QUALITY OF NUMERICAL RESULTS", out)
  if (!length(i0)) return(NULL)
  hi <- if (length(i1)) i1[1] - 1L else length(out)
  body <- out[i0[1]:hi]

  rows <- list()
  level <- NA_integer_
  sect <- NA_character_   # "by"/"with"/"var"/"rvar"/"intercept"/"mean"
  anchor <- NA_character_ # factor name for "by"/"with" sections
  add <- function(lhs, op, rhs) {
    rows[[length(rows) + 1L]] <<-
      list(lhs = lhs, op = op, rhs = rhs, level = level)
  }
  for (ln in body) {
    if (grepl("Within Level", ln))  { level <- 1L; sect <- NA; next }
    if (grepl("Between Level", ln)) { level <- 2L; sect <- NA; next }

    # Section headers (e.g. " FW       BY", " FW1      WITH", " Variances").
    mby <- regmatches(ln, regexec("^\\s*(\\S+)\\s+BY\\s*$", ln))[[1]]
    if (length(mby) == 2L) { sect <- "by"; anchor <- tolower(mby[2]); next }
    mwith <- regmatches(ln, regexec("^\\s*(\\S+)\\s+WITH\\s*$", ln))[[1]]
    if (length(mwith) == 2L) { sect <- "with"; anchor <- tolower(mwith[2]); next }
    if (grepl("^\\s*Residual Variances\\s*$", ln)) { sect <- "rvar"; next }
    if (grepl("^\\s*Variances\\s*$", ln))          { sect <- "var";  next }
    if (grepl("^\\s*Intercepts\\s*$", ln))         { sect <- "intercept"; next }
    if (grepl("^\\s*Means\\s*$", ln))              { sect <- "mean"; next }
    if (grepl("^\\s*Thresholds\\s*$", ln))         { sect <- "thr"; next }

    # Data rows: "<name>  <est>  <se>  <est/se>  <pval>".
    md <- regmatches(ln, regexec(
      "^\\s+(\\S+)\\s+(-?[0-9]+\\.[0-9]+)\\s+(-?[0-9]+\\.[0-9]+)", ln))[[1]]
    if (length(md) == 4L && !is.na(sect)) {
      nm <- tolower(md[2]); est <- as.numeric(md[3]); se <- as.numeric(md[4])
      key <- switch(sect,
        by        = list("=~", anchor, nm),  # factor =~ indicator
        with      = list("~~", anchor, nm),  # factor ~~ factor
        var       = list("~~", nm, nm),
        rvar      = list("~~", nm, nm),
        intercept = list("~1", nm, ""),
        mean      = list("~1", nm, ""),
        NULL)
      if (!is.null(key)) {
        rec <- list(lhs = key[[2]], op = key[[1]], rhs = key[[3]],
                    level = level, est = est, se = se)
        rows[[length(rows) + 1L]] <- rec
      }
    }
  }
  if (!length(rows)) return(NULL)
  do.call(rbind, lapply(rows, function(r)
    data.frame(lhs = r$lhs, op = r$op, rhs = r$rhs, level = r$level,
               est = r$est, se = r$se, stringsAsFactors = FALSE)))
}

# ---- per-fixture cross-check ----------------------------------------------
lookup_mplus <- function(mp, lhs, op, rhs, level) {
  hit <- mp[mp$op == op & mp$level == level &
            ((mp$lhs == lhs & mp$rhs == rhs) |
             (op == "~~" & mp$lhs == rhs & mp$rhs == lhs)), , drop = FALSE]
  if (nrow(hit)) hit[1, ] else NULL
}

cross_check_one <- function(fx_path, workdir) {
  j <- fromJSON(fx_path, simplifyVector = FALSE)
  id <- j$id
  ov <- unlist(j$ov_names)
  p <- length(ov)
  if (p > MPLUS_DV_CAP) {
    cat(sprintf("[%s] SKIP — %d observed > Mplus Demo %d-DV cap\n",
                id, p, MPLUS_DV_CAP))
    return(invisible(NA))
  }

  # Raw data: X rows (ov order) + cluster id as the last column. The fixture
  # stores 0-based ids; Mplus requires a positive integer cluster id, so shift
  # to 1-based (a relabeling; it does not change the clustering).
  X <- do.call(rbind, lapply(j$data$X, function(r) as.numeric(unlist(r))))
  clus <- as.integer(unlist(j$data$cluster_id)) + 1L
  dat <- cbind(X, clus)

  mdl <- parse_level_model(j$model)
  inp <- build_inp(id, ov, mdl)

  dat_path <- file.path(workdir, paste0(id, ".dat"))
  inp_path <- file.path(workdir, paste0(id, ".inp"))
  out_path <- file.path(workdir, paste0(id, ".out"))
  write.table(dat, dat_path, row.names = FALSE, col.names = FALSE)
  writeLines(inp, inp_path)

  status <- system2(mpdemo, c(shQuote(inp_path), shQuote(out_path)),
                    stdout = NULL, stderr = NULL,
                    timeout = 300)
  if (!file.exists(out_path)) {
    cat(sprintf("[%s] FAIL — Mplus produced no output (status=%s)\n", id, status))
    return(invisible(FALSE))
  }
  out <- readLines(out_path, warn = FALSE)
  term_ok <- any(grepl("THE MODEL ESTIMATION TERMINATED NORMALLY", out))
  if (!term_ok) {
    cat(sprintf("[%s] WARN — Mplus did not report normal termination\n", id))
  }

  # ---- chi-square / df ----
  chi <- parse_chisq(out)
  exp_chisq <- if (is.null(j$expected$chisq)) NA_real_ else
    as.numeric(j$expected$chisq)
  exp_df <- as.integer(j$expected$df)
  df_ok <- isTRUE(chi$df == exp_df)
  chisq_ok <- (is.na(exp_chisq) && (is.na(chi$value) || abs(chi$value) < TOL_CHISQ)) ||
    (!is.na(exp_chisq) && !is.na(chi$value) &&
       abs(chi$value - exp_chisq) <= TOL_CHISQ)
  cat(sprintf("[%s] chisq: mplus=%.3f lavaan=%s |diff|<=%.2g -> %s ; df: mplus=%s lavaan=%d -> %s\n",
              id, ifelse(is.na(chi$value), NA, chi$value),
              ifelse(is.na(exp_chisq), "NA(df=0)", sprintf("%.3f", exp_chisq)),
              TOL_CHISQ, ifelse(chisq_ok, "OK", "MISMATCH"),
              ifelse(is.na(chi$df), "NA", chi$df), exp_df,
              ifelse(df_ok, "OK", "MISMATCH")))

  # ---- parameter estimates / SEs (free rows only) ----
  mp <- parse_model_results(out)
  n_cmp <- 0L; n_miss <- 0L; max_de <- 0; max_ds <- 0
  worst <- ""
  if (!is.null(mp)) {
    for (pr in j$expected$params) {
      if (as.integer(pr$free) <= 0L) next   # skip fixed (markers, fixed-0)
      lhs <- pr$lhs; op <- pr$op; rhs <- pr$rhs; lvl <- as.integer(pr$level)
      hit <- lookup_mplus(mp, lhs, op, rhs, lvl)
      if (is.null(hit)) {
        n_miss <- n_miss + 1L
        next
      }
      de <- abs(hit$est - as.numeric(pr$est))
      ds <- abs(hit$se - as.numeric(pr$se))
      if (de > max_de) { max_de <- de; worst <- sprintf("%s%s%s/L%d", lhs, op, rhs, lvl) }
      if (ds > max_ds) max_ds <- ds
      n_cmp <- n_cmp + 1L
    }
  } else {
    n_miss <- sum(vapply(j$expected$params, function(pr)
      as.integer(pr$free) > 0L, logical(1)))
  }
  est_ok <- !is.null(mp) && (max_de <= TOL_EST) && (max_ds <= TOL_SE)
  cat(sprintf("    params: matched=%d unmatched=%d  max|d est|=%.4f max|d se|=%.4f (worst est: %s) -> %s\n",
              n_cmp, n_miss, max_de, max_ds, worst,
              ifelse(est_ok && n_miss == 0L, "OK", "CHECK")))

  invisible(term_ok && df_ok && chisq_ok && est_ok && n_miss == 0L)
}

# ---- driver ---------------------------------------------------------------
cat(sprintf("two-level Mplus cross-check — fixtures: %s\n", tl_dir))
if (is.na(mpdemo)) {
  cat("Mplus Demo CLI not found (looked for $MPDEMO, `mpdemo` on PATH,\n",
      "~/.local/bin/mpdemo, ~/mplusdemo/mpdemo). The generator is written and\n",
      "ready; re-run with `mpdemo` installed or pass its path as the first arg.\n",
      sep = "")
  quit(status = 0)
}
cat(sprintf("Mplus binary: %s\n\n", mpdemo))

fixtures <- sort(list.files(tl_dir, pattern = "\\.json$", full.names = TRUE))
fixtures <- fixtures[!grepl("\\.mplus\\.json$", fixtures)]
if (!length(fixtures)) {
  cat("No twolevel fixtures found; run regen_oracle_twolevel.R first.\n")
  quit(status = 0)
}

# `--keep` writes to a stable dir under the system temp ROOT (the per-session
# tempdir() is wiped on exit); otherwise use a throwaway session tempfile dir.
workdir <- if (keep_work)
  file.path(dirname(tempdir()), "magmaan_twolevel_mplus") else
  tempfile("twolevel_mplus_")
dir.create(workdir, showWarnings = FALSE, recursive = TRUE)

results <- vapply(fixtures, function(fx) {
  res <- cross_check_one(fx, workdir)
  if (is.na(res)) NA else isTRUE(res)
}, logical(1))

checked <- !is.na(results)
cat(sprintf("\nMplus cross-check: %d/%d checked fixtures fully matched the lavaan oracle",
            sum(results[checked]), sum(checked)))
if (any(!checked)) cat(sprintf(" (%d skipped by Mplus Demo limits)", sum(!checked)))
cat(".\n")
if (keep_work) cat(sprintf("scratch kept in %s\n", workdir))
if (any(checked) && any(!results[checked])) quit(status = 1)
