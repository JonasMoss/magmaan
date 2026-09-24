#!/usr/bin/env Rscript
# Two questions the earlier timing audits left open.
#
# A. magmaan's SB goes through the UGamma eigensolve; lavaan's SB needs only
#    tr(UGamma). magmaan already exposes the trace form to R
#    (robust_test_moments_both_breads_zc, the recipe experiments replications/01, replications/02, replications/03, replications/04
#    use). Does routing SB that way agree with the spectral answer, and what
#    does it save?
#
# B. pEBA needs the whole spectrum, so a trace path cannot help a pEBA request.
#    A nested pEBA4 test is the workload where BOTH engines must build and
#    eigensolve a difference spectrum, so it is a matched comparison with no
#    omitted-work assumption. Does magmaan's nested pEBA4 agree with lavaan
#    0.7-2's native nested FMG, and what does each side cost?
#
# Diagnostic only. No production estimator, benchmark claim or slide is changed
# here. Standalone phase means are not an instrumented partition: do not add or
# subtract them. Fits are outside every inference timer, on both sides.
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript diagnose_trace_sb_nested.R [--smoke] [--no-time] [--with-fiml]\n',
      '  --smoke    one case, one short batch\n',
      '  --no-time  correctness checks only (safe on a loaded machine)\n',
      '  --with-fiml  add FIML cells (comparator conventions UNRESOLVED, not parity)\n',
      'Writes results/trace-sb-nested/{global,nested,checks,metadata}.csv\n', sep = '')
  quit(status = 0)
}
smoke <- '--smoke' %in% args
do_time <- !('--no-time' %in% args)
# FIML is OFF by default and is not a parity claim. magmaan's FIML sb_ml does
# not cleanly match either lavaan FIML robust variant across p (1.5e-04 from
# yuan.bentler.mplus but 9.2e-02 from yuan.bentler at p=20; 5.6e-03 from both at
# p=10), and the two engines make different choices in estimating the bread,
# meat and saturated-H1 matrices under missing data. Settle those conventions
# before treating any lavaan FIML robust test as an oracle. The flag exists so
# the comparator traps documented below are not lost.
with_fiml <- '--with-fiml' %in% args
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
base <- dirname(script)
source(file.path(base, 'R', 'design.R'))
suppressPackageStartupMessages({library(magmaan); library(lavaan)})
core <- magmaan_core
outdir <- file.path(base, 'results', 'trace-sb-nested')
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Package fingerprints before and after, so a mid-run reinstall cannot be
# mistaken for a timing effect. A parallel `just r-dev` did exactly that once.
fingerprint <- function() {
  f <- unlist(lapply(c('magmaan', 'lavaan'), function(p)
    list.files(find.package(p), pattern = '\\.so$|^DESCRIPTION$',
               recursive = TRUE, full.names = TRUE)))
  tools::md5sum(f)
}
fp_before <- fingerprint()
started <- Sys.time()

# ── timing ──────────────────────────────────────────────────────────────────
# proc.time() cannot resolve a 0.2 ms call: time a batch and divide. min over
# batches, because contending load only ever inflates.
bench <- function(fun, reps = if (smoke) 10L else 40L,
                  batches = if (smoke) 1L else 7L) {
  fun()
  per <- vapply(seq_len(batches), function(b) {
    s <- proc.time()[['elapsed']]
    for (i in seq_len(reps)) fun()
    (proc.time()[['elapsed']] - s) / reps
  }, numeric(1))
  1000 * min(per)
}

# ── SB from trace moments, verbatim from experiment replications/01 ──────────────────────
sb_from_moments <- function(T, mom) {
  df <- as.integer(mom$df %||% NA_integer_)
  tr <- as.numeric(mom$trace %||% NA_real_)
  if (!is.finite(T) || !is.finite(tr) || !is.finite(df) || df <= 0)
    return(list(stat = NA_real_, df = NA_integer_, scale_c = NA_real_, p = NA_real_))
  c_scale <- tr / df
  stat <- if (c_scale > 0) T / c_scale else NA_real_
  list(stat = stat, df = df, scale_c = c_scale, trace = tr,
       p = if (is.finite(stat)) pchisq(stat, df, lower.tail = FALSE) else NA_real_)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# ── models ──────────────────────────────────────────────────────────────────
# Marker identification (the lavaan/magmaan default), so the population
# loadings in that parameterization are a_j / a_1. Labels on every non-marker
# loading; H0 pins them with `==`, which keeps npar equal to H1 -- the exact
# restriction map rejects npar(H1) != npar(H0), so a *fixed* value fails.
syntax_h1 <- function(p) {
  h <- p / 2
  paste(sprintf('f1 =~ x1 + %s', paste0('a', 2:h, '*x', 2:h, collapse = ' + ')),
        sprintf('f2 =~ x%d + %s', h + 1,
                paste0('b', 2:h, '*x', h + seq(2, h), collapse = ' + ')), sep = '\n')
}
# df_diff = 8: all free loadings of both factors at p=10, the first four of each
# at p=20. Constant across cells, and > 4 so that pEBA4's four blocks are not
# all singletons -- at df_diff <= 4 pEBA4 collapses onto `pall` exactly.
syntax_h0 <- function(p, pop_a, pop_b, k_per_factor = 4L) {
  ta <- pop_a / pop_a[1]; tb <- pop_b / pop_b[1]
  idx <- 2:(k_per_factor + 1L)
  paste(syntax_h1(p),
        paste(c(sprintf('a%d == %.12f', idx, ta[idx]),
                sprintf('b%d == %.12f', idx, tb[idx])), collapse = '\n'), sep = '\n')
}
mcar <- function(d, frac, seed) {
  set.seed(seed); m <- as.matrix(d)
  m[sample(length(m), round(frac * length(m)))] <- NA_real_
  as.data.frame(m)
}

cases <- data.frame(
  case = c('cont_p10', 'cont_p20', 'cont_p20_vm2', 'fiml_p10', 'fiml_p20'),
  p = c(10L, 20L, 20L, 10L, 20L), n = 500L,
  distribution = c('normal', 'normal', 'vm2', 'normal', 'normal'),
  missing = c(0, 0, 0, .10, .10), stringsAsFactors = FALSE)
if (!with_fiml) cases <- cases[cases$missing == 0, ]
if (smoke) cases <- cases[1, ]

global <- nested <- checks <- list()
for (j in seq_len(nrow(cases))) {
  cs <- cases[j, ]; p <- cs$p; n <- cs$n
  ctx <- prepare(p, cs$distribution)
  pa <- c(.4,.5,.6,.8,.4,.7,.8,.6,.6,.3)[seq_len(p/2)]
  pb <- c(.4,.7,.6,.4,.8,.8,.4,.7,.5,.6)[seq_len(p/2)]
  d_full <- draw_sample(ctx, n, cs$distribution, 20260917L + j * 100000L)
  d <- if (cs$missing > 0) mcar(d_full, cs$missing, 4242L + j) else d_full
  is_fiml <- cs$missing > 0
  s1 <- syntax_h1(p); s0 <- syntax_h0(p, pa, pb)
  ctrl <- list(max_iter = 4000L, ftol = 1e-12, gtol = 1e-8)
  est <- if (is_fiml) 'FIML' else 'ML'

  f1 <- magmaan(s1, d, estimator = est, control = ctrl)
  f0 <- magmaan(s0, d, estimator = est, control = ctrl)
  l1 <- cfa(s1, d, missing = if (is_fiml) 'ml' else 'listwise')
  l0 <- cfa(s0, d, missing = if (is_fiml) 'ml' else 'listwise')
  stopifnot(f1$converged, f0$converged,
            lavInspect(l1, 'converged'), lavInspect(l0, 'converged'))

  # ── A. global SB: spectral vs trace vs lavaan ────────────────────────────
  # FIML/ML2S inference reads the fit's own raw data and rejects a `data` arg.
  new_ic <- if (is_fiml) function() prepare_inference(f1) else
    function() prepare_inference(f1, d)
  ic <- new_ic()
  spec <- fmg_tests(ic, tests = 'sb_ml')
  T_ML <- spec$base_statistic[1]; df <- spec$df[1]
  # Comparator choice is load-bearing under FIML. lavaan SILENTLY forces
  # missing = "listwise" when satorra.bentler is requested at fit time
  # ("missing will be set to listwise for satorra.bentler style test"), so
  # cfa(missing = "ml", test = "satorra.bentler") is NOT a FIML fit. Calling
  # lavTest(test = "satorra.bentler") post hoc on a FIML fit sneaks past that
  # gate and returns lavaan's yuan.bentler numbers anyway -- it is not a
  # legitimate oracle. The supported FIML robust route is estimator = "MLR",
  # whose test is yuan.bentler.mplus, and that is what magmaan's FIML sb_ml
  # tracks. Complete data keeps satorra.bentler.
  lsb <- if (is_fiml) {
    lr <- cfa(s1, d, missing = 'ml', estimator = 'MLR')
    lr@test[[length(lr@test)]]
  } else lavTest(l1, test = 'satorra.bentler')[['satorra.bentler']]
  lsb_label <- if (is_fiml) 'yuan.bentler.mplus (MLR)' else 'satorra.bentler'

  # The trace route only exists for complete-data continuous ML; FIML has its
  # own trace-form scaling inside the estimator and no Zc route here.
  tr_p <- tr_scale <- NA_real_; t_trace <- NA_real_
  if (!is_fiml) {
    X <- as.matrix(d)
    trace_call <- function() {
      Zc <- core$robust_casewise_contributions(f1$partable, X)
      mom <- core$robust_test_moments_both_breads_zc(f1, Zc, n,
                                                    moments = 'structured')
      sb_from_moments(T_ML, mom$expected)
    }
    tr <- trace_call(); tr_p <- tr$p; tr_scale <- tr$scale_c
    if (do_time) t_trace <- bench(trace_call)
  }
  t_spec <- if (do_time) bench(function()
    fmg_tests(new_ic(), tests = 'sb_ml')) else NA_real_
  t_spec_warm <- if (do_time) bench(function()
    fmg_tests(ic, tests = 'sb_ml')) else NA_real_
  # FIML: time the whole supported route (fit + MLR robust test), because
  # lavaan offers no way to get a FIML robust test off an existing fit.
  t_lav <- if (!do_time) NA_real_ else if (is_fiml)
    bench(function() cfa(s1, d, missing = 'ml', estimator = 'MLR'), 3L, 2L) else
    bench(function() lavTest(l1, test = 'satorra.bentler'))
  t_lav_fit_only <- if (!do_time) NA_real_ else if (is_fiml)
    bench(function() cfa(s1, d, missing = 'ml'), 3L, 2L) else NA_real_

  global[[j]] <- data.frame(cs, df = df, T_ML = T_ML,
    p_spectral = spec$p_value[1], p_trace = tr_p, p_lavaan = lsb$pvalue,
    scale_spectral = T_ML / spec$chi2_equiv[1], scale_trace = tr_scale,
    scale_lavaan = lsb$scaling.factor, lavaan_test = lsb_label,
    ms_lavaan_fit_only = t_lav_fit_only,
    # Nonzero here is exactly where trace-SB and spectral-SB must disagree: the
    # FMG path averages max(lambda, 0), the trace is tr(M) untruncated.
    n_truncated = spec$n_truncated[1],
    ms_spectral_cold = t_spec, ms_spectral_warm = t_spec_warm,
    ms_trace = t_trace, ms_lavaan = t_lav, stringsAsFactors = FALSE)

  # ── B. nested pEBA4 ─────────────────────────────────────────────────────
  nt <- tryCatch(nestedTest(f1, f0, data = if (is_fiml) NULL else d),
                 error = function(e) e)
  if (inherits(nt, 'error')) {
    nested[[j]] <- data.frame(cs, df_diff = NA_integer_,
      error = conditionMessage(nt), stringsAsFactors = FALSE)
  } else {
    nd <- if (is_fiml) NULL else d
    mn <- fmg_nested(f1, f0, data = nd, tests = c('sb_ml', 'peba4_ml', 'pall_ml'))
    getp <- function(lab) mn$p_value[mn$label == lab]
    # lavaan's nested FMG hardcodes the delta restriction matrix
    # (lav_test_fmg.R:1022, lav_test_diff_a(..., method = "delta")), whereas
    # fmg_nested() defaults to A.method = "exact". That single option is the
    # whole magmaan/lavaan difference-spectrum gap, so parity must gate on
    # delta; exact is a different (analytic) restriction map, not an error.
    p_peba4_delta <- fmg_nested(f1, f0, data = nd, tests = 'peba4_ml',
                                A.method = 'delta')$p_value
    # lavaan's FMG gates on missing == "listwise" (lav_test_fmg.R:132-147), so
    # the FIML cells have NO lavaan pEBA comparator at all -- global or nested.
    lav_peba <- tryCatch({
      z <- lavTestLRT(l0, l1, test = 'peba4')
      as.numeric(z[2, grep('^Pr', names(z))[1]])
    }, error = function(e) NA_real_)
    lav_reason <- tryCatch({lavTestLRT(l0, l1, test = 'peba4'); ''},
                           error = function(e) conditionMessage(e))
    rp <- if (smoke) 5L else 20L; bt <- if (smoke) 1L else 5L
    t_m_peba <- if (do_time) bench(function()
      fmg_nested(f1, f0, data = nd, tests = 'peba4_ml'), rp, bt) else NA_real_
    t_m_peba_delta <- if (do_time) bench(function()
      fmg_nested(f1, f0, data = nd, tests = 'peba4_ml', A.method = 'delta'),
      rp, bt) else NA_real_
    t_l_peba <- if (do_time && is.finite(lav_peba)) bench(function()
      lavTestLRT(l0, l1, test = 'peba4'), rp, bt) else NA_real_
    nested[[j]] <- data.frame(cs, df_diff = nt$df_diff, T_diff = nt$T_diff,
      T_diff_lavaan = as.numeric(lavTestLRT(l0, l1)[2, 'Chisq diff']),
      lambda_min = min(nt$eigenvalues), lambda_max = max(nt$eigenvalues),
      p_sb = getp('sb_ml'), p_peba4 = getp('peba4_ml'), p_pall = getp('pall_ml'),
      p_peba4_delta = p_peba4_delta,
      p_peba4_lavaan = lav_peba, lavaan_error = lav_reason,
      ms_magmaan_peba4_delta = t_m_peba_delta,
      peba4_equals_pall = isTRUE(all.equal(getp('peba4_ml'), getp('pall_ml'))),
      ms_magmaan_peba4 = t_m_peba, ms_lavaan_peba4 = t_l_peba,
      error = '', stringsAsFactors = FALSE)
  }

  checks[[j]] <- data.frame(cs, df = df,
    sb_trace_vs_spectral = if (is_fiml) NA_real_ else abs(tr_p - spec$p_value[1]),
    sb_spectral_vs_lavaan = abs(spec$p_value[1] - lsb$pvalue),
    nested_Tdiff_vs_lavaan = if (inherits(nt, 'error')) NA_real_ else
      abs(nt$T_diff - as.numeric(lavTestLRT(l0, l1)[2, 'Chisq diff'])),
    # The parity-relevant one: magmaan delta vs lavaan. The `exact` column in
    # nested.csv is deliberately NOT a parity target.
    nested_peba4_delta_vs_lavaan = if (inherits(nt, 'error')) NA_real_ else
      abs(nested[[j]]$p_peba4_delta - nested[[j]]$p_peba4_lavaan),
    stringsAsFactors = FALSE)
  cat(sprintf('[%d/%d] %s done\n', j, nrow(cases), cs$case))
}

global <- do.call(rbind, global); nested <- do.call(rbind, nested)
checks <- do.call(rbind, checks)
fp_after <- fingerprint()
stopifnot(identical(fp_before, fp_after))

write.csv(global, file.path(outdir, 'global.csv'), row.names = FALSE)
write.csv(nested, file.path(outdir, 'nested.csv'), row.names = FALSE)
write.csv(checks, file.path(outdir, 'checks.csv'), row.names = FALSE)
write.csv(data.frame(args = paste(args, collapse = ' '), smoke = smoke,
  timed = do_time, started = format(started), finished = format(Sys.time()),
  magmaan = as.character(packageVersion('magmaan')),
  lavaan = as.character(packageVersion('lavaan')),
  R = paste(R.version$major, R.version$minor, sep = '.'),
  blas_threads = Sys.getenv('OPENBLAS_NUM_THREADS', 'unset')),
  file.path(outdir, 'metadata.csv'), row.names = FALSE)

cat('\n== global SB (ms, p-values) ==\n')
print(global[, c('case', 'df', 'lavaan_test', 'p_spectral', 'p_trace', 'p_lavaan',
                 'ms_spectral_cold', 'ms_spectral_warm', 'ms_trace', 'ms_lavaan',
                 'ms_lavaan_fit_only')], row.names = FALSE, digits = 6)
cat('\n== nested pEBA4 ==\n')
print(nested[, intersect(c('case', 'df_diff', 'T_diff', 'p_sb', 'p_peba4',
  'p_peba4_delta', 'p_peba4_lavaan', 'peba4_equals_pall', 'ms_magmaan_peba4',
  'ms_magmaan_peba4_delta', 'ms_lavaan_peba4'), names(nested))],
  row.names = FALSE, digits = 6)
if (any(nzchar(nested$lavaan_error)))
  cat('\nlavaan refusals:\n', paste(unique(nested$lavaan_error[
    nzchar(nested$lavaan_error)]), collapse = '\n'), '\n')
cat('\n== checks (absolute differences) ==\n')
print(checks, row.names = FALSE, digits = 3)
cat('\nWrote', outdir, '\n')
