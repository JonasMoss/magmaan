#!/usr/bin/env Rscript
# Preparation only: retained development recipes, never historical recovery.
main <- function(args = commandArgs(TRUE)) {
  if ('--help' %in% args) {
    cat('Usage: Rscript prepare_retained_psd_case.R --case round_trip|chain_loss',
        '--run-id NAME [--dry-run-plan]\n',
        'One seed-fixed draw; no fits, evaluations, Hessians or workers.\n',
        'Current-runtime development reconstruction, not historical recovery or confirmation.\n',
        'Use R_LIBS=~/.cache/magmaan-rlib/merge-124 read-only. Choose a fresh run ID.\n')
    return(invisible(NULL))
  }
  allowed <- c('--case', '--run-id', '--dry-run-plan')
  values <- list()
  i <- 1L
  while (i <= length(args)) {
    k <- args[i]
    if (!k %in% allowed || !is.null(values[[k]])) stop('unknown or repeated option: ', k)
    if (k == '--dry-run-plan') values[[k]] <- TRUE else {
      i <- i + 1L
      if (i > length(args) || startsWith(args[i], '--')) stop('missing value: ', k)
      values[[k]] <- args[i]
    }
    i <- i + 1L
  }
  case <- values[['--case']]; run_id <- values[['--run-id']]
  if (is.null(case) || !case %in% c('round_trip', 'chain_loss')) stop('select exactly one supported case')
  if (is.null(run_id) || !grepl('^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$', run_id)) stop('unsafe or missing run ID')
  script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
  here <- dirname(dirname(script))
  out <- file.path(here, 'results', paste0('retained-case-preparation-', run_id))
  if (file.exists(out)) stop('choose a fresh run ID')
  recipe <- if (case == 'round_trip') {
    list(case = case, population = 'eqcfa_l40', model = 'equal_loadings', n = 25L,
         rep = 3L, seed = 93325847L, transform = 'native', estimator = 'ML', route = 'psd_layered')
  } else {
    list(case = case, population = 'eqchain_b20', model = 'equal_paths', n = 25L,
         rep = 3L, seed = 233870947L, transform = 'x0.01', estimator = 'ML', route = 'two_stage')
  }
  controls <- if (case == 'round_trip') {
    list(psd_layered = list(control = list(start = 'layered'), preconditioning = 'diagonal',
                           unspecified_defaults = 'Current public binding defaults; not historical resolved values'))
  } else {
    make_control <- function(normalized) {
      ctl <- list(start = 'fabin3', start_transport = 'auto', normalize_sample = normalized,
                  max_iter = 5000L, nlopt = list(max_eval = 5000L, ftol_rel = 1e-12, xtol_rel = 1e-10),
                  coordinate_scaling = 'information')
      list(ordinary_optimizer = 'nlopt-lbfgs', psd_optimizer = 'nlopt-slsqp',
           ordinary_control = ctl, psd_control = ctl[setdiff(names(ctl), c('start', 'start_transport'))],
           preconditioning = 'diagonal', start_eigen_floor = 1e-6, feasibility_tol = 1e-6)
    }
    list(original_units = make_control(FALSE), normalized = make_control(TRUE))
  }
  plan <- list(recipe = recipe, draws = 1L, controls = controls, fits = 0L, output = out,
               interpretation = 'Current-runtime development reconstruction only')
  if (isTRUE(values[['--dry-run-plan']])) {
    dput(plan)
    return(invisible(plan))
  }
  runtime <- normalizePath(path.expand('~/.cache/magmaan-rlib/merge-124'), mustWork = TRUE)
  package_path <- normalizePath(find.package('magmaanlab', lib.loc = runtime))
  if (normalizePath(find.package('magmaanlab')) != package_path) stop('active magmaanlab must be merge-124')
  hash_files <- function(paths) {
    data.frame(path = paths, exists = file.exists(paths),
               md5 = vapply(paths, function(p) if (file.exists(p)) unname(tools::md5sum(p)) else NA_character_, ''),
               row.names = NULL)
  }
  inventory <- '/home/jonas/Files/research/magmaan/experiments/decisions/01-optimizer-defaults/results/task-125-inventory'
  inventory_files <- if (dir.exists(inventory)) list.files(inventory, recursive = TRUE, full.names = TRUE) else character()
  sources <- c(script, file.path(here, 'R', c('families.R', 'fits.R')),
               file.path(here, 'scripts', 'normalization_revisit.R'))
  provenance <- list(command = commandArgs(), session = sessionInfo(), package_path = package_path,
                     package_description = utils::packageDescription('magmaanlab', lib.loc = runtime),
                     source_hashes = hash_files(sources),
                     binary_hashes = hash_files(list.files(file.path(package_path, 'libs'), full.names = TRUE, recursive = TRUE)),
                     inventory_path = inventory, inventory_present = dir.exists(inventory),
                     inventory_hashes = hash_files(inventory_files),
                     git_head = system2('git', c('-C', shQuote(here), 'rev-parse', 'HEAD'), stdout = TRUE),
                     git_status = system2('git', c('-C', shQuote(here), 'status', '--short'), stdout = TRUE),
                     rng_kind = RNGkind(), options = options(),
                     historical_gaps = c('dirty historical source', 'historical binary', 'historical RNGkind'))
  dir.create(out, recursive = TRUE)
  # The study allows some CSV evidence by glob; these checkpoints stay wholly local.
  writeLines('*', file.path(out, '.gitignore'))
  saveRDS(list(plan = plan, provenance = provenance), file.path(out, 'preparation_manifest.rds'))
  tryCatch({
    env <- new.env(parent = globalenv())
    sys.source(file.path(here, 'R', 'families.R'), env)
    sys.source(file.path(here, 'R', 'fits.R'), env)
    pops <- env$family_constrained()
    pop <- pops[[which(vapply(pops, function(p) p$key == recipe$population, logical(1)))]]
    fm <- pop$models[[which(vapply(pop$models, function(m) m$key == recipe$model, logical(1)))]]
    set.seed(recipe$seed)
    rng_before <- .Random.seed
    # draw_moments resets to the frozen seed, never derives a seed from rep.
    moments <- env$draw_moments(pop, recipe$n, recipe$seed)
    rng_after <- .Random.seed
    model <- env$build_model(fm)
    sample <- env$sample_in_units(moments, env$transform_factors(recipe$transform, pop$p), fm$meanstructure)
    pt <- model$partable
    missing <- c('resolved numeric starts', 'normalized internal model/sample',
                 'affine equality map', 'PSD lift and inverse lift', 'lift round-trip residual',
                 'optimizer endpoint/status/evaluation counts', 'objective/gradient/Hessian',
                 'terminal Newton audit', 'PSD feasibility/admissibility', 'fallback stage/reason')
    checkpoint <- list(plan = plan, provenance = provenance, population = pop, model_recipe = fm,
                       model = model, partable = pt,
                       free_mapping = pt[, intersect(c('id', 'lhs', 'op', 'rhs', 'group', 'free', 'label', 'plabel'), names(pt)), drop = FALSE],
                       exposed_start_equality_bounds = pt[, intersect(c('id', 'free', 'ustart', 'start', 'label', 'plabel', 'lower', 'upper', 'op', 'lhs', 'rhs'), names(pt)), drop = FALSE],
                       exposure_gaps = setdiff(c('ustart', 'start', 'lower', 'upper'), names(pt)),
                       moments = moments, sample = sample, controls = controls,
                       rng_before = rng_before, rng_after = rng_after, missing = missing,
                       next_diagnostic = 'One bounded no-optimizer public preparation/lift round-trip probe, if an exposed API is available; otherwise a separately scoped API diagnostic.')
    saveRDS(checkpoint, file.path(out, 'checkpoint.rds'))
    write.csv(hash_files(file.path(out, c('preparation_manifest.rds', 'checkpoint.rds'))),
              file.path(out, 'artifact_hashes.csv'), row.names = FALSE)
    cat('Prepared ', case, ': one draw, ', length(controls), ' control(s), zero fits.\nWrote ', out, '\n', sep = '')
    invisible(checkpoint)
  }, error = function(e) {
    saveRDS(list(message = conditionMessage(e), call = conditionCall(e), plan = plan, provenance = provenance),
            file.path(out, 'failure.rds'))
    stop(e)
  })
}
if (sys.nframe() == 0L) main()
