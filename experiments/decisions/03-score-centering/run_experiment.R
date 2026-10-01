#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
usage <- 'Usage: Rscript run_experiment.R [options]
  --smoke                 one draw per cell, separate seeds; mechanics only
  --kind pilot|confirmation (default pilot)
  --reps R                draws per cell (default 100 pilot, 2000 confirmation)
  --seed-base S           default 302610011 pilot, 402610031 confirmation
  --run-id ID             immutable results/score-centering/<ID>; default kind
  --workers W             parallel workers, one math thread each (default 4)
  --families a,b          family filter; stable IDs/seeds survive filtering
  --library PATH          prepend an installed matched >=0.1.0 package library
  --summarize             summarize existing raw batches without fitting
  --help                  print this help

Thirty-six cells cover ML global/nested tests, a known grouped covariance target,
and prospective MCAR/MAR FIML score calibration. Both methods use the same fits.
Confirmation requires preregistered criteria and fresh seeds. Raw rows are local;
summary CSVs and provenance are frozen decision evidence. No defaults change.'
if ('--help' %in% args) { cat(usage, '\n'); quit(save = 'no') }
opt <- function(name, default = NULL) {
  if (!name %in% args) return(default)
  at <- match(name, args)
  if (at == length(args) || startsWith(args[at + 1], '--')) stop('Missing value for ', name)
  args[at + 1]
}
valid <- c('--smoke', '--kind', '--reps', '--seed-base', '--run-id', '--workers',
           '--families', '--library', '--summarize')
if (any(startsWith(args, '--') & !args %in% valid)) stop('Unknown option')
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(script)
source(file.path(here, '..', '..', '_support', 'R', 'helpers.R'))
set_single_threaded_math()
if (!is.null(opt('--library'))) .libPaths(c(normalizePath(opt('--library')), .libPaths()))
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(magmaan))
if (packageVersion('magmaanlab') < '0.1.0' || packageVersion('magmaan') < '0.1.0')
  stop('Use matched >=0.1.0 packages; provide --library PATH if needed.')
for (f in c('cells.R', 'compute.R', 'summaries.R')) source(file.path(here, 'R', f))
kind <- if ('--smoke' %in% args) 'smoke' else match.arg(opt('--kind', 'pilot'), c('pilot', 'confirmation'))
reps <- if (kind == 'smoke') 1L else as.integer(opt('--reps', if (kind == 'confirmation') '2000' else '100'))
seed_base <- as.integer(opt('--seed-base', switch(kind, smoke = '202610001', pilot = '302610011', confirmation = '402610031')))
workers <- as.integer(opt('--workers', '4'))
if (anyNA(c(reps, seed_base, workers)) || reps < 1 || reps >= 10000 || workers < 1 || seed_base < 1)
  stop('Require positive seed/workers and 1 <= reps < 10000.')
if (kind == 'confirmation' && (reps < 2000 || any(abs(seed_base - c(202610001L, 302610011L)) < 1000000L)))
  stop('Confirmation requires >=2000 draws per cell and fresh seeds.')
run_id <- opt('--run-id', kind)
if (!grepl('^[a-zA-Z0-9_-]+$', run_id)) stop('run-id must be a plain directory name')
cells <- centering_cells()
if (!is.null(opt('--families'))) {
  families <- strsplit(opt('--families'), ',')[[1]]
  if (!all(families %in% cells$family)) stop('Unknown family')
  cells <- cells[cells$family %in% families, ]
}
out <- file.path(here, 'results', 'score-centering', run_id)
raw_dir <- file.path(out, 'raw')
if (dir.exists(out) && !'--summarize' %in% args) stop('Run already exists; use a new --run-id or --summarize.')
if (!'--summarize' %in% args) {
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  binary <- list.files(file.path(find.package('magmaanlab'), 'libs'), pattern = '\\.so$', full.names = TRUE)
  hashes <- tools::md5sum(c(file.path(here, 'criteria', 'score_centering.md'), script,
    file.path(here, 'R', c('cells.R', 'compute.R', 'summaries.R')), binary))
  write_metadata(file.path(out, 'metadata.csv'), list(kind = kind, reps = reps, seed_base = seed_base,
    families = paste(unique(cells$family), collapse = ','), cells = nrow(cells), draws = nrow(cells) * reps,
    workers = workers, criteria_md5 = unname(hashes[1]), source_hashes = paste(hashes, collapse = ','),
    hash_files = paste(names(hashes), collapse = ','), git_head = git_scalar(c('rev-parse', 'HEAD')),
    release_tag = git_scalar(c('rev-parse', 'v0.1.0')), git_dirty = git_dirty(),
    magmaanlab_path = find.package('magmaanlab'), magmaan_path = find.package('magmaan'),
    native_md5 = paste(unname(tools::md5sum(binary)), collapse = ','),
    scope = 'ML policy; FIML prospective; fixed sampling groups; no pattern recentering'),
    packages = c('magmaanlab', 'magmaan', 'lavaan'))
  write_csv(cells, file.path(out, 'cells.csv'))
  start <- proc.time()[['elapsed']]; done <- 0L; total <- nrow(cells) * reps
  for (i in seq_len(nrow(cells))) for (batch in split(seq_len(reps), ceiling(seq_len(reps) / 50))) {
    cell <- cells[i, ]
    values <- parallel::mclapply(batch, function(rep) {
      seed <- seed_base + 10000L * cell$cell_id + rep
      tryCatch(run_centering_task(cell, rep, seed), error = function(e) {
        r <- centering_row(cell, rep, seed, 'task_error', 'raw')
        r$fit_error <- conditionMessage(e); r
      })
    }, mc.cores = workers, mc.preschedule = TRUE)
    write_csv(do.call(rbind, values), file.path(raw_dir,
      sprintf('cell_%02d_batch_%04d.csv', cell$cell_id, ceiling(batch[1] / 50))))
    done <- done + length(batch); elapsed <- proc.time()[['elapsed']] - start
    eta <- elapsed / done * (total - done)
    write_csv(data.frame(done = done, total = total, elapsed_s = elapsed, eta_s = eta), file.path(out, 'progress.csv'))
    cat(sprintf('%s %s N=%d %s: %d/%d draws, %.1f s elapsed, %.1f s ETA\n', kind,
      cell$family, cell$n, cell$role, done, total, elapsed, eta)); flush.console()
  }
  cat(sprintf('Measured %.3f seconds per draw with %d workers; 2000/cell estimate %.1f minutes.\n',
    elapsed / done, workers, elapsed / done * 36 * 2000 / 60))
}
meta <- read.csv(file.path(out, 'metadata.csv'), stringsAsFactors = FALSE)
kind <- meta$value[meta$key == 'kind']
files <- list.files(raw_dir, pattern = '^cell_.*\\.csv$', full.names = TRUE)
if (!length(files)) stop('No raw batches exist.')
raw <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
# An all-empty CSV error column is inferred as logical NA; these are successes.
for (name in c('fit_error', 'cov_error', 'test_error')) raw[[name]][is.na(raw[[name]])] <- ''
s <- centering_summaries(raw, kind)
hard_failure <- any(grepl('identity failed|disagrees with ordinary|projection/Gram',
  paste(raw$fit_error, raw$cov_error, raw$test_error)))
if (hard_failure) s$decisions$status <- 'blocked_construction_identity'
summary_files <- c(script, file.path(here, 'R', c('cells.R', 'compute.R', 'summaries.R')))
meta <- meta[!meta$key %in% c('summary_source_hashes', 'summarized_at_utc'), ]
meta <- rbind(meta, data.frame(key = c('summary_source_hashes', 'summarized_at_utc'),
  value = c(paste(unname(tools::md5sum(summary_files)), collapse = ','),
            format(Sys.time(), tz = 'UTC', usetz = TRUE))))
write_csv(meta, file.path(out, 'metadata.csv'))
for (name in names(s)) write_csv(s[[name]], file.path(out, paste0(name, '.csv')))
fail <- raw[nzchar(raw$fit_error) | nzchar(raw$cov_error) | nzchar(raw$test_error), ]
write_csv(fail[, c('cell_id', 'family', 'n', 'role', 'rep', 'seed', 'geometry', 'arm',
  'fit_error', 'cov_error', 'test_error')], file.path(out, 'failures.csv'))
checks <- data.frame(draws = length(unique(paste(raw$cell_id, raw$rep))),
  fit_failures = length(unique(paste(fail$cell_id[nzchar(fail$fit_error)], fail$rep[nzchar(fail$fit_error)]))),
  max_gram_identity_gap = max(c(0, raw$gram_identity_gap), na.rm = TRUE),
  max_projection_gap = max(c(0, raw$projection_gap), na.rm = TRUE),
  max_numerator_gap = max(c(0, raw$numerator_gap), na.rm = TRUE),
  max_policy_gap = max(c(0, raw$policy_gap), na.rm = TRUE),
  max_covariance_policy_gap = max(c(0, raw$covariance_policy_gap), na.rm = TRUE),
  median_task_seconds = median(raw$elapsed_s, na.rm = TRUE))
write_csv(checks, file.path(out, 'checks.csv'))
cat('Results: ', out, '\n', sep = '')
