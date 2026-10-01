#!/usr/bin/env Rscript
# Freeze derived moments and DWLS weights only; raw textbook data stays in the
# optional external corpus. Model preparation explicitly matches sem defaults.
suppressPackageStartupMessages({library(lavaan); library(jsonlite); library(textbookcorpus)})
script <- sub('^--file=', '', commandArgs(FALSE)[grepl('^--file=', commandArgs(FALSE))][1])
root <- normalizePath(file.path(dirname(script), '../../..'))
corpus <- file.path(root, 'external/textbook-corpus')
out <- file.path(root, 'cpp/tests/fixtures/textbook_mixed')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
pin <- readLines(file.path(root, 'cpp/tests/fixtures/lavaan_version.txt'), warn=FALSE)[1]
stopifnot(gsub('-', '.', pin, fixed=TRUE) == as.character(packageVersion('lavaan')))
manifest <- read.csv(file.path(corpus, 'manifest.csv'))
canon <- function(keys) vapply(strsplit(keys, '~~', fixed=TRUE), function(k) paste(sort(k), collapse='~~'), character(1))
ids <- c('newsom_2015_ex5_3a', 'newsom_2015_ex5_3b', 'newsom_2015_ex5_7a', 'newsom_2024_ex5_8b')
for (id in ids) {
  case <- load_case(file.path(corpus, manifest$case_dir[match(id, manifest$case_id)]), root=corpus)
  args <- textbookcorpus:::.lavaan_args(case, 'WLSMV')
  fit <- suppressWarnings(do.call(lavaan::sem, args))
  stopifnot(lavInspect(fit, 'converged'))
  ov <- lavNames(fit, 'ov'); ordered <- ov %in% lavNames(fit, 'ov.ord'); p <- length(ov)
  ss <- lavInspect(fit, 'sampstat'); th <- ss$th
  keys <- c(names(th), paste0(ov[!ordered], '~1'), paste0(ov[!ordered], '~~', ov[!ordered]))
  for (j in seq_len(p-1)) for (i in (j+1):p) keys <- c(keys, paste0(ov[i], '~~', ov[j]))
  obs <- lavInspect(fit, 'wls.obs'); ix <- match(canon(keys), canon(names(obs)))
  stopifnot(!anyNA(ix), !anyDuplicated(ix))
  pt <- parTable(fit); pt <- pt[pt$op != ':=', ]
  pt$ustart <- ifelse(pt$free == 0L, pt$est, pt$start)
  opts <- lavInspect(fit, 'options')
  payload <- list(`_meta`=list(fixture_kind='textbook_mixed_dwls',
    tool='cpp/tests/tools/regen_textbook_mixed_fixtures.R', lavaan_version=as.character(packageVersion('lavaan')),
    case_id=id, provenance=case$meta$provenance$citation,
    note='Derived summaries only. NACOV diagonal is sufficient for DWLS; full WLS and robust inference are not gated.'),
    input=args$model, ov_names=ov, ordered=ov[ordered],
    options=list(auto_cov_y=opts$auto.cov.y, meanstructure=opts$meanstructure,
      fixed_x=opts$fixed.x, parameterization=opts$parameterization),
    n=as.integer(lavInspect(fit, 'nobs')), R=unname(ss$cov[ov,ov]), mean=unname(ss$mean[ov]),
    ordered_mask=as.integer(ordered), thresholds=unname(th),
    threshold_ov=match(sub('\\|.*$', '', names(th)),ov)-1L,
    threshold_level=as.integer(sub('^.*\\|t', '', names(th))),
    moments=unname(obs[ix]), model_moments=unname(lavInspect(fit, 'wls.est')[ix]),
    NACOV_diag=unname(diag(lavInspect(fit, 'gamma'))[ix]),
    W_diag=unname(diag(lavInspect(fit, 'wls.v'))[ix]),
    partable=lapply(seq_len(nrow(pt)), function(i) as.list(pt[i,,drop=FALSE])),
    fit=as.list(fitMeasures(fit, c('fmin','chisq','df'))),
    implied=unname(lavInspect(fit, 'implied')$cov[ov,ov]))
  write_json(payload, file.path(out,paste0(id,'.json')), auto_unbox=TRUE, pretty=TRUE, digits=16, na='null')
  cat('wrote',id,'\n')
}
