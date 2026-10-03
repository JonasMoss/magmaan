#!/usr/bin/env Rscript
# Hand-written independent models, not mplus2lavaan or magmaan translations.
# Run from any directory; original Demo files stay in ignored scratch storage.
suppressPackageStartupMessages({ library(lavaan); library(jsonlite) })
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), '../../..'))
fixture_dir <- file.path(root, 'cpp/tests/fixtures')
pin <- trimws(readLines(file.path(fixture_dir, 'lavaan_version.txt'))[1])
version <- as.character(packageVersion('lavaan'))
if (gsub('-', '.', pin, fixed=TRUE) != version) stop('Pinned lavaan mismatch')
mpdemo <- '/home/jonas/mplusdemo/mpdemo'
if (!file.exists(mpdemo)) stop('Mplus Demo unavailable')
scratch <- path.expand('~/.cache/magmaan-logs/mplus-golden')
cases <- list()
add <- function(id, names, model, syntax, rules, analysis='', use=names) {
  input <- paste(paste0('TITLE: golden ', id, ';'), 'DATA: FILE = golden.dat;',
    'VARIABLE:', paste0('NAMES = ', names, ';'),
    paste0('USEVARIABLES = ', use, ';'), 'ANALYSIS: ESTIMATOR = ML;', analysis,
    'MODEL:', model, 'OUTPUT: TECH1;', sep='\n')
  stopifnot(all(nchar(strsplit(input, '\n')[[1]]) <= 90))
  cases[[length(cases)+1L]] <<- list(id=id, names=strsplit(names, ' ')[[1]],
    mplus_input=input, lavaan_syntax=syntax, rules=rules,
    meanstructure=id != 'nomean')
}
# MS02 marker; DF02 intercepts, DF03 latent means, DF04/DF12 variances.
marker <- paste('f =~ 1*y1 + y2 + y3 + y4', 'f ~~ f; f ~ 0*1',
  'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4',
  'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1', sep='\n')
add('cfa_marker', 'y1 y2 y3 y4', 'f BY y1-y4;', marker,
    c('MS02','DF02','DF03','DF04','DF12'))
# MS02/LB01 free marker and fixed factor variance; explicit y3 start.
add('cfa_std', 'y1 y2 y3 y4', 'f BY y1* y2 y3*0.7 y4; f@1;',
  paste('f =~ y1 + y2 + start(.7)*y3 + y4', 'f ~~ 1*f; f ~ 0*1',
    'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1', sep='\n'),
  c('MS02','LB01','DF02','DF03','DF12'))
# DF01 conditioned x; DF09 final latent disturbances covary.
add('mimic', 'y1 y2 y3 y4 y5 y6 x1 x2',
  'f1 BY y1-y3; f2 BY y4-y6; f1 f2 ON x1 x2;',
  paste('f1 =~ 1*y1 + y2 + y3; f2 =~ 1*y4 + y5 + y6',
    'f1 ~ x1 + x2; f2 ~ x1 + x2',
    'f1 ~~ f1 + f2; f2 ~~ f2; f1 ~ 0*1; f2 ~ 0*1',
    'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4; y5 ~~ y5; y6 ~~ y6',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1; y5 ~ 1; y6 ~ 1', sep='\n'),
  c('MS02','DF01','DF02','DF03','DF04','DF09','DF12'))
# DF08 only the two final observed dependents covary.
add('path_final', 'y1 y2 y3 x1', 'y1 y2 ON x1; y3 ON y1;',
  paste('y1 ~ x1; y2 ~ x1; y3 ~ y1', 'y1 ~~ y1; y2 ~~ y2 + y3; y3 ~~ y3',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1', sep='\n'), c('DF01','DF02','DF04','DF08'))
# DF10 final observed/latent residual covariance.
add('mixed_final', 'y1 y2 y3 y4 x1', 'f BY y1-y3; f ON x1; y4 ON x1;',
  paste('f =~ 1*y1 + y2 + y3; f ~ x1; y4 ~ x1',
    'f ~~ f + y4; f ~ 0*1',
    'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1', sep='\n'),
  c('MS02','DF01','DF02','DF03','DF04','DF10','DF12'))
# MS03 second-order definition order; DF09 no first-order residual covariances.
add('second_order', 'y1 y2 y3 y4 y5 y6',
  'f1 BY y1-y2; f2 BY y3-y4; f3 BY y5-y6; g BY f1-f3;',
  paste('f1 =~ 1*y1 + y2; f2 =~ 1*y3 + y4; f3 =~ 1*y5 + y6',
    'g =~ 1*f1 + f2 + f3', 'g ~~ g; f1 ~~ f1; f2 ~~ f2; f3 ~~ f3',
    'g ~ 0*1; f1 ~ 0*1; f2 ~ 0*1; f3 ~ 0*1',
    'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4; y5 ~~ y5; y6 ~~ y6',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1; y5 ~ 1; y6 ~ 1', sep='\n'),
  c('MS02','MS03','DF02','DF03','DF04','DF09','DF12'))
# LB04 equality excludes fixed marker; label must end its line (LB02/LB03).
add('equalities', 'y1 y2 y3 y4', 'f BY y1 y2-y4 (1);\ny1-y4 (2);',
  paste('f =~ 1*y1 + loading*y2 + loading*y3 + loading*y4',
    'f ~~ f; f ~ 0*1',
    'y1 ~~ residual*y1; y2 ~~ residual*y2; y3 ~~ residual*y3; y4 ~~ residual*y4',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1', sep='\n'),
  c('MS02','LB02','LB03','LB04','DF02','DF03','DF12'))
# LB06 shared label; DF08 both final residuals covary.
add('labels', 'y1 y2 x1 x2',
  'y1 ON x1 (b);\ny2 ON x1 (b);\ny1 y2 ON x2;',
  'y1 ~ b*x1 + x2; y2 ~ b*x1 + x2\ny1 ~~ y1 + y2; y2 ~~ y2\ny1 ~ 1; y2 ~ 1',
  c('LB06','DF01','DF02','DF04','DF08'))
# DF13 suppresses latent covariance; explicit WITH frees only y1-y4.
add('nocov', 'y1 y2 y3 y4 y5 y6',
  'f1 BY y1-y3; f2 BY y4-y6; y1 WITH y4;',
  paste('f1 =~ 1*y1 + y2 + y3; f2 =~ 1*y4 + y5 + y6',
    'f1 ~~ f1; f2 ~~ f2; f1 ~ 0*1; f2 ~ 0*1',
    'y1 ~~ y1 + y4; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4; y5 ~~ y5; y6 ~~ y6',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1; y5 ~ 1; y6 ~ 1', sep='\n'),
  c('MS02','DF02','DF03','DF04','DF12','DF13'), 'MODEL = NOCOVARIANCES;')
# MS11 requires expected information for NOMEANSTRUCTURE.
add('nomean', 'y1 y2 y3 y4', 'f BY y1-y4;',
  'f =~ 1*y1 + y2 + y3 + y4\nf ~~ f\ny1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4',
  c('MS02','MS11','DF04','DF12'),
  'MODEL = NOMEANSTRUCTURE; INFORMATION = EXPECTED;')
# MS05 pairs y2/y1, y3/x1 and y2/y3, y4/y5; DF08 adds other final pairs.
add('pon_pwith', 'y1 y2 y3 y4 y5 x1',
  'y2 y3 PON y1 x1; y1 ON x1; y2 y4 PWITH y3 y5; y4 y5 ON x1;',
  paste('y2 ~ y1; y3 ~ x1; y1 ~ x1; y4 ~ x1; y5 ~ x1',
    'y1 ~~ y1; y2 ~~ y2 + y3 + y4 + y5',
    'y3 ~~ y3 + y4 + y5; y4 ~~ y4 + y5; y5 ~~ y5',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1; y5 ~ 1', sep='\n'),
  c('MS05','DF01','DF02','DF04','DF08'))
# DF07 unmentioned y4 has free mean/variance, no covariance.
add('unmentioned', 'y1 y2 y3 y4', 'f BY y1-y3;',
  paste('f =~ 1*y1 + y2 + y3; f ~~ f; f ~ 0*1',
    'y1 ~~ y1; y2 ~~ y2; y3 ~~ y3; y4 ~~ y4',
    'y1 ~ 1; y2 ~ 1; y3 ~ 1; y4 ~ 1', sep='\n'),
  c('MS02','DF02','DF03','DF07','DF12'), use='y1-y4')
# NM03 NAMES-order range makes x1 an indicator, not a conditioned predictor.
add('range_order', 'y1 x1 y2 y3', 'f BY y1-y3;',
  paste('f =~ 1*y1 + x1 + y2 + y3; f ~~ f; f ~ 0*1',
    'y1 ~~ y1; x1 ~~ x1; y2 ~~ y2; y3 ~~ y3',
    'y1 ~ 1; x1 ~ 1; y2 ~ 1; y3 ~ 1', sep='\n'),
  c('NM03','MS02','DF02','DF03','DF12'))

make_data <- function(case, seed) {
  set.seed(seed)
  n <- 500L
  z <- matrix(rnorm(n*12), n, 12)
  x1 <- z[,1]; x2 <- .2*x1 + z[,2]
  f <- z[,3]; f2 <- .3*f + z[,4]
  # Marker/std/nomean CFA: independent indicator errors; equalities also
  # shares the non-marker population loadings and all residual variances.
  y <- outer(f, c(1,.8,.7,.9,.85,.75)) + .7*z[,5:10]
  if (case$id == 'equalities') y <- outer(f, c(1,.8,.8,.8,.8,.8)) + .7*z[,5:10]
  if (case$id %in% c('mimic','mixed_final')) {
    # Correlated final factor disturbances (DF09); mixed_final instead uses
    # a final observed disturbance correlated with f (DF10).
    f <- f + .35*x1 + .2*x2
    f2 <- f2 + .25*x1 - .2*x2
    y[,1:3] <- outer(f,c(1,.8,.7)) + .7*z[,5:7]
    y[,4:6] <- outer(f2,c(1,.85,.75)) + .7*z[,8:10]
    if (case$id == 'mixed_final') y[,4] <- .4*x1 + .2*z[,3] + z[,8]
  }
  if (case$id == 'second_order') {
    g <- z[,3]
    factors <- outer(g,c(1,.8,.7)) + .7*z[,4:6]
    # Separate indicator innovations from each other and the factor disturbances.
    for (j in 1:3)
      y[,(2*j-1):(2*j)] <- outer(factors[,j],c(1,.8)) + .6*z[,(5+2*j):(6+2*j)]
  }
  if (case$id == 'nocov') {
    # Independent factors; only the explicitly freed y1-y4 errors covary.
    y[,4:6] <- outer(z[,4],c(1,.85,.75)) + .7*z[,8:10]
    y[,4] <- y[,4] + .1*z[,5]
  }
  if (case$id %in% c('path_final','labels','pon_pwith')) {
    # path_final omits x2: its independent innovation joins y2's residual.
    # Only final residuals correlate; labels has equal .4 slopes on x1.
    y[,1] <- .4*x1 + z[,5]
    y[,2] <- .4*x1 + .2*x2 + z[,6]
    y[,3] <- .5*y[,1] + z[,7] + .15*z[,6]
    if (case$id == 'labels') y[,1] <- y[,1] + .3*x2
    if (case$id == 'pon_pwith') {
      y[,2] <- .5*y[,1] + z[,6]
      y[,3] <- .3*x1 + z[,7] + .15*z[,6]
      y[,4] <- .4*x1 + z[,8]; y[,5] <- .2*x1 + z[,9] + .1*z[,8]
    }
  }
  if (case$id == 'unmentioned') y[,4] <- z[,8]
  # NAMES-order x1 is a fourth indicator, with its own independent error.
  if (case$id == 'range_order') x1 <- .85*f + .7*z[,11]
  colnames(y) <- paste0('y',1:6)
  data <- cbind(y,x1=x1,x2=x2)[,case$names,drop=FALSE]
  if (case$meanstructure) data <- sweep(data,2,seq_len(ncol(data))*.1,'+')
  # Round once, then give both engines the exact same decimal observations.
  round(data,8)
}

parse_output <- function(out) {
  if (any(grepl('\\*\\*\\* ERROR',out))) stop('Demo input error')
  if (!any(grepl('THE MODEL ESTIMATION TERMINATED NORMALLY',out))) stop('Demo did not converge')
  if (!any(grepl('^Mplus VERSION 9.1',out))) stop('Unexpected Demo version')
  num <- function(x) as.numeric(regmatches(x,regexpr('[0-9]+(?:\\.[0-9]+)?',x,perl=TRUE)))
  free <- num(grep('Number of Free Parameters',out,value=TRUE))
  idx <- grep('^\\s*Chi-Square Test of Model Fit\\s*$',out)
  if (length(idx) != 1L || length(free) != 1L) stop('Missing/ambiguous fit statistics')
  block <- out[idx:min(idx+10L,length(out))]
  chi <- num(grep('^\\s*Value\\s',block,value=TRUE)[1])
  df <- num(grep('Degrees of Freedom',block,value=TRUE)[1])
  rows <- list(); section <- ''; active <- FALSE
  for (line in out) {
    s <- trimws(line)
    if (s == 'MODEL RESULTS') { active <- TRUE; next }
    if (grepl('^(STANDARDIZED MODEL RESULTS|QUALITY OF NUMERICAL RESULTS|TECHNICAL)',s)) active <- FALSE
    if (!active) next
    m <- regmatches(s,regexec('^(\\S+)\\s+(-?[0-9]+\\.[0-9]+)(?:\\s|$)',s))[[1]]
    if (length(m)) {
      rhs <- tolower(m[2])
      relation <- regmatches(section,regexec('^(\\S+)\\s+(BY|ON|WITH)$',section))[[1]]
      if (length(relation)) {
        lhs <- tolower(relation[2]); op <- c(BY='=~',ON='~',WITH='~~')[[relation[3]]]
      } else if (section %in% c('Means','Intercepts')) { lhs <- rhs; rhs <- ''; op <- '~1'
      } else if (section %in% c('Variances','Residual Variances')) { lhs <- rhs; op <- '~~'
      } else stop('Unmapped Demo result block: ',section)
      rows[[length(rows)+1L]] <- list(lhs=lhs,op=op,rhs=rhs,estimate=as.numeric(m[3]))
    } else if (nzchar(s) && !grepl('^(Estimate|Two-Tailed|S.E.)',s)) section <- s
  }
  if (!length(rows)) stop('No printed estimates parsed')
  list(npar=free,df=df,chisq=chi,estimates=rows)
}

unlist_options <- list(estimator='ML',information='expected',fixed.x=TRUE,
  likelihood='normal',auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,
  auto.fix.first=FALSE,auto.fix.single=FALSE,auto.th=FALSE,auto.delta=FALSE,
  auto.efa=FALSE)
unlink(scratch,recursive=TRUE)
dir.create(scratch,recursive=TRUE)
results <- list()
for (i in seq_along(cases)) {
  case <- cases[[i]]; seed <- 51400L+i
  dir <- file.path(scratch,case$id); dir.create(dir)
  data <- make_data(case,seed)
  write.table(data,file.path(dir,'golden.dat'),row.names=FALSE,col.names=FALSE,quote=FALSE)
  writeLines(case$mplus_input,file.path(dir,'golden.inp'))
  options <- c(unlist_options,list(meanstructure=case$meanstructure))
  fit <- do.call(lavaan::lavaan,c(list(model=case$lavaan_syntax,data=as.data.frame(data)),
    options,list(control=list(iter.max=2000))))
  if (!lavInspect(fit,'converged')) stop(case$id,': lavaan did not converge')
  pt <- parTable(fit)
  fail <- function(message) stop(case$id,': ',message,'; rules ',paste(case$rules,collapse=', '))
  fm <- fitMeasures(fit,c('npar','df','chisq','pvalue'))
  # Meaning fixtures require a proper fit to their intended population.
  variance_rows <- which(pt$op=='~~' & pt$lhs==pt$rhs)
  if (any(!is.finite(pt$est[variance_rows]) | pt$est[variance_rows] < 0))
    fail('data-quality gate: non-finite or negative lavaan variance estimate')
  if (fm['df'] > 0 && (!is.finite(fm['pvalue']) || fm['pvalue'] < .001))
    fail(sprintf('data-quality gate: lavaan chi-square p-value %.8g below .001',fm['pvalue']))
  old <- getwd(); setwd(dir)
  status <- system2(mpdemo,c('golden.inp','golden.out'),
    stdout='console.log',stderr='console.err',timeout=120)
  setwd(old)
  if (status != 0 || !file.exists(file.path(dir,'golden.out'))) stop(case$id,': Demo execution failed')
  mp <- parse_output(readLines(file.path(dir,'golden.out'),warn=FALSE))
  if (mp$npar != fm['npar'] || mp$df != fm['df'])
    fail(sprintf('counts disagree: Mplus npar/df %s/%s, lavaan %s/%s',mp$npar,mp$df,fm['npar'],fm['df']))
  chi_error <- abs(mp$chisq-fm['chisq'])
  if (chi_error > .002+1e-5*fm['chisq']) fail(sprintf('chi-square differs by %.8g',chi_error))
  margins <- numeric()
  for (j in seq_along(mp$estimates)) {
    row <- mp$estimates[[j]]
    hit <- which(pt$lhs==row$lhs & pt$op==row$op & pt$rhs==row$rhs)
    if (!length(hit) && row$op=='~~') hit <- which(pt$lhs==row$rhs & pt$op=='~~' & pt$rhs==row$lhs)
    if (length(hit)!=1L) fail(paste('unmatched estimate',row$lhs,row$op,row$rhs))
    error <- abs(row$estimate-pt$est[hit]); tolerance <- .001+2e-4*abs(pt$est[hit])
    if (error > tolerance) fail(sprintf('%s %s %s estimate differs: Mplus %.8g, lavaan %.8g',row$lhs,row$op,row$rhs,row$estimate,pt$est[hit]))
    mp$estimates[[j]]$lavaan_estimate <- pt$est[hit]
    mp$estimates[[j]]$absolute_error <- error
    margins <- c(margins,tolerance-error)
  }
  centered <- sweep(data,2,colMeans(data),'-')
  implied <- fitted(fit)
  rows <- lapply(seq_len(nrow(pt)),function(j) list(lhs=pt$lhs[j],op=pt$op[j],
    rhs=pt$rhs[j],free=pt$free[j],fixed_value=if(pt$free[j]==0L) pt$est[j] else NA_real_,
    label=pt$label[j],start=pt$start[j],ustart=pt$ustart[j],
    estimate=pt$est[j],se=pt$se[j]))
  results[[i]] <- list(id=case$id,mplus_input=case$mplus_input,
    lavaan_syntax=case$lavaan_syntax,options=options,rule_ids=case$rules,seed=seed,
    variables=case$names,n=500L,sample_cov=unname(crossprod(centered)/500),
    sample_mean=unname(colMeans(data)),lavaan=list(rows=rows,
      implied_variables=colnames(implied$cov),implied_cov=unname(implied$cov),
      implied_mean=if(case$meanstructure) unname(implied$mean) else NULL,
      npar=unname(fm['npar']),df=unname(fm['df']),chisq=unname(fm['chisq']),
      pvalue=unname(fm['pvalue'])),
    mplus=c(mp,list(chisq_absolute_error=unname(chi_error),
      minimum_estimate_margin=min(margins))))
  cat(sprintf('%s: npar=%d df=%d chi error=%.6g min estimate margin=%.6g\n',
    case$id,mp$npar,mp$df,chi_error,min(margins)))
}
output <- file.path(fixture_dir,'mplus/golden.json')
write_json(list(meta=list(lavaan_version=version,mplus_version='9.1 Demo',
  generator='cpp/tests/tools/regen_oracle_mplus.R',
  provenance='Hand-written paired models; synthetic complete raw observations; derived Demo values only',
  conventions='ML normal likelihood; N-divisor covariance; expected-information SEs; fixed x',
  tolerances=list(chisq_absolute=.002,chisq_relative=1e-5,
    estimate_absolute=.001,estimate_relative=2e-4)),cases=results),
  output,auto_unbox=TRUE,pretty=TRUE,digits=16,na='null',null='null')
if (file.info(output)$size >= 300000) stop('Fixture exceeds 300 KB')
cat('Wrote',output,'\n')

# Increment 2 extends the same frozen fixture with independently specified groups.
status <- system2(file.path(R.home('bin'),'Rscript'),
  shQuote(file.path(root,'cpp/tests/tools/regen_oracle_mplus_groups.R')))
if (status != 0L) stop('Multigroup oracle generation failed')
