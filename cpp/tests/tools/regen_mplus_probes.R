#!/usr/bin/env Rscript
# =============================================================================
# Mplus 9.1 Demo language probes: independent maintainer evidence, not an oracle
# exemption or a frontend policy. Base R + jsonlite; joint-X numeric gates
# also use pinned lavaan. No magmaan/Rcpp.
# Run: Rscript cpp/tests/tools/regen_mplus_probes.R
# Inputs, data and original Demo output remain under ~/.cache/magmaan-logs/
# mplus-probes. Only our input text and regex-derived facts are serialized.
# Seeds, ordering and serialization are fixed; no times or machine paths enter
# the fixture. Each invocation clears its own scratch tree before running.
# =============================================================================
suppressMessages(library(jsonlite))
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value = TRUE)[1])
root <- normalizePath(file.path(dirname(script), '../../..'))
scratch <- path.expand('~/.cache/magmaan-logs/mplus-probes')
mpdemo <- '/home/jonas/mplusdemo/mpdemo'
Sys.setenv(OPENBLAS_NUM_THREADS = '1', OMP_NUM_THREADS = '1', MKL_NUM_THREADS = '1')
if (!file.exists(mpdemo)) stop('Mplus Demo binary unavailable')
joint_x_only <- '--joint-x' %in% commandArgs(trailingOnly = TRUE)
if (joint_x_only) scratch <- paste0(scratch, '-joint-x')
categorical_only <- '--categorical' %in% commandArgs(trailingOnly = TRUE)
if (categorical_only) scratch <- paste0(scratch, '-categorical')
bracket_only <- '--brackets' %in% commandArgs(trailingOnly = TRUE)
if (bracket_only) scratch <- paste0(scratch, '-brackets')
probes <- list()
add <- function(id, variant, model = 'f BY y1-y3;', names = 'y1 y2 y3',
                variable = '', analysis = 'ESTIMATOR = ML;', data = '',
                title = NULL, kind = 'continuous', note = NULL) {
  probes[[length(probes) + 1L]] <<- list(id = id, variant = variant,
    model = model, names = names, variable = variable, analysis = analysis,
    data = data, title = title, kind = kind, note = note)
}
cfa <- 'f BY y1-y3;'
grouping <- 'GROUPING = g (1 = g1 2 = g2);'
mg <- function(id, variant, model = cfa, names = 'y1 y2 y3',
               variable = '', analysis = 'ESTIMATOR = ML;', ...) {
  add(id, variant, model, paste(names, 'g'), paste(grouping, variable), analysis,
      kind = 'groups', ...)
}
catadd <- function(id, variant, model = 'f BY u1-u3;', names = 'u1 u2 u3',
                   variable = '', analysis = 'ESTIMATOR = WLSMV;', groups = FALSE, ...) {
  ord <- paste(strsplit(names, ' ')[[1]][startsWith(strsplit(names, ' ')[[1]], 'u')], collapse = ' ')
  variable <- paste('CATEGORICAL =', ord, ';', variable)
  if (groups) mg(id, variant, model, names, variable, analysis, ...)
  else add(id, variant, model, names, variable, analysis, kind = 'ordinal', ...)
}
add('P-LX1', 'inline', 'y1 ON x1; !* comment *! y2 ON x1;', 'y1 y2 x1')
add('P-LX1', 'multiline', 'y1 ON x1; !* comment\ncontinued *! y2 ON x1;', 'y1 y2 x1')
for (a in c('GENE', 'GEN')) add('P-LX2', tolower(a), analysis = paste0('TYPE = ', a, ';'))
for (v in c('USEVAR', 'USE')) add('P-LX2', tolower(v), variable = paste0(v, ' = y1-y3;'))
add('P-LX2', 'esti', analysis = 'ESTI = ML;')
add('P-LX2', 'nocovar', analysis = 'MODEL = NOCOVAR;')
for (t in c('anal', 'model')) add('P-LX3', t, title = paste0('TITLE: test ', t, ': x'))
for (n in c(91, 200)) add('P-LX4', paste0('columns_', n),
  paste0('f BY y1 y2', paste(rep(' ', n - 14), collapse = ''), ' y3;'),
  note = 'MODEL body line is exactly the stated column count; the final indicator and semicolon extend the BY statement at the line end.')
add('P-NM1', 'mixed_range', 'y08-y11 z;', 'y08-y11 a1b-a3b z',
    variable = 'USEV = y08-y11 z;', kind = 'nm_range',
    note = 'Eight NAMES data columns; only four y variables and z analyzed to stay below the six-dependent-variable Demo cap. NAMES syntax unchanged.')
add('P-NM1', 'mixed_members', 'a1b a2b a3b z;', 'y08-y11 a1b-a3b z',
    variable = 'USEV = a1b a2b a3b z;', kind = 'nm_range',
    note = 'Analyze explicit candidate members of the mixed range and z (four dependent variables); original NAMES syntax unchanged.')
add('P-NM2', 'long_names', 'f BY longname_1 longname_2 y3;', 'longname_1 longname_2 y3')
add('P-NM2', 'shared_prefix', 'f BY abcdefgh1 abcdefgh2 y3;', 'abcdefgh1 abcdefgh2 y3')
add('P-NM3', 'names_order', 'f BY y1-y3;', 'y1 x1 y2 y3', 'USEV = y1-y3;')
add('P-NM3', 'use_order', 'f BY y1-y3;', 'y1 x1 y2 y3', 'USEV = y3 y1 y2 x1;')
add('P-NM3', 'mixed_range', 'f BY y1-y3; f-y3;', note = 'Range f-y3 spans latent f and observed y3.')
add('P-MS1', 'two_by', 'f BY y1 y2; f BY y3 y4;', 'y1 y2 y3 y4')
add('P-MS1', 'repeat_marker', 'f BY y1-y3; f BY y1;')
add('P-MS1', 'start_and_fix', 'f BY y1-y4*0.5 y2@1;', 'y1 y2 y3 y4')
for (m in c('', 'x1;', '[x1];', 'x1 WITH x2;')) add('P-MS2', c('baseline','variance','mean','covariance')[match(m,c('', 'x1;', '[x1];', 'x1 WITH x2;'))],
  paste('y1 ON x1 x2;', m), 'y1 x1 x2', 'MISSING = ALL (-99);', kind = 'missing_x')
add('P-LB1', 'factor_at', 'f BY y1-y3; f@;')
add('P-LB1', 'residual_at', 'f BY y1* y2-y3; y1@;')
add('P-LB2', 'label_range', 'y1-y3 ON x1-x2 (p1-p6);\nMODEL CONSTRAINT: NEW(d); d = p2 - p3;', 'y1 y2 y3 x1 x2')
add('P-LB3', 'slopes', 'y1 ON x1 (b); y2 ON x1 (b);', 'y1 y2 x1')
add('P-LB3', 'variances', 'y1 y2 (v);', 'y1 y2')
for (r in c('1-4','a2-a4','a1-a4')) {
  labels <- if (r == 'a2-a4') 'a2+a3+a4' else 'a1+a2+a3+a4'
  tail <- if (r == '1-4') '' else paste0('\nMODEL CONSTRAINT: NEW(r); r = ', labels, ';')
  add('P-LB4', r, paste0('f BY y1-y4 (', r, ');', tail), 'y1 y2 y3 y4')
}
add('P-LB5', 'trailing_indicator', 'f BY y1-y4 (1) y5;', 'y1 y2 y3 y4 y5')
add('P-LB6', 'fixed_label', 'f BY y1@1 (l1) y2-y3;\nMODEL CONSTRAINT: NEW(r); r = l1;')
add('P-LB6', 'fixed_label_own_line', 'f BY y1@1 (l1)\n  y2-y3;\nMODEL CONSTRAINT: NEW(r); r = 2*l1;', note = 'Label on an explicitly fixed loading, on its own line so LB03 does not interfere.')
add('P-LB6', 'case_label', 'y1 ON x1 (MiXeD);\nMODEL CONSTRAINT: NEW(r); r = mixed;', 'y1 x1')
add('P-LB6', 'long_label', 'y1 ON x1 (abcdefghi);\nMODEL CONSTRAINT: NEW(r); r = abcdefghi;', 'y1 x1')
add('P-DF1', 'latent_observed', 'f BY y1-y3; f ON x1; y4 ON x1;', 'y1 y2 y3 y4 x1')
add('P-DF2', 'unrelated', cfa, 'y1 y2 y3 y4 x1', 'USEV = y1-y4 x1;')
add('P-DF3', 'second_order', 'f1 BY y1-y2; f2 BY y3-y4; f3 BY f1 f2; f1 f2 ON x1;', 'y1 y2 y3 y4 x1')
two <- 'f1 BY y1-y3; f2 BY y4-y6; f1 ON x1; f2 ON x1;'
add('P-DF3', 'first_order', two, 'y1 y2 y3 y4 y5 y6 x1')
add('P-DF4', 'nocovariances', 'f1 BY y1-y3; f2 BY y4-y6; f1 f2 ON x1; y1 WITH y4;',
    'y1 y2 y3 y4 y5 y6 x1', analysis = 'MODEL = NOCOVARIANCES; ESTIMATOR = ML;')
add('P-DF5', 'single', analysis = 'MODEL = NOMEANSTRUCTURE; ESTIMATOR = ML;')
mg('P-DF5', 'groups', analysis = 'MODEL = NOMEANSTRUCTURE; ESTIMATOR = ML;')
for (a in c('MISSING','MEANSTRUCTURE','GENERAL MISSING H1')) add('P-DF6', gsub(' ', '_', tolower(a)), analysis = paste0('TYPE = ', a, ';'))
mg('P-MG1', 'second_order', 'f1 BY y1-y2; f2 BY y3-y4; f3 BY f1 f2;', 'y1 y2 y3 y4')
add('P-MG3', 'reversed_codes', cfa, 'y1 y2 y3 g', 'GROUPING = g (2 = b 1 = a);', kind = 'three_groups')
for (k in c(2,3)) add('P-MG4', paste0('count_',k), cfa, 'y1 y2 y3 g', paste0('GROUPING = g (',k,');'), kind = 'codes_5_7')
mg('P-MG5', 'named', 'y1 (a);', 'y1 y2')
mg('P-MG5', 'named_pair', 'y1 (a); y2 (a);', 'y1 y2')
mg('P-MG5', 'released', 'y1 (1);\nMODEL g2: y1;', 'y1 y2')
mg('P-MG5', 'reused', 'y1 (a);\nMODEL g2: y2 (a);', 'y1 y2')
mg('P-MG7', 'on_latent', 'f1 BY y1-y3; f2 BY y4-y5; y4 ON f1; y6 ON f1;', 'y1 y2 y3 y4 y5 y6')
add('P-DF7', 'final_dependents', 'y1 y2 ON x1; y3 ON y1;', 'y1 y2 y3 x1')
for (y in c('y1','y2')) mg('P-MG6', y, paste0(cfa, '\nMODEL g2: f BY ',y,';'))
mg('P-IV1', 'markers', 'f1 BY y1-y3; f2 BY y4-y6;', 'y1 y2 y3 y4 y5 y6', analysis = 'MODEL = CONFIGURAL METRIC SCALAR (MODEL); ESTIMATOR = ML;')
mg('P-IV1', 'variance', 'f1 BY y1* y2-y3; f2 BY y4* y5-y6; f1 f2@1;', 'y1 y2 y3 y4 y5 y6', analysis = 'MODEL = CONFIGURAL METRIC SCALAR (MODEL); ESTIMATOR = ML;', note = 'As run by the lane: f1 f2@1 frees f1 and fixes only f2, so f1 is unidentified.')
mg('P-IV1', 'variance_both', 'f1 BY y1* y2-y3; f2 BY y4* y5-y6; f1@1 f2@1;', 'y1 y2 y3 y4 y5 y6', analysis = 'MODEL = CONFIGURAL METRIC SCALAR (MODEL); ESTIMATOR = ML;', note = 'Variance identification for both factors (planner correction).')
for (p in c('DELTA','THETA')) catadd('P-MG2', tolower(p), 'f BY u1-u3; u4 ON x1;', 'u1 u2 u3 u4 x1', analysis = paste0('ESTIMATOR = WLSMV; PARAMETERIZATION = ',p,';'), groups = TRUE)
for (m in c('[u1$1-u1$2];','[u1$1-u4$1] (1);','[u1];')) catadd('P-CT1', c('within_range','across_range','bare')[match(m,c('[u1$1-u1$2];','[u1$1-u4$1] (1);','[u1];'))], paste('f BY u1-u4;',m), 'u1 u2 u3 u4')
catadd('P-CT2', 'predictor', 'y1 ON u1;', 'y1 u1')
catadd('P-CT3', 'unequal_categories', groups = TRUE, note = 'u1 category 3 is present only in group 1.')
catadd('P-CT4', 'cfa_ml', analysis = 'ESTIMATOR = ML;')
catadd('P-CT4', 'regression_ml', 'u1 ON x1 x2;', 'u1 x1 x2', analysis = 'ESTIMATOR = ML;')
growth <- 'i s | u1@0 u2@1 u3@2 u4@3;'
for (p in c('DELTA','THETA','ML')) catadd('P-GR1', tolower(p), growth, 'u1 u2 u3 u4', analysis = if (p == 'ML') 'ESTIMATOR = ML;' else paste0('ESTIMATOR = WLSMV; PARAMETERIZATION = ',p,';'))
for (p in c('DELTA','THETA','ML')) catadd('P-GR2', tolower(p), growth, 'u1 u2 u3 u4', groups = TRUE, analysis = if (p == 'ML') 'ESTIMATOR = ML;' else paste0('ESTIMATOR = WLSMV; PARAMETERIZATION = ',p,';'))
mg('P-GR2', 'continuous', 'i s | y1@0 y2@1 y3@2 y4@3;', 'y1 y2 y3 y4')
add('P-GR3', 'cubic', 'i s q c | y1@0 y2@1 y3@2 y4@3 y5@4;', 'y1 y2 y3 y4 y5')
add('P-GR3', 'free_time', 'i s q | y1@0 y2@1 y3 y4;', 'y1 y2 y3 y4')
add('P-GR3', 'piecewise', 'i s1 | y1@0 y2@1 y3@2; i s2 | y3@0 y4@1 y5@2;', 'y1 y2 y3 y4 y5', note = 'Two adjacent three-time segments share i and y3.')
for (v in c('before','after')) add('P-GR3', v, if (v == 'before') '[y1-y4] (1); i s | y1@0 y2@1 y3@2 y4@3;' else 'i s | y1@0 y2@1 y3@2 y4@3; [y1-y4] (1);', 'y1 y2 y3 y4')
for (v in c('explicit','implicit','new_free','derived')) {
  m <- if (v == 'derived') 'y1 ON x1 (p1); y2 ON x1 (q1);' else 'y1 ON x1 (p1); y2 ON x1 (p2); y3 ON x1 (p3);'
  cn <- switch(v, explicit = 'p1 = p2**2 + p3**2;', implicit = '0 = p1 - p2**2 - p3**2;', new_free = 'NEW(c); p2 = p1 + c;', derived = 'NEW(r); r = p1/q1;')
  add('P-CN1', v, paste0(m,'\nMODEL CONSTRAINT: ',cn), 'y1 y2 y3 x1')
}
for (v in c('forward','reverse')) add('P-CN2', v, paste0('y1 ON x1; y2 ON y1 x1; y3 ON y2 y1 x1;\nMODEL INDIRECT: y3 IND ',if (v == 'forward') 'y2 y1' else 'y1 y2',' x1;'), 'y1 y2 y3 x1')
add('P-CN2', 'factor', 'f BY y1-y3; f ON x; y ON f x;\nMODEL INDIRECT: y IND f x;', 'y1 y2 y3 y x')
add('P-CN2', 'continuous', 'm ON x; y ON m x;\nMODEL INDIRECT: y IND m x;', 'y m x')
for (v in c('empty','extra','wrapped')) add('P-DA1', v, 'y1 y2 y3;', kind = 'free_special')
for (v in c('99','9.9','-9','-9.0')) add('P-DA2', v, 'y1 y2 y3;', variable = paste0('MISSING = ALL (',v,');'), data = 'FORMAT = 3F2.1;', kind = 'fixed_special')

mg('P-MG8', 'group_only_covariance', paste0(cfa, '\nMODEL g2: y1 WITH y2;'))
mg('P-MG9', 'cumulative', paste0(cfa, '\nMODEL g2: f BY y2;\nMODEL g2: [y3];'))
mg('P-MG9', 'override', paste0(cfa, '\nMODEL g2: f BY y2@0.7;\nMODEL g2: f BY y2@0.8;'))
mg('P-MG9', 'release', paste0(cfa, '\nMODEL g2: f BY y2@0.7;\nMODEL g2: f BY y2;'))
mg('P-MG10', 'multi_label', paste0(cfa, '\nMODEL g1 g2: [y3];'))
add('P-MG11', 'fractional', cfa, 'y1 y2 y3 g',
    'GROUPING = g (-1 = g1 2.5 = g2);', kind = 'fractional_codes')
add('P-MG12', 'negative_integer', cfa, 'y1 y2 y3 g',
    'GROUPING = g (-1 = g1 2 = g2);', kind = 'negative_codes')
add('P-MG12', 'decimal_integer', cfa, 'y1 y2 y3 g',
    'GROUPING = g (-1.0 = g1 2.0 = g2);', kind = 'negative_codes')

make_data <- function(p, seed) {
  set.seed(seed)
  names <- strsplit(p$names, ' +')[[1]]
  if (p$kind == 'nm_range') names <- c(paste0('y', sprintf('%02d',8:11)), paste0('a',1:3,'b'), 'z')
  grouped <- 'g' %in% names
  codes <- if (p$kind == 'three_groups') 1:3 else if (p$kind == 'codes_5_7') c(5,7) else if (p$kind == 'fractional_codes') c(-1,2.5) else if (p$kind == 'negative_codes') c(-1,2) else 1:2
  n <- if (grouped) 500L * length(codes) else 500L
  latent <- rnorm(n)
  latent2 <- .3 * latent + sqrt(1-.3^2) * rnorm(n)
  slope <- .3*rnorm(n)
  quadratic <- .03*rnorm(n)
  cubic <- .005*rnorm(n)
  x <- matrix(0, n, length(names), dimnames = list(NULL, names))
  for (j in seq_along(names)) {
    nm <- names[j]
    x[,j] <- if (nm == 'g') rep(codes, each = 500) else if (startsWith(nm,'x')) rnorm(n) else .7 * latent + rnorm(n)
    if (grepl('f2 BY',p$model,fixed=TRUE) && nm %in% c('y4','y5','y6')) x[,j] <- .7*latent2+rnorm(n)
    if (p$id == 'P-IV2' && nm %in% c('u4', 'u5', 'u6')) x[,j] <- .7*latent2+rnorm(n)
    if (p$id == 'P-MG7' && nm == 'y4') x[,j] <- x[,j]+.25*latent
    if (p$id == 'P-MG7' && nm == 'y6') x[,j] <- .4*latent+rnorm(n)
    if (p$id == 'P-LB1' && p$variant == 'factor_at') x[,j] <- sqrt(.05)*latent+rnorm(n)
    if (p$id %in% c('P-GR1','P-GR2','P-GR3') && grepl('^[yu][1-5]$',nm)) {
      time <- as.integer(substring(nm,2))-1
      x[,j] <- latent+time*slope+rnorm(n,sd=.6)
      if(p$variant == 'cubic') x[,j] <- x[,j]+time^2*quadratic+time^3*cubic
    }
    if (startsWith(nm,'u')) x[,j] <- as.integer(cut(x[,j], c(-Inf,-.5,.5,Inf)))
  }
  if (p$id == 'P-IV2' && grepl('_binary_', p$variant)) {
    for (nm in names[startsWith(names, 'u')]) x[, nm] <- as.integer(x[, nm] > 1)
  }
  if (p$id == 'P-CT3') x[x[,'g'] == 2 & x[,'u1'] == 3, 'u1'] <- 2
  if (p$kind == 'missing_x') x[sample.int(n, n/10), 'x1'] <- -99
  x
}
write_data <- function(p, x, path) {
  if (p$kind == 'divisor_summary') {
    S <- matrix(c(2,.5,.5,1),2)
    if(grepl('CORR',p$data)) {
      sd <- sqrt(diag(S)); S <- cov2cor(S)
    }
    lines <- if(grepl('MEANS',p$data)) '3 4' else character()
    if(grepl('STDEVIATIONS',p$data)) lines <- c(lines,paste(sd,collapse=' '))
    lines <- c(lines,if(grepl('FULL',p$data)) apply(S,1,paste,collapse=' ') else c('1',paste(S[2,],collapse=' ')))
    if(!grepl('CORR|FULL',p$data)) lines <- c(if(grepl('MEANS',p$data)) '3 4', '2','.5 1')
    if(grepl('NGROUPS',p$data)) lines <- rep(lines,2)
    writeLines(lines,path)
  } else if (p$kind == 'summary') {
    S <- crossprod(scale(x,scale=FALSE))/nrow(x)
    lines <- character()
    if(p$variant=='with_means') lines <- c(lines,paste(colMeans(x),collapse=' '))
    if(p$id=='P-DA4') {
      if(p$variant=='sd') lines <- c(lines,paste(sqrt(diag(S)),collapse=' '))
      S <- cov2cor(S)
    }
    lines <- c(lines,vapply(seq_len(ncol(S)),function(j) paste(S[j,seq_len(j)],collapse=' '),''))
    writeLines(lines,path)
  } else if (p$kind == 'fixed_special') {
    # F2.1 supplies an implied decimal: field 99 is 9.9; field -9 is -0.9.
    fields <- matrix(sprintf('%02d', sample(10:89, length(x), replace = TRUE)), nrow(x))
    fields[1:50,1] <- if (startsWith(p$variant,'-')) '-9' else '99'
    writeLines(apply(fields,1,paste0,collapse=''),path)
  } else if (p$kind == 'free_special') {
    lines <- apply(round(x,6),1,paste,collapse=',')
    if (p$variant == 'empty') lines[1] <- '1,,3'
    if (p$variant == 'extra') lines <- paste0(lines, ',4')
    if (p$variant == 'wrapped') lines <- unlist(lapply(seq_len(nrow(x)), function(i) c(paste(round(x[i,1:2],6),collapse=' '),as.character(round(x[i,3],6)))))
    writeLines(lines,path)
  } else write.table(x,path,row.names=FALSE,col.names=FALSE,quote=FALSE)
}
number <- function(line) {
  z <- regmatches(line, gregexpr('[-+]?[0-9]+(?:\\.[0-9]+)?(?:[Ee][-+]?[0-9]+)?',line,perl=TRUE))[[1]]
  if (length(z)) as.numeric(z) else numeric()
}
parse_output <- function(out) {
  errors <- substr(trimws(grep('\\*\\*\\* ERROR',out,value=TRUE)),1,200)
  warnings <- substr(trimws(grep('\\*\\*\\* WARNING',out,value=TRUE)),1,200)
  diagnostics <- list()
  for (i in grep('\\*\\*\\* (ERROR|WARNING)',out)) {
    end <- i+1L
    while (end <= length(out) && nzchar(trimws(out[end])) && !grepl('\\*\\*\\*',out[end])) end <- end+1L
    diagnostics[[length(diagnostics)+1L]] <- list(header=substr(trimws(out[i]),1,200),message=paste(trimws(out[seq.int(i+1L,end-1L)]),collapse=' '))
  }
  chi <- list()
  for (i in grep('^\\s*Chi-Square Test of Model Fit\\s*$',out)) {
    blk <- out[i:min(i+10L,length(out))]
    val <- grep('^\\s*Value\\s',blk,value=TRUE)
    df <- grep('Degrees of Freedom',blk,value=TRUE)
    chi[[length(chi)+1L]] <- list(value=if(length(val)) number(val[1])[1] else NULL,df=if(length(df)) number(df[1])[1] else NULL)
  }
  nobs <- list()
  for (i in grep('^\\s*Number of observations',out)) {
    nums <- number(out[i])
    if (length(nums)) nobs[[length(nobs)+1L]] <- list(group='single',n=tail(nums,1))
    else for (ln in out[seq.int(i+1L,min(i+15L,length(out)))]) {
      m <- regmatches(ln,regexec('^\\s*Group\\s+(\\S+)\\s+([0-9]+)\\s*$',ln))[[1]]
      if(length(m)) nobs[[length(nobs)+1L]] <- list(group=m[2],n=as.integer(m[3]))
    }
  }
  free <- lapply(grep('Number of Free Parameters',out,value=TRUE), function(x) tail(number(x),1))
  # Summary variable lists are authoritative even when SAMPLE STATISTICS is not
  # requested. Retain the role and group/model section ordering explicitly.
  order <- list(); role <- NULL
  for (ln in out) {
    if (grepl('^\\s*(Observed dependent variables|Observed independent variables|Continuous latent variables)\\s*$',ln)) {role <- trimws(ln);next}
    if (!is.null(role)) {
      s <- trimws(ln)
      if (grepl('^(Continuous|Categorical|Binary and ordered categorical)$',s) || !nzchar(s)) next
      if (grepl('^[A-Z][A-Z0-9_]*( +[A-Z][A-Z0-9_]*)*$',s)) {
        order[[length(order)+1L]] <- list(role=role,variables=as.list(strsplit(s,' +')[[1]]))
      } else role <- NULL
    }
  }
  tech <- list(); results <- list(); group <- 'single'; section <- ''; in_tech <- FALSE; in_results <- FALSE
  model_family <- 'single'; matrix_name <- NULL; cols <- character(); rows <- list(); row_names <- character()
  flush <- function() {
    if (!is.null(matrix_name) && length(rows)) {
      key <- paste(model_family,group,matrix_name,sep='/')
      mat <- tech[[key]]
      if (is.null(mat)) mat <- list(model=model_family,group=group,name=matrix_name,columns=character(),rows=list())
      mat$columns <- unique(c(mat$columns,cols))
      row_names <- make.unique(row_names, sep = '#')
      for (r in seq_along(rows)) {
        old <- mat$rows[[row_names[r]]]
        if (is.null(old)) old <- list()
        for (j in seq_along(rows[[r]])) old[[cols[j]]] <- rows[[r]][j]
        mat$rows[[row_names[r]]] <- old
      }
      tech[[key]] <<- mat
    }
    rows <<- list(); row_names <<- character()
  }
  for (ln in out) {
    s <- trimws(ln)
    if (grepl('^MODEL RESULTS',s)) {flush();matrix_name <- NULL;in_results <- TRUE;in_tech <- FALSE;group <- 'single';model_family <- sub('^MODEL RESULTS(?: FOR THE (.*) MODEL)?$', '\\1', s, perl=TRUE);if(!nzchar(model_family)) model_family <- 'single';section <- '';next}
    if (grepl('^(STANDARDIZED MODEL RESULTS|QUALITY OF NUMERICAL RESULTS|TOTAL, TOTAL INDIRECT|TECHNICAL 1 OUTPUT)',s)) {in_results <- FALSE}
    if (startsWith(s, 'TECHNICAL 1 OUTPUT')) {flush();matrix_name <- NULL;in_tech <- TRUE;group <- 'single';model_family <- sub('^TECHNICAL 1 OUTPUT(?: FOR THE (.*) MODEL)?$', '\\1',s,perl=TRUE);if(!nzchar(model_family)) model_family <- 'single';next}
    if (grepl('^TECHNICAL [2-9]',s)) {flush();in_tech <- FALSE;matrix_name <- NULL}
    gm <- regmatches(s,regexec('^(?:Group|GROUP) (.+)$',s))[[1]]
    if (length(gm) && (in_results || in_tech)) {flush();matrix_name <- NULL;group <- gm[2];next}
    if (in_results) {
      md <- regmatches(s,regexec('^(\\S+)\\s+(-?[0-9]+\\.[0-9]+)(?:\\s|$)',s))[[1]]
      if (length(md)) results[[length(results)+1L]] <- list(model=model_family,group=group,block=section,parameter=md[2],estimate=as.numeric(md[3]))
      else if (nzchar(s) && !grepl('^(Estimate|Two-Tailed|S.E.)',s)) section <- s
    }
    if (in_tech) {
      if (grepl('^(PARAMETER SPECIFICATION|STARTING VALUES)',s)) {flush();matrix_name <- NULL;section <- if(startsWith(s,'PARAMETER SPECIFICATION')) 'PARAMETER SPECIFICATION' else 'STARTING VALUES';if(grepl(' FOR ',s)) group <- sub('.* FOR ','',s);next}
      if (section != 'PARAMETER SPECIFICATION') next
      if (grepl('^(NU|LAMBDA|THETA|ALPHA|BETA|PSI|TAU|DELTA|KAPPA|GAMMA|New/Additional Parameters)( [0-9]+)?$',s)) {flush();matrix_name <- s;cols <- character();next}
      if (is.null(matrix_name) || !nzchar(s) || grepl('^-+$',s)) next
      tok <- strsplit(s,' +')[[1]]
      if (all(grepl('^[A-Z][A-Z0-9_$]*$',tok))) {flush();cols <- make.unique(tok, sep = '#');next}
      if (length(cols) && all(grepl('^[0-9]+$',tok))) {
        row_names <- c(row_names,'vector');rows[[length(rows)+1L]] <- as.integer(tok);next
      }
      if (length(tok)>1 && length(cols)>=length(tok)-1 && all(grepl('^[0-9]+$',tok[-1]))) {
        row_names <- c(row_names,tok[1]);rows[[length(rows)+1L]] <- as.integer(tok[-1])
      }
    }
  }
  flush()
  list(status=if(length(errors)) 'error' else 'accepted',
       version=trimws(grep('^Mplus VERSION',out,value=TRUE)[1]),
       errors=as.list(errors),warnings=as.list(warnings),diagnostics=diagnostics,
       observations=nobs,free_parameters=free,chi_square=chi,variable_order=order,
       tech1=unname(tech),model_results=results,
       estimation_messages=as.list(trimws(grep('TERMINATED NORMALLY|NO CONVERGENCE|COULD NOT BE|NOT BE IDENTIFIED|NOT POSITIVE DEFINITE|PROBLEM INVOLVING|^ +Parameter [0-9]',out,value=TRUE))),
       integration=extract_section(out,'Integration Specifications',c('Input data file','SUMMARY OF DATA')),
       indirect_effects=extract_section(out,'TOTAL, TOTAL INDIRECT',c('TECHNICAL','QUALITY','MODEL COMMAND')),
       generated_model=extract_section(out,'MODEL COMMAND',c('TECHNICAL','DIAGRAM','Beginning Time')),
       sample_statistics=extract_section(out,'SAMPLE STATISTICS',c('THE MODEL','MODEL FIT','Beginning Time','Mplus VERSION')))
}
extract_section <- function(out, start, ends) {
  idx <- grep(start,out)
  if(!length(idx)) return(list())
  lapply(idx,function(i) {
    j <- i+1L
    while(j<=length(out) && !any(vapply(ends,function(e) grepl(e,out[j],fixed=TRUE),logical(1)))) j<-j+1L
    as.list(out[i:min(j-1L,length(out))])
  })
}
# Bind each result to the inventory's Settles cell without duplicating rule text.
inventory <- readLines(file.path(root,'project/grammar/mplus_source_inventory.md'))
probe_lines <- grep('^\\| P-',inventory,value=TRUE)
mg('P-MG13', 'group_only_regression', 'f BY y1-y3;\nMODEL g2: y4 ON x1;', 'y1 y2 y3 y4 x1')
mg('P-MG13', 'group_only_indicator', 'f BY y1-y3;\nMODEL g2: f BY y4;', 'y1 y2 y3 y4')
for (v in c('no_means','intercept','factor_mean','with_means')) add('P-DA3', v,
  paste('f BY y1-y3;', switch(v,intercept='[y1];',factor_mean='[f];','')),
  data=paste('TYPE = COVARIANCE',if(v=='with_means') 'MEANS' else '', '; NOBSERVATIONS = 500;'),kind='summary')
for (v in c('unit','sd')) add('P-DA4',v,
  data=paste('TYPE = CORRELATION',if(v=='sd') 'STDEVIATIONS' else '', '; NOBSERVATIONS = 500;'),kind='summary')

for (v in c('COVARIANCE','FULLCOV','CORRELATION','STDEVIATIONS CORRELATION','COVARIANCE MEANS','COVARIANCE MEANS; NGROUPS=2')) add('P-DA5',gsub('[ ;=]+','_',v),
  model='y1 y2; y1 WITH y2;',names='y1 y2',
  data=paste0('TYPE=',v,'; NOBSERVATIONS=',if(grepl('NGROUPS',v)) '5 7' else '5',';'),kind='divisor_summary')

# P-IV2 is isolated so existing fixtures and their data seeds remain unchanged.
for (parameterization in c('DELTA', 'THETA')) {
  for (categories in c('ordinal', 'binary')) {
    for (shortcut in c('CONFIGURAL', 'SCALAR', 'METRIC')) {
      catadd('P-IV2', paste(tolower(parameterization), categories,
                           tolower(shortcut), sep = '_'),
        'f1 BY u1-u3; f2 BY u4-u6;', 'u1 u2 u3 u4 u5 u6',
        analysis = paste0('ESTIMATOR = WLSMV; PARAMETERIZATION = ',
          parameterization, ';\nMODEL = ', shortcut, ' (MODEL);\n',
          'CONVERGENCE = 0.00000001; ITERATIONS = 10000;'),
        groups = TRUE, note = paste(categories, 'categorical shortcut TECH1 probe'))
    }
  }
}

# Append probes to preserve every existing seed. The same data serve all variants.
for (missing in c(FALSE, TRUE)) for (with in c(FALSE, TRUE)) for (means in c(FALSE, TRUE)) {
  add('P-MS08b', paste(if (missing) 'missing' else 'complete',
      if (with) 'with' else 'no_with', if (means) 'means' else 'no_means', sep = '_'),
      paste('f BY y1-y3; f ON x1 x2; x1 x2;',
            if (with) 'x1 WITH x2;' else '', if (means) '[x1 x2];' else ''),
      'y1 y2 y3 x1 x2', 'MISSING = ALL (-99);',
      analysis = 'ESTIMATOR = ML; CONVERGENCE = 0.00000001; ITERATIONS = 10000;',
      kind = if (missing) 'missing_x' else 'continuous')
}

add('P-LB7', 'distinct_lines', 'f BY y1-y3;\n[y1] (i1)\n[y2] (i2)\n[y3] (i3);')
add('P-LB7', 'shared_lines', 'f BY y1-y3;\n[y1] (i)\n[y2] (i)\n[y3] (i);')
add('P-LB7', 'numbers_lines', 'f BY y1-y3;\n[y1] (1)\n[y2] (1)\n[y3] (1);')
catadd('P-LB7', 'threshold_lines', 'f BY u1-u3;\n[u1$1] (t1)\n[u2$1] (t2);')
catadd('P-LB7', 'scale_lines', 'f BY u1-u3;\n{u1*} (s1)\n{u2*} (s2);')
add('P-LB7', 'variance_lines', 'f BY y1-y3;\ny1 (v1)\ny2 (v2)\ny3@4;')
add('P-LB7', 'modifiers_lines', 'f BY y1-y3;\n[y1@2]\n[y2@3]\n[y3@4];')
add('P-LB7', 'bare_same_line', 'f BY y1-y3; [y1] [y2] [y3];')
add('P-LB7', 'same_line', 'f BY y1-y3;\n[y1] (i1) [y2] (i2);')

settles <- setNames(lapply(probe_lines,function(x) trimws(strsplit(x,'|',fixed=TRUE)[[1]][3])),
                    vapply(probe_lines,function(x) trimws(strsplit(x,'|',fixed=TRUE)[[1]][2]),character(1)))
ids <- unique(vapply(probes,`[[`,character(1),'id'))
if (!setequal(ids,names(settles))) stop('Probe inventory mismatch')
if (bracket_only) probes <- Filter(function(p) p$id == 'P-LB7', probes)
if (joint_x_only) probes <- Filter(function(p) p$id == 'P-MS08b', probes)
if (categorical_only) probes <- Filter(function(p) p$id == 'P-IV2', probes)
unlink(scratch,recursive=TRUE)
dir.create(scratch,recursive=TRUE)
all_results <- list()
for (p in probes) {
  dir <- file.path(scratch,p$id,p$variant);dir.create(dir,recursive=TRUE)
  x <- make_data(p,58000L+match(p$id,ids))
  write_data(p,x,file.path(dir,'probe.dat'))
  variable <- paste0('NAMES = ',p$names,';\n',p$variable)
  input <- c(if(is.null(p$title)) paste0('TITLE: ',p$id,' ',p$variant,';') else p$title,
    paste0('DATA: FILE = probe.dat; ',p$data),'VARIABLE:',variable,
    'ANALYSIS:',p$analysis,'MODEL:',p$model,'OUTPUT: TECH1;')
  writeLines(input,file.path(dir,'probe.inp'))
  old <- getwd();setwd(dir)
  status <- system2(mpdemo,c('probe.inp','probe.out'),stdout='console.log',stderr='console.err',timeout=120)
  setwd(old)
  path <- file.path(dir,'probe.out')
  if(!file.exists(path) || status %in% c(124,137,139)) stop('Demo execution failed: ',p$id,'/',p$variant,' status ',status)
  out <- readLines(path,warn=FALSE)
  if(!any(grepl('Mplus VERSION',out))) stop('Demo did not produce a version header')
  parsed <- parse_output(out)
  if(status != 0 && parsed$status != 'error') stop('Demo execution failed without an input diagnostic: ',p$id,'/',p$variant)
  if(is.null(all_results[[p$id]])) all_results[[p$id]] <- list(settles=settles[[p$id]],variants=list())
  if (p$id == 'P-MS08b') {
    if (as.character(utils::packageVersion('lavaan')) !=
        gsub('-', '.', trimws(readLines(file.path(root, 'cpp/tests/fixtures/lavaan_version.txt'))[1]), fixed = TRUE))
      stop('Pinned lavaan mismatch')
    observed <- as.data.frame(x); observed[observed == -99] <- NA_real_
    syntax <- paste('f =~ 1*y1 + y2 + y3; f ~ x1 + x2;',
      'f ~~ f; f ~ 0*1; y1 ~~ y1; y2 ~~ y2; y3 ~~ y3;',
      'y1 ~ 1; y2 ~ 1; y3 ~ 1; x1 ~ 1; x2 ~ 1;',
      'x1 ~~ x1 + x2; x2 ~~ x2;')
    fit <- lavaan::lavaan(syntax, data = observed, fixed.x = FALSE,
      meanstructure = TRUE, missing = if (anyNA(observed)) 'ml' else 'listwise',
      auto.var = FALSE, auto.cov.lv.x = FALSE, auto.cov.y = FALSE,
      control = list(iter.max = 10000))
    pt <- lavaan::parTable(fit)
    fm <- lavaan::fitMeasures(fit, c('df', 'chisq'))
    stopifnot(lavaan::lavInspect(fit, 'converged'), fm['df'] == parsed$chi_square[[1]]$df,
      abs(fm['chisq'] - parsed$chi_square[[1]]$value) < .002,
      parsed$observations[[1]]$n == nrow(observed))
    for (row in parsed$model_results) {
      relation <- strsplit(row$block, ' +')[[1]]
      if (length(relation) == 2 && relation[2] %in% c('BY','ON','WITH')) {
        lhs <- tolower(relation[1]); rhs <- tolower(row$parameter)
        op <- c(BY='=~', ON='~', WITH='~~')[[relation[2]]]
      } else if (row$block %in% c('Means', 'Intercepts')) {
        lhs <- tolower(row$parameter); rhs <- ''; op <- '~1'
      } else if (row$block %in% c('Variances', 'Residual Variances')) {
        lhs <- rhs <- tolower(row$parameter); op <- '~~'
      } else stop('Unmapped Demo block: ', row$block)
      hit <- which(pt$lhs == lhs & pt$op == op & pt$rhs == rhs)
      if (!length(hit) && op == '~~') hit <- which(pt$lhs == rhs & pt$op == op & pt$rhs == lhs)
      stopifnot(length(hit) == 1, abs(pt$est[hit] - row$estimate) < .001)
    }
    parsed$numeric_gate <- list(lavaan_version = as.character(utils::packageVersion('lavaan')),
      syntax = syntax, data = unname(as.matrix(observed)), rows = pt[,c('lhs','op','rhs','free','est')],
      chisq = unname(fm['chisq']), df = unname(fm['df']))
  }
  all_results[[p$id]]$variants[[p$variant]] <- c(list(variable=variable,analysis=p$analysis,
    model=p$model,data=p$data,title=p$title,seed=58000L+match(p$id,ids),note=p$note),parsed)
  cat(p$id,p$variant,parsed$status,'\n')
}
write_fixture <- function(results, name) {
  fixture <- file.path(root, 'cpp/tests/fixtures/mplus', name)
  dir.create(dirname(fixture), recursive = TRUE, showWarnings = FALSE)
  writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = NA,
                   null = 'null'), fixture)
}
if (!joint_x_only && !categorical_only) write_fixture(all_results[names(all_results) == 'P-LB7'], 'probes_brackets.json')
if (!bracket_only && !categorical_only && !joint_x_only) write_fixture(all_results[!names(all_results) %in% c('P-IV2', 'P-MS08b', 'P-LB7')], 'probes.json')
if (!bracket_only && !joint_x_only) write_fixture(all_results[names(all_results) == 'P-IV2'], 'probes_categorical.json')
if (!bracket_only && !categorical_only) write_fixture(all_results[names(all_results) == 'P-MS08b'], 'probes_joint_x.json')
cat(length(unique(vapply(probes, `[[`, character(1), 'id'))), 'probes;', length(probes), 'variants\n')
