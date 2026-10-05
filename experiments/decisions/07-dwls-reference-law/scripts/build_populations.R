#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
here <- normalizePath(file.path(dirname(sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])), '..'))
source(file.path(here, '../../_support/R/helpers.R'))
if ('--help' %in% args) {
  cat('Build frozen derived population JSON: --corpus ROOT (default corpus_root()). No simulation.\n'); quit()
}
root <- if ('--corpus' %in% args) args[match('--corpus', args)+1] else corpus_root()
if (!corpus_available(root)) stop('Missing optional textbook corpus: ', root)
ids <- c(ptsd8='brown_2015_tab8_8_reliability_ptsd', mtmm9='brown_2015_tab6_3_mtmm_correlated_uniqueness',
  coping12='brown_2015_tab8_2_higher_order_coping', bifactor15='brown_2015_tab8_6_bifactor_perceived_control',
  long18='little_2013_ch5_tab5_5_configfi_2by3', long30='little_2013_ch8_tab8_bull_config',
  worland11='kline_2023_ch15_worland_sr_step2a', mdd9='brown_2015_tab7_14_invariance_mdd_scalar')
ledger <- populations <- list()
for (id in names(ids)) {
  cat('Building ', id, '\n'); flush.console()
  paths <- list.files(file.path(root, 'cases'), recursive=TRUE, full.names=TRUE)
  path <- paths[endsWith(paths, paste0('/', ids[[id]], '/expected/lavaan_ml.json'))]
  stopifnot(length(path)==1)
  solution <- jsonlite::fromJSON(path)
  stopifnot(isTRUE(solution$fit$converged))
  pt <- solution$parameter_table
  if ('group' %in% names(pt)) pt <- pt[pt$group==1,]
  pt <- pt[pt$op %in% c('=~','~','~~') & pt$rhs!='1',]
  latent <- unique(pt$lhs[pt$op=='=~']); obs <- setdiff(unique(pt$rhs[pt$op=='=~']), latent)
  vars <- c(latent,obs); q <- length(vars)
  directed <- covariance <- matrix(0,q,q,dimnames=list(vars,vars))
  for (i in seq_len(nrow(pt))) {
    a <- pt$lhs[i]; b <- pt$rhs[i]
    if (pt$op[i]=='=~') directed[b,a] <- pt$est[i]
    if (pt$op[i]=='~') directed[a,b] <- pt$est[i]
    if (pt$op[i]=='~~') covariance[a,b] <- covariance[b,a] <- pt$est[i]
  }
  inverse <- solve(diag(q)-directed)
  full <- inverse %*% covariance %*% t(inverse)
  original_min_rho <- min(eigen(cov2cor(full[obs,obs]), symmetric=TRUE, only.values=TRUE)$values)
  scale <- sqrt(diag(full)); stopifnot(all(is.finite(scale)),all(scale>0))
  directed <- directed*outer(1/scale,scale)
  covariance <- covariance/outer(scale,scale)
  # Null construction is deterministic and specified before draws are generated.
  load_rows <- lapply(latent,function(f) obs[obs %in% pt$rhs[pt$op=='=~' & pt$lhs==f]])
  names(load_rows) <- latent
  restrictions <- character()
  labels <- matrix('',q,q,dimnames=list(vars,vars))
  marker <- character()
  for (f in latent) {
    targets <- pt$rhs[pt$op=='=~' & pt$lhs==f]
    marker[f] <- targets[1]
    for (v in targets) labels[v,f] <- paste0('l_',f,'_',v)
  }
  if (id %in% c('ptsd8','mtmm9','coping12','bifactor15')) {
    selected <- if(id=='coping12') latent[1:3] else if(id=='bifactor15') latent[-1] else latent
    for (f in selected) {
      targets <- load_rows[[f]]
      targets <- if(id %in% c('ptsd8','coping12')) targets[2:3] else if(id=='mtmm9') targets[1:2] else targets
      directed[targets,f] <- mean(directed[targets,f])
      for(v in targets[-1]) restrictions <- c(restrictions,paste(labels[v,f],'==',labels[targets[1],f]))
    }
  }
  if(id=='long18') {
    for(i in seq_len(nrow(pt))) if(pt$op[i]=='~~' && pt$lhs[i]!=pt$rhs[i] &&
      pt$lhs[i] %in% obs && pt$rhs[i] %in% obs) {
      a <- pt$lhs[i]; b <- pt$rhs[i]; covariance[a,b] <- covariance[b,a] <- 0
      restrictions <- c(restrictions,paste0('r_',a,'_',b,' == 0'))
    }
  }
  if(id=='long30') {
    for(fs in list(latent[1:5],latent[6:10])) for(j in 1:3) {
      targets <- vapply(fs,function(f) load_rows[[f]][j], '')
      common <- mean(vapply(seq_along(fs),function(k) directed[targets[k],fs[k]],0))
      for(k in seq_along(fs)) directed[targets[k],fs[k]] <- common
      if(j>1) for(k in 2:5) restrictions <- c(restrictions,paste(labels[targets[k],fs[k]],'==',labels[targets[1],fs[1]]))
    }
  }
  if(id=='worland11') {
    cat('Original standardized Risk paths: ',paste(directed[c('Achieve','Adjust'),'Risk'],collapse=', '), '\n')
    cat('Original correlation minimum eigenvalue: ',original_min_rho,'\n')
    targets <- load_rows[['Cognitive']][2:3]
    directed[targets,'Cognitive'] <- mean(directed[targets,'Cognitive'])
    restrictions <- paste(labels[targets[1],'Cognitive'], '==', labels[targets[2],'Cognitive'])
  }
  # Reset observed residual variances after changing loadings. Preserve off-diagonals.
  inverse <- solve(diag(q)-directed)
  full <- inverse %*% covariance %*% t(inverse)
  diag(covariance)[match(obs,vars)] <- diag(covariance)[match(obs,vars)] + 1-diag(full)[match(obs,vars)]
  full <- inverse %*% covariance %*% t(inverse)
  rho <- full[obs,obs]
  # Rescale each latent so its first loading is one; equality constraints on
  # first loadings then become restrictions to the fixed marker constant.
  marker_value <- setNames(vapply(latent,function(f) directed[marker[f],f],0),latent)
  if(any(abs(marker_value)<1e-8)) stop('Zero marker: ',id)
  syntax <- character()
  for(f in latent) {
    targets <- pt$rhs[pt$op=='=~' & pt$lhs==f]
    terms <- vapply(targets,function(v) if(v==marker[f]) paste0('1*',v) else paste0(labels[v,f],'*',v),'')
    syntax <- c(syntax,paste(f,'=~',paste(terms,collapse=' + ')))
    restrictions <- gsub(labels[marker[f],f], '1', restrictions, fixed=TRUE)
  }
  for(i in seq_len(nrow(pt))) {
    a <- pt$lhs[i]; b <- pt$rhs[i]
    if(pt$op[i]=='~') syntax <- c(syntax,paste(a,'~',paste0('b_',a,'_',b,'*',b)))
    if(pt$op[i]=='~~' && a!=b) syntax <- c(syntax,paste(a,'~~',
      if(pt$est[i]==0 && pt$se[i]==0) paste0('0*',b) else paste0('r_',a,'_',b,'*',b)))
  }
  if(id=='mdd9') {
    # Group-specific labels are generated by the model constructor, not shared labels.
    syntax <- c(paste('DEPRESS =~',paste(obs,collapse=' + ')),'M1 ~~ M2')
  }
  min_residual <- min(eigen(covariance,symmetric=TRUE,only.values=TRUE)$values)
  min_rho <- min(eigen(rho,symmetric=TRUE,only.values=TRUE)$values)
  if(min_residual<=0 || min_rho<=0 || max(abs(diag(rho)-1))>1e-10)
    stop('Invalid null population ',id,': residual min=',min_residual,', rho min=',min_rho)
  z <- list(id=id,source_case=ids[[id]],source_md5=unname(tools::md5sum(path)),
    source_lavaan_version=solution$lavaan_version,observed=obs,groups=if(id=='mdd9') 2L else 1L,
    correlation=unname(rho),h1=paste(syntax,collapse='\n'),
    h0=paste(c(syntax,restrictions),collapse='\n'),restriction_df=length(restrictions),
    min_residual_eigenvalue=min_residual,min_correlation_eigenvalue=min_rho,
    marker_scale=marker_value,construction='standardized published coefficients; exact null; unit residual adjustment')
  populations[[id]] <- z
  ledger[[id]] <- data.frame(model=id,source_case=ids[[id]],source_md5=z$source_md5,
    items=length(obs),groups=z$groups,restriction_df=z$restriction_df,min_residual,min_rho)
}
# Publish only after every population passes the registered validity checks.
for(id in names(populations)) jsonlite::write_json(populations[[id]],
  file.path(here,'populations',paste0(id,'.json')),auto_unbox=TRUE,digits=16,pretty=TRUE)
write_csv(do.call(rbind,ledger),file.path(here,'populations','manifest.csv'))
