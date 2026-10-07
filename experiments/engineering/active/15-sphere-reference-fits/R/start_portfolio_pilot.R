# Paired provenance/cost pilot only; no endpoint is reused as a start.
run_start_portfolio_pilot <- function(args, here) {
  higher <- '--start-portfolio-higher-order-pilot' %in% args
  if (any(args %in% c('--help','-h'))) {
    cat('Usage: Rscript run_experiment.R --start-portfolio-pilot [--help]\n',
        'Fixed start-portfolio-pilot-v1: seed 202610071, n=200, three-factor CFA.\n',
        'One draw in native/mixed units; precompute FABIN3/layered before four stock ULS fits.\n',
        'Run serially under external timeout 180s; no retries or reference refinement.\n',sep='')
    cat('Higher-order mode: --start-portfolio-higher-order-pilot; seeds 202610072/073, four inputs, eight fits.\n')
    return(invisible(NULL))
  }
  stopifnot(all(args %in% c('--start-portfolio-pilot','--start-portfolio-higher-order-pilot')))
  library(magmaanlab)
  needed <- c('estimate_start_values','estimate_evaluate_at')
  stopifnot(all(needed %in% names(magmaan_core)))
  `%||%` <- function(x,y) if(is.null(x)||!length(x)) y else x
  out <- file.path(here,if(higher)'results/start-portfolio-higher-order-pilot-v1' else 'results/start-portfolio-pilot-v1')
  if(dir.exists(out)) stop('Preserve existing evidence; run ID already exists')
  dir.create(out,recursive=TRUE)
  write_out <- function(x,name) write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
  elapsed <- function() proc.time()[['elapsed']]
  capture <- function(expr) {
    warnings <- character(); begin <- elapsed()
    value <- tryCatch(withCallingHandlers(expr,warning=function(w) {
      warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')
    }),error=identity)
    list(value=value,seconds=elapsed()-begin,warnings=paste(warnings,collapse=' | '),
         error=if(inherits(value,'error'))conditionMessage(value) else '')
  }
  clock <- elapsed()
  model_text <- 'F1 =~ 1*x1 + x2 + x3\nF2 =~ 1*x4 + x5 + x6\nF3 =~ 1*x7 + x8 + x9'
  if(higher) model_text <- paste(model_text,'G =~ 1*F1 + F2 + F3','F1 ~~ 0*F2 + 0*F3','F2 ~~ 0*F3',sep='\n')
  spec <- model_spec(model_text,fixed_x=FALSE,meanstructure=FALSE)
  lambda <- matrix(0,9,3)
  for(j in 1:3) lambda[(3*j-2):(3*j),j] <- c(1,.8,.6)
  phi <- if(higher) .5*tcrossprod(c(1,.8,.6))+diag(.5,3) else matrix(.25,3,3)
  if(!higher) diag(phi) <- 1
  sigma <- lambda %*% phi %*% t(lambda)+diag(.6,9)
  seeds <- if(higher)c(202610072L,202610073L) else 202610071L
  RNGkind('Mersenne-Twister','Inversion','Rejection')
  samples <- list();draws <- list()
  for(seed in seeds) {
    set.seed(seed);draw <- matrix(rnorm(200*9),200,9) %*% chol(sigma)
    colnames(draw) <- paste0('x',1:9);draws[[as.character(seed)]] <- draw
    for(unit in c('native','mixed')) {
      key <- if(higher)paste(seed,unit,sep='_') else unit
      x <- sweep(draw,2,if(unit=='native')rep(1,9) else rep(c(1,.1,10),3),'*')
      samples[[key]] <- list(S=list(cov(x)*199/200),nobs=200L)
    }
  }
  saveRDS(list(draws=draws,lambda=lambda,phi=phi,sigma=sigma,model_text=model_text),file.path(out,'population.rds'))
  writeLines(model_text,file.path(out,'model.txt'))
  for(nm in c('lambda','phi','sigma')) write_out(as.data.frame(get(nm)),nm)
  saveRDS(list(draw=draw,samples=samples,spec=spec),file.path(out,'inputs.rds'))
  input_rows <- do.call(rbind,lapply(names(samples),function(unit) {
    g <- expand.grid(row=1:9,col=1:9);data.frame(unit=unit,g,value=sprintf('%.17g',as.vector(samples[[unit]]$S[[1]])))
  }))
  write_out(input_rows,'inputs')
  hashes <- list();producers <- list();producer_rows <- list()
  for(unit in names(samples)) {
    path <- file.path(out,paste0(unit,'-sample.csv'))
    write.csv(input_rows[input_rows$unit==unit,],path,row.names=FALSE)
    hashes[[unit]] <- unname(tools::md5sum(path))
    for(method in c('fabin3','layered')) {
      key <- paste(unit,method,sep='_')
      z <- capture(magmaan_core$estimate_start_values(spec$partable,samples[[unit]],start=method,transport='native'))
      producers[[key]] <- z
      producer_rows[[key]] <- data.frame(unit=unit,method=method,seconds=z$seconds,error=z$error,warnings=z$warnings,
        attributes=paste(capture.output(dput(attributes(z$value))),collapse=' '))
    }
  }
  saveRDS(producers,file.path(out,'producers.rds'))
  write_out(do.call(rbind,producer_rows),'producers')
  write_out(data.frame(key=c('seed','n','covariance','rng','package_path','package_version','R','command','source_md5','inputs_md5','native_md5','mixed_md5','identification'),
    value=c(paste(seeds,collapse='/'),'200','centered sample covariance with denominator n',paste(RNGkind(),collapse='/'),find.package('magmaanlab'),as.character(packageVersion('magmaanlab')),R.version.string,paste(commandArgs(),collapse=' '),unname(tools::md5sum(file.path(here,'R/start_portfolio_pilot.R'))),unname(tools::md5sum(file.path(out,'inputs.csv'))),paste(unlist(hashes)[grepl('native',names(hashes))],collapse='/'),paste(unlist(hashes)[grepl('mixed',names(hashes))],collapse='/'),'not verified by pre-TASK-33.3 runtime')),'metadata')
  rows <- list()
  for(unit in names(samples)) for(arm in c('default','layered')) {
    key <- paste(unit,if(arm=='default')'fabin3' else 'layered',sep='_')
    expected <- producers[[key]]$value
    cat(unit,arm,'starting; elapsed',elapsed()-clock,'s\n');flush.console()
    fit <- if(inherits(expected,'error')) list(value=expected,seconds=0,warnings='',error=conditionMessage(expected)) else capture({
      if(arm=='default') fit_model(spec,samples[[unit]],estimator='ULS')
      else fit_model(spec,samples[[unit]],estimator='ULS',control=list(start=as.numeric(expected)))
    })
    f <- fit$value;returned <- !inherits(f,'error')
    audit <- if(returned) capture(magmaan_core$estimate_evaluate_at(spec$partable,samples[[unit]],f$theta,estimator='ULS',
      bounds=list(lower=rep(-Inf,length(f$theta)),upper=rep(Inf,length(f$theta))),
      audit_options=list(verified_newton=TRUE,reported_objective=f$fmin))) else list(value=NULL,seconds=0,error='no returned endpoint',warnings='')
    ev <- if(inherits(audit$value,'error'))NULL else audit$value
    a <- ev$newton_audit;interval <- a$distance_interval_derived_inputs;source <- a$derived_interval_input_errors
    pt <- ev$partable
    primitive <- if(is.null(pt))numeric() else pt$est[pt$op=='~~' & pt$lhs==pt$rhs]
    saveRDS(list(fit=fit,audit=audit),file.path(out,paste0(unit,'-',arm,'.rds')))
    rows[[length(rows)+1L]] <- data.frame(unit=unit,arm=arm,attempted=TRUE,returned=returned,
      finite=returned && all(is.finite(f$theta)) && is.finite(f$fmin),
      start_exact=returned && identical(as.numeric(f$start$theta),as.numeric(expected)),
      duplicate=identical(as.numeric(producers[[paste0(unit,'_fabin3')]]$value),as.numeric(producers[[paste0(unit,'_layered')]]$value)),
      default_converged=if(returned)f$converged %||% NA else NA,
      legacy_converged=ev$converged_compatibility %||% NA,
      reported_objective=if(returned)f$fmin else NA,objective=ev$fmin %||% NA,
      objective_consistency=ev$verified_convergence$objective_consistency$status %||% 'unavailable',
      verified_status=ev$verified_convergence$status %||% 'unavailable',
      interval_status=interval$status %||% source$status %||% 'unavailable',decision=interval$decision %||% 'unresolved',
      distance=interval$distance %||% NA,lower=interval$lower %||% NA,upper=interval$upper %||% NA,
      curvature_lower=source$curvature_lower_bound %||% NA,primitive_variance_min=if(length(primitive))min(primitive) else NA,
      admissibility=paste(capture.output(dput(ev$diagnostics$admissibility)),collapse=' '),
      optimizer_status=if(returned)as.character(f$optimizer_status %||% 'unavailable') else 'error',
      iterations=if(returned)f$iterations %||% NA else NA,f_evals=if(returned)f$f_evals %||% NA else NA,
      fit_seconds=fit$seconds,audit_seconds=audit$seconds,error=fit$error,audit_error=audit$error,
      warnings=paste(fit$warnings,audit$warnings,sep=' | '))
    write_out(do.call(rbind,rows),'attempts')
  }
  d <- do.call(rbind,rows)
  write_out(d[c('unit','arm','returned','finite','start_exact','duplicate','default_converged','legacy_converged','objective','objective_consistency','verified_status','interval_status','distance','lower','upper','curvature_lower','primitive_variance_min','optimizer_status','iterations','f_evals','fit_seconds','audit_seconds','error','audit_error','warnings')],'observations')
  write_out(do.call(rbind,lapply(split(d,d$unit),function(x) {
    good <- which(x$verified_status=='passed' & x$objective_consistency=='passed' & is.finite(x$objective))
    data.frame(unit=x$unit[1],attempts=nrow(x),qualified=length(good),selected=if(length(good))x$arm[good[which.min(x$objective[good])]] else 'no-qualified-endpoint')
  })),'summary')
  write_out(data.frame(total_seconds=elapsed()-clock,construction_seconds=sum(vapply(producers,function(z)z$seconds,0)),fit_seconds=sum(d$fit_seconds),audit_seconds=sum(d$audit_seconds)),'timing')
  cat('Completed attempted fits:',out,'elapsed',elapsed()-clock,'s\n')
}
