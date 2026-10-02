# Design-specific, sample-only candidates for the two-factor study. This is an
# engineering comparison, not a general start policy or a new SEM fitter.
uls_study_point <- function(pt) {
  get <- function(a, op, b) {
    z <- pt$est[pt$lhs == a & pt$op == op & pt$rhs == b]
    if (!length(z) && op == '~~') z <- pt$est[pt$lhs == b & pt$op == op & pt$rhs == a]
    if (!length(z)) stop('missing point row: ', a, op, b)
    z[1]
  }
  loading <- matrix(0, 6, 2)
  loading[1:3, 1] <- vapply(ov_names[1:3], function(n) get('X', '=~', n), 0.0)
  loading[4:6, 2] <- vapply(ov_names[4:6], function(n) get('Y', '=~', n), 0.0)
  vx <- get('X', '~~', 'X'); vy <- get('Y', '~~', 'Y')
  if (any(pt$lhs == 'Y' & pt$op == '~' & pt$rhs == 'X')) {
    beta <- get('Y', '~', 'X'); cross <- beta * vx; vy <- vy + beta^2 * vx
  } else cross <- get('X', '~~', 'Y')
  list(loading = loading, phi = matrix(c(vx, cross, cross, vy), 2),
    residual = vapply(ov_names, function(n) get(n, '~~', n), 0.0))
}

uls_study_transport <- function(point, spec, unit = rep(1, 6)) {
  # Diagonal observed-variable transformation and marker re-identification.
  loading <- point$loading / unit
  anchors <- c(loading[1, 1], loading[4, 2])
  if (any(!is.finite(anchors)) || any(anchors == 0)) stop('singular marker transport')
  loading <- sweep(loading, 2, anchors, '/')
  phi <- point$phi * outer(anchors, anchors)
  pt <- spec$partable
  pt$est <- vapply(seq_len(nrow(pt)), function(i) {
    a <- pt$lhs[i]; b <- pt$rhs[i]
    switch(pt$op[i],
      '=~' = loading[match(b, ov_names), match(a, c('X', 'Y'))],
      '~' = phi[2, 1] / phi[1, 1],
      '~~' = if (a %in% ov_names) point$residual[match(a, ov_names)] / unit[match(a, ov_names)]^2
        else if (a != b) phi[1, 2]
        else if (a == 'X') phi[1, 1]
        else if (any(pt$op == '~')) phi[2, 2] - phi[1, 2]^2 / phi[1, 1] else phi[2, 2],
      stop('unsupported study point row'))
  }, 0.0)
  if (any(!is.finite(pt$est))) stop('nonfinite study start')
  theta <- pt$est[pt$free > 0][order(pt$free[pt$free > 0])]
  list(partable = pt, theta = theta)
}

uls_signed_moment_point <- function(sample, sign_x, sign_y) {
  unit <- sqrt(diag(sample$S[[1]])); s <- sample$S[[1]] / outer(unit, unit)
  lx <- c(sign_x * s[1, 3] / s[2, 3], 1, sign_x * s[1, 3] / s[1, 2])
  ly <- c(sign_y * s[4, 6] / s[5, 6], 1, sign_y * s[4, 6] / s[4, 5])
  vx <- sign_x * s[1, 2] * s[2, 3] / s[1, 3]
  vy <- sign_y * s[4, 5] * s[5, 6] / s[4, 6]
  w <- outer(unit[1:3], unit[4:6])^2; directions <- outer(lx, ly)
  cross <- sum(w * directions * s[1:3, 4:6]) / sum(w * directions^2)
  loading <- matrix(0, 6, 2); loading[1:3, 1] <- lx; loading[4:6, 2] <- ly
  loading <- loading * unit; phi <- matrix(c(vx, cross, cross, vy), 2)
  list(loading = loading, phi = phi,
    residual = diag(sample$S[[1]] - loading %*% phi %*% t(loading)))
}

run_uls_reliability_study <- function(args, here) {
  if (any(args %in% c('--help', '-h'))) {
    cat(paste('Usage: Rscript run_experiment.R --ordinary --uls-study [options]',
      '--smoke: first retained dataset; no fresh draws. Otherwise five retained + fresh draws.',
      '--run-id NAME --fresh-reps N (default 10 per population) --seed-base N (default 742260101)',
      '--designs ernst,weak_marker --reference-digits 60|90 --python PATH',
      '16 pipelines: regression marker/sphere, covariance full/profiled; mixed/standardized coordinates;',
      'L-BFGS/PORT-NLS. Six starts: shared FABIN3/layered and four signed moment recipes.',
      'Weights undo standardization exactly; original mixed-unit ULS is the fixed target.',
      'Production audit unchanged; diagnostic objective selection cannot certify a fit.',
      'Independent mpmath references run after all cold fits; no fitted reference enters starts.',
      'Single-threaded serial runner; run under nice -n 10. Fresh output IDs required.', sep='\n'), '\n')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--uls-study','--smoke','--run-id','--fresh-reps','--seed-base','--designs','--reference-digits','--python')
  if (any(startsWith(args,'--') & !args %in% known)) stop('unknown ULS study option')
  value <- function(k, default) {
    i <- match(k,args); if (is.na(i)) return(default)
    if (i == length(args) || startsWith(args[i+1L],'--')) stop('missing value for ',k)
    args[i+1L]
  }
  run <- value('--run-id','uls-reliability'); reps <- as.integer(value('--fresh-reps','10'))
  seed_base <- as.integer(value('--seed-base','742260101')); digits <- as.integer(value('--reference-digits','60'))
  designs <- parse_csv_arg(value('--designs','ernst,weak_marker'))
  if (!grepl('^[A-Za-z0-9_-]+$',run) || anyNA(c(reps,seed_base,digits)) || reps < 0 || reps > 100 ||
      seed_base < 1 || seed_base > 2e9 || !digits %in% c(60,90) || !length(designs) ||
      anyDuplicated(designs) || any(!designs %in% names(designs_all()))) stop('invalid ULS study options')
  require_pkg('magmaanlab'); out <- file.path(here,'results',run)
  if (dir.exists(out)) stop('choose a fresh run-id')
  reference_dir <- file.path(here,'results/uls-unit-final-60')
  frozen <- read.csv(file.path(reference_dir,'covariances.csv'))
  regression <- magmaanlab::model_spec(model_syntax)
  covariance <- magmaanlab::model_spec('X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y')
  # evaluate_at() defaults to a positive-variance preset. The study's domain
  # is unrestricted; explicitly infinite bounds prevent its active-box
  # first-order fallback from replacing the Newton accuracy assessment.
  unbounded <- list(lower = rep(-Inf, max(regression$partable$free)),
    upper = rep(Inf, max(regression$partable$free)))
  samples <- list(); tasks <- list()
  for (i in if ('--smoke' %in% args) 1L else sort(unique(frozen$case_id))) {
    z <- frozen[frozen$case_id == i,]; s <- matrix(0,6,6); s[cbind(z$row,z$col)] <- z$value
    dimnames(s) <- list(ov_names,ov_names)
    samples[[length(samples)+1L]] <- list(S=list(s),nobs=100L)
    tasks[[length(tasks)+1L]] <- data.frame(case_id=i, role='retained', design='ernst', rep=i, seed=NA_integer_)
  }
  if (!'--smoke' %in% args && reps > 0) for (design in designs) for (rep in seq_len(reps)) {
    seed <- seed_base + match(design,names(designs_all()))*100000L + 10000L + rep
    data <- draw_data(design_sigma(designs_all()[[design]]),100L,seed)
    data[] <- sweep(as.matrix(data),2,c(.01,100,2,.3,10,.1),'*')
    samples[[length(samples)+1L]] <- magmaanlab::df_to_data(data,regression,scaling='n')
    tasks[[length(tasks)+1L]] <- data.frame(case_id=length(samples),role='fresh',design=design,rep=rep,seed=seed)
  }
  tasks <- do.call(rbind,tasks)
  arms <- expand.grid(route=c('regression_marker','regression_sphere','covariance_full','covariance_profiled'),
    coordinates=c('mixed','standardized'),optimizer=c('nlopt-lbfgs','port-nls'),stringsAsFactors=FALSE)
  dir.create(out,recursive=TRUE)
  write_out <- function(d,name) write_csv(d,file.path(out,paste0(name,'.csv')))
  source_files <- c(file.path(here,'run_experiment.R'),file.path(here,'R',c('uls_reliability_study.R','designs.R','fit.R','ordinary_program.R')),
    file.path(here,'scripts/uls_unit_reference.py'),file.path(here,'../../../_support/R/helpers.R'))
  package_files <- list.files(find.package('magmaanlab'),recursive=TRUE,full.names=TRUE)
  ref <- magmaan_cache_ref()
  meta <- list(lane='uls_reliability',seed_base=seed_base,seed_rule='base + stable design index*100000 + 10000 + rep',
    planned_fits=nrow(tasks)*nrow(arms)*6L, fresh_reps=reps, designs=designs, reference_digits=digits,
    source_md5=paste(names(tools::md5sum(source_files)),tools::md5sum(source_files),collapse=';'),
    package_md5=paste(tools::md5sum(package_files[grepl('\\.(so|rdb|rdx)$',package_files)]),collapse=';'),
    input_md5=unname(tools::md5sum(file.path(reference_dir,'covariances.csv'))),git_head=ref$git_head,git_dirty=ref$git_dirty,
    judge='common original-marker unscaled ULS evaluate_at with explicit infinite bounds; native audit retained separately; objective 1e-6(1+|f|), standardized sigma 1e-5',
    starts='all six constructed once before fitting; shared FABIN3(auto), layered(native), signed moments (+,+),(+,-),(-,+),(-,-)',
    controls='stock backend controls; explicit numeric start; coordinate_scaling sample_units; no polishing',
    target='unrestricted mixed-unit covariance-only ULS; no PSD/barriers; standardized-coordinate W exactly restores mixed target',
    portfolio='diagnostic lowest original objective, never a production acceptance override',threads=1,evidence_role='exploratory; fresh draws repeat candidate comparisons, no default decision')
  write_metadata(file.path(out,'metadata.csv'),values=meta,packages='magmaanlab')
  write_out(tasks,'tasks'); write_out(arms,'pipelines')
  rows <- starts_rows <- covariances <- reference_seeds <- transports <- list(); t0 <- proc.time()[['elapsed']]
  cat(sprintf('%d datasets, %d cold fits; %s\n',nrow(tasks),meta$planned_fits,out))
  for (case_id in seq_along(samples)) {
    sample <- samples[[case_id]]; s <- sample$S[[1]]; sd <- sqrt(diag(s))
    saveRDS(list(task=tasks[case_id,],sample=sample),file.path(out,sprintf('sample_%03d.rds',case_id)))
    starts <- list()
    for (method in c('fabin3','layered')) {
      theta <- magmaanlab::magmaan_core$estimate_start_values(regression$partable,sample,start=method,
        transport=if(method=='layered')'native' else 'auto')
      pt <- regression$partable; pt$est <- pt$ustart
      pt$est[pt$free>0] <- theta[pt$free[pt$free>0]]
      starts[[method]] <- uls_study_point(pt)
    }
    for (sx in c(1,-1)) for(sy in c(1,-1)) starts[[paste0('signed_',sx,'_',sy)]] <- uls_signed_moment_point(sample,sx,sy)
    points <- list(); best <- NULL; best_value <- Inf
    for (start_id in names(starts)) {
      start <- starts[[start_id]]; original <- uls_study_transport(start,regression)
      starts_rows[[length(starts_rows)+1L]] <- data.frame(case_id=case_id,start_id=start_id,parameter=seq_along(original$theta),value=original$theta)
      expected_sigma <- start$loading %*% start$phi %*% t(start$loading) + diag(start$residual)
      expected_objective <- sum((expected_sigma-s)[lower.tri(s,diag=TRUE)]^2)/2
      for (j in seq_len(nrow(arms))) {
        arm <- arms[j,]; spec <- if(startsWith(arm$route,'regression')) regression else covariance
        unit <- if(arm$coordinates=='mixed')rep(1,6) else sd
        transformed <- uls_study_transport(start,spec,unit)
        target_sample <- list(S=list(s/outer(unit,unit)),nobs=100L)
        weights <- diag(outer(unit,unit)[lower.tri(s,diag=TRUE)]^2)
        pre <- magmaanlab::magmaan_core$evaluate_at(spec$partable,target_sample,transformed$theta,estimator='WLS',W=weights,bounds=unbounded)
        pre_sigma <- magmaanlab::magmaan_core$model_implied(pre)$sigma[[1]] * outer(unit,unit)
        objective_gap <- abs(pre$fmin-expected_objective)/(1+abs(expected_objective))
        sigma_gap <- max(abs((pre_sigma-expected_sigma)/outer(sd,sd)))
        if(objective_gap>1e-10 || sigma_gap>1e-10)stop('start transport changes fixed target')
        transports[[length(transports)+1L]] <- data.frame(case_id=case_id,start_id=start_id,arm,objective_gap=objective_gap,standardized_sigma_gap=sigma_gap)
        ctl <- list(start=as.numeric(transformed$theta),coordinate_scaling='sample_units')
        target <- spec; free <- target$partable$free
        target$partable$ustart[free>0] <- transformed$theta[free[free>0]]
        warnings <- character(); fit_time <- proc.time()[['elapsed']]
        fit <- tryCatch(withCallingHandlers({
          if(arm$route=='regression_sphere') magmaanlab::frontier_fit_sphere(target,target_sample,estimator='WLS',W=weights,
            optimizer=arm$optimizer,control=ctl,start='user',polish=FALSE)
          else if(arm$route=='covariance_profiled') magmaanlab::magmaan_core$fit_wls_snlls(target,target_sample,W=weights,optimizer=arm$optimizer,control=ctl)
          else magmaanlab::fit_model(target,target_sample,estimator='WLS',W=weights,optimizer=arm$optimizer,control=ctl)
        },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=identity)
        returned <- !inherits(fit,'error') || inherits(fit,'magmaan_sphere_condition')
        native <- if(returned && arm$route=='regression_sphere')fit$gauge$native_audit else NULL
        pt <- if(!returned)NULL else if(arm$route=='regression_sphere')fit$gauge$sphere_partable else fit$partable
        checked <- point <- sigma <- implicit <- NULL; message <- if(inherits(fit,'error'))conditionMessage(fit) else ''
        if(!is.null(pt)) {
          checked <- tryCatch({
            point <- uls_study_point(pt); point$loading <- point$loading*unit; point$residual <- point$residual*unit^2
            restored <- uls_study_transport(point,regression)
            implicit <- magmaanlab::magmaan_core$evaluate_at(regression$partable,sample,restored$theta,estimator='ULS')
            magmaanlab::magmaan_core$evaluate_at(regression$partable,sample,restored$theta,estimator='ULS',bounds=unbounded)
          },error=identity)
          if(inherits(checked,'error'))message <- paste(message,conditionMessage(checked))
        }
        available <- !is.null(checked) && !inherits(checked,'error')
        objective <- if(available)checked$fmin else NA_real_
        reported <- if(!returned)NA_real_ else if(arm$route=='regression_sphere')fit$gauge$fmin_sphere else fit$fmin
        consistent <- available && is.finite(reported) && abs(objective-reported)<=1e-6*(1+abs(objective))
        if(available)sigma <- magmaanlab::magmaan_core$model_implied(checked)$sigma[[1]]
        native_audit <- if(!returned)'unavailable' else if(is.null(native))fit$verdict$status else native$status
        na <- if(available)checked$diagnostics$newton_accuracy else NULL
        if (available && checked$verdict$status == 'passed' &&
            (na$status != 'available' || na$condition > 1e12 || na$distance > .01))
          stop('unrestricted point verdict bypasses the Newton audit')
        record <- cbind(tasks[case_id,],start_id=start_id,arm,returned=returned,objective=objective,objective_consistent=consistent,
          common_audit=if(available)checked$verdict$status else 'unavailable', common_newton_status=na$status %||% 'unavailable',
          common_condition=na$condition %||% NA_real_,native_audit=native_audit,native_newton_status=if(is.null(native))fit$diagnostics$newton_accuracy$status %||% 'unavailable' else native$newton_accuracy$status,
          implicit_bound_audit=implicit$verdict$status %||% 'unavailable',implicit_bound_criterion=implicit$verdict$criterion %||% '',
          n_nonlinear=fit$n_nonlinear %||% NA_integer_,n_linear=fit$n_linear %||% NA_integer_,
          sigma_pd=if(available)checked$diagnostics$sigma_pd_all else NA, f_evals=fit$f_evals %||% NA_integer_,
          seconds=proc.time()[['elapsed']]-fit_time,message=message,warnings=paste(unique(warnings),collapse=' | '))
        rows[[length(rows)+1L]] <- record; points[[length(points)+1L]] <- list(point=point,sigma=sigma)
        if(consistent && objective<best_value){best <- point;best_value <- objective}
      }
    }
    if(is.null(best))stop('no finite consistent candidate for reference refinement')
    scales <- c(.01,100,2,.3,10,.1); l <- best$loading/scales
    reference_seed <- c(l[c(1,3),1]/l[2,1],l[c(4,6),2]/l[5,2],best$phi[1,1]*l[2,1]^2,
      best$phi[1,2]*l[2,1]*l[5,2],best$phi[2,2]*l[5,2]^2)
    reference_seeds[[case_id]] <- data.frame(case_id=case_id,parameter=1:7,value=reference_seed)
    covariances[[case_id]] <- expand.grid(case_id=case_id,row=1:6,col=1:6);covariances[[case_id]]$value<-as.vector(s)
    saveRDS(points,file.path(out,sprintf('endpoints_%03d.rds',case_id)))
    write_out(do.call(rbind,rows),'fits')
    cat(sprintf('ULS dataset %d/%d, %d fits; %.1fs elapsed\n',case_id,nrow(tasks),length(rows),proc.time()[['elapsed']]-t0))
  }
  fits <- do.call(rbind,rows)
  write_out(do.call(rbind,starts_rows),'starts');write_out(do.call(rbind,transports),'transport_checks')
  write_out(do.call(rbind,covariances),'covariances');write_out(do.call(rbind,reference_seeds),'reference_starts')
  meta$fit_elapsed_s <- proc.time()[['elapsed']]-t0
  write_metadata(file.path(out,'metadata.csv'),values=meta,packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/uls_unit_reference.py')),'--run-dir',shQuote(normalizePath(out)),'--digits',digits))
  if(status!=0)stop('reference refinement failed; cold evidence retained')
  references <- read.csv(file.path(out,'reference_summary.csv')); parameters <- read.csv(file.path(out,'reference_parameters.csv'))
  checks <- core_checks <- list()
  for(case_id in seq_along(samples)) {
    ref_row <- references[references$case_id==case_id,]; d <- fits[fits$case_id==case_id,]
    d$reference_available <- tolower(ref_row$finite_local_minimum)=='true'
    d$reference_objective <- ref_row$objective; d$objective_gap <- d$objective-ref_row$objective
    d$standardized_sigma_gap <- NA_real_; d$reference_match <- FALSE
    if(d$reference_available[1]){
      p <- parameters[parameters$case_id==case_id,]; pt <- regression$partable
      key<-function(z)paste(z$lhs,z$op,z$rhs);pt$est<-p$est[match(key(pt),key(p))]
      theta<-pt$est[pt$free>0][order(pt$free[pt$free>0])]
      ev<-magmaanlab::magmaan_core$evaluate_at(pt,samples[[case_id]],theta,estimator='ULS',bounds=unbounded)
      if(abs(ev$fmin-ref_row$objective)>1e-8*(1+abs(ev$fmin)))stop('reference disagrees with original core objective')
      ref_sigma<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
      unit<-sqrt(diag(samples[[case_id]]$S[[1]])); points<-readRDS(file.path(out,sprintf('endpoints_%03d.rds',case_id)))
      d$standardized_sigma_gap<-vapply(points,function(p)if(is.null(p$sigma))NA_real_ else max(abs((p$sigma-ref_sigma)/outer(unit,unit))),0.0)
      d$reference_match<-d$objective_consistent & is.finite(d$objective_gap) & abs(d$objective_gap)<=1e-6*(1+abs(ref_row$objective)) &
        is.finite(d$standardized_sigma_gap) & d$standardized_sigma_gap<=1e-5
      core_checks[[length(core_checks)+1L]]<-data.frame(case_id=case_id,objective_gap=abs(ev$fmin-ref_row$objective),audit=ev$verdict$status,
        newton_status=ev$diagnostics$newton_accuracy$status,condition=ev$diagnostics$newton_accuracy$condition)
    }
    checks[[case_id]]<-d
  }
  checks<-do.call(rbind,checks);write_out(checks,'reference_checks');write_out(do.call(rbind,core_checks),'reference_core_checks')
  # Summaries keep matching and certification distinct, including fresh problems
  # with unresolved reference refinement. Lowest-objective selection is diagnostic.
  groups<-c('role','design','route','coordinates','optimizer','start_id')
  write_out(aggregate(list(fits=rep(1L,nrow(checks)),returned=checks$returned,reference_available=checks$reference_available,
    matches=checks$reference_match,common_passes=checks$common_audit=='passed',native_passes=checks$native_audit=='passed'),checks[groups],sum),'summary')
  portfolios<-list()
  for(d in split(checks,interaction(checks[c('case_id','route','coordinates','optimizer')],drop=TRUE))) {
    for(portfolio in c('fabin3','shared','signed','all')) {
      ids<-switch(portfolio,fabin3='fabin3',shared=c('fabin3','layered'),signed=grep('^signed_',unique(d$start_id),value=TRUE),all=unique(d$start_id))
      z<-d[d$start_id%in%ids & d$objective_consistent & is.finite(d$objective),]
      selected<-if(nrow(z))z[which.min(z$objective),] else NULL
      portfolios[[length(portfolios)+1L]]<-cbind(d[1,c('case_id','role','design','route','coordinates','optimizer')],portfolio=portfolio,
        calls=length(ids),reference_available=d$reference_available[1],returned=!is.null(selected),
        selected_start=if(is.null(selected))'' else selected$start_id,objective=if(is.null(selected))NA_real_ else selected$objective,
        match=if(is.null(selected))FALSE else selected$reference_match,common_pass=if(is.null(selected))FALSE else selected$common_audit=='passed',
        native_pass=if(is.null(selected))FALSE else selected$native_audit=='passed',seconds=sum(d$seconds[d$start_id%in%ids]))
    }
  }
  portfolio<-do.call(rbind,portfolios);write_out(portfolio,'portfolio_checks')
  write_out(aggregate(list(datasets=rep(1L,nrow(portfolio)),reference_available=portfolio$reference_available,returned=portfolio$returned,
    matches=portfolio$match,common_passes=portfolio$common_pass,native_passes=portfolio$native_pass,calls=portfolio$calls,seconds=portfolio$seconds),
    portfolio[c('role','design','route','coordinates','optimizer','portfolio')],sum),'portfolio_summary')
  write_out(aggregate(list(fits=rep(1L,nrow(checks))),checks[c('role','route','coordinates','optimizer','common_newton_status','native_newton_status')],sum),'audit_summary')
  cat('Wrote ULS study to',out,'\n')
}
