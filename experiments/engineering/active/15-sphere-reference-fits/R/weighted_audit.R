# Fixed-weight mechanism checks and bounded fresh route confirmation.
run_weighted_audit <- function(args,here) {
  value <- function(k,default) {
    i <- match(k,args); if(is.na(i)) return(default)
    if(i==length(args) || startsWith(args[i+1L],'--')) stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: Rscript run_experiment.R --ordinary --weighted-audit --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
      'Identity/uneven diagonal/dense/NT and DWLS/ADF/DLS recipes; covariance/means, ties, feedback, groups and mixed units.\n',
      'PORT-NLS, actual default starts, fixed sample-unit controls; opt-in construction bounds and independent 90-digit checks.\n',
      'Small representative subset, no default adoption or global/basin recovery claim.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--weighted-audit','--run-id','--smoke','--reps','--seed-base','--python')
  if(any(startsWith(args,'--') & !args %in% known)) stop('unknown weighted-audit option')
  smoke <- '--smoke' %in% args
  reps <- as.integer(value('--reps',if(smoke)'1' else '2'))
  seed <- as.integer(value('--seed-base',if(smoke)'864051101' else '864051201'))
  run <- value('--run-id','weighted-audit')
  if(is.na(reps) || reps<1 || reps>20 || is.na(seed) || seed<1 || !grepl('^[A-Za-z0-9_-]+$',run)) stop('invalid options')
  out <- file.path(here,'results',run); if(dir.exists(out)) stop('choose a fresh run ID')
  require_pkg('magmaanlab'); dir.create(out,recursive=TRUE)
  write_out <- function(x,name) write_csv(x,file.path(out,paste0(name,'.csv')))
  matrix_rows <- function(x,id,name) {
    x <- as.matrix(x); if(!length(x)) return(NULL)
    cbind(id=id,name=name,expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x))),
      value=sprintf('%.17g',as.vector(x)))
  }
  artifacts <- producers <- points <- intervals <- fits <- samples <- raw_rows <- recipes <- cases <- producer_failures <- list()
  retain_weights <- function(w,id,prefix,destination) {
    # An empty retained list is the implicit identity representation.
    for(g in seq_along(w)) {
      kind <- match(w[[g]]$kind,c('identity','diagonal','dense_factor','normal_theory_root'))
      destination[[length(destination)+1L]] <- matrix_rows(kind,id,paste0(prefix,g,'_kind'))
      for(k in c('diagonal','factor','root')) if(length(w[[g]][[k]]))
        destination[[length(destination)+1L]] <- matrix_rows(w[[g]][[k]],id,paste0(prefix,g,'_',k))
    }
    destination
  }
  retain <- function(ev,pt,cid,method,chart,target,seconds) {
    id <- length(intervals)+1L
    a <- if(chart=='sphere') ev$gauge$native_audit else ev$newton_audit
    e <- a$derived_interval_input_errors
    z <- a$distance_interval_derived_inputs
    if(is.null(e)) stop('missing construction-bound artifact')
    if(is.null(z)) z <- list(status=e$status,decision='unresolved',lower=0,upper=Inf)
    intervals[[id]] <<- data.frame(point_id=id,case_id=cid,method=method,chart=chart,target=target,
      source_status=e$status,matrix_bound=sprintf('%.17g',e$matrix),vector_bound=sprintf('%.17g',e$vector),
      curvature_bound=sprintf('%.17g',e$curvature),curvature_lower=sprintf('%.17g',e$curvature_lower_bound),
      interval_status=z$status,decision=z$decision,lower=sprintf('%.17g',z$lower),upper=sprintf('%.17g',z$upper),
      recomputed_objective=sprintf('%.17g',if(chart=='sphere')a$fmin else ev$fmin),
      selected_passed=if(chart=='sphere')a$converged else ev$converged,
      selected_status=if(chart=='sphere')a$status else ev$verified_convergence$status,seconds=seconds)
    pp <- pt[c('lhs','op','rhs','group','free','est')]; pp$est <- sprintf('%.17g',pp$est)
    points[[id]] <<- cbind(point_id=id,pp)
    if(chart=='marker') {
      for(name in c('derivative_basis','gradient','equilibrated_factor','factor_scale','metric_score_residual',
                   'curvature_equilibrated_hessian','curvature_coordinate_map'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(a[[name]],id,name)
    } else {
      for(name in c('point','tangent_basis','reduced_gradient','equilibrated_factor','factor_scale','metric_score_residual'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(a[[name]],id,name)
      for(name in c('equilibrated_hessian','coordinate_map'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(a$curvature_system[[name]],id,paste0('curvature_',name))
      for(name in c('offset','rest_basis'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(a$input_map[[name]],id,paste0('map_',name))
      for(k in seq_along(a$input_map$spheres)) for(name in c('basis','units','parameters','offset'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(a$input_map$spheres[[k]][[name]],id,paste0('sphere_',k,'_',name))
    }
    artifacts <<- retain_weights(a$retained_ls_weights,id,'weight_',artifacts)
  }
  families <- c('regular','means','shared','feedback','feedback_ridge','groups','mixed')
  start_clock <- proc.time()[['elapsed']]
  for(family in families) for(rep in seq_len(reps)) {
    cid <- length(cases)+1L; means <- family %in% c('means','shared','feedback','feedback_ridge','groups')
    groups <- if(family=='groups')2L else 1L; p <- if(family %in% c('feedback','feedback_ridge'))6L else 4L
    ov <- if(p==6L)c('x1','x2','x3','y1','y2','y3') else paste0('x',1:4)
    syntax <- if(p==6L) paste0('X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~ ',
      if(family=='feedback')'.2*Y' else 'Y','\nY ~ X\nX ~ .2*1\nY ~ -.1*1') else
      if(family=='shared')'f =~ x1 + a*x2 + a*x3 + x4\nf ~ .2*1' else 'f =~ x1 + x2 + x3 + x4'
    spec <- magmaanlab::model_spec(syntax,fixed_x=FALSE,meanstructure=means,
      group_labels=if(groups==2L)c('g1','g2') else NULL)
    set.seed(seed+10000L*match(family,families)+rep)
    X <- lapply(seq_len(groups),function(g) {
      n <- if(g==2L)180L else 120L
      # Skewed latent scores make empirical mean/covariance cross-blocks nonzero.
      latent <- matrix((rnorm(n*if(p==6L)2L else 1L)^2-1)/sqrt(2),n)
      if(p==6L) {
        latent <- latent%*%t(solve(diag(2)-matrix(c(0,.3,.2,0),2)))
        L <- matrix(0,6,2); L[1:3,1] <- c(1,.8,.7); L[4:6,2] <- c(1,.9,.6)
      } else L <- matrix(c(1,.8,if(family=='shared').8 else .7,.9),4,1)
      x <- latent%*%t(L)+matrix(rnorm(n*p),n,p)
      if(means) x <- sweep(x,2,seq_len(p)/10,'+')
      units <- if(family=='mixed')c(.01,100,.3,10) else rep(sqrt(g),p)
      x <- sweep(x,2,units,'*'); colnames(x) <- ov; x
    })
    sample <- magmaanlab::magmaan_core$data_sample_stats_from_raw(X)
    if(!means) sample$mean <- NULL
    frame <- if(groups==1L)as.data.frame(X[[1]]) else do.call(rbind,lapply(seq_len(groups),function(g)
      cbind(as.data.frame(X[[g]]),study_group=paste0('g',g))))
    data <- magmaanlab::df_to_data(frame,spec,group=if(groups==2L)'study_group' else NULL)
    if(!means)data$mean <- NULL
    for(g in seq_len(groups))stopifnot(all(data$S[[g]]==sample$S[[g]]))
    cases[[cid]] <- data.frame(case_id=cid,family=family,rep=rep,has_means=means,seed=seed+10000L*match(family,families)+rep)
    for(g in seq_len(groups)) {
      samples[[length(samples)+1L]] <- matrix_rows(sample$S[[g]],cid,paste0('S_',g))
      samples[[length(samples)+1L]] <- matrix_rows(nrow(X[[g]]),cid,paste0('n_',g))
      if(means) samples[[length(samples)+1L]] <- matrix_rows(sample$mean[[g]],cid,paste0('mean_',g))
      raw_rows[[length(raw_rows)+1L]] <- matrix_rows(X[[g]],cid,paste0('X_',g))
    }
    # Preserve the originating data even if an unexpected adapter error stops
    # the pilot before all endpoint artifacts have been collected.
    for(name in c('cases','samples','raw_rows')) write_out(do.call(rbind,get(name)),name)
    methods <- if(family %in% c('regular','means')) c('ULS','uneven','dense','GLS','DWLS','WLS','DLS0','DLS40','DLS1') else c('GLS','DWLS','DLS40')
    initial <- magmaanlab::magmaan_core$estimate_start_values(spec$partable,sample,start='fabin3',transport='native')
    unbounded <- list(lower=rep(-Inf,length(initial)),upper=rep(Inf,length(initial)))
    for(method in methods) {
      a <- switch(method,DLS0=0,DLS40=.4,DLS1=1,.5)
      W <- NULL
      if(method %in% c('uneven','dense')) W <- lapply(sample$S,function(S) {
        r <- nrow(S)*(nrow(S)+1)/2+if(means)nrow(S) else 0
        d <- 10^seq(-2,2,length.out=r); w <- diag(d)
        if(method=='dense')w <- w+.03*sqrt(d)%o%sqrt(d)
        w
      }) else {
        producer <- tryCatch(magmaanlab::magmaan_core$fixed_moment_weight_impl(spec$partable,sample,
          if(startsWith(method,'DLS'))'DLS' else method,raw_data=X,dls_a=a),error=identity)
        if(inherits(producer,'error')) {
          producer_failures[[length(producer_failures)+1L]] <- data.frame(case_id=cid,method=method,message=conditionMessage(producer))
          write_out(do.call(rbind,producer_failures),'producer_failures')
          next
        }
        rid <- length(recipes)+1L
        recipes[[rid]] <- data.frame(id=rid,case_id=cid,method=method,a=a)
        producers <- retain_weights(producer$retained_ls_weights,rid,'weight_',producers)
        for(g in seq_along(producer$W)) producers[[length(producers)+1L]] <- matrix_rows(producer$W[[g]],rid,paste0('supplied_',g))
        if(!method %in% c('ULS','GLS')) W <- producer$W
      }
      estimator <- if(method %in% c('ULS','GLS'))method else 'WLS'
      t0 <- proc.time()[['elapsed']]
      advertised <- if(startsWith(method,'DLS'))'DLS' else if(method %in% c('uneven','dense'))'WLS' else method
      fit <- tryCatch(suppressWarnings(magmaanlab::fit_model(spec,data,estimator=advertised,
        W=if(method %in% c('uneven','dense'))W else NULL,dls_a=a,
        optimizer='port-nls',control=list(coordinate_scaling='sample_units'),bounds=unbounded)),error=identity)
      seconds <- proc.time()[['elapsed']]-t0
      fits[[length(fits)+1L]] <- data.frame(case_id=cid,method=method,chart='marker',returned=!inherits(fit,'error'),seconds=seconds,
        legacy_passed=if(inherits(fit,'error'))NA else fit$converged,message=if(inherits(fit,'error'))conditionMessage(fit) else '')
      saveRDS(fit,file.path(out,paste0('fit_',cid,'_',method,'.rds')))
      if(!inherits(fit,'error')) for(target in c(0,.05)) {
        theta <- fit$theta; row <- which(spec$partable$lhs=='x1' & spec$partable$op=='~~' & spec$partable$rhs=='x1' & spec$partable$group==1L)[1]
        theta[spec$partable$free[row]] <- theta[spec$partable$free[row]]+target*sample$S[[1]][1,1]
        settings <- list(retain_newton_artifacts=TRUE,verified_newton=TRUE)
        if(target==0)settings$reported_objective <- fit$fmin
        ev <- magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,theta,estimator=estimator,W=W,bounds=unbounded,
          audit_options=settings)
        retain(ev,ev$partable,cid,method,'marker',target,seconds)
      }
      if(family=='regular' && method=='GLS' && !inherits(fit,'error')) {
        row <- which(spec$partable$lhs=='x1' & spec$partable$op=='~~' & spec$partable$rhs=='x1')[1]
        index <- spec$partable$free[row]
        for(goal in c(.0099,.0101)) {
          shift <- .001*sample$S[[1]][1,1]
          for(k in 1:4) {
            theta <- fit$theta; theta[index] <- theta[index]+shift
            ev <- magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,theta,estimator='GLS',bounds=unbounded,
              audit_options=list(retain_newton_artifacts=TRUE,verified_newton=TRUE))
            if(k<4) shift <- shift*goal/ev$newton_audit$diagnostics$distance
          }
          retain(ev,ev$partable,cid,method,'marker',paste0('near_',goal),0)
        }
        # Profile the diagonal variances at zero free loadings. This gives a
        # stationary saddle with a deficient gradient metric, without relying
        # on the ULS-specific rule of matching each sample variance.
        control <- magmaanlab::model_spec('f =~ NA*x1 + x2 + x3 + x4\nf ~~ 1*f',fixed_x=FALSE,meanstructure=FALSE)
        theta <- magmaanlab::magmaan_core$estimate_start_values(control$partable,sample,start='simple')
        theta[control$partable$free[control$partable$op=='=~']] <- 0
        S <- sample$S[[1]]; ij <- which(lower.tri(S,diag=TRUE),arr.ind=TRUE)
        diagonals <- which(ij[,1]==ij[,2]); residual <- -S[lower.tri(S,diag=TRUE)]; residual[diagonals] <- 0
        profiled <- diag(S)-solve(producer$W[[1]][diagonals,diagonals],(producer$W[[1]]%*%residual)[diagonals])
        for(j in 1:4) {
          r <- which(control$partable$op=='~~' & control$partable$lhs==paste0('x',j) & control$partable$rhs==paste0('x',j))
          theta[control$partable$free[r]] <- profiled[j]
        }
        ev <- magmaanlab::magmaan_core$estimate_evaluate_at(control$partable,sample,theta,estimator='GLS',
          bounds=list(lower=rep(-Inf,length(theta)),upper=rep(Inf,length(theta))),
          audit_options=list(retain_newton_artifacts=TRUE,verified_newton=TRUE))
        retain(ev,ev$partable,cid,method,'marker','saddle_rank',0)
      }
      # Check normalization-chain composition on a small weighted subset.
      if(family %in% c('regular','means') && method %in% c('GLS','DWLS','dense')) {
        t0 <- proc.time()[['elapsed']]
        sphere <- tryCatch(suppressWarnings(magmaanlab::frontier_fit_sphere(spec,sample,estimator=estimator,W=W,
          optimizer='port-nls',control=list(coordinate_scaling='sample_units',verified_newton=TRUE),bounds=unbounded,polish=FALSE)),error=identity)
        seconds <- proc.time()[['elapsed']]-t0
        native <- if(is.null(sphere$gauge))NULL else sphere$gauge$native_audit
        fits[[length(fits)+1L]] <- data.frame(case_id=cid,method=method,chart='sphere',returned=!is.null(native),seconds=seconds,
          legacy_passed=if(is.null(native))NA else native$compatibility_assessment$converged,
          message=if(inherits(sphere,'error'))conditionMessage(sphere) else '')
        saveRDS(sphere,file.path(out,paste0('sphere_',cid,'_',method,'.rds')))
        if(!is.null(native)) retain(sphere,sphere$gauge$sphere_partable,cid,method,'sphere',0,seconds)
      }
    }
    cat(sprintf('Weighted case %d/%d (%s); %.1fs\n',cid,length(families)*reps,family,proc.time()[['elapsed']]-start_clock))
  }
  for(name in c('artifacts','producers','points','intervals','fits','samples','raw_rows','recipes','cases'))
    write_out(do.call(rbind,get(name)),name)
  write_out(if(length(producer_failures))do.call(rbind,producer_failures) else
    data.frame(case_id=integer(),method=character(),message=character()),'producer_failures')
  sources <- file.path(here,c('run_experiment.R','R/weighted_audit.R','scripts/weighted_audit_reference.py',
    '../../../../cpp/src/estimate/frontier/newton_input_bounds.cpp','../../../../cpp/src/estimate/frontier/newton_adapters.cpp',
    '../../../../cpp/src/estimate/gmm/moment_quadratic.cpp','../../../../cpp/src/estimate/gmm/dls_weight.cpp',
    '../../../../cpp/src/estimate/gmm/detail_weight_inverse.hpp','../../../../cpp/include/magmaan/estimate/gmm/weight.hpp',
    '../../../../cpp/src/estimate/frontier/detail_newton_interval.hpp','../../../../r-package/src/fit.cpp'))
  ref <- magmaan_cache_ref()
  write_metadata(file.path(out,'metadata.csv'),values=list(lane='weighted_audit',seed_base=seed,reps=reps,
    command=paste(commandArgs(),collapse=' '),elapsed_s=proc.time()[['elapsed']]-start_clock,threads=1,
    source_md5=paste(tools::md5sum(sources),collapse=';'),git_head=ref$git_head,git_dirty=ref$git_dirty,
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),
    start='actual API default, no start override',controls='PORT-NLS stock controls, sample_units, unrestricted, no polish',
    weight_route='advertised complete-data fit_model ULS/GLS/DWLS/WLS/DLS; custom W only for uneven/dense; sphere supplied fixed W',
    scope='conditional numerical verification, original objective, weight recipes; no optimality or default decision'),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/weighted_audit_reference.py')),'--run-dir',shQuote(normalizePath(out))))
  if(status!=0)stop('weighted independent checks failed; inputs retained')
  comparison <- read.csv(file.path(out,'comparisons.csv'),stringsAsFactors=FALSE)
  endpoints <- comparison[comparison$target=='0',]
  terminal <- merge(do.call(rbind,fits),endpoints,by=c('case_id','method','chart'),all.x=TRUE)
  terminal <- merge(terminal,do.call(rbind,cases)[c('case_id','family')],by='case_id',suffixes=c('','.case'))
  terminal$family <- terminal$family.case; terminal$family.case <- NULL
  yes <- function(x)!is.na(x) & tolower(as.character(x))=='true'
  terminal$passed <- yes(terminal$selected_passed)
  terminal$loss <- yes(terminal$legacy_passed) & !terminal$passed
  summary <- do.call(rbind,lapply(split(terminal,list(terminal$family,terminal$chart),drop=TRUE),function(x)
    data.frame(family=x$family[1],chart=x$chart[1],fits=nrow(x),returned=sum(x$returned),
      legacy_passed=sum(yes(x$legacy_passed)),verified_passed=sum(x$passed),
      failed=sum(x$returned & !is.na(x$selected_passed) & !x$passed),
      unchecked=sum(x$returned & is.na(x$selected_passed)),legacy_losses=sum(x$loss),
      median_fit_seconds=median(x$seconds))))
  write_out(summary,'family_summary')
  write_out(terminal[!terminal$passed,c('case_id','family','method','chart','returned','legacy_passed',
      'selected_passed','source_status','reference_positive','reference_distance','loss','message')],'failures')
  cat('Weighted evidence saved: ',out,'\n',sep='')
}
