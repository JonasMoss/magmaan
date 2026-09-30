validate_methods <- function(output) {
  results<-list();audit<-list()
  check<-function(name,value,tolerance) {
    results[[length(results)+1L]]<<-data.frame(check=name,error=value,tolerance=tolerance,
      passed=is.finite(value)&&abs(value)<=tolerance)
  }
  for(distribution in distributions) {
    X<-generate(300,distribution,2026112751+match(distribution,distributions))
    u<-fit_ml(X);id<-index_of(u,'loading');b<-u$theta[id]+.1
    r<-fit_ml(X,'loading',b,u$theta);ctx<-magmaanlab::prepare_inference(r,X)
    for(kind in c('expected','observed')) {
      g<-project_parameter(ctx,id,kind)
      nested<-magmaanlab::nested_score_test(u,r,X,sensitivity=kind)
      check(paste(distribution,kind,'nested_score'),g$score-nested$statistic_sandwich,1e-6)
      check(paste(distribution,kind,'scalar_scale'),g$score-g$ordinary/g$scale,1e-9)
      # Independent block algebra in validation only; production uses projection.
      a<-setdiff(seq_len(ncol(g$H)),id)
      v<-solve(g$H[a,a],g$H[a,id]);efficient<-g$rows[,id]-as.vector(g$rows[,a]%*%v)
      h<-g$H[id,id]-sum(g$H[id,a]*v)
      check(paste(distribution,kind,'efficient_score'),g$score-sum(efficient)^2/sum(efficient^2),1e-9)
      check(paste(distribution,kind,'scale_ratio'),g$scale-sum(efficient^2)/h,1e-9)
      # Landed profile helper uses centered meat. Check c_unc=c_center+S_naive/n.
      prof<-core$frontier_profile_lrt_parameter_ml(u,id,b,raw_data=X,
        optimizer='nlopt-slsqp',control=list(max_iter=2000L,ftol=1e-12,gtol=1e-8),
        reference=if(kind=='expected') 'robust_scaled' else 'misspec_scaled')
      c0<-if(kind=='expected') prof$scaling_factor else prof$misspec_scaling_factor
      check(paste(distribution,kind,'centered_meat_identity'),g$scale-c0-g$ordinary/nrow(X),1e-4)
      audit[[length(audit)+1L]]<-data.frame(distribution=distribution,sensitivity=kind,
        uncentered_scale=g$scale,landed_centered_scale=c0,predicted_gap=g$ordinary/nrow(X),
        actual_gap=g$scale-c0)
    }
    V<-wald_covariance(u,X,'wald_O');policy<-magmaanlab::policy_inference(u,X)
    check(paste(distribution,'policy_covariance'),max(abs(V-policy$covariance)),1e-9)
    # Full-model score at the restricted fit, independent likelihood derivative.
    S<-crossprod(X)/nrow(X)
    objective<-function(theta) {
      f<-r;f$theta<-theta;Sigma<-core$model_implied(f)$sigma[[1]]
      -.5*nrow(X)*(as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)+sum(diag(solve(Sigma,S))))
    }
    eps<-1e-5
    fd<-vapply(seq_along(r$theta),function(j) {
      plus<-minus<-r$theta;plus[j]<-plus[j]+eps;minus[j]<-minus[j]-eps
      (objective(plus)-objective(minus))/(2*eps)
    },numeric(1))
    check(paste(distribution,'full_score_derivative'),max(abs(fd-magmaanlab::scores(ctx)$score))/nrow(X),1e-6)
    if(distribution=='normal') {
      lu<-lavaan::cfa(syntax,data=as.data.frame(X),std.lv=TRUE,meanstructure=FALSE,se='none',test='standard')
      lc<-lavaan::cfa(paste(syntax,sprintf('b == %.17g',b),sep='\n'),
        data=as.data.frame(X),std.lv=TRUE,meanstructure=FALSE,se='none',test='standard')
      check('lavaan_convergence',as.numeric(!lavaan::lavInspect(lu,'converged')||!lavaan::lavInspect(lc,'converged')),0)
      check('normal_profile_LR_lavaan',lr_statistic(u,r,nrow(X))-
        2*(lavaan::fitMeasures(lu,'logl')-lavaan::fitMeasures(lc,'logl')),1e-5)
    }
    # This is a numerical smoke at an interior sample, not a coverage gate.
    for(method in c('score_E','lr_E','score_O','lr_O')) {
      profile<-make_profile(u,X,'loading',method)
      se<-sqrt(wald_covariance(u,X,'wald_naive')[id,id])
      ci<-invert_profile(profile,u$theta[id],1.96*se)
      gap<-max(abs(c(profile$evaluate(ci$lower)$statistic,profile$evaluate(ci$upper)$statistic)-critical))
      check(paste(distribution,method,'endpoint'),gap,1e-4)
      check(paste(distribution,method,'truth_inversion'),as.numeric(
        (ci$lower<=truth[['loading']]&&ci$upper>=truth[['loading']])!=
        (profile$evaluate(truth[['loading']])$statistic<=critical)),0)
    }
    # Empirical fourth moments give a Monte Carlo error scale for the covariance
    # check. Gamma has all moments; standardized t(10) has finite fourth moments.
    big<-generate(100000,distribution,2026122751+match(distribution,distributions))
    Sigma<-population();z<-numeric()
    for(i in 1:8) for(j in 1:i) {
      products<-big[,i]*big[,j]
      z<-c(z,(mean(products)-Sigma[i,j])/(sd(products)/sqrt(nrow(big))))
    }
    check(paste(distribution,'population_covariance_max_mc_z'),max(abs(z)),5)
  }
  # Analytic total-information normalization in the one-variable normal model.
  set.seed(2026122755);x<-rnorm(250);x<-x-mean(x)
  f<-magmaanlab::fit_model('x ~~ v*x',data.frame(x=x),meanstructure=FALSE)
  check('normal_variance_total_information',core$infer_information_expected(f)[1,1]-250/(2*f$theta[1]^2),1e-7)
  out<-do.call(rbind,results)
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  write.csv(out,file.path(output,'validation.csv'),row.names=FALSE)
  write.csv(do.call(rbind,audit),file.path(output,'api_audit.csv'),row.names=FALSE)
  print(out,row.names=FALSE,digits=3)
  if(!all(out$passed)) stop('Numerical validation failed; do not run the simulation')
  invisible(out)
}
