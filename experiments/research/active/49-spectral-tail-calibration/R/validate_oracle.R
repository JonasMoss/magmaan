base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
source(file.path(base,'oracle.R'));source(file.path(base,'spectral.R'));source(file.path(base,'mv.R'))
set.seed(20260928)
for(model in c('one_factor','two_factor')) for(p in c(8,12)) for(dist in c('normal','skewed')) {
  o <- cfa_oracle(model,p,dist);q <- nrow(o$H)
  stopifnot(o$df==p*(p+1)/2-2*p-if(model=='two_factor')1 else 0,
            max(abs(o$H%*%o$Gamma_normal-diag(q)))<1e-10,
            max(abs(o$lambda-1))<1e-9,o$projected_excess<1e-20,
            max(abs(crossprod(o$Delta,o$H%*%o$W)))<1e-10)
  # Finite differences of the actual covariance map, holding residual variances fixed.
  eps <- 1e-6
  for(a in seq_len(p)) {
    plus <- minus <- o$L;k <- ceiling(a/(p/ncol(o$L)))
    plus[a,k] <- plus[a,k]+eps;minus[a,k] <- minus[a,k]-eps
    d <- (plus%*%o$Phi%*%t(plus)-minus%*%o$Phi%*%t(minus))/(2*eps)
    stopifnot(max(abs(d[o$idx]-o$Delta[,a]))<1e-9)
  }
  # Moment-space residual projector is an independent route to the same spectrum.
  R <- o$H-o$H%*%o$Delta%*%solve(crossprod(o$Delta,o$H%*%o$Delta))%*%t(o$Delta)%*%o$H
  C <- chol(o$Gamma);ev <- eigen(C%*%R%*%t(C),symmetric=TRUE,only.values=TRUE)$values
  stopifnot(max(abs(ev[seq_len(o$df)]-sort(o$lambda,decreasing=TRUE)))<1e-9,
            max(abs(tail(ev,q-o$df)))<1e-9)
}
# Check that the generator tangent agrees with the actual fitted CFA interface.
# Synthetic observations have exactly the population covariance (denominator n).
for(model in c('one_factor','two_factor')) for(p in c(8,12)) {
  o <- cfa_oracle(model,p,'normal');n <- 100L
  Z <- scale(matrix(rnorm(n*p),n,p),scale=FALSE)
  X <- as.data.frame(sqrt(n)*qr.Q(qr(Z))%*%chol(o$Sigma));names(X) <- paste0('x',seq_len(p))
  syntax <- if(model=='one_factor')paste('f =~',paste(names(X),collapse='+')) else
    paste(paste('f1 =~',paste(names(X)[1:(p/2)],collapse='+')),
          paste('f2 =~',paste(names(X)[(p/2+1):p],collapse='+')),sep='\n')
  f <- magmaanlab::fit_model(syntax,X,estimator='ML')
  stopifnot(isTRUE(f$converged))
  c <- magmaanlab::score_components(f,X,sensitivity='expected',metric='expected')
  s <- magmaanlab::project_scores(c)
  W <- s$projection%*%solve(chol(s$metric))*sqrt(n)
  stopifnot(ncol(W)==o$df,max(abs(tcrossprod(W)-tcrossprod(o$W)))<1e-4,
            max(abs(c$metric/n-o$H))<1e-4)
}
# Check non-Gaussian fourth moments before projection, where they do NOT vanish.
# Each entry is compared with its own Monte Carlo standard error.
for(dist in c('normal','skewed')) {
  o <- cfa_oracle('two_factor',8,dist);n <- 100000L
  Z <- matrix(rnorm(n*ncol(o$M)),n,ncol(o$M))
  if(dist=='skewed')Z <- (Z^2-1)/sqrt(2)
  X <- Z%*%t(o$M)
  V <- vapply(seq_len(nrow(o$idx)),function(a)
    X[,o$idx[a,1]]*X[,o$idx[a,2]]-o$Sigma[o$idx[a,1],o$idx[a,2]],numeric(n))
  for(pair in list(c(1,1),c(1,8),c(10,10),c(20,25))) {
    product <- V[,pair[1]]*V[,pair[2]]
    stopifnot(abs(mean(product)-o$Gamma[pair[1],pair[2]])<6*sd(product)/sqrt(n))
  }
  rows <- fixed_covariance_scores(X,rep(0,8),o$Sigma)
  stopifnot(max(abs(rows-V%*%o$H))<1e-9)
  # Trace check of oracle-transformed score covariance using known zero mean.
  lengths <- rowSums((rows%*%o$W)^2)
  stopifnot(abs(mean(lengths)-o$tau1)<6*sd(lengths)/sqrt(n))
}
# When the lower bound binds, constrained MV is exactly the SB tail.
for(q in c(1,20,60)) {
  b <- mv_constrained_moment(25,10,20)
  stopifnot(b==25^2/20,
    abs(mv_moment_tail(q,25,b)-pchisq(q*20/25,20,lower.tail=FALSE))<1e-14,
    mv_constrained_moment(25,100,20)==100)
}
cat('Passed: population tangent derivatives, Gaussian information identity,\n',
    'independent residual-projector spectrum, fitted CFA geometry, nonnormal fourth moments,\n',
    'oracle score transport and trace, constrained-MV/SB identity.\n')
# IG roots preserve covariance and target marginal moments but change dependence.
for(p in c(8,12)) for(dist in c('ig_moderate','ig_severe','ig_moderate_cholesky','ig_severe_cholesky')) {
  o <- cfa_oracle('two_factor',p,dist);target <- if(grepl('moderate',dist))c(2,7) else c(3,21)
  skew <- as.numeric(o$M^3%*%o$calibration$generator_skewness)/diag(o$Sigma)^1.5
  kurt <- as.numeric(o$M^4%*%o$excess)/diag(o$Sigma)^2
  stopifnot(max(abs(tcrossprod(o$M)-o$Sigma))<1e-10,
            max(abs(skew-target[1]))<1e-7,max(abs(kurt-target[2]))<1e-7,
            min(o$lambda)>1-1e-9,max(o$lambda)>1.001)
  R <- o$H-o$H%*%o$Delta%*%solve(crossprod(o$Delta,o$H%*%o$Delta))%*%t(o$Delta)%*%o$H
  C <- chol(o$Gamma);ev <- eigen(C%*%R%*%t(C),symmetric=TRUE,only.values=TRUE)$values
  stopifnot(max(abs(ev[seq_len(o$df)]-sort(o$lambda,decreasing=TRUE)))<1e-8)
}
o <- cfa_oracle('two_factor',8,'ig_moderate_cholesky');n <- 100000L
X <- magmaanlab::magmaan_core$sim_ig_draw(o$calibration,n=n,reps=1L,seed_base=20260929)$draws[[1]]
Y <- fixed_covariance_scores(X,rep(0,8),o$Sigma)%*%o$W
lengths <- rowSums(Y^2)
stopifnot(abs(mean(lengths)-o$tau1)<6*sd(lengths)/sqrt(n))
cat('Passed: IG covariance and marginal-moment targets, heterogeneous spectrum,\n',
    'independent residual-projector spectrum and native-draw oracle trace.\n')
