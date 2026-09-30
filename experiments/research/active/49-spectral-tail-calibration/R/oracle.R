# Population geometry for this study's generator X = M z, E zz' = I.
# Components are normal, standardized chi-square1, or calibrated native IG Pearson marginals.
# Cov(vech XX') = Gaussian pairings + sum_r kappa_r vech(m_r m_r')^2.
cfa_oracle <- function(model,p,distribution) {
  stopifnot(model%in%c('one_factor','two_factor'),p>=4,p%%2==0,
            distribution%in%c('normal','skewed','ig_moderate','ig_severe','ig_moderate_cholesky','ig_severe_cholesky'))
  k <- if(model=='one_factor') 1L else 2L
  loading <- rep(seq(.45,.8,length.out=p/k),k)
  L <- matrix(0,p,k); L[cbind(seq_len(p),rep(seq_len(k),each=p/k))] <- loading
  Phi <- if(k==1) matrix(1,1,1) else matrix(c(1,.4,.4,1),2,2)
  M <- cbind(L%*%t(chol(Phi)),diag(sqrt(1-loading^2)))
  Sigma <- tcrossprod(M); P <- solve(Sigma)
  idx <- which(lower.tri(Sigma,diag=TRUE),arr.ind=TRUE);q <- nrow(idx)
  vech <- function(x)x[idx]
  E <- lapply(seq_len(q),function(a) {
    x <- matrix(0,p,p);x[idx[a,1],idx[a,2]] <- 1;x[idx[a,2],idx[a,1]] <- 1;x
  })
  PE <- lapply(E,function(e)P%*%e)
  H <- outer(seq_len(q),seq_len(q),Vectorize(function(a,b).5*sum(PE[[a]]*t(PE[[b]]))))
  i <- idx[,1];j <- idx[,2]
  Gamma_normal <- Sigma[i,i]*Sigma[j,j]+Sigma[i,j]*Sigma[j,i]
  U <- vapply(seq_len(ncol(M)),function(r)vech(tcrossprod(M[,r])),numeric(q))
  calibration <- NULL
  if(startsWith(distribution,'ig_')) {
    target <- if(grepl('moderate',distribution))c(2,7) else c(3,21)
    calibration <- magmaanlab::magmaan_core$sim_ig_calibrate(
      Sigma,rep(target[1],p),rep(target[2],p),root=if(grepl('cholesky',distribution))'cholesky' else 'symmetric',generator_family='pearson')
    M <- calibration$root
    U <- vapply(seq_len(ncol(M)),function(r)vech(tcrossprod(M[,r])),numeric(q))
    excess <- calibration$generator_excess_kurtosis
  } else excess <- rep(if(distribution=='normal')0 else 12,ncol(M))
  extra <- sweep(U,2,sqrt(excess),'*')
  Gamma <- Gamma_normal+tcrossprod(extra)
  Delta <- vapply(seq_len(p),function(a) {
    dL <- matrix(0,p,k);dL[a,ceiling(a/(p/k))] <- 1
    v <- dL%*%Phi%*%t(L);vech(v+t(v))
  },numeric(q))
  Delta <- cbind(Delta,vapply(seq_len(p),function(a) {
    v <- matrix(0,p,p);v[a,a] <- 1;vech(v)
  },numeric(q)))
  if(k==2) Delta <- cbind(Delta,vech(L%*%matrix(c(0,1,1,0),2,2)%*%t(L)))
  tangent <- qr(Delta);rank <- tangent$rank;Q <- qr.Q(tangent,complete=TRUE)
  K <- Q[,seq_len(rank),drop=FALSE];D <- Q[,(rank+1):q,drop=FALSE]
  # Reuse the public matrix-composition interface for nuisance projection.
  co <- magmaanlab::score_components_from_matrices(rep(0,q),matrix(0,1,q),H,
                                                  nuisance=K,directions=D)
  so <- magmaanlab::project_scores(co)
  W <- so$projection%*%solve(chol(so$metric))
  meat <- H%*%Gamma%*%H
  A <- crossprod(W,meat%*%W);A <- (A+t(A))/2
  lambda <- eigen(A,symmetric=TRUE,only.values=TRUE)$values
  stopifnot(min(lambda)>0,max(abs(crossprod(W,H%*%W)-diag(q-rank)))<1e-9)
  list(lambda=lambda,A=A,H=H,Gamma=Gamma,Gamma_normal=Gamma_normal,U=U,
       Delta=Delta,W=W,Sigma=Sigma,M=M,L=L,Phi=Phi,idx=idx,
       calibration=calibration,excess=excess,tangent_rank=rank,df=q-rank,tau1=sum(lambda),tau2=sum(lambda^2),
       projected_excess=max(abs(tcrossprod(crossprod(W,H%*%extra)))))
}

mv_constrained_moment <- function(tau1,tau2,df) {
  if(any(!is.finite(c(tau1,tau2,df))) || tau1<=0 || tau2<=0 || df<1)
    stop('Invalid constrained MV moments or rank')
  max(tau2,tau1^2/df)
}
