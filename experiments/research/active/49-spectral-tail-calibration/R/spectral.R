# Experimental estimators, deliberately local to this study. No package defaults.
# Ledoit & Wolf (2020), equations (4.7)--(4.9), p<n branch.
# Covariance-loss shrinkage is a heuristic for mixture weights, NOT QuEST
# population eigenvalue recovery. Transformed SEM rows need not satisfy MP.
nonlinear_weights <- function(lambda, n) {
  d <- length(lambda)
  if (d >= n || min(lambda) <= 0) stop('ANS requires positive spectrum and d < n')
  h <- lambda * n^(-1/3)
  x <- sweep(outer(lambda, lambda, '-'), 2, h, '/')
  a <- 1-x^2/5
  term <- a * log(abs((sqrt(5)-x)/(sqrt(5)+x)))
  term[abs(a)<1e-12] <- 0
  density <- rowMeans(sweep(3/(4*sqrt(5))*pmax(a,0),2,h,'/'))
  hilbert <- rowMeans(sweep(-3*x/(10*pi)+3*term/(4*sqrt(5)*pi),2,h,'/'))
  lambda / ((pi*d/n*lambda*density)^2+(1-d/n-pi*d/n*lambda*hilbert)^2)
}

# Kong & Valiant (2017): increasing-index cycles. Unbiased for independent,
# zero-mean rows with a FIXED transform; not the fully symmetrized U-statistic.
# Pair differences remove unknown means without shared sample centering.
cycle_moments <- function(Y, K=6L) {
  m <- nrow(Y)%/%2L
  if(m<K) stop('Too few independent pairs')
  D <- (Y[seq_len(m),,drop=FALSE]-Y[m+seq_len(m),,drop=FALSE])/sqrt(2)
  G <- tcrossprod(D); U <- G; U[lower.tri(U,diag=TRUE)] <- 0
  ans <- numeric(K); ans[1] <- mean(diag(G)); P <- U
  for(k in 2:K) {
    ans[k] <- sum(P*t(G))/choose(m,k)
    if(k<K) P <- P%*%U
  }
  ans
}

# Equal-mass positive spectral reconstruction, explicit approximate moment fit.
# Enforces exactly d nonnegative mixture weights; records fitting residual.
moment_weights <- function(moments, raw) {
  K <- length(moments); d <- length(raw)
  if(any(!is.finite(moments)) || any(moments<=0)) stop('Nonpositive cycle moment')
  scale <- moments[1]/d
  target <- moments / scale^(seq_len(K)); denom <- pmax(target,d)
  objective <- function(x) {
    predicted <- vapply(seq_len(K),function(k)sum(x^k),numeric(1))
    sum(((predicted-target)/denom)^2)
  }
  gradient <- function(x) {
    residual <- (vapply(seq_len(K),function(k)sum(x^k),numeric(1))-target)/denom^2
    Reduce('+',lapply(seq_len(K),function(k)2*residual[k]*k*x^(k-1)))
  }
  z <- optim(pmax(raw/scale,1e-6),objective,gradient,method='L-BFGS-B',
             lower=0,upper=max(2,2*max(raw/scale)),control=list(maxit=300,factr=1e8))
  if((z$convergence!=0 && z$value>1e-16) || !is.finite(z$value)) stop('Moment reconstruction did not converge')
  list(weights=z$par*scale,residual=sqrt(z$value))
}

# Lugannani--Rice from an exact positive-spectrum CGF, including the t=0 limit.
spectral_saddle <- function(q, lambda) {
  lambda <- lambda[lambda>0]; mu <- sum(lambda)
  if(q<=0) return(1)
  if(abs(q-mu)<1e-6*mu) return(.5-8*sum(lambda^3)/(6*sqrt(2*pi)*(2*sum(lambda^2))^1.5))
  kp <- function(t)sum(lambda/(1-2*t*lambda))
  lo <- -1; while(kp(lo)>q) lo <- lo*2
  t <- uniroot(function(t)kp(t)-q,c(lo,(1-1e-10)/(2*max(lambda))),tol=1e-12)$root
  K <- -.5*sum(log1p(-2*t*lambda)); kpp <- 2*sum((lambda/(1-2*t*lambda))^2)
  w <- sign(t)*sqrt(2*(t*q-K)); u <- t*sqrt(kpp)
  pnorm(w,lower.tail=FALSE)+dnorm(w)*(1/u-1/w)
}

# Direct truncated CGF pilot: only right of the estimated mean. Below it,
# return NA rather than invent a CDF from a polynomial that may not be a CGF.
# The runner records a separate nonrejection decision there, never a fake p-value.
cycle_cgf_tail <- function(q, moments) {
  if(any(!is.finite(moments)) || any(moments<=0)) stop('Nonpositive cycle moment')
  if(q<=moments[1]) return(NA_real_)
  k <- seq_along(moments); b <- 2^(k-1)*moments/k
  kp <- function(t)sum(k*b*t^(k-1))
  hi <- 1/sqrt(moments[2]); while(kp(hi)<q) hi <- hi*2
  t <- uniroot(function(t)kp(t)-q,c(0,hi),tol=1e-12)$root
  K <- sum(b*t^k); kpp <- sum((k*(k-1)*b*t^pmax(0,k-2))[k>=2])
  if(t<1e-8/sqrt(moments[2])) return(.5)
  w <- sqrt(2*(t*q-K)); u <- t*sqrt(kpp)
  pnorm(w,lower.tail=FALSE)+dnorm(w)*(1/u-1/w)
}

fmg_tail <- function(q, lambda, method='all') {
  magmaanlab:::infer_fmg_test(q,length(lambda),lambda,method=method,param=4)$p_value
}

# Gaussian covariance score in column-major vech order, fixed evaluation point.
# Used only to transport an independently generated calibration sample; checked
# against core rows in every replication before using it.
fixed_covariance_scores <- function(X, mu, sigma) {
  precision <- solve(sigma); Z <- sweep(as.matrix(X),2,mu,'-')%*%precision
  idx <- which(lower.tri(sigma,diag=TRUE),arr.ind=TRUE)
  vapply(seq_len(nrow(idx)),function(k) {
    i <- idx[k,1]; j <- idx[k,2]
    (Z[,i]*Z[,j]-precision[i,j]) * if(i==j) .5 else 1
  },numeric(nrow(X)))
}
