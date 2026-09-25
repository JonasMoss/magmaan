# Run from anywhere: Rscript .../R/validate.R
base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
source(file.path(base,'spectral.R'))
# Independent brute force cycle enumeration, orders 1--6 (no sample centering).
set.seed(9); Y <- matrix(rnorm(16*3),16,3); D <- (Y[1:8,]-Y[9:16,])/sqrt(2)
a <- cycle_moments(Y)
for(k in 2:6) {
  b <- mean(apply(combn(8,k),2,function(idx) {
    prod(vapply(seq_len(k),function(j)sum(D[idx[j],]*D[idx[j%%k+1],]),numeric(1)))
  }))
  stopifnot(abs(a[k]-b)<1e-10)
}
stopifnot(max(abs(cycle_moments(Y+matrix(rep(c(2,3,4),each=16),16))-a))<1e-9)
# Fixed-transform independent-data Monte Carlo checks unbiased moments.
set.seed(10)
mc <- replicate(1500,cycle_moments(matrix(rnorm(60*3),60,3)%*%diag(sqrt(c(.5,1,2)))) )
truth <- vapply(1:6,function(k)sum(c(.5,1,2)^k),numeric(1))
stopifnot(all(abs(rowMeans(mc)-truth)<5*apply(mc,1,sd)/sqrt(ncol(mc))))
# Spectrum reconstruction and scale equivariance; exact-gamma saddlepoint check.
lambda <- seq(.5,2,length.out=12); moments <- vapply(1:6,function(k)sum(lambda^k),numeric(1))
w <- moment_weights(moments,lambda)
stopifnot(w$residual<1e-8)
q <- qchisq(.95,12)*2
stopifnot(abs(spectral_saddle(q,rep(2,12))-.05)<.001,
          abs(spectral_saddle(24,rep(2,12))-pchisq(12,12,lower.tail=FALSE))<.003,
          max(abs(nonlinear_weights(3*lambda,100)-3*nonlinear_weights(lambda,100)))<1e-9,
          abs(cycle_cgf_tail(2*q,2^(1:6)*moments)-cycle_cgf_tail(q,moments))<1e-8)
# Gaussian identity population: ANS should remove much empirical dispersion.
set.seed(12); X <- matrix(rnorm(300*60),300,60)
raw <- eigen(crossprod(X)/300,symmetric=TRUE,only.values=TRUE)$values
shrunk <- nonlinear_weights(raw,300)
stopifnot(mean((shrunk-1)^2)<mean((raw-1)^2),all(shrunk>0))
cat('Passed: brute-force cycles, location invariance, unbiasedness Monte Carlo,\n',
    'positive reconstruction, gamma saddlepoint, scale equivariance, ANS identity check.\n')
