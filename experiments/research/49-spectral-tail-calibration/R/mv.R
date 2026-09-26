# Translation-invariant all-distinct U-statistic for tr(Cov(Y)^2).
# Kernel: ((Yi-Yj)'(Yk-Yl))^2/4 over ordered distinct i,j,k,l.
# Expanding gives U2 - 2 U3 + U4. Centering is algebraically invariant,
# not the invalid shortcut of treating sample-centered rows as independent.
# For centered rows: G1=0, sum off(G)^2=C-D, sum off(G)=-T,
# sum(rowSums(off(G))^2)=D. This costs O(n*d*min(n,d)), not O(n^4).
# Unbiased for iid rows with a fixed transform, not estimated SEM projections.
mv_cross_moments <- function(Y) {
  if(!is.matrix(Y) || any(!is.finite(Y)) || nrow(Y)<4 || ncol(Y)<1)
    stop('Supply at least four finite observation rows')
  n <- nrow(Y); Z <- sweep(Y,2,colMeans(Y),'-')
  norm2 <- rowSums(Z^2); T <- sum(norm2); D <- sum(norm2^2)
  C <- if(ncol(Z)<=n) sum(crossprod(Z)^2) else sum(tcrossprod(Z)^2)
  tau2 <- (C-D)/(n*(n-1)) - 2*(2*D-C)/(n*(n-1)*(n-2)) +
    (T^2+2*C-6*D)/(n*(n-1)*(n-2)*(n-3))
  if(!is.finite(tau2) || tau2<=0 || T<=0) stop('Degenerate or nonpositive covariance moment')
  c(tau1=T/(n-1),tau2=tau2)
}

mv_moment_tail <- function(q,tau1,tau2) {
  if(any(!is.finite(c(q,tau1,tau2))) || q<0 || tau1<=0 || tau2<=0)
    stop('Invalid MV statistic or moments')
  pchisq(q*tau1/tau2,df=tau1^2/tau2,lower.tail=FALSE)
}
