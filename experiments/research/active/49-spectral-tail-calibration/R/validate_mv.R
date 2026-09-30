base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
source(file.path(base,'mv.R'))
set.seed(77); Y <- matrix(rnorm(7*3),7,3)+4
# Independent O(n^4) positive kernel; each set of four has three pairings.
brute <- mean(apply(combn(nrow(Y),4),2,function(i) {
  mean(c(sum((Y[i[1],]-Y[i[2],])*(Y[i[3],]-Y[i[4],]))^2,
         sum((Y[i[1],]-Y[i[3],])*(Y[i[2],]-Y[i[4],]))^2,
         sum((Y[i[1],]-Y[i[4],])*(Y[i[2],]-Y[i[3],]))^2))/4
}))
m <- mv_cross_moments(Y)
stopifnot(abs(m['tau2']-brute)<1e-12,
          max(abs(mv_cross_moments(Y+100)-m))<1e-10,
          abs(mv_cross_moments(3*Y)['tau2']-81*m['tau2'])<1e-9,
          max(abs(mv_cross_moments(Y[7:1,])-m))<1e-12)
# Exercise d>n Gram route and rotation invariance.
Q <- qr.Q(qr(matrix(rnorm(100),10,10)))
wide <- cbind(Y,matrix(0,7,7))%*%Q
stopifnot(max(abs(mv_cross_moments(wide)-m))<1e-11)
stopifnot(abs(mv_moment_tail(10,5,5)-pchisq(10,5,lower.tail=FALSE))<1e-14)
# Fixed-transform normal and skewed rows with nonzero means; independent
# covariance oracle diag(.5,1,2). No fitted SEM is involved in this check.
set.seed(78)
for(dist in c('normal','skewed')) {
  out <- replicate(3000,{
    Z <- matrix(rnorm(60*3),60,3)
    if(dist=='skewed') Z <- (Z^2-1)/sqrt(2)
    Y <- sweep(Z,2,sqrt(c(.5,1,2)),'*')+3
    m <- mv_cross_moments(Y)
    c(m,plug=sum(cov(Y)^2))
  })
  target <- c(3.5,5.25)
  stopifnot(all(abs(rowMeans(out)[1:2]-target)<5*apply(out[1:2,],1,sd)/sqrt(ncol(out))),
            mean(out['plug',])>mean(out['tau2',]))
}
stopifnot(inherits(try(mv_cross_moments(matrix(0,5,2)),silent=TRUE),'try-error'))
cat('Passed: U4 enumeration, translation/scale/permutation/rotation identities,\n',
    'wide Gram route, gamma tail, normal/skewed unbiasedness, degenerate input.\n')
