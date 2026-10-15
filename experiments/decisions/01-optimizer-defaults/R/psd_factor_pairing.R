key<-function(d,cols)do.call(paste,c(d[cols],sep="|"))
psd_factor_best <- function(x) {
x$key<-key(x,c("family","population","model","estimator","N","replication","seed"))
stopifnot(!anyDuplicated(paste(x$key,x$arm)))
x$best<-FALSE
for(ids in split(seq_len(nrow(x)),x$key)) {
 d<-x[ids,];stopifnot(setequal(d$arm,c("triangular","square")))
 success<-d$converged & is.finite(d$fmin)
 if(any(success)) {best<-min(d$fmin[success]);x$best[ids]<-success & d$fmin<=best+1e-6*(1+abs(best))}
}
x
}
psd_factor_pair <- function(x, candidate="square", baseline="triangular") {
a<-x[x$arm==candidate,]; b<-x[x$arm==baseline,]
p<-merge(a,b,by="key",suffixes=c("_square","_triangular"))
for(k in c("family","role","estimator"))p[[k]]<-p[[paste0(k,"_square")]]
p$success_win<-p$converged_square & !p$converged_triangular
p$success_loss<-!p$converged_square & p$converged_triangular
p$best_win<-p$best_square & !p$best_triangular
p$best_loss<-!p$best_square & p$best_triangular
p$win<-p$success_win | (p$converged_square & p$converged_triangular & p$best_win)
p$loss<-p$success_loss | (p$converged_square & p$converged_triangular & p$best_loss)
p
}
