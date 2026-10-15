key<-function(d,cols)do.call(paste,c(d[cols],sep="|"))
psd_factor_best <- function(x) {
x$key<-key(x,c("family","population","model","estimator","N","replication","seed"))
stopifnot(!anyDuplicated(paste(x$key,x$arm)))
x$best<-FALSE
for(ids in split(seq_len(nrow(x)),x$key)) {
 d<-x[ids,];stopifnot(setequal(d$arm,unique(x$arm)))
 success<-d$converged & is.finite(d$fmin)
 if(any(success)) {best<-min(d$fmin[success]);x$best[ids]<-success & d$fmin<=best+1e-6*(1+abs(best))}
}
x
}
psd_factor_pair <- function(x, candidate="square", baseline="triangular") {
a<-x[x$arm==candidate,]; b<-x[x$arm==baseline,]
p<-merge(a,b,by="key",suffixes=paste0("_",c(candidate,baseline)))
for(k in c("family","role","estimator"))p[[k]]<-p[[paste0(k,"_",candidate)]]
cget <- function(k) p[[paste0(k,"_",candidate)]]
bget <- function(k) p[[paste0(k,"_",baseline)]]
p$success_win<-cget("converged") & !bget("converged")
p$success_loss<-!cget("converged") & bget("converged")
p$best_win<-cget("best") & !bget("best")
p$best_loss<-!cget("best") & bget("best")
p$win<-p$success_win | (cget("converged") & bget("converged") & p$best_win)
p$loss<-p$success_loss | (cget("converged") & bget("converged") & p$best_loss)
p
}
