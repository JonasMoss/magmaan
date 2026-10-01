summarize_intervals <- function(x,timing,out) {
  mean0<-function(z) if(length(z)) mean(z) else NA_real_
  sdse<-function(z) if(length(z)>1L) sd(z)/sqrt(length(z)) else NA_real_
  keys<-c('n','distribution','target','method')
  groups<-split(x,interaction(x[keys],drop=TRUE))
  summary<-do.call(rbind,lapply(groups,function(z) {
    v<-z[z$valid,];m<-nrow(v);N<-nrow(z);p<-mean0(v$covered)
    data.frame(z[1,keys],attempts=N,valid=m,failures=N-m,coverage=p,
      coverage_mcse=if(m) sqrt(p*(1-p)/m) else NA_real_,
      successful_covering=sum(v$covered),successful_covering_rate=sum(v$covered)/N,
      lower_miss=mean0(v$lower_miss),upper_miss=mean0(v$upper_miss),
      mean_width=mean0(v$width),median_width=if(m) median(v$width) else NA_real_,
      mean_asymmetry=mean0(v$asymmetry),mean_seconds=mean(z$seconds),
      mean_refits=mean(z$refits),out_of_domain=sum(v$out_of_domain),
      candidate_inadmissible=sum(z$candidate_inadmissible),
      unrestricted_inadmissible=sum(!z$unrestricted_admissible,na.rm=TRUE),
      max_endpoint_gap=if(any(is.finite(v$endpoint_gap))) max(v$endpoint_gap,na.rm=TRUE) else NA_real_)
  }))
  write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
  bycell<-split(x,interaction(x[c('n','distribution','target')],drop=TRUE))
  comparisons<-list(c('score_E','wald_O'),c('lr_E','wald_O'),c('lr_E','score_E'),
                    c('wald_E','wald_O'),c('score_O','score_E'),c('lr_O','lr_E'))
  paired<-do.call(rbind,lapply(bycell,function(z) do.call(rbind,lapply(comparisons,function(pair) {
    a<-z[z$method==pair[1],];b<-z[z$method==pair[2],]
    both<-merge(a,b,by='replicate',suffixes=c('_a','_b'))
    both<-both[both$valid_a & both$valid_b,]
    d<-as.numeric(both$covered_a)-as.numeric(both$covered_b)
    data.frame(z[1,c('n','distribution','target')],method=pair[1],reference=pair[2],
      common_valid=nrow(both),coverage_difference=mean0(d),difference_mcse=sdse(d),
      width_difference=mean0(both$width_a-both$width_b))
  }))))
  write.csv(paired,file.path(out,'paired.csv'),row.names=FALSE)
  common<-do.call(rbind,lapply(bycell,function(z) {
    ids<-Reduce(intersect,lapply(primary,function(m) z$replicate[z$method==m & z$valid]))
    do.call(rbind,lapply(primary,function(m) {
      v<-z[z$method==m & z$replicate%in%ids,]
      data.frame(z[1,c('n','distribution','target')],method=m,common_valid=nrow(v),
                 coverage=mean0(v$covered),mean_width=mean0(v$width))
    }))
  }))
  write.csv(common,file.path(out,'common_valid.csv'),row.names=FALSE)
  failure<-x[!x$valid,]
  failure$category<-ifelse(grepl('sensitivity',failure$error),'sensitivity',
    ifelse(grepl('disconnected|bracket',failure$error),'acceptance_set',
      ifelse(grepl('endpoint',failure$error),'endpoint',
        ifelse(grepl('verdict|fallback|orientation',failure$error),'fit','other'))))
  counts<-if(nrow(failure)) aggregate(rep(1,nrow(failure)),failure[c(keys,'category')],sum) else
    data.frame(n=integer(),distribution=character(),target=character(),method=character(),category=character(),x=integer())
  names(counts)[ncol(counts)]<-'count'
  write.csv(counts,file.path(out,'failures.csv'),row.names=FALSE)
  write.csv(failure,file.path(out,'failed_intervals.csv'),row.names=FALSE)
  write.csv(aggregate(timing[c('shared_fit_seconds','total_seconds')],timing[c('n','distribution')],mean),
            file.path(out,'timing_summary.csv'),row.names=FALSE)
}
