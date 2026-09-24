summarize_run <- function(out) {
  combine<-function(stem) {
    paths<-list.files(out,paste0('^',stem,'-[0-9]+\\.csv$'),full.names=TRUE)
    if(!length(paths)) return(data.frame())
    x<-do.call(rbind,lapply(paths,read.csv,stringsAsFactors=FALSE))
    write.csv(x,file.path(out,paste0(stem,'.csv')),row.names=FALSE,na='');x
  }
  samples<-combine('samples'); checks<-combine('checks'); diagnostics<-combine('diagnostics')
  combine('stages'); errors<-combine('errors');combine('values');combine('metadata')
  planned<-read.csv(file.path(out,'planned.csv'))
  if(!nrow(samples)) stop('No timings; inspect errors CSV.')
  groups<-split(samples,interaction(samples$case,samples$boundary,samples$workload,drop=TRUE))
  rows<-lapply(groups,function(d) {
    ms<-tapply(d$ms,d$engine,median)
    paired<-merge(d[d$engine=='magmaan',],d[d$engine=='lavaan',],by=c('session','batch'))
    cc<-checks[checks$case==d$case[1] &
      (checks$boundary==d$boundary[1] | checks$boundary=='all') &
      (checks$workload==d$workload[1] | checks$workload=='all'),]
    expected_sessions<-planned$session[planned$case==d$case[1]]
    complete<-setequal(unique(d$session),expected_sessions) &&
      (!nrow(errors) || !d$case[1] %in% errors$case)
    valid<-nrow(paired)>0 && nrow(cc)>0 && all(cc$passed) && complete
    ratios<-if(valid) paired$ms.y/paired$ms.x else NA_real_
    data.frame(case=d$case[1],boundary=d$boundary[1],workload=d$workload[1],
      magmaan_ms=if('magmaan'%in%names(ms))unname(ms['magmaan']) else NA_real_,
      lavaan_ms=if('lavaan'%in%names(ms))unname(ms['lavaan']) else NA_real_,
      accepted=valid,paired_ratio=median(ratios),
      ratio_q25=if(valid)unname(quantile(ratios,.25)) else NA_real_,
      ratio_q75=if(valid)unname(quantile(ratios,.75)) else NA_real_,
      sessions=length(unique(d$session)),batches=nrow(d))
  })
  summary<-do.call(rbind,rows)
  write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE,na='')
  print(summary[summary$boundary=='raw',],row.names=FALSE)
}
