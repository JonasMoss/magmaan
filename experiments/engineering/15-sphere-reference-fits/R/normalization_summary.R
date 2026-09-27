summarize_normalization_revisit <- function(x,out) {
 write <- function(z,name)write.csv(z,file.path(out,paste0(name,'.csv')),row.names=FALSE)
 summarize <- function(z,groups,cols){z$attempts<-1L;aggregate(z[c('attempts',cols)],z[groups],function(v)sum(v,na.rm=TRUE))}
 metrics<-c('returned','certified','accepted','target_available','objective_match','better_objective','target_reached','historical_hit')
 write(summarize(x,c('batch','method','arm'),metrics),'summary')
 write(summarize(x,c('batch','design','method','arm'),metrics),'by_family')
 keys<-c('batch','design','n','rep','domain','seed','method')
 stopifnot(!anyDuplicated(x[c(keys,'arm')]))
 p<-merge(x[x$arm=='original_units',],x[x$arm=='normalized',],by=keys,suffixes=c('_original','_normalized'))
 stopifnot(nrow(p)*2==nrow(x))
 p$gain<-!p$target_reached_original & p$target_reached_normalized
 p$loss<-p$target_reached_original & !p$target_reached_normalized
 p$acceptance_gain<-!p$accepted_original & p$accepted_normalized
 p$acceptance_loss<-p$accepted_original & !p$accepted_normalized
 p$verdict_changed<-p$certified_original!=p$certified_normalized
 p$objective_changed<-is.finite(p$objective_original)&is.finite(p$objective_normalized)&abs(p$objective_normalized-p$objective_original)>1e-6*(1+abs(p$objective_original))
 write(p,'paired')
 write(p[p$gain|p$loss|p$acceptance_gain|p$acceptance_loss|p$verdict_changed|p$objective_changed,],'changes')
 write(summarize(p,c('batch','method'),c('gain','loss','acceptance_gain','acceptance_loss','verdict_changed','objective_changed')),'paired_summary')
 write(x[x$target_source=='refined_finite_witness'|x$known_near_pole_case,],'hard_cases')
}
