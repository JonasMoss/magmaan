# Diagnostic coordinates avoid dividing by latent total variances. The older
# standardized extent can explode when a total variance cancels to zero at a
# finite signed-ML solution. Primitive contributions below are scale invariant.
escape_metrics <- function(pt, sample) {
  blank <- list(record=data.frame(component_extent=NA_real_, latent_total_x=NA_real_,
    latent_total_y=NA_real_, disturbance_y=NA_real_, scaled_path=NA_real_), sigma=NULL)
  if(is.null(pt))return(blank)
  tryCatch({
    target<-strong_marker_model(pt,sample)
    tr<-magmaanlab::frontier_reidentify(pt,target,pole_tol=0)
    ev<-magmaanlab::magmaan_core$evaluate_at(target$partable,sample,tr$theta,estimator='ML')
    p<-ev$partable; S<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
    value<-function(a,op,b)p$est[p$lhs==a & p$op==op & p$rhs==b][1]
    energy<-vapply(c('X','Y'),function(f){z<-p[p$lhs==f&p$op=='=~',];sum(z$est^2/diag(sample$S[[1]])[z$rhs])},0)
    vx<-value('X','~~','X');psi<-value('Y','~~','Y');b<-value('Y','~','X')
    phi<-matrix(c(vx,b*vx,b*vx,b*b*vx+psi),2)
    contributions<-phi*sqrt(outer(energy,energy))
    residual<-p[p$op=='~~' & p$lhs %in% ov_names,]
    by<-b*sqrt(energy[2]/energy[1]);dy<-psi*energy[2]
    extent<-max(abs(contributions),abs(dy),abs(by),abs(residual$est)/diag(sample$S[[1]])[residual$lhs])
    list(record=data.frame(component_extent=extent,latent_total_x=contributions[1,1],
      latent_total_y=contributions[2,2],disturbance_y=dy,scaled_path=by),sigma=S)
  },error=function(e)blank)
}

inspection_charts <- function() {
  out<-list()
  for(x in 1:3)for(y in 1:3){
    line<-function(f,prefix,k)paste(f,'=~',paste(paste0(ifelse(1:3==k,'1*','NA*'),prefix,1:3),collapse=' + '))
    out[[paste0('marker_',x,y)]]<-magmaanlab::model_spec(paste(line('X','x',x),line('Y','y',y),'Y ~ X',sep='\n'))
  }
  out$effect_coded<-magmaanlab::model_spec(model_syntax,effect_coding=TRUE)
  # Fixing disturbance variances to +1 excludes the negative-variance ML sector.
  out$std_lv_positive<-magmaanlab::model_spec(model_syntax,std_lv=TRUE)
  out$sphere<-magmaanlab::model_spec(model_syntax)
  out
}
