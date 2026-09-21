args<-commandArgs(TRUE);stopifnot(length(args)==2L)
a<-read.csv(file.path(args[1],'raw.csv'));b<-read.csv(file.path(args[2],'raw.csv'))
stopifnot(nrow(a)==1701L,nrow(b)==1701L,
 identical(a[,c('model','n','rep','units','seed','profile')],b[,c('model','n','rep','units','seed','profile')]))
summary<-function(x,backtracking){
 x$eligible<-x$interior==1 & x$status=='available'
 do.call(rbind,lapply(split(x,x$profile),function(z){
  q<-z[z$eligible,]
  data.frame(backtracking=backtracking,profile=z$profile[1],attempts=nrow(z),eligible=nrow(q),
   pass_003=sum(q$distance<=.003),pass_01=sum(q$distance<=.01),pass_03=sum(q$distance<=.03),
   old_pass_eligible=sum(q$old_residual<=.001),negative_rc=sum(z$rc<0),negative_rc_newton_pass=sum(q$rc<0 & q$distance<=.01),
   max_d=max(q$distance),median_evals=median(z$evals),median_fit_ms=median(z$fit_ms),
   median_hessian_audit_ms=median(q$hessian_audit_ms))
 }))
}
s<-rbind(summary(a,10),summary(b,60));write.csv(s,file.path(args[2],'summary.csv'),row.names=FALSE)
print(s,row.names=FALSE)
stopifnot(max(c(a$hessian_fd_rel,b$hessian_fd_rel),na.rm=TRUE)<1e-5)
# Pair terminal results under current and candidate controls without optimizer restarts.
z<-merge(subset(b,profile=='current'),subset(b,profile=='f12_x10'),by=c('model','n','rep','units','seed'),suffixes=c('_current','_candidate'))
z$twice_loglik_improvement<-2*z$n*(z$fmin_current-z$fmin_candidate)
z$predicted_twice_loglik_improvement<-z$distance_current^2
write.csv(z,file.path(args[2],'paired-controls.csv'),row.names=FALSE)
q<-subset(z,interior_current==1 & status_current=='available' & distance_current>.01)
print(q[,c('model','n','rep','units','distance_current','distance_candidate','twice_loglik_improvement','predicted_twice_loglik_improvement')],row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(args[2],'summary-session.txt'))
