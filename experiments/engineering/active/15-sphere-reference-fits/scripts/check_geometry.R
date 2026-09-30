#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
for(f in c('designs.R','fit.R','start_design.R','expanded_designs.R','geometry_designs.R'))source(file.path(here,'R',f))
# Verify the independent closure sequence against the library's covariance
# map and objective, including the library's one-half discrepancy convention.
d<-geometry_designs()$nonattainment; spec<-magmaanlab::model_spec(d$syntax)
sample<-sample_moments(exact_data(d$sigma));S<-d$sigma;last<-Inf
for(t in c(1,2,4,8,16,32,64)) {
 p<-spec$partable;v<-p$ustart
 load<-c(1,.3/t^2,.4/t^2);names(load)<-paste0('x',1:3)
 residual<-1-c(t,.3/t,.4/t)^2;names(residual)<-names(load)
 v[p$op=='=~']<-load[p$rhs[p$op=='=~']]
 v[p$op=='~~' & p$lhs=='X']<-t^2
 ii<-p$op=='~~' & p$lhs!='X';v[ii]<-residual[p$lhs[ii]]
 theta<-numeric(max(p$free));theta[p$free[p$free>0]]<-v[p$free>0]
 ev<-magmaanlab::magmaan_core$evaluate_at(p,sample,theta,estimator='ML')
 Sigma<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
 expected<-S;expected[2,3]<-expected[3,2]<-.12/t^2
 F<-as.numeric(determinant(expected,logarithm=TRUE)$modulus)-as.numeric(determinant(S,logarithm=TRUE)$modulus)+sum(diag(solve(expected,S)))-3
 stopifnot(max(abs(Sigma-expected))<1e-10,abs(ev$fmin-F/2)<1e-10,F<last,
   ev$diagnostics$admissibility$implied_sigma_pd)
 last<-F
}
f<-read.csv(file.path(here,'results','geometry-witnesses','fits.csv'))
stopifnot(nrow(f)==118)
for(name in c('interior','marker_pole','residual_face','stdlv_pole'))for(domain in c('ML','PSD')) {
 z<-f[f$design==name & f$domain==domain & f$route=='alternate_marker',]
 stopifnot(any(z$screened & z$matches_covariance))
 z<-f[f$design==name & f$domain==domain & f$route=='sphere',]
 stopifnot(all(z$screened & z$matches_covariance))
}
# A known nonattainment example must not be certified by the exploratory screen.
z<-f[f$design=='nonattainment' & f$domain=='ML',]
stopifnot(!any(z$screened))
z<-f[f$design=='nonattainment' & f$domain=='PSD',]
stopifnot(all(z$screened),all(z$psd_position=='near_PSD_face'),diff(range(z$objective))<1e-8)
# Both chart poles must remain visible after a successful alternative-chart fit.
z<-f[f$design %in% c('marker_pole','stdlv_pole') & f$route=='sphere',]
stopifnot(all(z$chart_position=='translation_failed_or_near_boundary'))
# Feasibility/face nominations should be invariant under changes of units.
d<-geometry_designs()$residual_face;data<-exact_data(d$sigma);sample<-sample_moments(data)
spec<-geometry_marker(d);fit<-magmaanlab::frontier_fit_ml_psd(spec,data,preconditioning='none',control=list(start='fabin3',start_transport='auto'))
a<-geometry_endpoint(fit$partable,d,sample,'PSD')
u<-c(.2,3,7,.5);names(u)<-colnames(d$sigma);p<-fit$partable
# Normalize the factor by the new unit of the fixed second indicator.
p$est[p$op=='=~']<-p$est[p$op=='=~']*u[p$rhs[p$op=='=~']]/u[2]
ii<-p$op=='~~' & p$lhs!='X';p$est[ii]<-p$est[ii]*u[p$lhs[ii]]^2
p$est[p$op=='~~' & p$lhs=='X']<-p$est[p$op=='~~' & p$lhs=='X']*u[2]^2
d$sigma<-d$sigma*outer(u,u);sample$S[[1]]<-sample$S[[1]]*outer(u,u)
b<-geometry_endpoint(p,d,sample,'PSD')
stopifnot(a$psd_position==b$psd_position,abs(a$min_scaled_primitive-b$min_scaled_primitive)<1e-9)
cat('Analytic closure sequence, chart-pole recovery, PSD boundary and unit-invariance checks passed.\n')
