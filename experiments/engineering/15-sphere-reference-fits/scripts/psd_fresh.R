#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R','start_design.R','expanded_designs.R','geometry_designs.R'))source(file.path(here,'R',f))
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/psd_fresh.R [--smoke] [--run-id NAME]\nRetained 60 fresh draws (smoke: 6). PSD SLSQP none/diagonal.\nOrdinary FABIN3-auto and positive spectral; sphere canonical, positive spectral, three random starts.\nAll arms run independently on every draw. Sphere-only best-observed references.\n');quit(save='no')}
smoke<-'--smoke' %in% args
run<-if('--run-id' %in% args)args[match('--run-id',args)+1L] else if(smoke)'psd-fresh-smoke' else 'psd-fresh'
stopifnot(!is.na(run),grepl('^[a-zA-Z0-9_-]+$',run))
out<-file.path(here,'results',run);if(file.exists(file.path(out,'fits.csv')))stop('choose a fresh run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out<-function(x,name)write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
tasks<-expand.grid(design=names(designs_all()),n=c(20L,100L),rep=1:10,stringsAsFactors=FALSE)
tasks$seed<-902609281L+match(tasks$design,names(designs_all()))*100000L+tasks$n*100L+tasks$rep
if(smoke)tasks<-head(tasks,6)
spec<-magmaanlab::model_spec(model_syntax)
d<-list(blocks=list(X=ov_names[1:3],Y=ov_names[4:6]),structural='Y ~ X',syntax=model_syntax,std_lv=FALSE)
rows<-parameters<-list();t0<-proc.time()[['elapsed']]
for(i in seq_len(nrow(tasks))){
 task<-tasks[i,];data<-draw_data(design_sigma(designs_all()[[task$design]]),task$n,task$seed)
 sample<-sample_moments(data);d$sigma<-sample$S[[1]]
 positive<-spectral_start(spec,sample)
 for(route in c('ordinary','sphere')){
  recipes<-if(route=='ordinary')list(fabin3_auto=NULL,spectral_positive=positive) else
   c(list(sphere_canonical=NULL,spectral_positive=positive),setNames(lapply(1:3,function(j)random_start(spec,sample,task$seed+400000L+j)),paste0('random_',1:3)))
  for(scaling in c('none','diagonal'))for(arm in names(recipes)){
   ctl<-if(is.null(recipes[[arm]]))list(start='fabin3',start_transport='auto') else list(start_transport='native')
   z<-run_fit(spec,data,sample,'PSD',route,'nlopt-slsqp',arm,recipes[[arm]],preconditioning=scaling,control=ctl)
   id<-length(rows)+1L
   rows[[id]]<-cbind(fit_id=id,task,transform='native',domain='PSD',route=route,backend='nlopt-slsqp',
     preconditioning=scaling,start_id=arm,z$record,geometry_endpoint(z$partable,d,sample,'PSD'))
   if(!is.null(z$partable))parameters[[length(parameters)+1L]]<-cbind(fit_id=id,z$partable[c('lhs','op','rhs','est')])
  }
 }
 cat(sprintf('[%d/%d] %.1fs\n',i,nrow(tasks),proc.time()[['elapsed']]-t0))
}
fits<-do.call(rbind,rows);refs<-reference_rows(fits);cmp<-compare_rows(fits,refs)
write_out(fits,'fits');write_out(do.call(rbind,parameters),'parameters');write_out(refs,'references');write_out(cmp,'comparisons')
paired<-lapply(split(cmp,interaction(cmp$design,cmp$n,cmp$rep,cmp$preconditioning,drop=TRUE)),function(z){
 ordinary<-z[z$route=='ordinary',];hit<-ordinary$comparison=='matches_sphere_reference'
 cbind(z[1,c('design','n','rep','seed','preconditioning','reference_label','chart_status')],
  reference_available=is.finite(z$best_objective[1]),fabin3_hit=any(hit & ordinary$start_id=='fabin3_auto'),
  spectral_hit=any(hit & ordinary$start_id=='spectral_positive'),portfolio_hit=any(hit),
  better_than_reference=any(ordinary$comparison=='better_than_sphere_reference'),
  any_screened=any(ordinary$screened),
  screened_face=any(ordinary$screened & ordinary$psd_position=='near_PSD_face'))
})
paired<-do.call(rbind,paired);write_out(paired,'paired')
write_out(aggregate(paired[c('reference_available','fabin3_hit','spectral_hit','portfolio_hit','better_than_reference','any_screened','screened_face')],paired['preconditioning'],sum),'summary')
write_out(aggregate(list(fits=cmp$fit_id),cmp[c('route','preconditioning','start_id','label','comparison','psd_position')],length),'labels')
ref<-magmaan_cache_ref();write_metadata(file.path(out,'metadata.csv'),values=list(tasks=nrow(tasks),fits=nrow(fits),
 seed_rule='902609281 + design_index*100000 + N*100 + rep; random start seed=draw seed+400000+j',
 selection='all arms on every draw; no reference-based fitting; previously retained fresh batch',
 reference='sphere-only; canonical + positive spectral + three random starts; both PSD scaling choices pooled',
 psd_face='heuristic 1e-7 on invariant scaled primitive eigenvalues; admissibility and accuracy remain separate',
 sphere_polish=FALSE,git_head=ref$git_head,git_dirty=ref$git_dirty),packages='magmaanlab')
cat('Wrote ',out,'\n',sep='')
