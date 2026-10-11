# TASK-157 bounded before/after measurements; raw RSS samples remain ignored.
# R_LIBS must point to the chosen read-only library. Runs six serial probes.
import subprocess,time,os,csv,pathlib,argparse
root = pathlib.Path(__file__).resolve().parents[4]
os.chdir(root)
parser = argparse.ArgumentParser(description='Six bounded worker probes; requires R_LIBS, Unix and /proc.')
parser.add_argument('--output', default='experiments/decisions/09-fit-indices/results/task157-check')
base = parser.parse_args().output
if not os.environ.get('R_LIBS'):
    raise SystemExit('Set R_LIBS to the chosen read-only library')
if os.path.exists(base):
    raise SystemExit('Use a fresh --output directory; existing evidence is preserved')
os.makedirs(base)
env=dict(os.environ,TASK157_OUTPUT=base,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
for count,est in [(50,'ML'),(50,'FIML'),(400,'FIML')]:
 for mode in ['before','after']:
  name=f'{mode}-{est}-{count}'
  with open(f'{base}/{name}.log','w') as log,open(f'{base}/{name}-rss.csv','w') as mem:
   p=subprocess.Popen(['nice','-n','10','Rscript','experiments/decisions/09-fit-indices/scripts/check_batches.R',mode,est,str(count),f'{base}/{name}'],stdout=log,stderr=log,env=env)
   w=csv.writer(mem);w.writerow(['seconds','pid','role','rss_mib','completed']);start=time.monotonic()
   while p.poll() is None:
    procs={}
    for item in os.listdir('/proc'):
     if item.isdigit():
      try:
       s=open(f'/proc/{item}/stat').read().split();procs[int(item)]=int(s[3])
      except (OSError,ValueError):pass
    ids={p.pid}
    while True:
     more={i for i,parent in procs.items() if parent in ids}
     if more <= ids:break
     ids |= more
    completed=len(__import__('glob').glob(f'{base}/{name}/raw/*/rep_*.rds'))
    for pid in ids:
     try:
      rss=int(open(f'/proc/{pid}/statm').read().split()[1])*os.sysconf('SC_PAGE_SIZE')/1048576
      w.writerow([round(time.monotonic()-start,3),pid,'parent' if pid==p.pid else 'child',rss,completed])
     except OSError:pass
    mem.flush();time.sleep(1)
   if p.returncode:raise SystemExit(f'{name} failed {p.returncode}')
  print(name,'complete',flush=True)

# Every field except timing must match, including unavailable rows and reasons.
subprocess.run(['Rscript', '-e', """
base <- Sys.getenv('TASK157_OUTPUT')
cases <- data.frame(estimator=c('ML','FIML','FIML'),reps=c(50L,50L,400L))
rows <- lapply(seq_len(nrow(cases)),function(j) {
  e <- cases$estimator[j]; n <- cases$reps[j]
  for (i in seq_len(n)) {
    read <- function(mode) readRDS(file.path(base,paste0(mode,'-',e,'-',n),
      'raw',paste0('batch_check_',e),sprintf('rep_%05d.rds',i)))
    x <- read('before'); y <- read('after'); x$seconds <- y$seconds <- NULL
    stopifnot(identical(x,y),nrow(x)==24L)
  }
  data.frame(model=if (n==400L) 'cfa_long18' else 'cfa_ptsd8',
    estimator=e,replicates=n,rows=24L*n,identical_except_seconds=TRUE)
})
write.csv(do.call(rbind,rows),file.path(base,'parity.csv'),row.names=FALSE)
library_files <- list.files(find.package('magmaanlab'),recursive=TRUE,
  full.names=TRUE,pattern='\\\\.(so|rdb)$')
metadata <- data.frame(key=c('before_ref','after_source_md5','library_build',
  'library_version','library_path','seed_base','batch_size','probe','parity',
  'rss_sampling','slope'),value=c('2e02a4ca46fb379200c967925b8a001786a71712',
  unname(tools::md5sum('experiments/decisions/09-fit-indices/run_experiment.R')),
  packageDescription('magmaanlab')$Built,as.character(packageVersion('magmaanlab')),
  find.package('magmaanlab'),2026101101,25,
  'cfa_long18; normal; delta=-.1; N=2000; FIML MAR=.30; 400 draws; one worker',
  'cfa_ptsd8; normal; delta=-.1; N=200; 50 draws each ML/FIML',
  'one-second /proc RSS samples per parent/descendant',
  'OLS RSS on completed draws 25..375; max RSS per completed draw per role'))
metadata <- rbind(metadata,data.frame(key=paste0('library_md5:',library_files),
  value=unname(tools::md5sum(library_files))))
write.csv(metadata,file.path(base,'metadata.csv'),row.names=FALSE)
"""], check=True, env=env)
# Slope against completed draws within each process; startup/exit samples excluded.
subprocess.run(['Rscript', '-e', """
base <- Sys.getenv('TASK157_OUTPUT')
rows <- lapply(c('before','after'),function(mode) {
  x <- read.csv(file.path(base,paste0(mode,'-FIML-400-rss.csv')))
  do.call(rbind,lapply(split(x,x$role),function(z) {
    z <- z[z$completed >= 25 & z$completed <= 375,]
    # One observation per completed draw avoids weighting slow draws more heavily.
    by_draw <- aggregate(rss_mib~completed,z,max)
    batch_peaks <- if (z$role[1]=='child') do.call(rbind,
      lapply(split(x[x$role=='child',],x$pid[x$role=='child']),function(b)
        data.frame(batch_end=25*ceiling((min(b$completed)+1)/25),peak=max(b$rss_mib)))) else NULL
    data.frame(mode=mode,role=z$role[1],peak_mib=max(x$rss_mib[x$role==z$role[1]]),
      slope_mib_per_draw=coef(lm(rss_mib~completed,by_draw))[2],
      batch_peak_slope_mib_per_draw=if(is.null(batch_peaks)) NA_real_ else
        coef(lm(peak~batch_end,batch_peaks))[2],
      first_mib=by_draw$rss_mib[1],last_mib=tail(by_draw$rss_mib,1),
      samples=nrow(z))
  }))
})
write.csv(do.call(rbind,rows),file.path(base,'memory.csv'),row.names=FALSE)
"""], check=True, env=env)
