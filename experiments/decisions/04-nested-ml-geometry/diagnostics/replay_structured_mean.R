# Run from the repository root; fixed development replays, never production.
source('experiments/decisions/04-nested-ml-geometry/R/structured_mean.R')
out <- 'experiments/decisions/04-nested-ml-geometry/results/structured-mean/replay-2026-10-05'
dir.create(file.path(out,'raw'),recursive=TRUE,showWarnings=FALSE)
cell <- structured_cells()[2,]
# Parent-side inference preparation before forking exercises PID cache rebuilding.
parent <- structured_draw(cell,16,1526120047L)
all <- list(parent)
for(batch in 1:10) {
 x <- parallel::mclapply(1:20,function(r) structured_draw(cell,r,1526120031L+r),mc.cores=2,mc.preschedule=TRUE)
 all <- c(all,x)
 cat('batch',batch,'failed arms',sum(vapply(x,function(z) sum(nzchar(z$error)),integer(1))),'\n')
}
raw <- do.call(rbind,all)
write.csv(raw,file.path(out,'raw','draws.csv'),row.names=FALSE)
print(raw[nzchar(raw$error),c('rep','seed','arm','error','error_call','error_stage')])
# Verify that an arm error is captured without suppressing the other arms.
original <- structured_lr_reference
structured_lr_reference <- function(...) stop('diagnostic injected reference failure')
z <- structured_draw(cell,16,1526120047L)
stopifnot(all(nzchar(z$error[c(1,3)])),!nzchar(z$error[2]),all(nzchar(z$error_call[c(1,3)])))
cat('condition capture and arm isolation passed; total draws',length(all),'\n')
write.csv(data.frame(draws=length(all),workers=2,batches=10,failed_arms=sum(nzchar(raw$error)),capture_isolation_passed=TRUE),file.path(out,'checks.csv'),row.names=FALSE)
