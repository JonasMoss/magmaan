#!/usr/bin/env Rscript
# Curate existing local evidence; no fitting, simulation or policy selection.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1L])
root <- dirname(dirname(normalizePath(script)))
if (identical(commandArgs(TRUE), "--help")) {
  cat("Usage: Rscript scripts/summarize.R\nReads the retained original runs and writes compact overview CSVs with source hashes.\n")
  quit(status=0L)
}
out <- file.path(root,"results","overview")
sources <- character()
read_result <- function(name) {
  p <- file.path(root,"results",name)
  if (!file.exists(p)) stop("Missing historical result: ", p,
    "; recover that run before rebuilding the retained overview.")
  sources <<- c(sources,p)
  read.csv(p,stringsAsFactors=FALSE)
}
r <- read_result("global/sem-paired-score-fmg-n200-500/replications.csv")
meta <- read_result("global/sem-paired-score-fmg-n200-500/metadata.csv")
# Only generating-model conditions with a justified estimator-level null.
r <- r[r$missingness != "mar_30" | r$distribution == "normal", ]
labels <- c(p_lrt_mlr="Scalar robust",p_lrt_peba4="LR/D pEBA4",
            p_score_sb="Score SB",p_score_peba4="Score pEBA4")
cells <- do.call(rbind,lapply(split(r,interaction(r$cell_id,r$estimator)),function(z)
  do.call(rbind,lapply(names(labels),function(method) {
    good <- z$fit_ok & is.finite(z[[method]])
    if (startsWith(method,"p_score")) good <- good & z$flip_ok & z$flip_nominal_geometry
    good[is.na(good)] <- FALSE
    data.frame(cell_id=z$cell_id[1],estimator=z$estimator[1],method=unname(labels[method]),
      distribution=z$distribution[1],missingness=z$missingness[1],attempts=nrow(z),
      usable=sum(good),rejected=sum(z[[method]][good] <= .05),
      rate=if(sum(good)) mean(z[[method]][good] <= .05) else NA_real_)
  }))))
overview <- do.call(rbind,lapply(split(cells,interaction(cells$estimator,cells$method)),function(z) {
  rates <- z$rate[is.finite(z$rate)]
  data.frame(estimator=z$estimator[1],method=z$method[1],cells=nrow(z),
    min_rate=min(rates),max_rate=max(rates),mean_cell_error=mean(abs(rates-.05)),
    min_usable=min(z$usable),attempts_per_cell=max(z$attempts),
    unavailable=sum(z$attempts-z$usable))
}))
pseudo <- read_result("global/robust-score-pseudonull-r1000/summary.csv")
pseudo <- pseudo[pseudo$beta==5 & pseudo$n==2000,
  c("n","beta","geometry","attempted","usable","reject_peba4")]
# Expected versus observed H0 sensitivity on the same FIML fits (expected metric).
panel <- read_result("global/robust-score-sem-r1000/null_summary.csv")
panel_meta <- read_result("global/robust-score-sem-r1000/metadata.csv")
panel_labels <- c("Legacy-score FMG SB"="expected_sb",
  "Legacy-score FMG pEBA(4)"="expected_peba4",
  "Observed-score FMG SB"="observed_sb",
  "Observed-score FMG pEBA(4)"="observed_peba4",
  "LR/D FMG SB"="lr_sb", "LR/D FMG pEBA(4)"="lr_peba4")
panel <- panel[panel$method %in% names(panel_labels),
  c("model_id","expected_df","distribution","missingness","n","method",
    "attempted","finite","rejection_le_05")]
panel$method <- unname(panel_labels[panel$method])
pseudo_grid <- do.call(rbind,lapply(c("pseudonull-information","pseudonull-information-n10000"),
  function(run) {
    z <- read_result(paste0("global/",run,"/summary.csv"))
    z[z$geometry %in% c("expected-H0","observed-H0/expected-metric"),
      c("n","beta","geometry","attempted","usable","reject_sb","reject_peba4")]
  }))
bases <- read_result("global/ml2s-rls-smoke/summary.csv")
bases <- bases[bases$null_contract=="pseudo-null",]
bases <- do.call(rbind,lapply(split(bases,bases$n),function(z)
  data.frame(n=z$n[1],cells=nrow(z),min_usable=min(z$usable),
    max_sb_size_difference=max(abs(z$rejection_sb_ml-z$rejection_sb_rls)),
    max_peba4_size_difference=max(abs(z$rejection_peba4_ml-z$rejection_peba4_rls)),
    max_all_size_difference=max(abs(z$rejection_all_ml-z$rejection_all_rls)))))
legacy <- do.call(rbind,lapply(c("legacy-mlr","two-stage"),function(lane) {
  m <- read_result(paste0(lane,"/metadata.csv"));v <- setNames(m$value,m$key)
  data.frame(lane=lane,reps=v[["reps"]],cell_filter=v[["cells_filter"]],
    scope="single smoke cell; nonnormal MAR stress, not calibration evidence")
}))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
artifacts <- list(sem_cells=cells,sem_overview=overview,pseudo_null=pseudo,
                  ml2s_bases=bases,legacy_scope=legacy,sem_metadata=meta,
                  sensitivity_panel=panel,sensitivity_panel_metadata=panel_meta,
                  pseudo_null_grid=pseudo_grid)
for (name in names(artifacts)) write.csv(artifacts[[name]],file.path(out,paste0(name,".csv")),row.names=FALSE)
write.csv(data.frame(source=sub(paste0(root,"/"),"",sources,fixed=TRUE),
  md5=unname(tools::md5sum(sources))),file.path(out,"provenance.csv"),row.names=FALSE)
cat("Wrote retained overview to",out,"\n")
