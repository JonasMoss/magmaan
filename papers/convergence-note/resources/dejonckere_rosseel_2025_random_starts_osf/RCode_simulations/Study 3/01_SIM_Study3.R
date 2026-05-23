
#############################################
##
##              STUDY 3
##
#############################################

setwd("/home/semlab/Julie/New Test Rstart/New Test Rstart")

load("DATA_MSST_SIM10.RData") # gehaald uit folder: New - Jan - 24

## -------------------
## POPULATION MODEL 
## -------------------

popmodel.msst <- '

  ## factor loadings 
   #  state
  eta1 =~ 1*y11 + la211*y21 + la311*y31
  eta2 =~ 1*y12 + la221*y22 + la321*y32
  eta3 =~ 1*y13 + la231*y23 + la331*y33
  
   # trait
  theta1 =~ 1*eta1 + ga21*eta2 + ga31*eta3
  
  ## intercepts
  y11 ~ 0*1
  y21 ~ 0*1
  y31 ~ 0*1
  
  y12 ~ 0*1
  y22 ~ 0*1
  y32 ~ 0*1
  
  y13 ~ 0*1
  y23 ~ 0*1
  y33 ~ 0*1
  
  eta1 ~ 0*1
  eta2 ~ 0*1
  eta3 ~ 0*1
  
  theta1 ~ 0*1

  # (residual) observed variances
  y11 ~~ eps11*y11
  y21 ~~ eps21*y21
  y31 ~~ eps31*y31
  
  y12 ~~ eps12*y12
  y22 ~~ eps22*y22
  y32 ~~ eps32*y32
  
  y13 ~~ eps13*y13
  y23 ~~ eps23*y23
  y33 ~~ eps33*y33
  
  # factor variances
  eta1 ~~ psi1*eta1
  eta2 ~~ psi2*eta2
  eta3 ~~ psi3*eta3
  theta1 ~~ vartheta1*theta1
  

  
  # true parameter values
  la211     := 0.5   # first nonindicator lambda state 1
  la311     := 1.3   # second nonindicator lambda state 1
  la221     := 0.5   # first nonindicator lambda state 2
  la321     := 1.3
  la231     := 0.5
  la331     := 1.3
  ga21      := 0.5   # first nonindicator lambda trait 1
  ga31      := 1.3
  psi1      := 0.6   # factor residual variances
  psi2      := 0.25
  psi3      := 0.7
  vartheta1 := 0.6   # variance trait?
  #vareta1   := 1.8   # vareta = is variance latent state !!!!
  #vareta2   := 1.8
  #vareta3   := 1.8
  
  # total variance not equal to latent state residual! But has same name atm!
  vareta1 := 1^2    * vartheta1 + psi1   # 1.2  
  vareta2 := ga21^2 * vartheta1 + psi2   # 0.4
  vareta3 := ga31^2 * vartheta1 + psi3   # 1.714
  
  vary11 := 1^2     * vareta1 + eps11
  vary21 := la211^2 * vareta1 + eps21
  vary31 := la311^2 * vareta1 + eps31
  
  vary12 := 1^2     * vareta2 + eps12
  vary22 := la221^2 * vareta2 + eps22
  vary32 := la321^2 * vareta2 + eps32
  
  vary13 := 1^2     * vareta3 + eps13
  vary23 := la231^2 * vareta3 + eps23
  vary33 := la331^2 * vareta3 + eps33
  
  # Reliability
  rely11 := 1^2     * vareta1 / vary11
  rely21 := la211^2 * vareta1 / vary21
  rely31 := la311^2 * vareta1 / vary31
  
  rely12 := 1^2     * vareta2 / vary12
  rely22 := la221^2 * vareta2 / vary22
  rely32 := la321^2 * vareta2 / vary32
  
  rely13 := 1^2     * vareta3 / vary13
  rely23 := la231^2 * vareta3 / vary23
  rely33 := la331^2 * vareta3 / vary33
  
  # Specificity
  spey11 := 1^2     * psi1 / vary11
  spey21 := la211^2 * psi1 / vary21
  spey31 := la311^2 * psi1 / vary31
  
  spey12 := 1^2     * psi2 / vary12
  spey22 := la221^2 * psi2 / vary22
  spey32 := la321^2 * psi2 / vary32
  
  spey13 := 1^2     * psi3 / vary13
  spey23 := la231^2 * psi3 / vary23
  spey33 := la331^2 * psi3 / vary33
  
  # Consistency
  cony11 := 1^2     * 1^2 * vartheta1 / vary11
  cony21 := la211^2 * 1^2 * vartheta1 / vary21
  cony31 := la311^2 * 1^2 * vartheta1 / vary31
  
  cony12 := 1^2     * ga21^2 * vartheta1 / vary12
  cony22 := la221^2 * ga21^2 * vartheta1 / vary22
  cony32 := la321^2 * ga21^2 * vartheta1 / vary32
  
  cony13 := 1^2     * ga31^2 * vartheta1 / vary13
  cony23 := la231^2 * ga31^2 * vartheta1 / vary23
  cony33 := la331^2 * ga31^2 * vartheta1 / vary33
'


## -------------------
## MODEL for analysis
## -------------------

model.msst <- '

  # factor loadings states
  eta1   =~ 1*y11  + la211*y21 + la311*y31
  eta2   =~ 1*y12  + la221*y22 + la321*y32
  eta3   =~ 1*y13  + la231*y23 + la331*y33
  theta1 =~ 1*eta1 + ga21*eta2 + ga31*eta3

  
  # intercepts
  y11 ~ 0*1
  y21 ~ 0*1
  y31 ~ 0*1
  
  y12 ~ 0*1
  y22 ~ 0*1
  y32 ~ 0*1
  
  y13 ~ 0*1
  y23 ~ 0*1
  y33 ~ 0*1
  
  eta1 ~ 0*1
  eta2 ~ 0*1
  eta3 ~ 0*1
  
  theta1 ~ 0*1

  # (residual) observed variances
  y11 ~~ eps11*y11
  y21 ~~ eps21*y21
  y31 ~~ eps31*y31
  
  y12 ~~ eps12*y12
  y22 ~~ eps22*y22
  y32 ~~ eps32*y32
  
  y13 ~~ eps13*y13
  y23 ~~ eps23*y23
  y33 ~~ eps33*y33
  
  # factor variances
  eta1   ~~ psi1*eta1
  eta2   ~~ psi2*eta2
  eta3   ~~ psi3*eta3
  theta1 ~~ vartheta1*theta1
  
  vareta1 := 1^2    * vartheta1 + psi1
  vareta2 := ga21^2 * vartheta1 + psi2
  vareta3 := ga31^2 * vartheta1 + psi3
  
  vary11 := 1^2     * vareta1 + eps11
  vary21 := la211^2 * vareta1 + eps21
  vary31 := la311^2 * vareta1 + eps31
  
  vary12 := 1^2     * vareta2 + eps12
  vary22 := la221^2 * vareta2 + eps22
  vary32 := la321^2 * vareta2 + eps32
  
  vary13 := 1^2     * vareta3 + eps13
  vary23 := la231^2 * vareta3 + eps23
  vary33 := la331^2 * vareta3 + eps33
  
  # Reliability
  rely11 := 1^2     * vareta1 / vary11
  rely21 := la211^2 * vareta1 / vary21
  rely31 := la311^2 * vareta1 / vary31
  
  rely12 := 1^2     * vareta2 / vary12
  rely22 := la221^2 * vareta2 / vary22
  rely32 := la321^2 * vareta2 / vary32
  
  rely13 := 1^2     * vareta3 / vary13
  rely23 := la231^2 * vareta3 / vary23
  rely33 := la331^2 * vareta3 / vary33
  
  # Specificity
  spey11 := 1^2     * psi1 / vary11
  spey21 := la211^2 * psi1 / vary21
  spey31 := la311^2 * psi1 / vary31
  
  spey12 := 1^2     * psi2 / vary12
  spey22 := la221^2 * psi2 / vary22
  spey32 := la321^2 * psi2 / vary32
  
  spey13 := 1^2     * psi3 / vary13
  spey23 := la231^2 * psi3 / vary23
  spey33 := la331^2 * psi3 / vary33
  
  # Consistency
  cony11 := 1^2     * 1^2 * vartheta1 / vary11
  cony21 := la211^2 * 1^2 * vartheta1 / vary21
  cony31 := la311^2 * 1^2 * vartheta1 / vary31
  
  cony12 := 1^2     * ga21^2 * vartheta1 / vary12
  cony22 := la221^2 * ga21^2 * vartheta1 / vary22
  cony32 := la321^2 * ga21^2 * vartheta1 / vary32
  
  cony13 := 1^2     * ga31^2 * vartheta1 / vary13
  cony23 := la231^2 * ga31^2 * vartheta1 / vary23
  cony33 := la331^2 * ga31^2 * vartheta1 / vary33
'

## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

FIT_MSST      <- lavaan::sem(model.msst)               
OV.NAMES_MSST <- lavNames(FIT_MSST, "ov") # observed variables
LV.NAMES_MSST <- lavNames(FIT_MSST, "lv") # latent variables
COEF.NAMES_MSST    <- c(names(coef(FIT_MSST)), "nr_sets_needed_conv")  # (free) parameter names

col.names_msst     <- c(COEF.NAMES_MSST, 
                        "optim", 
                        "se", 
                        "ov.var.pos", 
                        "lv.var.pos")

NPAR_MSST <- length(COEF.NAMES_MSST)      
NVAR_MSST <- length(OV.NAMES_MSST) 

n_cov_elements_msst <- NVAR_MSST * (NPAR_MSST + 1) / 2  
n_eig_values_mssst <- NVAR_MSST                     
NDATA_MSST <- n_cov_elements_msst + n_eig_values_mssst

PTE_MSST <- parTable(FIT_MSST)
ov.var.idx_msst <- which(PTE_MSST$op == "~~" & PTE_MSST$lhs %in% OV.NAMES_MSST & PTE_MSST$lhs == PTE_MSST$rhs) 
lv.var.idx_msst <- which(PTE_MSST$op == "~~" & PTE_MSST$lhs %in% LV.NAMES_MSST & PTE_MSST$lhs == PTE_MSST$rhs) 
beta.idx_msst <- which(PTE_MSST$op == "~")

NRESULTS_MSST <- length(col.names_msst)
REP <- 1000
N   <- 10


# --------------------------

# select converged cases
idx.tmp10 <- which(DATA_MSST_SIM10[,"optim"] == 1| DATA_MSST_SIM10[,"optim"] == 1L & DATA_MSST_SIM10[,"se"] == 0L)
tmp10 <- 1:1000

nc.idx.msst10 <- setdiff(tmp10, idx.tmp10)
nc.L.msst10  <- length(nc.idx.msst10)


## -------------------
## DATA CONTAINERS   
## -------------------

RESULTS_MSST_RSTART_NEW           <- matrix(as.numeric(NA), 
                                         nrow = nc.L.msst10, 
                                         ncol = length(COEF.NAMES_MSST))
colnames(RESULTS_MSST_RSTART_NEW) <- COEF.NAMES_MSST




## -------------------
## SIMULATION START
## -------------------
start_time <- Sys.time()
for(i in 1:length(nc.idx.msst10)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  nc.idx.msst10[i]
  set.seed(this.seed)
  
  
  Data       <- simulateData(model = popmodel.msst, 
                             sample.nobs = N, 
                             empirical = FALSE)
  
  
    fit.start3 <- try(lavaan::sem(model.msst, 
                                  data = Data,
                                  rstarts = 150), 
                      silent = TRUE)
    
    first.conv3 <- which(sapply(fit.start3@optim$x.rstarts, attr, "converged"))[1]
      
    RESULTS_MSST_RSTART_NEW[i, ]  <- c(fit.start3@optim$x.rstarts[[first.conv3]][1:(length(COEF.NAMES_MSST)-1)], first.conv3)
}
end_time <- Sys.time()
end_time-start_time
save(RESULTS_MSST_RSTART_NEW, file = "RESULTS_MSST_RSTART_NEW.RData")


