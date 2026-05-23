
#############################################
##
##              STUDY 2
##
#############################################

setwd("/home/semlab/Julie/New Test Rstart/New Test Rstart")

load("DATA_C.Rdata")
load("ML_COMPLEX.RData")


## -------------------
## POPULATION MODEL 
## -------------------

pop.model.3LV <- '

  # factor loadings
    eta1 =~ 1*y1 + 1*y2 + 1*y3 + 0.3*y4
    eta2 =~ 1*y4 + 1*y5 + 1*y6 + 0.3*y7
    eta3 =~ 1*y7 + 1*y8 + 1*y9 + 0.3*y6

  # latent (residual) variances
    eta1 ~~ 0.49*eta1
    eta2 ~~ 0.3136*eta2
    eta3 ~~ 0.3136*eta3

  # regression part
    eta2 ~ 0.1*eta1
    eta3 ~ 0.2*eta2

  # residual variances
    y1 ~~ 0.51*y1 
    y2 ~~ 0.51*y2 
    y3 ~~ 0.51*y3 
    y4 ~~ 0.2895*y4 
    y5 ~~ 0.51*y5 
    y6 ~~ 0.2895*y6
    y7 ~~ 0.2895*y7
    y8 ~~ 0.51*y8
    y9 ~~ 0.51*y9
'

## -------------------
## MODEL for analysis
## -------------------

model.3LV <- '

  # factor loadings
    eta1 =~ y1 + y2 + y3 + y4
    eta2 =~ y4 + y5 + y6 + y7
    eta3 =~ y7 + y8 + y9 + y6

  # latent (residual) variances
    eta1 ~~ eta1
    eta2 ~~ eta2
    eta3 ~~ eta3

  # regression part
    eta2 ~ eta1
    eta3 ~ eta2

  # residual variances
    y1 ~~ y1 
    y2 ~~ y2 
    y3 ~~ y3 
    y4 ~~ y4 
    y5 ~~ y5 
    y6 ~~ y6
    y7 ~~ y7
    y8 ~~ y8
    y9 ~~ y9
'

## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

# parameter values
FIT_C      <- lavaan::sem(model.3LV)               
OV.NAMES_C <- lavNames(FIT_C, "ov") # observed variables
LV.NAMES_C <- lavNames(FIT_C, "lv") # latent variables
COEF.NAMES_C    <- c(names(coef(FIT_C)), "nr_sets_needed_conv")   # (free) parameter names

col.names_c     <- c(COEF.NAMES_C, 
                     "beta.se1", 
                     "beta.se2",
                     "optim", 
                     "se", 
                     "ov.var.pos", 
                     "lv.var.pos")

NPAR_C <- length(COEF.NAMES_C)      
NVAR_C <- length(OV.NAMES_C) 

n_cov_elements_c <- NVAR_C * (NVAR_C + 1) / 2  
n_eig_values_c <- NVAR_C                     
NDATA_C <- n_cov_elements_c + n_eig_values_c

PTE_C <- parTable(FIT_C)
ov.var.idx_c <- which(PTE_C$op == "~~" & PTE_C$lhs %in% OV.NAMES_C 
                      & PTE_C$lhs == PTE_C$rhs) 
lv.var.idx_c <- which(PTE_C$op == "~~" & PTE_C$lhs %in% LV.NAMES_C 
                      & PTE_C$lhs == PTE_C$rhs) 
beta.idx_c <- which(PTE_C$op == "~")

NRESULTS_C <- length(col.names_c)
REP <- 1000
N   <- ss <- 20




# select converged cases
idx.tmp <- which(ML_COMPLEX[,"optim"] == 1)
tmp <- 1:1000

nc.idx.complex <- setdiff(tmp, idx.tmp)
nc.L.complex  <- length(nc.idx.complex)


## -------------------
## DATA CONTAINERS   
## -------------------

RESULTS_STUDY2_RSTART_NEW  <- matrix(as.numeric(NA), 
                                     nrow = nc.L.complex, 
                                     ncol = length(COEF.NAMES_C))
colnames(RESULTS_STUDY2_RSTART_NEW) <- COEF.NAMES_C

DATA_C_NC  <- matrix(as.numeric(NA), nrow = nc.L.complex, ncol = NDATA_C) 


## -------------------
## SIMULATION START
## -------------------

start_time <- Sys.time()
for(i in 1:length(nc.idx.complex)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  nc.idx.complex[i]
  set.seed(this.seed)
  
  Data      <- simulateData(pop.model.3LV, 
                            sample.nobs = N, 
                            empirical = FALSE)
  
  #COV         <- cov(Data)
  #EV          <- eigen(COV)$values
  #DATA_C_NC[i, ] <- c(lav_matrix_vech(COV), EV)
  
  
  
    fit.start2 <- try(lavaan::sem(model.3LV, 
                                 data   = Data, 
                                 rstarts = 150), 
                     silent = TRUE)
    
    if(!inherits(fit.start2, "try-error")){
    first.conv2 <- which(sapply(fit.start2@optim$x.rstarts, attr, "converged"))[1]
    
    RESULTS_STUDY2_RSTART_NEW[i,] <- c(fit.start2@optim$x.rstarts[[first.conv2]][1:(length(COEF.NAMES_C)-1)], first.conv2)
	}
  
}

end_time <- Sys.time()
end_time - start_time 
save(RESULTS_STUDY2_RSTART_NEW, file = "RESULTS_STUDY2_RSTART_NEW.RData")
#save(DATA_C_NC, file = "DATA_C_NC.RData")




