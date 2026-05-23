## ---------------------------
##
## Script name:          00_Pre_Simulation_Study1.R
##
## Purpose of script:    Script to define population and analysis model, 
##                       parameter values from output, extract function for data        
##
## ---------------------------
## Notes:
##      1. Model specification
##      2. Utilities
##      3. Data extract function
##      4. First sim run OpenMx and lavaan
##         a. population model 1
##         b. population model 2
## ---------------------------

library("lavaan")

## -------------------------------------------------------------------------- ##
##                          1. Model Specification                            ##      
## -------------------------------------------------------------------------- ##

## -------------------
## POPULATION MODEL 1
## -------------------

pop.model <- '
   
   FX =~ 1*y1 + 0.8*y2 + 0.6*y3
   FY =~ 1*y4 + 0.7*y5 + 0.9*y6

   # regression
   FY ~ 0.1*FX        
   
   # factor (residual) variances
   FX ~~ 1*FX
   FY ~~ 1*FY

   # exo
   y1 ~~ ((1 - 0.1)*1*1.0^2/0.1)*y1       #rel 0.1
   y2 ~~ ((1 - 0.1)*1*0.8^2/0.1)*y2
   y3 ~~ ((1 - 0.1)*1*0.6^2/0.1)*y3

   # endo
   y4 ~~ ((1 - 0.1)*1.01*1.0^2/0.1)*y4
   y5 ~~ ((1 - 0.1)*1.01*0.7^2/0.1)*y5
   y6 ~~ ((1 - 0.1)*1.01*0.9^2/0.1)*y6
'



## -------------------
## POPULATION MODEL 2
## -------------------

pop.model2 <- '
   
   FX =~ 1*y1 + 0.8*y2 + 0.6*y3
   FY =~ 1*y4 + 0.7*y5 + 0.9*y6

   # regression
   FY ~ 0.7*FX # total variance = 0.7^2 + 1 = 1.49
   
   
   # factor (residual) variances
   FX ~~ 1*FX
   FY ~~ 1*FY

   # exo
   y1 ~~ ((1 - 0.1)*1*1.0^2/0.1)*y1       #rel 0.1
   y2 ~~ ((1 - 0.1)*1*0.8^2/0.1)*y2
   y3 ~~ ((1 - 0.1)*1*0.6^2/0.1)*y3

   # endo
   y4 ~~ ((1 - 0.1)*1.49*1.0^2/0.1)*y4
   y5 ~~ ((1 - 0.1)*1.49*0.7^2/0.1)*y5
   y6 ~~ ((1 - 0.1)*1.49*0.9^2/0.1)*y6
'


## -------------------
## MODEL for analysis
## -------------------

model <- '
    
    FX =~ y1 + y2 + y3
    FY =~ y4 + y5 + y6

    # regression part 
    FY ~ FX
    
    # factor variances
    FX ~~ FX
    FY ~~ FY
    
    # observed residual variances
    y1 ~~ y1
    y2 ~~ y2
    y3 ~~ y3
    y4 ~~ y4
    y5 ~~ y5
    y6 ~~ y6
'


## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

# parameter values
FIT1       <- lavaan::sem(model)               
OV.NAMES   <- lavNames(FIT1, "ov") # observed variables
LV.NAMES   <- lavNames(FIT1, "lv") # latent variables
COEF.NAMES <- names(coef(FIT1))  # (free) parameter names

col.names  <- c(COEF.NAMES, 
                "beta.se", 
                "optim", 
                "se", 
                "ov.var.pos", 
                "lv.var.pos")

NPAR <- length(COEF.NAMES)      
NVAR <- length(OV.NAMES) 

n_cov_elements <- NVAR * (NVAR + 1) / 2  
n_eig_values   <- NVAR                     
NDATA          <- n_cov_elements + n_eig_values

PTE1       <- parTable(FIT1)
ov.var.idx <- which(PTE1$op == "~~" & PTE1$lhs %in% OV.NAMES & PTE1$lhs == PTE1$rhs) 
lv.var.idx <- which(PTE1$op == "~~" & PTE1$lhs %in% LV.NAMES & PTE1$lhs == PTE1$rhs) 
beta.idx   <- which(PTE1$op == "~")

OV <- c("y1","y2","y3","y4","y5","y6")

NRESULTS <- length(col.names)
REP <- 1000
N <- 20


## -------------------------------------------------------------------------- ##
##                        3. Data extract function                            ##      
## -------------------------------------------------------------------------- ##

extract_results_from_fit <- function(fit) {
  
  OPTIM <- SE <- OV.VAR.POS <- LV.VAR.POS <- 0L; BETA.SE <- NA
  
  PT <- parTable(fit)
  THETA <- PT$est[   PT$free > 0L ]
  
  
  if(lavInspect(fit, "converged")) {
    OPTIM <- 1L
  } 
  if(!all(is.na(PT$se[PT$free > 0]))) {
    SE <- 1L
    BETA.SE <- PT$se[ beta.idx ]
  }
  if(all(PT$est[ov.var.idx] >= 0)) {  
    OV.VAR.POS <- 1
  }
  if(all(PT$est[lv.var.idx] >= 0)) { 
    LV.VAR.POS <- 1
  }
  
  c(THETA, BETA.SE, OPTIM, SE, OV.VAR.POS, LV.VAR.POS)
}



## -------------------------------------------------------------------------- ##
##                   4. Simulation OpenMx vs lavaan                           ##      
## -------------------------------------------------------------------------- ##

## ------------------------  POPULATION MODEL 1 ----------------------------- ##

## -------------------
## DATA CONTAINERS   
## -------------------

DATA                  <- matrix(as.numeric(NA), nrow = REP, ncol = NDATA) 

DATA_LAVAAN           <- matrix(as.numeric(NA), nrow = REP, ncol = NRESULTS)
colnames(DATA_LAVAAN) <- col.names

STATUS_OPENMX           <- matrix(as.numeric(NA), nrow  = REP, ncol = 1)
colnames(STATUS_OPENMX) <- c("converged")


## -------------------
## SIMULATION   
## -------------------

for(i in seq_len(REP)){
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  i
  set.seed(this.seed)
  
  
  # generate data
  Data      <- simulateData(pop.model, sample.nobs = N, empirical = FALSE)
  dataRaw   <- mxData(observed=Data, type="raw" )
  
  COV       <- cov(Data)
  EV        <- eigen(COV)$values
  DATA[i, ] <- c(lav_matrix_vech(COV), EV)
  
  #------------------------------------------------------
  # Package 1: lavaan 
  #------------------------------------------------------
  fit.Lavaan <- try(lavaan::sem(model, 
                                data = Data,
                                baseline = FALSE), 
                    silent = TRUE)
  if(!inherits(fit.Lavaan, "try-error")){
    DATA_LAVAAN[i, ]  <- extract_results_from_fit(fit.Lavaan)
  }
  

  #------------------------------------------------------
  # Package 3: OpenMx
  #------------------------------------------------------
  # residual variances
  resVars      <- mxPath( from = OV, arrows = 2,
                          free = TRUE,
                          values = diag(lavInspect(fit.Lavaan, "start")$theta),
                          labels=c("e1","e2","e3","e4","e5","e6"))
  # latent variances
  latVars      <- mxPath( from=c("FX","FY"), arrows = 2, 
                          connect = "unique.pairs",
                          free = TRUE, 
                          values = diag(lavInspect(fit.Lavaan, "start")$psi),
                          labels=c("varFX", "varFY") )
  #factor loadings for X variables   
  facLoadsX    <- mxPath( from = "FX", to = c("y1","y2","y3"), arrows = 1,
                          free = c(F,T,T), 
                          values = lavInspect(fit.Lavaan, "start")$lambda[1:3,1],
                          labels = c("l1","l2","l3") )
  # factor loadings for Y variables
  facLoadsY    <- mxPath( from = "FY", to = c("y4","y5","y6"), arrows = 1,
                          free = c(F,T,T), 
                          values = lavInspect(fit.Lavaan, "start")$lambda[4:6,2],
                          labels = c("l4","l5","l6") )
  # regression
  Beta         <- mxPath( from = "FX", to = "FY", arrows = 1,
                          free = TRUE,
                          values = 0,
                          labels = c("betayx") )
  # means/intercepts
  means        <- mxPath( from = "one",
                          to = c("y1","y2","y3","y4","y5","y6", "FX","FY"),
                          arrows = 1,
                          free = c(T,T,T,T,T,T,F,F), 
                          values = c(colMeans(Data), 0, 0),
                          labels=c("my1","my2","my3", "my4","my5","my6",
                                   "mfx","mfy") )
  
  Model <- mxModel("Simple Model", type="RAM",
                   manifestVars = OV,
                   latentVars = c("FX","FY"),
                   dataRaw, resVars, latVars, 
                   facLoadsX, facLoadsY, Beta, means)
  
  fit.OpenMx <- try(mxRun(Model), silent = TRUE)
  
  # https://openmx.ssri.psu.edu/thread/781
  if(!inherits(fit.OpenMx, "try-error") && 
     all(fit.OpenMx$output[["gradient"]]<0.001)  && # lavInspect(fit.Lavaan, "options")$optim.dx.tol
     is.positive.definite(fit.OpenMx$output[["hessian"]]) &&  
     !all(is.na(fit.OpenMx$output$standardErrors))){
    
    STATUS_OPENMX[i, 1] <- 1
    
  } else {
    STATUS_OPENMX[i, 1] <- 0
  }
}

# save data
save(DATA, file = "DATA.RData")
save(STATUS_OPENMX, file = "STATUS_OPENMX.RData")
save(DATA_LAVAAN, file = "DATA_LAVAAN.RData")


# extract cases that did not converge for both Lavaan and OpenMX
idem_Lavaan_OpenMx <- which(DATA_LAVAAN[,"optim"]==0 & STATUS_OPENMX[,1] ==0 | 
                              DATA_LAVAAN[,"optim"]==1 & STATUS_OPENMX[,1] ==1)
OPTIM.NEW <- DATA_LAVAAN[idem_Lavaan_OpenMx,]
DATA_PART2 <- DATA[idem_Lavaan_OpenMx,]

save(OPTIM.NEW, file = "OPTIM.NEW.RData")
save(DATA_PART2, file = "DATA_PART2.RData")


## ------------------------  POPULATION MODEL 2 ----------------------------- ##

## -------------------
## DATA CONTAINERS   
## -------------------

DATA_CONV_EST07   <- matrix(as.numeric(NA), nrow = REP, 
                            ncol = NRESULTS)
colnames(DATA_CONV_EST07) <- col.names
DATA_EST07        <- matrix(as.numeric(NA), nrow = REP, ncol = NDATA) 
COV_EST07         <- vector("list", length = REP)

STATUS_OpenMx07           <- matrix(as.numeric(NA), nrow  = REP, ncol = 1)
colnames(STATUS_OpenMx07) <- c("converged")

## -------------------
## SIMULATION   
## -------------------

for(i in seq_len(REP)) {
  
  cat("sample size = ", sprintf("%4d", N), " rep = ", 
      sprintf("%4d", i), "\n")
  
  this.seed <-  i
  set.seed(this.seed)
  
  Data      <- simulateData(pop.model2, sample.nobs = N, empirical = FALSE)
  dataRaw   <- mxData(observed=Data, type="raw" )
  
  COV       <- cov(Data)
  EV        <- eigen(COV)$values
  COV_EST07[[i]] <- COV
  DATA_EST07[i, ] <- c(lav_matrix_vech(COV), EV)
  
  # first fit with lavaan
  fit07 <- try(lavaan::sem(model, data = Data), silent = TRUE)
  
  if(!inherits(fit07, "try-error")){
    DATA_CONV_EST07[i,] <- extract_results_from_fit(fit07) 
  }
  
  # ---------------------------------------------------------------------- #
  # fit with OpenMx
  resVars      <- mxPath( from = OV, arrows = 2,
                          free = TRUE,
                          values = diag(lavInspect(fit07, "start")$theta),
                          labels=c("e1","e2","e3","e4","e5","e6"))
  
  # latent variances
  latVars      <- mxPath( from=c("FX","FY"), arrows = 2, 
                          connect = "unique.pairs",
                          free = TRUE, 
                          values = diag(lavInspect(fit07, "start")$psi),
                          labels=c("varFX", "varFY") )
  #factor loadings for X variables   
  facLoadsX    <- mxPath( from = "FX", to = c("y1","y2","y3"), arrows = 1,
                          free = c(F,T,T), 
                          values = lavInspect(fit07, "start")$lambda[1:3,1],
                          labels = c("l1","l2","l3") )
  # factor loadings for Y variables
  facLoadsY    <- mxPath( from = "FY", to = c("y4","y5","y6"), arrows = 1,
                          free = c(F,T,T), 
                          values = lavInspect(fit07, "start")$lambda[4:6,2],
                          labels = c("l4","l5","l6") )
  # regression
  Beta         <- mxPath( from = "FX", to = "FY", arrows = 1,
                          free = TRUE,
                          values = 0,
                          labels = c("betayx") )
  # means/intercepts
  means        <- mxPath( from = "one",
                          to = c("y1","y2","y3","y4","y5","y6", "FX","FY"),
                          arrows = 1,
                          free = c(T,T,T,T,T,T,F,F), 
                          values = c(colMeans(Data), 0, 0),
                          labels=c("my1","my2","my3", "my4","my5","my6",
                                   "mfx","mfy") )
  
  Model <- mxModel("Simple Model", type="RAM",
                   manifestVars = OV,
                   latentVars = c("FX","FY"),
                   dataRaw, resVars, latVars, facLoadsX, facLoadsY, Beta, means)
  
  fit.OpenMx <- try(mxRun(Model), silent = TRUE)
  
  # check whether converged to lavaan standards or not
  # https://openmx.ssri.psu.edu/thread/781
  if(!inherits(fit.OpenMx, "try-error") && 
     all(fit.OpenMx$output[["gradient"]]<0.001)  && # lavInspect(fit.Lavaan, "options")$optim.dx.tol
     is.positive.definite(fit.OpenMx$output[["hessian"]]) &&  
     !all(is.na(fit.OpenMx$output$standardErrors))){
    
    STATUS_OpenMx07[i, 1] <- 1
    
  } else {
    STATUS_OpenMx07[i, 1] <- 0
  }
}

save(DATA_CONV_EST07, file = "DATA_CONV_EST07.RData")
save(DATA_EST07, file = "DATA_EST07.RData")
save(COV_EST07, file = "COV_EST07.RData")
save(STATUS_OpenMx07, file = "STATUS_OpenMx07.RData")

idem_lavaan_OpenMx07 <- which(DATA_CONV_EST07[,"optim"]==0 & 
                                STATUS_OpenMx07[,1] ==0 | 
                                DATA_CONV_EST07[,"optim"]==1 & 
                                STATUS_OpenMx07[,1] ==1)
OPTIM.NEW07          <- DATA_CONV_EST07[idem_lavaan_OpenMx07,]
save(OPTIM.NEW07, file = "OPTIM.NEW07.RData")


## ----------------------------------END------------------------------------- ##