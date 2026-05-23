## ---------------------------
##
## Script name:        Pre_Simulation_Study1.R
##
## Purpose of script:  Script to define population and analysis model, 
##                     parameter values from output, extract function for data,
##                     target functions, model-based target function. 
##                     Run pre simulation.
##
##
## Authors:            Julie De Jonckere and Yves Rosseel
##                     Department of Data Analysis, Ghent University
##
##
## ---------------------------
## Notes:
##      1. Model specification
##      2. Utilities
##      3. Data extract function
##      4. Functions for (existing) target matrices
##      5. Model-based target function
## ---------------------------


## -------------------------------------------------------------------------- ##
##                          1. Model Specification                            ##      
## -------------------------------------------------------------------------- ##

library("lavaan")

## -------------------
## POPULATION MODEL 1
## -------------------

pop.model.1 <- '

               # factor loadings
               Y =~ 1*y1 + 0.8*y2 + 0.6*y3
               X =~ 1*x1 + 0.8*x2 + 0.6*x3

               # regression part
               Y ~ 0.1*X
'

## -------------------
## POPULATION MODEL 2
## -------------------

pop.model.2 <- '

               # factor loadings
               Y =~ 1*y1 + 0.8*y2 + 0.6*y3
               X =~ 1*x1 + 0.8*x2 + 0.6*x3

               # regression part
               Y ~ 0.3*X
'


## -------------------
## MODEL for analysis
## -------------------
model <- '
           # factor loadings
           Y =~ y1 + y2 + y3
           X =~ x1 + x2 + x3

           # regression part
           Y ~ X

           # factor variances
           Y ~~ Y 
           X ~~ X

           # observed residual variances
           y1 ~~ y1 
           y2 ~~ y2 
           y3 ~~ y3 
           x1 ~~ x1 
           x2 ~~ x2 
           x3 ~~ x3 
'


## -------------------------------------------------------------------------- ##
##                               2. Utilities                                 ##      
## -------------------------------------------------------------------------- ##

# parameter values
FIT      <- sem(model)               
OV.NAMES <- lavNames(FIT, "ov") # observed variables
LV.NAMES <- lavNames(FIT, "lv") # latent variables
COEF.NAMES    <- names(coef(FIT))  # (free) parameter names
col.names     <- c(COEF.NAMES, 
                   "beta.se", 
                   "optim", 
                   "se", 
                   "ov.var.pos", 
                   "lv.var.pos")
col.names.l   <- c(COEF.NAMES, 
                   "beta.se", 
                   "optim", 
                   "se", 
                   "ov.var.pos", 
                   "lv.var.pos", 
                   "lambda")
col.names.l.c <- c(COEF.NAMES, 
                   "beta.se", 
                   "optim", 
                   "se", 
                   "ov.var.pos", 
                   "lv.var.pos", 
                   "lambda", 
                   "CA1", 
                   "CA2",
                   "mCA")
NPAR <- length(COEF.NAMES)      
NVAR <- length(OV.NAMES) 

n_cov_elements <- NVAR * (NVAR + 1) / 2  
n_eig_values <- NVAR                     
NDATA <- n_cov_elements + n_eig_values

PTE <- parTable(FIT)
ov.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% OV.NAMES & PTE$lhs == PTE$rhs) 
lv.var.idx <- which(PTE$op == "~~" & PTE$lhs %in% LV.NAMES & PTE$lhs == PTE$rhs) 
beta.idx <- which(PTE$op == "~") 


## -------------------------------------------------------------------------- ##
##                        3. Data extract function                            ##      
## -------------------------------------------------------------------------- ##

extract_results_from_fit_MB <- function(fit) {
  
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
##                      4. Functions target matrices                          ##      
## -------------------------------------------------------------------------- ##

## -------------------------
## Identity matrix
## -------------------------


cov_est_ID <- function(data   = NULL, 
                       lambda = NULL,
                       lavmodel = NULL,
                       lavsamplestats = NULL){
  
  # get number of indicators/observed variables
  P         <- dim(data)[2] 
  
  # get sample (co)variance matrix
  lambda.idx <- which(names(lavmodel@GLIST) == "lambda")
  ov.names <- lavmodel@dimNames[[lambda.idx]][[1]]
  S   <- lavsamplestats@cov[[1]]
  colnames(S) <- rownames(S) <- ov.names
  
  # construct target
  id_target <- diag(P)
  
  # construct adjusted (co)variance matrix 
  id_cov    <- (1-lambda)*S + lambda*id_target
  
  return(id_cov)
}


## ----------------------------
## Constant Correlation matrix
## ----------------------------
# Based on code from https://github.com/ArdiaD/RiskPortfolios/blob/master/R/covEstimation.R


cov_est_CC <- function(data= NULL, 
                       lambda = NULL,
                       lavmodel = NULL,
                       lavsamplestats = NULL){
  
  # get number of indicators/observed variables
  P <- dim(data)[2] 
  
  # get sample (co)variance matrix
  lambda.idx <- which(names(lavmodel@GLIST) == "lambda")
  ov.names <- lavmodel@dimNames[[lambda.idx]][[1]]
  S   <- lavsamplestats@cov[[1]]
  colnames(S) <- rownames(S) <- ov.names
  
  # construct target
  var <- diag(S)
  sqrtvar <- sqrt(var)
  outerSqrtVar <- outer(sqrtvar, sqrtvar)
  
  rBar <- (sum(sum(S/outerSqrtVar)) - P)/(P * (P - 1))
  prior <- rBar * outerSqrtVar
  diag(prior) <- var
  
  # construct adjusted (co)variance matrix
  cc_cov <-  (1-lambda)*S + lambda*prior 
  
  return(cc_cov)
}



## -------------------------
## Common Variance matrix
## -------------------------

cov_est_CV <- function(data = NULL, 
                       lambda = NULL,
                       lavmodel = NULL,
                       lavsamplestats = NULL) {
  
  # get number of indicators/observed variables
  P <- dim(data)[2]
  
  # get sample (co)variance matrix
  lambda.idx <- which(names(lavmodel@GLIST) == "lambda")
  ov.names <- lavmodel@dimNames[[lambda.idx]][[1]]
  S   <- lavsamplestats@cov[[1]]
  colnames(S) <- rownames(S) <- ov.names
  
  # construct target 
  meanvar <- mean(diag(S))
  prior <- meanvar * diag(P)
  
  # construct adjusted (co)variance matrix
  cv_cov <- (1-lambda)*S + lambda*prior
  
  return(cv_cov)
}


## -------------------------------------------------------------------------- ##
##                    5. Model-based target function                          ##      
## -------------------------------------------------------------------------- ##
lav_samplestats_modify_cov <- function(lavsamplestats = NULL,
                                       lavmodel       = NULL,
                                       lambda         = 0.1,
                                       cor.shrink.tol = 0.1,
                                       cor            = 0.2,
                                       reliability    = 0.8) {
  
  # number of groups
  ngroups <- lavmodel@ngroups
  
  # no multilevel yet
  stopifnot(lavmodel@multilevel == FALSE)
  
  # list of modified sample statistics
  SAMPSTAT.NEW <- vector("list", length = ngroups)
  
  # all groups
  for(g in seq_len(ngroups)) {
    
    # sample variance-covariance matrix
    S <- lavsamplestats@cov[[g]]        
    
    # LAMBDA matrix for this group    
    lambda.idx <- which(names(lavmodel@GLIST) == "lambda")[g]
    LAMBDA <- lavmodel@GLIST[[lambda.idx]]
    ov.names <- lavmodel@dimNames[[lambda.idx]][[1]]
    
    nvar <- nrow(LAMBDA)
    nfac <- ncol(LAMBDA)
    
    
    # compute correlation matrix latent variables
    Lambda1 <- LAMBDA
    Lambda1[ lavmodel@m.free.idx[[lambda.idx]] ] <- 1
    COR.sumscores <- stats::cov2cor( t(Lambda1) %*% S %*% Lambda1 )  
    # shrink small values <|0.1| to zero
    small.idx <- which(abs(COR.sumscores) < cor.shrink.tol)
    if(length(small.idx) > 0L) {
      COR.sumscores[small.idx] <- 0
    }
    
    # 'ideal' correlation matrix (using cor as value, only adjusting
    # the sign diagonal = 1, off-diagonal = default cor 
    # option: default cor could be mean correlation of latent variables
    COR.unsigned <- matrix(cor, nfac, nfac)
    diag(COR.unsigned) <- 1
    VETA <- COR.unsigned * sign(COR.sumscores)
    
    # create Lambda matrix where lambda = 0.7
    L07 <- Lambda1 * 0.7
    
    # determine diagonal values of Theta based on reliability
    if(reliability >= 1.0) {            # if REL = 1, is THETA = 0 (no error variance)
      THETA <- matrix(0, nvar, nvar)  
    } else {
      tmp <- diag(L07 %*% VETA %*% t(L07))   
      theta.diag <- tmp/reliability - tmp 
      # no zero or negative theta values on the diagonal
      stopifnot(all(theta.diag > 0))
      THETA <- matrix(0, nvar, nvar)
      diag(THETA) <- theta.diag    
    }
    
    # compute model-implied Sigma
    Sigma <- L07 %*% VETA %*% t(L07) + THETA
    
    # rescale, so that the variances are the same as in 'S'
    Sigma.cor <- stats::cov2cor(Sigma)
    S.diag.sqrt <- sqrt(diag(S))
    Sigma.rescaled <- t(Sigma.cor * S.diag.sqrt) * S.diag.sqrt  # diagonal as in S
    
    # modifify sample statistics
    S.new <- (1 - lambda) * S + lambda * Sigma.rescaled
    colnames(S.new) <- rownames(S.new) <- ov.names
    
    SAMPSTAT.NEW[[g]] <- list(cov = S.new)
  }
  
  # return modified sample statistics
  list(SAMPSTAT.NEW, COR.sumscores[2,1])
  #COR.sumscores
}




lav_samplestats_modify_target <- function(lavsamplestats = NULL,
                                          lavmodel       = NULL,
                                          cor.shrink.tol = 0.1,
                                          cor            = 0,
                                          reliability    = 0.8) {
  
  # number of groups
  ngroups <- lavmodel@ngroups
  
  # no multilevel yet
  stopifnot(lavmodel@multilevel == FALSE)
  
  # list of modified sample statistics
  SAMPSTAT.NEW <- vector("list", length = ngroups)
  
  # all groups
  for(g in seq_len(ngroups)) {
    
    # sample variance-covariance matrix
    S <- lavsamplestats@cov[[g]]        
    
    # LAMBDA matrix for this group    
    lambda.idx <- which(names(lavmodel@GLIST) == "lambda")[g]
    LAMBDA <- lavmodel@GLIST[[lambda.idx]]
    ov.names <- lavmodel@dimNames[[lambda.idx]][[1]]
    
    nvar <- nrow(LAMBDA)
    nfac <- ncol(LAMBDA)
    
    
    # compute correlation matrix latent variables
    Lambda1 <- LAMBDA
    Lambda1[ lavmodel@m.free.idx[[lambda.idx]] ] <- 1
    COR.sumscores <- stats::cov2cor( t(Lambda1) %*% S %*% Lambda1 ) 
    # shrink small values <|0.1| to zero
    small.idx <- which(abs(COR.sumscores) < cor.shrink.tol)
    if(length(small.idx) > 0L) {
      COR.sumscores[small.idx] <- 0
    }
    
    # 'ideal' correlation matrix (using cor as value, only adjusting
    # the sign diagonal = 1, off-diagonal = default cor 
    # option: default cor could be mean correlation of latent variables (see word doc)
    COR.unsigned <- matrix(cor, nfac, nfac)
    diag(COR.unsigned) <- 1
    VETA <- COR.unsigned * sign(COR.sumscores)
    
    # create Lambda matrix where lambda = 0.7
    L07 <- Lambda1 * 0.7
    
    # determine diagonal values of Theta based on reliability
    if(reliability >= 1.0) {            # if REL = 1, is THETA = 0 (no error variance)
      THETA <- matrix(0, nvar, nvar)  
    } else {
      tmp <- diag(L07 %*% VETA %*% t(L07))   
      theta.diag <- tmp/reliability - tmp 
      # no zero or negative theta values on the diagonal
      stopifnot(all(theta.diag > 0))
      THETA <- matrix(0, nvar, nvar)
      diag(THETA) <- theta.diag    
    }
    
    # compute model-implied Sigma
    Sigma <- L07 %*% VETA %*% t(L07) + THETA
    
    # rescale, so that the variances are the same as in 'S'
    Sigma.cor <- stats::cov2cor(Sigma)
    S.diag.sqrt <- sqrt(diag(S))
    Sigma.rescaled <- t(Sigma.cor * S.diag.sqrt) * S.diag.sqrt  # diagonal as in S
    colnames(Sigma.rescaled) <- rownames(Sigma.rescaled) <- ov.names
    
    SAMPSTAT.NEW[[g]] <- list(cov = Sigma.rescaled)
  }
  
  # return modified sample statistics
  SAMPSTAT.NEW[[g]]
  #COR.sumscores
}


## --------------------------------- END ------------------------------------ ##