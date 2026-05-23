
datageneration<-function(N, indirect, direct, omega, Nrun){
  # N: sample size
  # indirect: size indirect effect
  # direct: size direct effect
  # omega: McDonald's omega
  # Nrun: numer of replications
  
  # create folder 
  if (!file.exists(paste(getwd(),'/N=',N,'/indirect=',indirect,'/direct=',direct,'/omega=',omega,sep=''),recursive=TRUE)) {
    dir.create(paste(getwd(),'/N=',N,'/indirect=',indirect,'/direct=',direct,'/omega=',omega,sep=''),recursive=TRUE)
  }  
  
  # regression coefficients
  #----------------------------------------------
  a<- sqrt(indirect)
  b<- a
  c<- direct 
  
  # (error) variances of latent variables
  # see Ledgerwood & Shrout, 2011
  sdEta1 <-1
  ressdEta2 <-sqrt(1-a^2)
  ressdEta3 <-sqrt(1-(b^2+c^2 +2*a*b*c))
  
  for(r in 1:Nrun){
    # latent variables
    #--------------------------------------------
    eta1<-rnorm(N,0,sdEta1)
    eta2<-rnorm(N,a*eta1,ressdEta2)
    eta3<-rnorm(N,b*eta2+c*eta1,ressdEta3)
    
    # observed indicators
    #---------------------------------------------
    sdX<-rep(1,3*3) # 3 latent variables, 3 indicators for each measurement model
    
    if(omega==0.55){
      lambda1<-0.562
      lambda2<-0.662
      lambda3<-0.378
    }
    if(omega==0.85){
      lambda1<-0.839
      lambda2<-0.895
      lambda3<-0.680
    }    

    lambda<-rep(c(lambda1,lambda2,lambda3),3)
    lambda

    X<-matrix(NA, nrow = N, ncol= 3*3)
    for(i in 1:N){
      for(j in 1:3){ # measurement model eta1
        X[i,j]<-rnorm(1,sdX[j]*(lambda[j]*eta1[i]),sdX[j]*sqrt(1-lambda[j]^2))        
      }
      for(j in 4:6){ # measurement model eta2
        X[i,j]<-rnorm(1,sdX[j]*(lambda[j]*eta2[i]),sdX[j]*sqrt(1-lambda[j]^2))       
      }
      for(j in 7:9){ # measurement model eta3
        X[i,j]<-rnorm(1,sdX[j]*(lambda[j]*eta3[i]),sdX[j]*sqrt(1-lambda[j]^2))       
      }
    }

    save(X, file=paste(getwd(),'/N=',N,'/indirect=',indirect,'/direct=',direct,'/omega=',omega,'/Dat', r,'.RData', sep=''))
  
  }
}

