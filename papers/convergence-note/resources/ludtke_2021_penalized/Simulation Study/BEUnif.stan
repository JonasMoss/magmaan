data {
  int<lower = 1> J; // number of items
  real N; // number of persons
  matrix[J,J] S; // scatter matrix
  vector[J] xbar; // sample means
  int<lower = 1, upper = 3> dd[J]; // dimension identifier
}
transformed data{
  real epsilon = 0.01;
  real n1epsilon = -1 + epsilon;
  real p1epsilon = 1 - epsilon;
}
parameters {
  vector[J] alpha;  // intercept measurement model
  vector<lower = epsilon>[J] sdX; // standard deviation observed indicators
  vector<lower = epsilon, upper= p1epsilon>[3] lambda1; // first standardized factor loading
  vector<lower = n1epsilon, upper= p1epsilon>[6] lambda2; // remaining  loadings
  real<lower=n1epsilon, upper=p1epsilon> correl12;
  real<lower=n1epsilon, upper= p1epsilon> correl13;
  real<lower=n1epsilon, upper= p1epsilon> correl23;
}
transformed parameters{
  real a;
  real b;
  real c;
  real ab;
  corr_matrix[3] correl;
  cov_matrix[J]  Sigma;
  vector[J] lambda;
  lambda[1] = lambda1[1];
  lambda[4] = lambda1[2];
  lambda[7] = lambda2[3];
  lambda[2:3] = lambda2[1:2];
  lambda[5:6] = lambda2[3:4];
  lambda[8:9] = lambda2[5:6];
  correl[2,1] = correl12;
  correl[1,2] = correl12;
  correl[3,1] = correl13;
  correl[1,3] = correl13;
  correl[3,2] = correl23;
  correl[2,3] = correl23;
  correl[1,1]=1;
  correl[2,2]=1;
  correl[3,3]=1;
  a = correl[1,2];
  b = (correl[2,3]-correl[1,3]*correl[1,2])/(1-correl[1,2]^2);
  c = (correl[1,3]-correl[2,3]*correl[1,2])/(1-correl[1,2]^2);
  ab = a*b;
  // diagonal Sigma
  for(j in 1:J){
    Sigma[j,j] = sdX[j]^2*lambda[j]^2+sdX[j]^2*(1-lambda[j]^2);
  }
  // off-diagonal
  for(j in 2:J){
    for (k in 1:(j-1)) {
      Sigma[j,k] = sdX[j]*lambda[j]*sdX[j]*lambda[k]*correl[dd[j],dd[k]];
      Sigma[k,j] = Sigma[j,k];
    }
  }
}
model {
  alpha ~ normal(0, 10); 
  S ~ wishart(N-1, Sigma);
  xbar ~ multi_normal(alpha, (1/N)*Sigma);
  correl12 ~ uniform(n1epsilon, p1epsilon);
  correl13 ~ uniform(n1epsilon, p1epsilon);
  correl23 ~ uniform(n1epsilon, p1epsilon);
  lambda1 ~ uniform(epsilon, p1epsilon);
  lambda2 ~ uniform(n1epsilon, p1epsilon);
  sdX  ~ normal(0, 10);  
}