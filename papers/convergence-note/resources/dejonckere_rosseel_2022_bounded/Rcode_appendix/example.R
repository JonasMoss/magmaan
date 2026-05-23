library("lavaan")

# define the population model
pop.model <- '
    # factor loadings
    Y =~ 1*y1 + 0.8*y2 + 0.6*y3
    X =~ 1*x1 + 0.8*x2 + 0.6*x3

    # regression part
    Y ~ 0.25*X
'

# set seed
set.seed(8)

# generate random sample (N=20)
Data <- simulateData(pop.model, sample.nobs = 20L)

# define the model to be fitted
model <- '
    Y =~ y1 + y2 + y3
    X =~ x1 + x2 + x3
    Y ~ X
'

# fit sem using ML (without bounds)
fit <- sem(model, data = Data, estimator = "ML", 
           optim.attempts = 1L, verbose = TRUE)
summary(fit)


