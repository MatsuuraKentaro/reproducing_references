library(cmdstanr)

d <- read.csv('input/CTW_Imbens_Rubin_Chap10.csv')
X <- matrix(d$X, nrow(d), 1)
data <- list(N = nrow(d), J = max(d$G), K = ncol(X), Y_obs = d$Y, 
             W = d$W, i2j = d$G, X = X)
model <- cmdstan_model(stan_file = 'model/chap10-model.stan')
fit <- model$sample(data = data, seed = 123, parallel_chains=4,
                    iter_sampling = 5000, iter_warmup = 1000)

## output
tau_ms <- fit$draws(variables = 'tau_fs')
print(sprintf('tau_fs: %.2f (%.2f)', mean(tau_ms), sd(tau_ms)))

# tau_fs: 8.47 (1.68)