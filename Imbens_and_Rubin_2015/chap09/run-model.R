library(cmdstanr)

d <- read.csv('input/STAR_Imbens_Rubin_Chap9.csv')
data <- list(N = nrow(d), J = max(d$schoolID), Y_obs = d$Y, 
             W = d$W, i2j = d$schoolID)
model <- cmdstan_model(stan_file = 'model/chap09-model.stan')
fit <- model$sample(data = data, seed = 123, parallel_chains=4,
                    iter_sampling = 5000, iter_warmup = 1000)

## output
tau_ms <- fit$draws(variables = 'tau_fs')
print(sprintf('tau_fs: %.3f (%.3f)', mean(tau_ms), sd(tau_ms)))

sigma_ms <- fit$draws(variables = 'sigma')
print(sprintf('log(sqrt(sigma[1,1])): %.2f (%.2f)', 
              mean(log(sqrt(sigma_ms[,,1]))), sd(log(sqrt(sigma_ms[,,1])))))
print(sprintf('log(sqrt(sigma[2,2])): %.2f (%.2f)', 
              mean(log(sqrt(sigma_ms[,,4]))), sd(log(sqrt(sigma_ms[,,4])))))

# tau_fs: 0.224 (0.069)
# log(sqrt(sigma[1,1])): -0.77 (0.28)
# log(sqrt(sigma[2,2])): -0.79 (0.28)