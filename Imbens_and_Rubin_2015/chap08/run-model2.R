library(cmdstanr)

d <- haven::read_dta('http://www.nber.org/~rdehejia/data/nsw_dw.dta')

data <- list(N = nrow(d), Y_obs = d$re78/1000, W = d$treat, Rho = 0)
model <- cmdstan_model(stan_file = 'model/chap08-model2.stan')
fit <- model$sample(data = data, seed = 123, parallel_chains=4,
                    iter_sampling = 5000, iter_warmup = 1000)

for (var in c('tau_fs', 'tau_quant_25', 'tau_quant_50', 'tau_quant_75')) {
  ms <- fit$draws(variables = var)
  print(sprintf('%s: %.2f (%.2f)', var, mean(ms), sd(ms)))
}

# tau_fs: 1.79 (0.49)
# tau_quant_25: 0.63 (0.35)
# tau_quant_50: 1.64 (0.55)
# tau_quant_75: 3.08 (0.64)