library(dplyr)
library(cmdstanr)

d <- haven::read_dta('http://www.nber.org/~rdehejia/data/nsw_dw.dta')
X <- d %>% 
  mutate(re74pos = if_else(re74 == 0, 0, 1),
         re75pos = if_else(re75 == 0, 0, 1),
         re74 = re74/1000,
         re75 = re75/1000,
         intercept = 1) %>% 
  select(intercept, age, education, married, nodegree, 
         black, re74, re74pos, re75, re75pos)

data <- list(N = nrow(d), K = ncol(X), Y_obs = d$re78/1000, W = d$treat, X = X)
model <- cmdstan_model(stan_file = 'model/chap08-model3.stan')
fit <- model$sample(data = data, seed = 123, parallel_chains=4,
                    iter_sampling = 5000, iter_warmup = 1000)

for (var in c('tau_fs', 'tau_quant_25', 'tau_quant_50', 'tau_quant_75')) {
  ms <- fit$draws(variables = var)
  print(sprintf('%s: %.2f (%.2f)', var, mean(ms), sd(ms)))
}

# tau_fs: 1.57 (0.50)
# tau_quant_25: 0.44 (0.34)
# tau_quant_50: 1.41 (0.55)
# tau_quant_75: 2.88 (0.64)