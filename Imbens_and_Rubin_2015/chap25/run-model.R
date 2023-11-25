library(dplyr)
library(cmdstanr)

d <- read.csv('input/flu.csv') %>% 
  filter(female == 1) %>% 
  mutate(age = (age - 65)/10,
         intercept = 1) %>% 
  rename(heart = heart.disease, 
         Z = treatment.assigned, 
         W_obs = treatment.received,
         Y_obs = outcome)

d_00 <- d %>% filter(Z == 0, W_obs == 0)
d_01 <- d %>% filter(Z == 0, W_obs == 1)
d_10 <- d %>% filter(Z == 1, W_obs == 0)
d_11 <- d %>% filter(Z == 1, W_obs == 1)
X    <- d    %>% select(intercept, age, copd, heart)
X_00 <- d_00 %>% select(intercept, age, copd, heart)
X_01 <- d_01 %>% select(intercept, age, copd, heart)
X_10 <- d_10 %>% select(intercept, age, copd, heart)
X_11 <- d_11 %>% select(intercept, age, copd, heart)

data <- list(N = nrow(d), K = ncol(X_00), 
             N_00 = nrow(d_00), N_01 = nrow(d_01), 
             N_10 = nrow(d_10), N_11 = nrow(d_11), 
             Y_obs_00 = d_00$Y_obs, Y_obs_01 = d_01$Y_obs, 
             Y_obs_10 = d_10$Y_obs, Y_obs_11 = d_11$Y_obs, 
             X = X, X_00 = X_00, X_01 = X_01, X_10 = X_10, X_11 = X_11)
model <- cmdstan_model(stan_file = 'model/model.stan')

# setting of initial values
fit_MAP <- model$optimize(data = data, seed = 1, iter = 10000)
beta_co_c_ini <- fit_MAP$draws(variables = 'beta_co_c') %>% as.vector()
beta_co_t_ini <- fit_MAP$draws(variables = 'beta_co_t') %>% as.vector()
beta_nt_ini <- fit_MAP$draws(variables = 'beta_nt') %>% as.vector()
beta_at_ini <- fit_MAP$draws(variables = 'beta_at') %>% as.vector()
gamma_nt_ini <- fit_MAP$draws(variables = 'gamma_nt') %>% as.vector()
gamma_at_ini <- fit_MAP$draws(variables = 'gamma_at') %>% as.vector()
init_fun <- function(chain_id) {
  set.seed(chain_id)
  list(beta_co_c=beta_co_c_ini, beta_co_t=beta_co_t_ini, 
       beta_nt=beta_nt_ini, beta_at=beta_at_ini,
       gamma_nt=gamma_nt_ini, gamma_at=gamma_at_ini)
}

fit <- model$sample(data = data, seed = 123, parallel_chains=4, init=init_fun,
                    iter_sampling = 2000, iter_warmup = 1000)

fit$save_object(file='output/result-model.RDS')
write.table(fit$summary(), file='output/fit-summary.csv',
            sep=',', quote=TRUE, row.names=FALSE)
