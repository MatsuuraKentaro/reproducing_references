library(dplyr)
library(cmdstanr)

d <- read.csv('input/flu.csv') |> 
  filter(female == 1) |> 
  mutate(subjid = row_number(),
         age = (age - 65)/10) |> 
  rename(heart = heart.disease, 
         Z = treatment.assigned, 
         W_obs = treatment.received,
         Y_obs = outcome)

d_00 <- d |> filter(Z == 0, W_obs == 0)
d_01 <- d |> filter(Z == 0, W_obs == 1)
d_10 <- d |> filter(Z == 1, W_obs == 0)
d_11 <- d |> filter(Z == 1, W_obs == 1)
X    <- d |> select(age, copd, heart)

data <- list(N = nrow(d),
             N_00 = nrow(d_00), N_01 = nrow(d_01),
             N_10 = nrow(d_10), N_11 = nrow(d_11),
             i_00 = d_00$subjid, i_01 = d_01$subjid,
             i_10 = d_10$subjid, i_11 = d_11$subjid,
             D = ncol(X), X = X, Y = d$Y, N_prior = 30.0)
model <- cmdstan_model(stan_file = 'model/model.stan')

fit <- model$sample(data = data, seed = 123, parallel_chains = 4, 
                    iter_sampling = 2000, iter_warmup = 1000)

fit$save_object(file = 'output/result-model.RDS')
write.csv(fit$summary(), file = 'output/fit-summary.csv',
          quote=TRUE, row.names=FALSE)
