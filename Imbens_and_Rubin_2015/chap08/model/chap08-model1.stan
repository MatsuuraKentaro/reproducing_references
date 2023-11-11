data {
  int<lower=1> N;                   // number of observations
  vector<lower=0>[N] Y_obs;         // observed outcome
  array[N] int<lower=0, upper=1> W; // treatment assignment indicator
  real<lower=-1, upper=1> Rho;      // correlation between Y_0 and Y_1
}

parameters {
  real mu_c;
  real mu_t;
  real<lower=0> s2;
}

transformed parameters {
  real s = sqrt(s2);
}

model {
  mu_c ~ normal(0, 100);
  mu_t ~ normal(0, 100);
  s2 ~ inv_gamma(1, 0.01);

  for (i in 1:N) {
    if (W[i] == 0) {
      Y_obs[i] ~ normal(mu_c, s);
    } else if (W[i] == 1) {
      Y_obs[i] ~ normal(mu_t, s);
    }
  }
}

generated quantities {
  vector[N] Y_0;
  vector[N] Y_1;
  vector[N] tau_individual;
  real tau = mu_t - mu_c;
  real tau_fs;
  real tau_quant_25;
  real tau_quant_50;
  real tau_quant_75;
  for (i in 1:N) {
    if (W[i] == 0) { 
      Y_0[i] = Y_obs[i];
      Y_1[i] = normal_rng(
        mu_t + Rho*(Y_obs[i] - mu_c), s*sqrt(1 - square(Rho)));
    } else if (W[i] == 1) {
      Y_0[i] = normal_rng(
        mu_c + Rho*(Y_obs[i] - mu_t), s*sqrt(1 - square(Rho)));
      Y_1[i] = Y_obs[i];
    }
  }
  tau_individual[1:N] = Y_1[1:N] - Y_0[1:N];
  tau_fs = mean(tau_individual);
  tau_quant_25 = quantile(Y_1, 0.25) - quantile(Y_0, 0.25);
  tau_quant_50 = quantile(Y_1, 0.50) - quantile(Y_0, 0.50);
  tau_quant_75 = quantile(Y_1, 0.75) - quantile(Y_0, 0.75);
}
