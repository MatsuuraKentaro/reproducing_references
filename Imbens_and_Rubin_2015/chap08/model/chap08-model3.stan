data {
  int<lower=1> N;                   // number of observations
  int K;                            // number of covariates
  vector[N] Y_obs;                  // log of observed outcome
  array[N] int<lower=0, upper=1> W; // treatment assignment indicator
  matrix[N, K] X;                   // covariates
}

parameters {
  vector[K] beta_c;
  vector[K] beta_t;
  real<lower=0> s2_c;
  real<lower=0> s2_t;
}

transformed parameters {
  real s_c = sqrt(s2_c);
  real s_t = sqrt(s2_t);
}

model {
  beta_c[1:K] ~ normal(0, 100);
  beta_t[1:K] ~ normal(0, 100);
  s2_c ~ inv_gamma(1, 0.01);
  s2_t ~ inv_gamma(1, 0.01);

  for (i in 1:N) {
    if (W[i] == 0) {
      Y_obs[i] ~ normal(X[i]*beta_c, s_c);
    } else if (W[i] == 1) {
      Y_obs[i] ~ normal(X[i]*beta_t, s_t);
    }
  }
}

generated quantities {
  vector[N] Y_0;
  vector[N] Y_1;
  vector[N] tau_individual;
  real tau_fs;
  real tau_quant_25;
  real tau_quant_50;
  real tau_quant_75;
  for (i in 1:N) {
    if (W[i] == 0) { 
      Y_0[i] = Y_obs[i];
      Y_1[i] = normal_rng(X[i]*beta_t, s_t);
    } else if (W[i] == 1) {
      Y_0[i] = normal_rng(X[i]*beta_c, s_c);
      Y_1[i] = Y_obs[i];
    }
  }
  tau_individual[1:N] = Y_1[1:N] - Y_0[1:N];
  tau_fs = mean(tau_individual);
  tau_quant_25 = quantile(Y_1, 0.25) - quantile(Y_0, 0.25);
  tau_quant_50 = quantile(Y_1, 0.50) - quantile(Y_0, 0.50);
  tau_quant_75 = quantile(Y_1, 0.75) - quantile(Y_0, 0.75);
}
