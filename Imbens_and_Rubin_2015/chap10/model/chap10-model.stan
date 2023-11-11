data {
  int<lower=1> N;                     // number of observations
  int<lower=1> J;                     // number of pairs
  int<lower=1> K;                     // number of covariates
  vector[N] Y_obs;                    // observed outcome
  array[N] int<lower=0, upper=1> W;   // treatment assignment indicator
  array[N] int<lower=1, upper=J> i2j; // this maps i to j
  matrix[N, K] X;                     // covariates
}

parameters {
  real mu_all;
  vector[J] mu;
  vector[K] beta;
  real gamma;
  real<lower=0> s_c;
  real<lower=0> s_t;
  real<lower=0> s_mu;
}

model {
  mu[1:J] ~ normal(mu_all, s_mu);

  for (i in 1:N) {
    if (W[i] == 0) {
      Y_obs[i] ~ normal(mu[i2j[i]] + X[i]*beta, s_c);
    } else if (W[i] == 1) {
      Y_obs[i] ~ normal(mu[i2j[i]] + X[i]*beta + gamma, s_t);
    }
  }
}

generated quantities {
  vector[N] Y_0;
  vector[N] Y_1;
  vector[N] tau_individual;
  real tau_fs;
  for (i in 1:N) {
    if (W[i] == 0) {
      Y_0[i] = Y_obs[i];
      Y_1[i] = normal_rng(mu[i2j[i]] + X[i]*beta + gamma, s_t);
    } else if (W[i] == 1) {
      Y_0[i] = normal_rng(mu[i2j[i]] + X[i]*beta, s_c);
      Y_1[i] = Y_obs[i];
    }
  }
  tau_individual[1:N] = Y_1[1:N] - Y_0[1:N];
  tau_fs = mean(tau_individual);
}
