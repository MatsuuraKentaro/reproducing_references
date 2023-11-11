data {
  int<lower=1> N;                     // number of observations
  int<lower=1> J;                     // number of schools
  vector[N] Y_obs;                    // observed outcome
  array[N] int<lower=0, upper=1> W;   // treatment assignment indicator
  array[N] int<lower=1, upper=J> i2j; // this maps classID to schoolID
}

parameters {
  vector[J] mu_c;
  vector[J] mu_t;
  real gamma_c;
  real gamma_t;
  real<lower=0> s_y;
  real<lower=0> s_mu_c;
  real<lower=0> s_mu_t;
  real<lower=-1, upper=1> rho;
}

transformed parameters {
  matrix[2, 2] sigma = [
    [square(s_mu_c), rho*s_mu_c*s_mu_t],
    [rho*s_mu_c*s_mu_t, square(s_mu_t)]];
}

model {
  for (j in 1:J) {
    [mu_c[j], mu_t[j]]' ~ multi_normal([gamma_c, gamma_t]', sigma);
  }

  for (i in 1:N) {
    if (W[i] == 0) {
      Y_obs[i] ~ normal(mu_c[i2j[i]], s_y);
    } else if (W[i] == 1) {
      Y_obs[i] ~ normal(mu_t[i2j[i]], s_y);
    }
  }
}

generated quantities {
  vector[N] Y_0;
  vector[N] Y_1;
  vector[N] tau_individual;
  real tau = gamma_t - gamma_c;
  real tau_fs;
  for (i in 1:N) {
    if (W[i] == 0) {
      Y_0[i] = Y_obs[i];
      Y_1[i] = normal_rng(mu_t[i2j[i]], s_y);
    } else if (W[i] == 1) {
      Y_0[i] = normal_rng(mu_c[i2j[i]], s_y);
      Y_1[i] = Y_obs[i];
    }
  }
  tau_individual[1:N] = Y_1[1:N] - Y_0[1:N];
  tau_fs = mean(tau_individual);
}
