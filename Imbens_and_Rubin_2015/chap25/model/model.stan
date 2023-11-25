functions {
  real pp_lpmf(array[] int y, vector log_p1, vector p2) {
    return sum(log_p1) + bernoulli_lpmf(y | p2);
  }

  real mix_pp_lpmf(int y, real log_p1, real p2, real log_q1, real q2) {
    vector[2] ps;
    ps[1] = log_p1 + bernoulli_lpmf(y | p2);
    ps[2] = log_q1 + bernoulli_lpmf(y | q2);
    return log_sum_exp(ps);
  }
}

data {
  int<lower=1> N;     // number of observations
  int<lower=1> N_00;  // number of observations at (0, 0)
  int<lower=1> N_01;  // number of observations at (0, 1)
  int<lower=1> N_10;  // number of observations at (1, 0)
  int<lower=1> N_11;  // number of observations at (1, 1)
  int K;              // number of covariates
  array[N_00] int Y_obs_00;  // observed outcome at (0, 0)
  array[N_01] int Y_obs_01;  // observed outcome at (0, 1)
  array[N_10] int Y_obs_10;  // observed outcome at (1, 0)
  array[N_11] int Y_obs_11;  // observed outcome at (1, 1)
  matrix[N, K] X;        // covariates
  matrix[N_00, K] X_00;  // covariates at (0, 0)
  matrix[N_01, K] X_01;  // covariates at (0, 1)
  matrix[N_10, K] X_10;  // covariates at (1, 0)
  matrix[N_11, K] X_11;  // covariates at (1, 1)
}

parameters {
  vector[K] beta_co_c;
  vector[K] beta_co_t;
  vector[K] beta_nt;
  vector[K] beta_at;
  vector[K] gamma_nt;
  vector[K] gamma_at;
}

transformed parameters {
  matrix[N_00,3] log_p_type_00;  // 1: co, 2:nt, 3:at
  matrix[N_01,3] log_p_type_01;  // 1: co, 2:nt, 3:at
  matrix[N_10,3] log_p_type_10;  // 1: co, 2:nt, 3:at
  matrix[N_11,3] log_p_type_11;  // 1: co, 2:nt, 3:at
  vector[N_00] p_y_00_nt   = inv_logit(X_00*beta_nt);
  vector[N_00] p_y_00_co_c = inv_logit(X_00*beta_co_c);
  vector[N_01] p_y_01_at   = inv_logit(X_01*beta_at);
  vector[N_10] p_y_10_nt   = inv_logit(X_10*beta_nt);
  vector[N_11] p_y_11_at   = inv_logit(X_11*beta_at);
  vector[N_11] p_y_11_co_t = inv_logit(X_11*beta_co_t);
  {
    vector[N_00] xg_00_nt = X_00*gamma_nt;
    vector[N_00] xg_00_at = X_00*gamma_at;
    vector[N_01] xg_01_nt = X_01*gamma_nt;
    vector[N_01] xg_01_at = X_01*gamma_at;
    vector[N_10] xg_10_nt = X_10*gamma_nt;
    vector[N_10] xg_10_at = X_10*gamma_at;
    vector[N_11] xg_11_nt = X_11*gamma_nt;
    vector[N_11] xg_11_at = X_11*gamma_at;
    for (n in 1:N_00) log_p_type_00[n] = log_softmax([0, xg_00_nt[n], xg_00_at[n]]')';
    for (n in 1:N_01) log_p_type_01[n] = log_softmax([0, xg_01_nt[n], xg_01_at[n]]')';
    for (n in 1:N_10) log_p_type_10[n] = log_softmax([0, xg_10_nt[n], xg_10_at[n]]')';
    for (n in 1:N_11) log_p_type_11[n] = log_softmax([0, xg_11_nt[n], xg_11_at[n]]')';
  }
}

model {
  // prior
  real prior_at = 2*sum(log_inv_logit(X*beta_at) + log1m_inv_logit(X*beta_at));
  real prior_nt = 2*sum(log_inv_logit(X*beta_nt) + log1m_inv_logit(X*beta_nt));
  real prior_co = sum(log_inv_logit(X*beta_co_c) + log1m_inv_logit(X*beta_co_c)
                    + log_inv_logit(X*beta_co_t) + log1m_inv_logit(X*beta_co_t));
  real prior_type = sum(log_p_type_00) + sum(log_p_type_01) 
                  + sum(log_p_type_10) + sum(log_p_type_11);
  target += 30.0/(12 * N) * (prior_at + prior_nt + prior_co + prior_type);

  // likelihood  
  for (n in 1:N_00) {
    Y_obs_00[n] ~ mix_pp(log_p_type_00[n,1], p_y_00_co_c[n], 
                         log_p_type_00[n,2], p_y_00_nt[n]);
  }
  Y_obs_01[1:N_01] ~ pp(log_p_type_01[1:N_01,3], p_y_01_at[1:N_01]);
  Y_obs_10[1:N_10] ~ pp(log_p_type_10[1:N_10,2], p_y_10_nt[1:N_10]);
  for (n in 1:N_11) {
    Y_obs_11[n] ~ mix_pp(log_p_type_11[n,1], p_y_11_co_t[n], 
                         log_p_type_11[n,3], p_y_11_at[n]);
  }
}

generated quantities {
  vector[N_00] Y_0_00;
  vector[N_00] Y_1_00;
  vector[N_11] Y_0_11;
  vector[N_11] Y_1_11;
  array[N_00] int Co_00;
  array[N_11] int Co_11;
  vector[N_00] p_y_00_co_t = inv_logit(X_00*beta_co_t);
  vector[N_11] p_y_11_co_c = inv_logit(X_11*beta_co_c);
  vector[N_00] tau_individual_00;
  vector[N_11] tau_individual_11;
  int N_co = 0;
  real sum_tau_co = 0.0;
  real tau_late;
  
  for (n in 1:N_00) {
    vector[2] ps;
    real p_co;
    ps[1] = log_p_type_00[n,1] + bernoulli_lpmf(Y_obs_00[n] | p_y_00_co_c[n]);
    ps[2] = log_p_type_00[n,2] + bernoulli_lpmf(Y_obs_00[n] | p_y_00_nt[n]);
    p_co = exp(ps[1] - log_sum_exp(ps));
    Co_00[n] = bernoulli_rng(p_co);
    Y_0_00[n] = Y_obs_00[n];
    if (Co_00[n] == 1) {
      Y_1_00[n] = bernoulli_rng(p_y_00_co_t[n]);
    } else {
      Y_1_00[n] = bernoulli_rng(p_y_00_nt[n]);
    }
  }

  for (n in 1:N_11) {
    vector[2] ps;
    real p_co;
    ps[1] = log_p_type_11[n,1] + bernoulli_lpmf(Y_obs_11[n] | p_y_11_co_t[n]);
    ps[2] = log_p_type_11[n,3] + bernoulli_lpmf(Y_obs_11[n] | p_y_11_at[n]);
    p_co = exp(ps[1] - log_sum_exp(ps));
    Co_11[n] = bernoulli_rng(p_co);
    if (Co_11[n] == 1) {
      Y_0_11[n] = bernoulli_rng(p_y_11_co_c[n]);
    } else {
      Y_0_11[n] = bernoulli_rng(p_y_11_at[n]);
    }
    Y_1_11[n] = Y_obs_11[n];
  }
  tau_individual_00[1:N_00] = Y_1_00[1:N_00] - Y_0_00[1:N_00];
  tau_individual_11[1:N_11] = Y_1_11[1:N_11] - Y_0_11[1:N_11];
  
  for (n in 1:N_00) {
    if (Co_00[n] == 1) {
      sum_tau_co += tau_individual_00[n];
      N_co += 1;
    }
  }
  for (n in 1:N_11) {
    if (Co_11[n] == 1) {
      sum_tau_co += tau_individual_11[n];
      N_co += 1;
    }
  }
  tau_late = (N_co != 0 ? sum_tau_co/N_co : -10000);
}
