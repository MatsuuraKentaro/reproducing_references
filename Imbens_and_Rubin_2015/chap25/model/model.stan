functions {
  real mixture_co_g_lpmf(int y, int g, vector xb_g, real p_1, real p_2) {
    return log_sum_exp(
      categorical_logit_lpmf(1 | xb_g) + bernoulli_lpmf(y | p_1),
      categorical_logit_lpmf(g | xb_g) + bernoulli_lpmf(y | p_2)
    );
  }
  
  int bernoulli_co_rng(int y, int g, vector xb_g, real p_1, real p_2) {
    vector[2] lp = [
      categorical_logit_lpmf(1 | xb_g) + bernoulli_lpmf(y | p_1),
      categorical_logit_lpmf(g | xb_g) + bernoulli_lpmf(y | p_2)]';
    return bernoulli_rng(softmax(lp)[1]);
  }
}

data {
  int<lower=1> N;        // number of observations
  int<lower=1> N_00;     // number of observations for (0, 0)
  int<lower=1> N_01;     // number of observations for (0, 1)
  int<lower=1> N_10;     // number of observations for (1, 0)
  int<lower=1> N_11;     // number of observations for (1, 1)
  array[N_00] int i_00;  // indices of observations for (0, 0)
  array[N_01] int i_01;  // indices of observations for (0, 1)
  array[N_10] int i_10;  // indices of observations for (1, 0)
  array[N_11] int i_11;  // indices of observations for (1, 1)
  int D;                 // number of covariates
  matrix[N, D] X;        // covariates
  array[N] int Y;        // observed outcome
  real N_prior;          // weight of prior
}

parameters {
  vector[2] a_g_raw;
  matrix[D,2] b_g_raw;
  real a_co_A0;
  real a_co_A1;
  real a_nt;
  real a_at;
  vector[D] b_co_A0;
  vector[D] b_co_A1;
  vector[D] b_nt;
  vector[D] b_at;
}

transformed parameters {
  vector[3] a_g = append_row(0, a_g_raw);   // 1: co, 2: nt, 3: at
  matrix[D,3] b_g = append_col(rep_vector(0, D), b_g_raw);
}

model {
  matrix[3,N] Xb_g = rep_matrix(a_g, N) + (X*b_g)';  // NOTE: transpose for softmax
  vector[N_00] p_co_A0 = inv_logit(a_co_A0 + X[i_00]*b_co_A0);
  vector[N_00] p_nt    = inv_logit(a_nt    + X[i_00]*b_nt);
  vector[N_11] p_co_A1 = inv_logit(a_co_A1 + X[i_11]*b_co_A1);
  vector[N_11] p_at    = inv_logit(a_at    + X[i_11]*b_at);

  // prior
  real lp_prior = 0;
  lp_prior += categorical_logit_glm_lpmf(1 | X, a_g, b_g);  // should be 4 times?
  lp_prior += categorical_logit_glm_lpmf(2 | X, a_g, b_g);  // should be 4 times?
  lp_prior += categorical_logit_glm_lpmf(3 | X, a_g, b_g);  // should be 4 times?
  lp_prior += bernoulli_logit_glm_lpmf(1 | X, a_co_A0, b_co_A0);
  lp_prior += bernoulli_logit_glm_lpmf(0 | X, a_co_A0, b_co_A0);
  lp_prior += bernoulli_logit_glm_lpmf(1 | X, a_co_A1, b_co_A1);
  lp_prior += bernoulli_logit_glm_lpmf(0 | X, a_co_A1, b_co_A1);
  lp_prior += 2*bernoulli_logit_glm_lpmf(1 | X, a_nt, b_nt);
  lp_prior += 2*bernoulli_logit_glm_lpmf(0 | X, a_nt, b_nt);
  lp_prior += 2*bernoulli_logit_glm_lpmf(1 | X, a_at, b_at);
  lp_prior += 2*bernoulli_logit_glm_lpmf(0 | X, a_at, b_at);
  target += N_prior/(12 * N) * lp_prior;
  
  // likelihoood
  // (0, 0): mixture of co and nt
  for (n in 1:N_00) {
    Y[i_00[n]] ~ mixture_co_g(2, Xb_g[:,i_00[n]], p_co_A0[n], p_nt[n]);
  }

  // (0, 1): at
  rep_array(3, N_01) ~ categorical_logit_glm(X[i_01], a_g, b_g);
  Y[i_01] ~ bernoulli_logit_glm(X[i_01], a_at, b_at);

  // (1, 0): nt
  rep_array(2, N_10) ~ categorical_logit_glm(X[i_10], a_g, b_g);
  Y[i_10] ~ bernoulli_logit_glm(X[i_10], a_nt, b_nt);
  
  // (1, 1): mixture of co and at
  for (n in 1:N_11) {
    Y[i_11[n]] ~ mixture_co_g(3, Xb_g[:,i_11[n]], p_co_A1[n], p_at[n]);
  }
}

generated quantities {
  int N_co = 0;
  real tau_late;
  {
    matrix[3,N] Xb_g = rep_matrix(a_g, N) + (X*b_g)';  // NOTE: transpose for softmax
    vector[N_00] p_co_A0 = inv_logit(a_co_A0 + X[i_00]*b_co_A0);
    vector[N_00] p_nt    = inv_logit(a_nt    + X[i_00]*b_nt);
    vector[N_11] p_co_A1 = inv_logit(a_co_A1 + X[i_11]*b_co_A1);
    vector[N_11] p_at    = inv_logit(a_at    + X[i_11]*b_at);
    vector[N_00] p_co_A1_00 = inv_logit(a_co_A1 + X[i_00]*b_co_A1);
    vector[N_11] p_co_A0_11 = inv_logit(a_co_A0 + X[i_11]*b_co_A0);
    vector[N_00] Y_0_00;
    vector[N_00] Y_1_00;
    vector[N_11] Y_0_11;
    vector[N_11] Y_1_11;
    array[N_00] int Co_00;
    array[N_11] int Co_11;
    vector[N_00] tau_individual_00;
    vector[N_11] tau_individual_11;
    real sum_tau_co = 0.0;
    
    for (n in 1:N_00) {
      Co_00[n] = bernoulli_co_rng(Y[i_00[n]], 2, Xb_g[:,i_00[n]], p_co_A0[n], p_nt[n]);
      Y_0_00[n] = Y[i_00[n]];
      if (Co_00[n] == 1) {
        Y_1_00[n] = bernoulli_rng(p_co_A1_00[n]);
      } else {
        Y_1_00[n] = bernoulli_rng(p_nt[n]);
      }
    }
    
    for (n in 1:N_11) {
      Co_11[n] = bernoulli_co_rng(Y[i_11[n]], 3, Xb_g[:,i_11[n]], p_co_A1[n], p_at[n]);
      if (Co_11[n] == 1) {
        Y_0_11[n] = bernoulli_rng(p_co_A0_11[n]);
      } else {
        Y_1_00[n] = bernoulli_rng(p_at[n]);
      }
      Y_1_11[n] = Y[i_11[n]];
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
}
