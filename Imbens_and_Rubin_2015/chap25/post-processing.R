library(dplyr)
library(cmdstanr)
library(ggplot2)
library(patchwork)
library(latex2exp)

d <- read.csv('input/flu.csv') %>% filter(female == 1)
N <- nrow(d)

fit <- readRDS('output/result-model.RDS')

print_summary <- function(ms, display_names) {
  if (is.vector(ms)) {
    cat(sprintf('%s: %.2f (%.2f, %.2f)\n', display_names[1],
                mean(ms), quantile(ms, 0.025), quantile(ms, 0.975)))
  } else {
    for (i in 1:ncol(ms)) {
      cat(sprintf('%s: %.2f (%.2f, %.2f)\n', display_names[i],
                  mean(ms[,i]), quantile(ms[,i], 0.025), quantile(ms[,i], 0.975)))
    }
  }
}

tau_ms <- fit$draws(variables = 'tau_late', format = 'matrix')
tau_ms <- tau_ms[tau_ms > -9999]
N_co_ms <- fit$draws(variables = 'N_co', format = 'matrix')[tau_ms > -9999]
ITT_W_ms <- N_co_ms / N
ITT_Y_ms <- tau_ms * ITT_W_ms
beta_co_c_ms <- fit$draws(variables = 'beta_co_c', format = 'matrix')
beta_co_t_ms <- fit$draws(variables = 'beta_co_t', format = 'matrix')
beta_nt_ms <- fit$draws(variables = 'beta_nt', format = 'matrix')
beta_at_ms <- fit$draws(variables = 'beta_at', format = 'matrix')
gamma_nt_ms <- fit$draws(variables = 'gamma_nt', format = 'matrix')
gamma_at_ms <- fit$draws(variables = 'gamma_at', format = 'matrix')

sink('output/result-summary.txt')
print_summary(tau_ms, 'tau_late')
print_summary(ITT_W_ms, 'ITT_W')
print_summary(ITT_Y_ms, 'ITT_Y')
print_summary(beta_co_c_ms, c('beta_co_c,intercept', 'beta_co_c,age', 'beta_co_c,copd', 'beta_co_c,heart'))
print_summary(beta_co_t_ms, c('beta_co_t,intercept', 'beta_co_t,age', 'beta_co_t,copd', 'beta_co_t,heart'))
print_summary(beta_nt_ms, c('beta_nt,intercept', 'beta_nt,age', 'beta_nt,copd', 'beta_nt,heart'))
print_summary(beta_at_ms, c('beta_at,intercept', 'beta_at,age', 'beta_at,copd', 'beta_at,heart'))
print_summary(gamma_nt_ms, c('gamma_nt,intercept', 'gamma_nt,age', 'gamma_nt,copd', 'gamma_nt,heart'))
print_summary(gamma_at_ms, c('gamma_at,intercept', 'gamma_at,age', 'gamma_at,copd', 'gamma_at,heart'))
sink()

d_plot <- data.frame(tau_late = tau_ms, pi_co = ITT_W_ms)
p_tau <- ggplot(d_plot, aes(x = tau_late)) +
  theme(text = element_text(size = 18)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.05, fill = 'white', color = 'black') +
  geom_density(fill = 'black', alpha = 0.3) +
  xlim(range(density(tau_ms)$x)) +
  labs(x = TeX('$\\tau_{late}$'))
p_pi_co <- ggplot(d_plot, aes(x = pi_co)) +
  theme(text = element_text(size = 18)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.01, fill = 'white', color = 'black') +
  geom_density(fill = 'black', alpha = 0.3) +
  xlim(range(density(ITT_W_ms)$x)) +
  labs(x = TeX('$\\pi_{co}$'))
p <- p_tau + p_pi_co
ggsave(p, file = 'output/result-fig.png', dpi = 300, w = 8, h = 5)
