library(dplyr)
library(cmdstanr)
library(ggplot2)
library(patchwork)
library(latex2exp)

d <- read.csv('input/flu.csv') |> filter(female == 1)
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
a_co_A0_ms <- fit$draws(variables = 'a_co_A0', format = 'matrix')
a_co_A1_ms <- fit$draws(variables = 'a_co_A1', format = 'matrix')
a_nt_ms <- fit$draws(variables = 'a_nt', format = 'matrix')
a_at_ms <- fit$draws(variables = 'a_at', format = 'matrix')
b_co_A0_ms <- fit$draws(variables = 'b_co_A0', format = 'matrix')
b_co_A1_ms <- fit$draws(variables = 'b_co_A1', format = 'matrix')
b_nt_ms <- fit$draws(variables = 'b_nt', format = 'matrix')
b_at_ms <- fit$draws(variables = 'b_at', format = 'matrix')
a_g_raw_ms <- fit$draws(variables = 'a_g_raw', format = 'matrix')
b_g_raw_ms <- fit$draws(variables = 'b_g_raw', format = 'matrix')

sink('output/result-summary.txt')
print_summary(tau_ms, 'tau_late')
print_summary(ITT_W_ms, 'ITT_W')
print_summary(ITT_Y_ms, 'ITT_Y')
print_summary(a_co_A0_ms, 'a_co_A0,intercept')
print_summary(b_co_A0_ms, c('b_co_A0,age', 'b_co_A0,copd', 'b_co_A0,heart'))
print_summary(a_co_A0_ms, c('a_co_A1,intercept'))
print_summary(b_co_A0_ms, c('b_co_A1,age', 'b_co_A1,copd', 'b_co_A1,heart'))
print_summary(a_nt_ms, 'a_nt,intercept')
print_summary(b_nt_ms, c('b_nt,age', 'b_nt,copd', 'b_nt,heart'))
print_summary(a_at_ms, 'a_at,intercept')
print_summary(b_at_ms, c('b_at,age', 'b_at,copd', 'b_at,heart'))
print_summary(a_g_raw_ms, c('a_g nt,intercept', 'a_g at,intercept'))
print_summary(b_g_raw_ms, c('b_g nt,age', 'b_g nt,copd', 'b_g nt,heart', 'b_g at,age', 'b_g at,copd', 'b_g at,heart'))
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
