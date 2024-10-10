data {
  int<lower=1> N;      // Number of observations
}

transformed data {
  real mu_signal =  45;                 // Signal location
  real<lower=0> sigma_signal = 5;       // Signal scale
  real beta_back = 20;                  // Background rate
  real<lower=0, upper=1> lambda = 0.95; // Background probability
}

generated quantities {
  array[N] real<lower=0> y = rep_array(-1, N);

  for (n in 1:N) {
    if (bernoulli_rng(lambda)) {
      y[n] = exponential_rng(1 / beta_back);
    } else {
      // Truncate signal to positive values
      while (y[n] < 0) {
        y[n] = cauchy_rng(mu_signal, sigma_signal);
      }
    }
  }
}
