data {
  int<lower=1> N;          // Number of observations
  array[N] int<lower=0> y; // Positive integer observations
}

parameters {
  real<lower=0> mu;              // Poisson intensity
  real<lower=0, upper=1> lambda; // Main component probability
}

model {
  // Prior model
  mu ~ normal(0, 15 / 2.57); // 0 <~ mu <~ 15
  // Implicit uniform prior density function for lambda

  // Observational model
  for (n in 1:N) {
    if (y[n] == 0) {
      target += log_mix(lambda, poisson_lpmf(y[n] | mu), 0);
    } else {
      target += log(lambda) + poisson_lpmf(y[n] | mu);
    }
  }
}

generated quantities {
  // Initialize predictive variables with inflated value
  array[N] int<lower=0> y_pred = rep_array(0, N);

  for (n in 1:N) {
    // If we sample the non-inflating component then replace initial
    // value with a Poisson sample
    if (bernoulli_rng(lambda)) {
      y_pred[n] = poisson_rng(mu);
    }
  }
}
