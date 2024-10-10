data {
  int<lower=1> N;   // Number of observations
  real<lower=0> mu; // Poisson intensity
  real<lower=0, upper=1> lambda; // Main component probability
}

generated quantities {
  // Initialize predictive variables with inflated value
  array[N] int<lower=0> y = rep_array(0, N);

  for (n in 1:N) {
    // If we sample the non-inflating component then replace initial
    // value with a Poisson sample
    if (bernoulli_rng(lambda)) {
      y[n] = poisson_rng(mu);
    }
  }
}
