data {
  int<lower=1> N; // Number of observations
}

transformed data {
  real alpha = 3;
  real beta = 2;
  simplex[3] lambda = [0.75, 0.15, 0.10]';
}

generated quantities {
  // Initialize predictive variables with inflated value
  array[N] real<lower=0, upper=1> y;

  for (n in 1:N) {
    int z = categorical_rng(lambda);

    if (z == 1) {
      y[n] = beta_rng(alpha, beta);
    } else if (z == 2) {
      y[n] = 0;
    } else {
      y[n] = 1;
    }
  }
}
