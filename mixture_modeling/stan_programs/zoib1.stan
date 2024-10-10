data {
  int<lower=1> N;                    // Number of observations
  array[N] real<lower=0, upper=1> y; // Unit-interval valued observations
}

transformed data {
  int<lower=0> N_zero = 0;
  int<lower=0> N_one = 0;
  int<lower=0> N_else = N;

  for (n in 1:N) {
    if (y[n] == 0) N_zero += 1;
    if (y[n] == 1) N_one  += 1;
  }

  N_else -= N_one + N_zero;
}

parameters {
  real<lower=0> alpha; // Beta shape
  real<lower=0>  beta; // Beta scale
  simplex[3]   lambda; // Component probabilities
}

model {
  // Prior model
  alpha ~ normal(0, 10 / 2.57); // 0 <~ alpha <~ 10
  beta ~ normal(0, 10 / 2.57);  // 0 <~ beta  <~ 10
  // Implicit uniform prior density function for lambda

  // Observational model
  target += multinomial_lpmf({N_else, N_zero, N_one} | lambda);

  for (n in 1:N) {
    if (0 < y[n] && y[n] < 1) {
      target += beta_lpdf(y[n] | alpha, beta);
    }
  }
}

generated quantities {
  array[N] real<lower=0, upper=1> y_pred;

  for (n in 1:N) {
    int z = categorical_rng(lambda);

    if (z == 1) {
      y_pred[n] = beta_rng(alpha, beta);
    } else if (z == 2) {
      y_pred[n] = 0;
    } else {
      y_pred[n] = 1;
    }
  }
}
