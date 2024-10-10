data {
  int<lower=1> N;  // Number of observations
  array[N] real y; // Observations
}

transformed data {
  int K = 3; // Number of components
}

parameters {
  array[K] real mu;             // Component locations
  array[K] real<lower=0> sigma; // Component scales
  simplex[K] lambda;            // Component probabilities
}

model {
  // Prior model
  mu ~ normal(0, 10 / 2.32);    // -10 <~  mu[k]   <~ +10
  sigma ~ normal(0, 10 / 2.57); //   0 <~ sigma[k] <~ +10

  // Implicit uniform prior density function for lambda

  // Observational model
  for (n in 1:N) {
    vector[K] lpds;
    for (k in 1:K) {
      lpds[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lpds);
  }
}

generated quantities {
  array[N] real y_pred;

  for (n in 1:N) {
    int z = categorical_rng(lambda);
    y_pred[n] = normal_rng(mu[z], sigma[z]);
  }
}
