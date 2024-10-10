data {
  int<lower=1> N;      // Number of observations
}

transformed data {
  int K = 3;                                    // Number of components
  array[K] real mu = {-4, 1, 3};                // Component locations
  array[K] real<lower=0> sigma = {2, 0.5, 0.5}; // Component scales
  simplex[K] lambda = [0.3, 0.5, 0.2]';         // Component probabilities

}

generated quantities {
  array[N] real y;

  for (n in 1:N) {
    int z = categorical_rng(lambda);
    y[n] = normal_rng(mu[z], sigma[z]);
  }
}
