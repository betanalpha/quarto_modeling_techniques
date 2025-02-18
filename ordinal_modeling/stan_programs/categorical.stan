data {
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations
  array[N] int<lower=1, upper=K> y; // Observed categoriesObserved categories
}

parameters {
  simplex[K] p; // Category probabilities
}
model {
  // Prior model
  p ~ dirichlet(rep_vector(1, K));
  
  // Observational model
  y ~ categorical(p);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = categorical_rng(p);
}