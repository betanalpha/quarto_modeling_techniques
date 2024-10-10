data {
  int<lower=1> N_zero; // Number of zero observations
  int<lower=1> N_one;  // Number of one observations
  int<lower=1> N_else; // Number of non-zero/one observations
}

parameters {
  simplex[3] lambda; // Component probabilities
}

model {
  // Prior model
  // Implicit uniform prior density function for lambda

  // Observational model
  target += multinomial_lpmf({N_else, N_zero, N_one} | lambda);
}
