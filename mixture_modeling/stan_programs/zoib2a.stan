data {
  // Number of non-zero/one observations
  int<lower=1> N_else;
  // Non-zero/one observations
  array[N_else] real<lower=0, upper=1> y_else;
}

parameters {
  real<lower=0> alpha; // Beta shape
  real<lower=0>  beta; // Beta scale
}

model {
  // Prior model
  alpha ~ normal(0, 10 / 2.57); // 0 <~ alpha <~ 10
  beta ~ normal(0, 10 / 2.57);  // 0 <~ beta  <~ 10

  // Observational model
  target += beta_lpdf(y_else | alpha, beta);
}
