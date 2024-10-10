data {
  // Signal and background observations
  int<lower=1> N;
  array[N] real<lower=0> y;
}

parameters {
  real mu_signal;                // Signal location
  real<lower=0> sigma_signal;    // Signal scale
  real<lower=0> beta_back;       // Background scale
  real<lower=0, upper=1> lambda; // Background probability
}

model {
  // Prior model
  mu_signal ~ normal(50, 50 / 2.32);   // 0 <~ mu_signal    <~ 100
  sigma_signal ~ normal(0, 25 / 2.57); // 0 <~ sigma_signal <~  25
  beta_back ~ normal(0, 50 / 2.57);    // 0 <~ beta_back    <~  50
  // Implicit uniform prior density function for lambda

  // Observational model
  for (n in 1:N) {
    target += log_mix(lambda,
                      exponential_lpdf(y[n] | 1 / beta_back),
                      cauchy_lpdf(y[n] | mu_signal, sigma_signal));
  }
}

generated quantities {
  array[N] real<lower=0, upper=1> p;
  array[N] int<lower=0, upper=1> z_pred;

  for (n in 1:N) {
    vector[2] xs = [   log(lambda)
                     + exponential_lpdf(y[n] | 1 / beta_back),
                       log(1 - lambda)
                     + cauchy_lpdf(y[n] | mu_signal, sigma_signal) ]';
    p[n] = softmax(xs)[1];
    z_pred[n] = bernoulli_rng(p[n]);
  }
}