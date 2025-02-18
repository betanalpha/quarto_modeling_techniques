data {
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations
  array[N] int<lower=1, upper=K> y; // Observed categories
}

parameters {
  ordered[K - 1] cut_points; // Interior cut points
}

model {
  // Implicit uniform prior model

  // Observational model
  y ~ ordered_logistic(zeros_vector(N), cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordered_logistic_rng(0, cut_points);
}
