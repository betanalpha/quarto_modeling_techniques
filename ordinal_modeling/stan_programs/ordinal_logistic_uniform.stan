functions {
  // Ordinal probability mass function assuming a
  // latent standard logistic density function.
  real ordinal_logistic_lpmf(int[] y, vector c) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_lpmf(y | p);
  }

  // Ordinal pseudo-random number generator assuming
  // a latent standard logistic density function.
  int ordinal_logistic_rng(vector c) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_rng(p);
  }
}

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
  y ~ ordinal_logistic(cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordinal_logistic_rng(cut_points);
}
