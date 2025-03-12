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

  // Derive cut points from baseline probabilities
  // and latent logistic density function.
  vector derived_cut_points(vector p) {
    int K = num_elements(p);
    vector[K - 1] c;

    real cum_sum = 0;
    for (k in 1:(K - 1)) {
      cum_sum += p[k];
      c[k] = logit(cum_sum);
    }

    return c;
  }
}

data {
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations
  array[N] int<lower=1, upper=K> y; // Observed categories
}

parameters {
  simplex[K] p; // Category probabilities
}

transformed parameters {
  // Interior cut points
  ordered[K - 1] cut_points = derived_cut_points(p);
}

model {
  // Prior model
  p ~ dirichlet(rep_vector(1, K));

  // Observational model
  y ~ ordinal_logistic(cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordinal_logistic_rng(cut_points);
}
