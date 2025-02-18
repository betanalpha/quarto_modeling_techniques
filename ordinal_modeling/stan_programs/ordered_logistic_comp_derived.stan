functions {
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
  int<lower=1> I;                   // Number of items
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations

  array[N] int<lower=1, upper=I> item_idx; // Observed items
  array[N] int<lower=1, upper=K> y;        // Observed categories
}

parameters {
  vector[I] gamma; // Item qualities
  simplex[K] p;    // Baseline category probabilities
}

transformed parameters {
  // Interior cut points
  ordered[K - 1] cut_points = derived_cut_points(p);
}

model {
  // Prior model
  gamma ~ normal(0, 5 / 2.32);
  p ~ dirichlet(rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(gamma[item_idx], cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordered_logistic_rng(gamma[item_idx[n]], cut_points);
}
