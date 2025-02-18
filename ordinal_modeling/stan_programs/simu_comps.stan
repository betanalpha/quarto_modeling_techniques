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
  int<lower=1> I; // Number of items
  int<lower=1> K; // Number of ordinal categories
  int<lower=1> N; // Number of observations
}

transformed data {
  simplex[K] p = dirichlet_rng(rep_vector(1, K));
  simplex[I] lambda = dirichlet_rng(rep_vector(10, I));
}

generated quantities {
  // Item qualities
  array[I] real gamma = normal_rng(zeros_vector(I), 1.5);

  // Interior cut points
  ordered[K - 1] cut_points = derived_cut_points(p);

  // Observations
  array[N] int<lower=1, upper=I> item_idx;
  array[N] int<lower=1, upper=K> y;
  
  for (n in 1:N) {
    item_idx[n] = categorical_rng(lambda);
    y[n] = ordered_logistic_rng(gamma[item_idx[n]], cut_points);
  }
}
