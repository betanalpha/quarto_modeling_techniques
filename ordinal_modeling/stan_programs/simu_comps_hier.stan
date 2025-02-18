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
  int<lower=1> J; // Number of contexts
  int<lower=1> K; // Number of ordinal categories
  int<lower=1> N; // Number of observations
}

transformed data {
  simplex[I] lambda_item = dirichlet_rng(rep_vector(10, I));
  simplex[I] lambda_context = rep_vector(1.0 / J, J);

  simplex[K] mu_p = dirichlet_rng(rep_vector(3, K));
  real<lower=0> tau_p = abs(normal_rng(0, 2));
  vector[K] alpha = mu_p / tau_p + rep_vector(1, K);

  // Interior cut points
  array[J] ordered[K - 1] cut_points;

  for (j in 1:J) {
    vector[K] p = dirichlet_rng(alpha);
    cut_points[j] = derived_cut_points(p);
  }
}

generated quantities {
  // Item qualities
  array[I] real gamma = normal_rng(zeros_vector(I), 1.5);

  // Observations
  array[N] int<lower=1, upper=I> item_idx;
  array[N] int<lower=1, upper=J> context_idx;
  array[N] int<lower=1, upper=K> y;
  
  for (n in 1:N) {
    item_idx[n] = categorical_rng(lambda_item);
    context_idx[n] = categorical_rng(lambda_context);
    y[n] = ordered_logistic_rng(gamma[item_idx[n]],
                                cut_points[context_idx[n]]);
  }
}
