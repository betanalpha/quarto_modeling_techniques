functions {
  // Ordinal probability mass function assuming a
  // latent shifted logistic density function.
  //
  // Positive gamma shifts baseline ordinal
  // probabilities towards larger values.
  real ordinal_shifted_logistic_lpmf(int y, vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return categorical_lpmf(y | p);
  }

  // Ordinal pseudo-random number generator assuming
  // a latent standard logistic density function.
  int ordinal_shifted_logistic_rng(vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
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
  int<lower=1> I; // Number of items
  int<lower=1> J; // Number of contexts
  int<lower=1> K; // Number of ordinal categories
  int<lower=1> N; // Number of observations

  array[N] int<lower=1, upper=I> item_idx;    // Observed items
  array[N] int<lower=1, upper=J> context_idx; // Observed contexts
  array[N] int<lower=1, upper=K> y;           // Observed categories
}

parameters {
  vector[I] gamma; // Item qualities

  simplex[K] mu_p;       // Population simplex baseline
  real<lower=0> tau_p;   // Population simplex scale
  array[J] simplex[K] p; // Baseline category probabilities
}

transformed parameters {
  // Interior cut points
  array[J] ordered[K - 1] cut_points;
  for (j in 1:J)
    cut_points[j] = derived_cut_points(p[j]);
}

model {
  vector[K] alpha = mu_p / tau_p + rep_vector(1, K);

  // Prior model
  gamma ~ normal(0, 5 / 2.32);

  mu_p ~ dirichlet(rep_vector(5, K));
  tau_p ~ normal(0, 5 / 2.57);
  for (j in 1:J)
    p[j] ~ dirichlet(alpha);

  // Observational model
  for (n in 1:N)
    y[n] ~ ordinal_shifted_logistic(cut_points[context_idx[n]],
                                    gamma[item_idx[n]]);
}

generated quantities {
  ordered[K - 1] mu_cut_points = derived_cut_points(mu_p);

  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordinal_shifted_logistic_rng(cut_points[context_idx[n]],
                                             gamma[item_idx[n]]);
}
