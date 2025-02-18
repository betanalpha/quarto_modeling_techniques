functions {
  // Log probability density function over cut point
  // induced by a Dirichlet probability density function
  // over baseline probabilities and latent logistic
  // density function.
  real induced_dirichlet_lpdf(vector c, vector alpha) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p;
    real logJ = 0;

    // Induced ordinal probabilities
    p[1] = Pi[1];
    for (k in 2:(K - 1))
      p[k] = Pi[k] - Pi[k - 1];
    p[K] = 1 - Pi[K - 1];

    // Log Jacobian correction
    for (k in 1:(K - 1)) {
      if (c[k] >= 0)
        logJ += -c[k] - 2 * log(1 + exp(-c[k]));
      else
        logJ += +c[k] - 2 * log(1 + exp(+c[k]));
    }

    return dirichlet_lpdf(p | alpha) + logJ;
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
  vector[I] gamma;                    // Item qualities

  simplex[K] mu_p;                    // Population simplex baseline
  real<lower=0> tau_p;                // Population simplex scale
  array[J] ordered[K - 1] cut_points; // Individual interior cut points
}

model {
  vector[K] alpha = mu_p / tau_p + rep_vector(1, K);

  // Prior model
  gamma ~ normal(0, 5 / 2.32);

  mu_p ~ dirichlet(rep_vector(5, K));
  tau_p ~ normal(0, 5 / 2.57);
  for (j in 1:J)
    cut_points[j] ~ induced_dirichlet(alpha);

  // Observational model
  y ~ ordered_logistic(gamma[item_idx], cut_points[context_idx]);
}

generated quantities {
  ordered[K - 1] mu_cut_points = derived_cut_points(mu_p);

  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordered_logistic_rng(gamma[item_idx[n]],
                                     cut_points[context_idx[n]]);
}
