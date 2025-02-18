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
}

data {
  int<lower=1> I;                   // Number of items
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations

  array[N] int<lower=1, upper=I> item_idx; // Observed items
  array[N] int<lower=1, upper=K> y;        // Observed categories
}

parameters {
  vector[I] gamma;           // Item qualities
  ordered[K - 1] cut_points; // Interior cut points
}

model {
  // Prior model
  gamma ~ normal(0, 5 / 2.32);
  cut_points ~ induced_dirichlet(rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(gamma[item_idx], cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordered_logistic_rng(gamma[item_idx[n]], cut_points);
}
