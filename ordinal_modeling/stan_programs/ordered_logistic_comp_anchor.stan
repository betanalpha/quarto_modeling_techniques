data {
  int<lower=1> I;                   // Number of items
  int<lower=1> K;                   // Number of ordinal categories
  int<lower=1> N;                   // Number of observations

  array[N] int<lower=1, upper=I> item_idx; // Observed items
  array[N] int<lower=1, upper=K> y;        // Observed categories
}

parameters {
  vector[I - 1] gamma_free;  // Free item qualities
  ordered[K - 1] cut_points; // Interior cut points
}

transformed parameters {
  vector[I] gamma = append_row([0]', gamma_free);
}

model {
  // Implicit uniform prior model
  
  // Observational model
  y ~ ordered_logistic(gamma[item_idx], cut_points);
}

generated quantities {
  array[N] int<lower=1, upper=K> y_pred;
  for (n in 1:N)
    y_pred[n] = ordered_logistic_rng(gamma[item_idx[n]], cut_points);
}