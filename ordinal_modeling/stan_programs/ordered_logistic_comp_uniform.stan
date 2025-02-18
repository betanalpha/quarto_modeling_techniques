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
  // Implicit uniform prior model
  
  // Observational model
  y ~ ordered_logistic(gamma[item_idx], cut_points);
}