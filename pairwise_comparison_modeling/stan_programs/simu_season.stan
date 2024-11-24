data {
  int<lower=1> N_games; // Number of games

  // Matchups
  int<lower=1> N_teams;
  array[N_games] int<lower=1, upper=N_teams> home_idx;
  array[N_games] int<lower=1, upper=N_teams> away_idx;

  int<lower=1> N_weeks; // Number of weeks
  array[N_games] int<lower=1, upper=N_weeks> week;
}

generated quantities {
  // Baseline log score
  real alpha = 4;

  // Team skills
  array[N_teams] real beta_off
    = normal_rng(rep_vector(0.0, N_teams), 0.75 / 2.32);

  array[N_teams] real beta_def
    = normal_rng(rep_vector(0.0, N_teams), 0.65 / 2.32);

  // Home-field advantage
  real eta_off = 0.05;
  real eta_def = 0.20;

  // Baseline discrimination
  real<lower=0> gamma = 0.985;

  // Game outcomes
  array[N_games] int<lower=0> y_home;
  array[N_games] int<lower=0> y_away;

  for (n in 1:N_games) {
    // Simulate scores
    real mu =  alpha + eta_off
             + pow(gamma, week[n]) * (  beta_off[home_idx[n]]
                                      - beta_def[away_idx[n]]);
    y_home[n] = poisson_log_rng(mu);

    mu =  alpha - eta_def
        + pow(gamma, week[n]) * (  beta_off[away_idx[n]]
                                 - beta_def[home_idx[n]]);
    y_away[n] = poisson_log_rng(mu);
  }
}
