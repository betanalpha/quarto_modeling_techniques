data {
  int<lower=1> N_games; // Number of games

  // Matchups
  int<lower=1> N_teams;
  array[N_games] int<lower=1, upper=N_teams> home_idx;
  array[N_games] int<lower=1, upper=N_teams> away_idx;

  int<lower=1> N_weeks; // Number of weeks
  array[N_games] int<lower=1, upper=N_weeks> week;

  // Game scores
  array[N_games] int<lower=0> y_home;
  array[N_games] int<lower=0> y_away;

  // Playoff games
  int<lower=1> N_playoff_games; // Number of playoff games
  array[N_playoff_games] int<lower=1, upper=N_teams> playoff_home_idx;
  array[N_playoff_games] int<lower=1, upper=N_teams> playoff_away_idx;
  array[N_playoff_games] int playoff_week;
}

parameters {
  real alpha;                         // Baseline log score
  real eta_off;                       // Offensive home-field advantage
  real eta_def;                       // Defensive home-field advantage
  real<lower=0> gamma0;               // Discrimination mantissa
  vector[N_teams - 1] delta_off_free; // Relative team offensive skills
  vector[N_teams - 1] delta_def_free; // Relative team defensive skills
}

transformed parameters {
  // Relative skills for all teams
  vector[N_teams] delta_off = append_row([0]', delta_off_free);
  vector[N_teams] delta_def = append_row([0]', delta_def_free);
}

model {
  // Prior model
  alpha ~ normal(3.77, 0.55 / 2.32); // 25 <~  exp(alpha) <~ 75

  // 0.78 <~ exp(eta_off/eta_def) <~ 1.28
  eta_off ~ normal(0, 0.25 / 2.32);
  eta_def ~ normal(0, 0.25 / 2.32);

  gamma0 ~ normal(0.9, 0.1 / 2.57); // 0.8  <~ gamma0 <~ 1.0

  // -2 <~ delta_off/delta_def <~ +2
  delta_off_free ~ normal(0, 2 / 2.32);
  delta_def_free ~ normal(0, 2 / 2.32);

  // Observational model
  y_home ~ poisson_log(  alpha + eta_off
                       + to_vector(pow(gamma0, week))
                         .* (  delta_off[home_idx]
                             - delta_def[away_idx] ) );
  y_away ~ poisson_log(  alpha - eta_def
                       + to_vector(pow(gamma0, week))
                         .* (  delta_off[away_idx]
                             - delta_def[home_idx] ) );
}

generated quantities {
  array[N_playoff_games] int<lower=0> y_home_playoff_pred;
  array[N_playoff_games] int<lower=0> y_away_playoff_pred;

  for (n in 1:N_playoff_games) {
    real mu =  alpha + eta_off
             + pow(gamma0, playoff_week[n])
               * (  delta_off[playoff_home_idx[n]]
                  - delta_def[playoff_away_idx[n]]);
    y_home_playoff_pred[n] = poisson_log_rng(mu);

    mu =  alpha - eta_def
        + pow(gamma0, playoff_week[n])
          * (  delta_off[playoff_away_idx[n]]
             - delta_def[playoff_home_idx[n]]);
    y_away_playoff_pred[n] = poisson_log_rng(mu);
  }
}
