data {
  int<lower=1> N_players; // Number of players
  int<lower=1> N_games;   // Number of games

  // Game outcomes
  array[N_games] int<lower=1, upper=N_players> player1_idx;
  array[N_games] int<lower=1, upper=N_players> player2_idx;
  array[N_games] int<lower=0, upper=1> y;

  // Anchor configuration
  int<lower=0, upper=N_players - 1> N_anchors;
  array[N_anchors] int<lower=1, upper=N_players> anchor_idx;
}

parameters {
  // Relative player skills
  vector[N_players - N_anchors] delta_free;
}

transformed parameters {
  // Relative player skills with anchor skills
  vector[N_players] delta = rep_vector(0, N_players);

  {
    array[N_anchors + 2] int boundary_idx
      = append_array(append_array({0}, anchor_idx), {N_players + 1});

    for (n in 1:(N_anchors + 1)) {
      int idx1 = boundary_idx[n]     + 1;
      int idx2 = boundary_idx[n + 1] - 1;
      if (idx1 <= idx2) {
        delta[idx1:idx2] = delta_free[(idx1 - n + 1):(idx2 - n + 1)];
      }
    }
  }
}

model {
  // Prior model
  // -sqrt(2) * 10 <~ beta_free[i] <~ + sqrt(2) * 10
  delta_free ~ normal(0, sqrt(2) * 10 / 2.32);

  // Observational model
  y ~ bernoulli_logit(delta[player1_idx] - delta[player2_idx]);
}

generated quantities {
  array[N_games] int<lower=0, upper=1> y_pred;
  array[N_players] int<lower=0> win_counts_pred
    = rep_array(0, N_players);
  array[N_players] int<lower=0> loss_counts_pred
    = rep_array(0, N_players);
  array[N_players] real win_freq_pred;

  for (n in 1:N_games) {
    int idx1 = player1_idx[n];
    int idx2 = player2_idx[n];

    y_pred[n] = bernoulli_logit_rng(delta[idx1] - delta[idx2]);
    if (y_pred[n]) {
      win_counts_pred[idx1] += 1;
      loss_counts_pred[idx2] += 1;
    } else {
      win_counts_pred[idx2] += 1;
      loss_counts_pred[idx1] += 1;
    }
  }

  for (i in 1:N_players) {
    win_freq_pred[i] =   (1.0 * win_counts_pred[i])
                       / (win_counts_pred[i] + loss_counts_pred[i]);
  }
}
