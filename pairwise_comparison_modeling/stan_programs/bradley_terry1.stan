data {
  int<lower=1> N_players; // Number of players
  int<lower=1> N_games;   // Number of games

  // Game outcomes
  array[N_games] int<lower=1, upper=N_players> player1_idx;
  array[N_games] int<lower=1, upper=N_players> player2_idx;
  array[N_games] int<lower=0, upper=1> y;
}

parameters {
  vector[N_players] alpha; // Player skills
}

model {
  // Implicit uniform prior density function for the alpha components

  // Observational model
  y ~ bernoulli_logit(alpha[player1_idx] - alpha[player2_idx]);
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

    y_pred[n] = bernoulli_logit_rng(alpha[idx1] - alpha[idx2]);
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
