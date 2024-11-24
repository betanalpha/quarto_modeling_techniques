data {
  int<lower=1> N_players; // Number of players
  int<lower=1> N_games;   // Number of games

  // Matchup probabilities
  simplex[N_players] player1_probs;
  array[N_players] simplex[N_players] pair_probs;
}

generated quantities {
  // Player skills
  array[N_players] real alpha
    = normal_rng(rep_vector(0, N_players), log(sqrt(2)) / 2.32);

  // Game outcomes
  array[N_games] int<lower=1, upper=N_players> player1_idx;
  array[N_games] int<lower=1, upper=N_players> player2_idx;
  array[N_games] int<lower=0, upper=1> y;

  for (n in 1:N_games) {
    // Simulate first player
    player1_idx[n] = categorical_rng(player1_probs);

    // Simulate second player
    player2_idx[n] = categorical_rng(pair_probs[player1_idx[n]]);

    // Simulate winner
    y[n] = bernoulli_logit_rng(  alpha[player1_idx[n]]
                               - alpha[player2_idx[n]]);
  }
}
