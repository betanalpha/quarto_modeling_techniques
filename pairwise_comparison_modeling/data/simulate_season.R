################################################################################
# Setup
################################################################################

par(family="serif", las=1, bty="l",
    cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 1))

library(rstan)
rstan_options(auto_write = TRUE)            # Cache compiled Stan programs
options(mc.cores = parallel::detectCores()) # Parallelize chains
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

util <- new.env()
source('mcmc_analysis_tools_rstan.R', local=util)
source('mcmc_visualization_tools.R', local=util)

################################################################################
# Configure season schedule
################################################################################

# Construct quadruple round robin pairings
N_teams <- 10
N_rounds <- (45 / 5) * 8


home <- c()
away <- c()
round <- c()

t <- 1:N_teams

for (n in 1:N_rounds) {
  round <- c(round, rep(n, N_teams / 2))
  
  # Allocate teams
  if (n %% 2 == 1) {
    home <- c(home, head(t, N_teams / 2))
    away <- c(away, rev(tail(t, N_teams / 2)))
  } else {
    home <- c(home, rev(tail(t, N_teams / 2)))
    away <- c(away, head(t, N_teams / 2))
  }
  # Permute configuration around anchor
  t <- c( t[1], t[N_teams], t[2:(N_teams - 1)])
}

# Randomly permute rounds
set.seed(25848384)

round_week <- sample(1:N_rounds, N_rounds)
week <- round_week[round]

s <- sort(week, index.return=TRUE)

week <- s$x
home <- home[s$ix]
away <- away[s$ix]

N_weeks <- N_rounds
N_games <- length(week)

################################################################################
# Simulate game outcomes
################################################################################

simu <- stan(file="../stan_programs/simu_season.stan",
             algorithm="Fixed_param", seed=8438338,
             data=list("N_games" = N_games,
                       "N_teams" = N_teams,
                       "home_idx" = home,
                       "away_idx" = away,
                       "N_weeks" = N_weeks,
                       "week" = week),
             warmup=0, iter=1, chains=1, refresh=0)

data <- list("N_games" = N_games,
             "N_teams" = N_teams,
             "home_idx" = home,
             "away_idx" = away,
             "N_weeks" = N_weeks,
             "week" = week,
             "y_home" = extract(simu)$y_home[1,],
             "y_away" = extract(simu)$y_away[1,])

alpha_true <- extract(simu)$alpha
beta_off_true <- extract(simu)$beta_off
beta_def_true <- extract(simu)$beta_def
eta_off_true <- extract(simu)$eta_off
eta_def_true <- extract(simu)$eta_def
gamma_true <- extract(simu)$gamma

home_idx <- home
away_idx <- away
y_home <- extract(simu)$y_home[1,]
y_away <- extract(simu)$y_away[1,]

stan_rdump(c("N_games", "N_teams",
             "home_idx", "away_idx",
             "N_weeks", "week", 
             "y_home", "y_away"), file="season.data.R")

data <- read_rdump('season.data.R')

# Playoffs
wins <- rep(0, data$N_teams)
ties <- rep(0, data$N_teams)
losses <- rep(0, data$N_teams)

for (n in 1:data$N_games) {
  if (data$y_home[n] > data$y_away[n]) {
    wins[data$home_idx[n]] <- wins[data$home_idx[n]] + 1
    losses[data$away_idx[n]] <- losses[data$away_idx[n]] + 1
  } else if (data$y_home[n] < data$y_away[n]) {
    losses[data$home_idx[n]] <- losses[data$home_idx[n]] + 1
    wins[data$away_idx[n]] <- wins[data$away_idx[n]] + 1
  } else if (data$y_home[n] == data$y_away[n]) {
    ties[data$home_idx[n]] <- ties[data$home_idx[n]] + 1
    ties[data$away_idx[n]] <- ties[data$away_idx[n]] + 1
  }
}


standings <- data.frame(1:data$N_teams, wins, losses, ties)
names(standings) <- c("Team", "Wins", "Losses", "Ties")

print(standings[rev(order(wins, ties)),], row.names=FALSE)

# First Place: Team 8
# Second Place: Team 9
# Third Place: Team 7
# Fourth Place: Team 5

# First round of play-offs
#   Team 8 (Home) vs Team 5 (Away)
#   Team 9 (Home) vs Team 7 (Away)

# Game One
s_home <- exp(alpha_true + eta_off_true +
              gamma_true**(data$N_weeks + 1) * (beta_off_true[8] - beta_def_true[5]))

s_away <- exp(alpha_true - eta_def_true +
              gamma_true**(data$N_weeks + 1) * (beta_off_true[5] - beta_def_true[8]))

s_home - s_away

# Hypothetical Bet
# Team 9 gives 21 points to Team 5
# Bet 110, win 100 if y_home      > y_away + 21
# Bet 110, win 100 if y_away + 21 < y_home

# Game Two
s_home <- exp(alpha_true + eta_off_true +
                gamma_true**(data$N_weeks + 1) * (beta_off_true[9] - beta_def_true[7]))

s_away <- exp(alpha_true - eta_def_true +
                gamma_true**(data$N_weeks + 1) * (beta_off_true[7] - beta_def_true[9]))

s_home - s_away

# Hypothetical Bet
# Team 8 gives 12 points to Team 7
# Bet 110, win 100 if y_home      > y_away + 12
# Bet 110, win 100 if y_away + 12 < y_home