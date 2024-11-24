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
# Simulate data
################################################################################

N_questions <- 150
N_students <- 500
N_answers <- N_students * 100

simu <- stan(file="../stan_programs/simu_irt_test.stan",
             algorithm="Fixed_param", seed=8438338,
             data=list("N_questions" = N_questions,
                       "N_students" = N_students),
             warmup=0, iter=1, chains=1, refresh=0)

question_idx <- extract(simu)$question_idx[1,]
student_idx <- extract(simu)$student_idx[1,]
y <- extract(simu)$y[1,]

data <- list('N_questions' = N_questions,
             'N_students' = N_students,
             'N_answers' = N_answers,
             'question_idx' = question_idx,
             'student_idx' = student_idx,
             'y' = y)

stan_rdump(c('N_questions', 'N_students', 'N_answers',
             'question_idx', 'student_idx', 'y'), 
           'irt.data.R')

beta_true <- extract(simu)$beta[1,]
gamma_true <- extract(simu)$gamma[1,]
alpha_true <- extract(simu)$alpha[1,]

truth <- list('beta_true' = beta_true,
              'alpha_true' = alpha_true)

stan_rdump(c('beta_true', 'gamma_true', 'alpha_true'), 
           'irt.truth.R')