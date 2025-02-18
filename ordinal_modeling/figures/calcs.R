################################################################################
# Dirichlet samples
################################################################################


K <- 5
rho <- c(0.1, 0.2, 0.4, 0.2, 0.1)
tau <- 0.05

alpha <- rho / tau + rep(1, K)

ps <- sapply(alpha, function(a) rgamma(1, a, 1))
ps <- ps / sum(ps)
cat(sprintf("%.5f,", ps))

################################################################################
# Cut points to ordinal probabilities
################################################################################

cut_points <- c(-1.06, -0.4, 0.4, 1.06)
K <- length(cut_points) + 1

mu <- +0.25
sigma <- 0.8

cdf <- function(x) {
  pnorm(x, mu, sigma)
}
cdf_evals <- c(0, cdf(cut_points), 1)

ps <- cdf_evals[2:(K + 1)] - cdf_evals[1:K]
cat(sprintf("%.5f,", ps))

################################################################################
# Ordinal probabilities to cut points
################################################################################

# Target probabilities
ps <- c(0.05, 0.325, 0.25, 0.325, 0.05)
cum_probs <- cumsum(ps)

K <- length(ps)

xs <- seq(-2, 2, 0.00001)

# Unimodal latent
sigma <- 0.5

cdf <- function(x) {
  pnorm(x, 0, sigma)
}
cdf_evals <- cdf(xs)

cut_points <- sapply(cum_probs[1:(K - 1)], 
                     function(P) xs[which(cdf_evals > P)[1]])
cat(sprintf("%.5f,", cut_points))


# Bimodal latent
l <- 0.6
mu1 <- -0.75
mu2 <- +0.75
sigma <- 0.4

cdf <- function(x) {
  l * pnorm(x, mu1, sigma) + (1 - l) * pnorm(x, mu2, sigma)
}

cdf_evals <- cdf(xs)

cut_points <- sapply(cum_probs[1:(K - 1)], 
                     function(P) xs[which(cdf_evals > P)[1]])
cat(sprintf("%.5f,", cut_points))
