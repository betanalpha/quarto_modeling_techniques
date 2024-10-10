lpdf1 <- function(q) {
  mu <- -1.75
  sigma <- 0.2
  lpdf <- -0.5 * ((q - mu) / sigma)**2
  lpdf <- lpdf - 0.5 * log(6.283185307179586) - log(sigma)
  lpdf
}

lpdf2 <- function(q) {
  mu <- 1
  sigma <- 0.4
  lpdf <- -0.5 * ((q - mu) / sigma)**2
  lpdf <- lpdf - 0.5 * log(6.283185307179586) - log(sigma)
  lpdf
}

target_lpdf <- function(x) {
  x1 <- log(0.03) + lpdf1(x)
  x2 <- log(0.97) + lpdf2(x)
  if (x1 > x2) {
    lpdf <- x1 + log(1 + exp(x2 - x1))
  } else {
    lpdf <- x2 + log(1 + exp(x1 - x2))
  }
  (lpdf)
}

xs <- seq(-3, 3, 0.1)
ps <- sapply(xs, function(x) exp(target_lpdf(x)))
plot(xs, ps)

n_transitions <- 300
set.seed(5849586)

sigma <- 0.1

D <- 1
mcmc_samples <- matrix(0, nrow=n_transitions + 1, ncol=3)

# Seed the Markov chain from a diffuse sample
#mcmc_samples[1, 1] <- rnorm(1, 0, 3)
mcmc_samples[1, 1] <- -2
mcmc_samples[1, 2] <- 0
mcmc_samples[1, 3] <- 1.5

for (n in 1:n_transitions) {
  for (c in 1:3) {
    q0 <- mcmc_samples[n, c] # Initial point
    qp <- rnorm(1, q0, sigma)  # Proposal
  
    # Compute acceptance probability
    accept_prob <- min(1, exp(target_lpdf(qp) - target_lpdf(q0)))
    #mcmc_samples[n, D + 1] <- accept_prob
  
    # Apply Metropolis correction
    u = runif(1, 0, 1)
    if (accept_prob > u)
      mcmc_samples[n + 1, c] <- qp
    else
      mcmc_samples[n + 1, c] <- q0
  }
}

plot(0:n_transitions, mcmc_samples[,1], type="l", col=util$c_dark, ylim=c(-3, 3))
lines(0:n_transitions, mcmc_samples[,2], col=util$c_mid)
lines(0:n_transitions, mcmc_samples[,3], col=util$c_light)

cat(sprintf("%.3f,", mcmc_samples[,1]), "\n")

cat(sprintf("%.3f,", mcmc_samples[,2]), "\n")

cat(sprintf("%.3f,", mcmc_samples[,3]), "\n")
