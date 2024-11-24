data {
  int<lower=1> N_questions; // Number of questions
  int<lower=1> N_students;  // Number of students
  int<lower=1> N_answers;   // Number of answers

  // Testing configuration and outcomes
  array[N_answers] int<lower=1, upper=N_questions> question_idx;
  array[N_answers] int<lower=1, upper=N_students>  student_idx;
  array[N_answers] int<lower=0, upper=1> y;
}

parameters {
  // Relative question discriminations
  vector<lower=0>[N_questions - 1] gamma_free;

  vector[N_questions] beta;  // Question difficulties

  // Non-centered, relative student abilities
  vector[N_students] alpha_tilde;

  real delta_mu;       // Difference in student cohort ability locations
  real<lower=0> tau1;  // First student cohort ability variation
  real<lower=0> tau2;  // Second student cohort ability variation
}

transformed parameters {
  vector[N_questions] gamma = append_row([1]', gamma_free);

  vector[N_students] alpha;
  alpha[1:(N_students / 2)]
    = tau1 * alpha_tilde[1:(N_students / 2)];
  alpha[(N_students / 2):N_students]
    = delta_mu + tau2 * alpha_tilde[(N_students / 2):N_students];
}

model {
  // Prior model
  gamma_free ~ normal(0, 10 / 2.57);
  beta ~ normal(0, 3 / 2.32);

  alpha_tilde ~ normal(0, 1);

  delta_mu ~ normal(0, 3 / 2.32);
  tau1 ~ normal(0, 3 / 2.57);
  tau2 ~ normal(0, 3 / 2.57);

  // Observational model
  y ~ bernoulli_logit(   gamma[question_idx]
                      .* (alpha[student_idx] - beta[question_idx]) );
}

generated quantities {
  array[N_questions] real mean_q_pred = rep_array(0, N_questions);
  array[N_questions] real var_q_pred  = rep_array(0, N_questions);
  array[N_students]  real mean_s_pred = rep_array(0, N_students);
  array[N_students]  real var_s_pred  = rep_array(0, N_students);

  {
    array[N_questions] real N_answers_q = rep_array(0, N_questions);
    array[N_students]  real N_answers_s = rep_array(0, N_students);

    for (n in 1:N_answers) {
      int i = question_idx[n];
      int j = student_idx[n];
      real delta = 0;

      int y_pred = bernoulli_logit_rng(gamma[i] * (alpha[j] - beta[i]));

      N_answers_q[i] += 1;
      delta = y_pred - mean_q_pred[i];
      mean_q_pred[i] += delta / N_answers_q[i];
      var_q_pred[i] += delta * (y_pred - mean_q_pred[i]);

      N_answers_s[j] += 1;
      delta = y_pred - mean_s_pred[j];
      mean_s_pred[j] += delta / N_answers_s[j];
      var_s_pred[j] += delta * (y_pred - mean_s_pred[j]);
    }

    for (i in 1:N_questions)
      var_q_pred[i] /= N_answers_q[i] - 1;
    for (j in 1:N_students)
      var_s_pred[j] /= N_answers_s[j] - 1;
  }
}
