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
  vector[N_questions]    beta;       // Question difficulties
  vector[N_students - 1] alpha_free; // Relative student abilities
}

transformed parameters {
  vector[N_students] alpha = append_row([0]', alpha_free);
}

model {
  // Prior model
  beta       ~ normal(0, 3 / 2.32); // -3 <~  beta[i] <~ +3
  alpha_free ~ normal(0, 3 / 2.32); // -3 <~ alpha[j] <~ +3

  // Observational model
  y ~ bernoulli_logit(alpha[student_idx] - beta[question_idx]);
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

      int y_pred = bernoulli_logit_rng(alpha[j] - beta[i]);

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
