data {
  int<lower=1> N_questions; // Number of questions
  int<lower=1> N_students;  // Number of students
}

transformed data {
  int<lower=1> N_answers = N_students * 100; // Number of answers
}

generated quantities {
  // Question difficulties
  array[N_questions] real beta;

  // Question discriminations
  array[N_questions] real<lower=0> gamma
    = abs(normal_rng(rep_vector(0, N_questions), 2 / 2.57));

  // Student abilities
  array[N_students] real alpha;

  // Responses
  array[N_answers] int<lower=1, upper=N_questions> question_idx;
  array[N_answers] int<lower=1, upper=N_students>  student_idx;
  array[N_answers] int<lower=0, upper=1> y;

  // Varying difficulties across questions that are shared across
  // tests and unique to each test
  beta[1:50]    = normal_rng(rep_vector(-1, 50), 1.5 / 2.32);
  beta[51:100]  = normal_rng(rep_vector( 0, 50), 1.5 / 2.32);
  beta[101:150] = normal_rng(rep_vector(+1, 50), 1.5 / 2.32);

  // Varying abilities across student cohorts
  alpha[1:250]   = normal_rng(rep_vector(0.75, 250), 1.5 / 2.32);
  alpha[251:500] = normal_rng(rep_vector(2.25, 250), 1.5 / 2.32);

  {
    int a_idx = 1;

    // Test One
    for (j in 1:(N_students / 2)) {
      for (i in 1:100) {
        question_idx[a_idx] = i;
        student_idx[a_idx] = j;
        y[a_idx] = bernoulli_logit_rng(gamma[i] * (alpha[j] - beta[i]));
        a_idx += 1;
      }
    }

    // Test Two
    for (j in (N_students / 2 + 1):N_students) {
      for (i in 51:150) {
        question_idx[a_idx] = i;
        student_idx[a_idx] = j;
        y[a_idx] = bernoulli_logit_rng(gamma[i] * (alpha[j] - beta[i]));
        a_idx += 1;
      }
    }
  }
}
