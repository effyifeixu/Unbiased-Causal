Metalearner-Based Average Treatment Effect Estimation: MLR vs. UMLR
================

## Packages

``` r
library(quadprog)
library(MASS)
library(knitr)
library(glmnet)
library(DoubleML)
library(mlr3)
library(mlr3learners)
```

------------------------------------------------------------------------

## Algorithms

### Gaussian Kernel

``` r
gaussian_kernel <- function(X1, X2, sigma) {
  X1 <- as.matrix(X1);  X2 <- as.matrix(X2)
  combined <- rbind(X1, X2)
  sq_dist  <- as.matrix(dist(combined))^2
  n1 <- nrow(X1)
  K <- sq_dist[seq_len(n1), -seq_len(n1), drop = FALSE]
  exp(-K / (2 * sigma^2))
}
```

### Kernel Ridge Regression (KRR)

``` r
kernel_ridge_regression <- function(X, y, lambda, sigma) {
  X <- as.matrix(X)
  K <- gaussian_kernel(X, X, sigma)
  n <- nrow(X)
  alpha <- solve(K + lambda * diag(n)) %*% y
  fit <- as.vector(K %*% alpha)
  list(alpha = alpha, X_train = X, sigma = sigma,
       predict = fit, lambda = lambda)
}

predict_krr <- function(model, X_new) {
  K_new <- gaussian_kernel(as.matrix(X_new), model$X_train, model$sigma)
  as.vector(K_new %*% model$alpha)
}
```

### Kernel Ridge Regression — Unbiased Prediction (KRR-UP)

``` r
kernel_ridge_regression_eq <- function(X, y, lambda, sigma) {
  X <- as.matrix(X)
  K <- gaussian_kernel(X, X, sigma)
  n <- nrow(X)

  Dmat <- K + lambda * diag(n)
  dvec <- y

  my <- mean(y)
  young <- which(y <= my)
  old   <- which(y >  my)

  K_young <- gaussian_kernel(matrix(X[young, ], nrow = length(young)), X, sigma)
  K_old <- gaussian_kernel(matrix(X[old,   ], nrow = length(old)),   X, sigma)

  A1 <- matrix(rep(1, length(young)), nrow = 1) %*% K_young
  A2 <- matrix(rep(1, length(old)),   nrow = 1) %*% K_old

  Amat <- rbind(A1, A2)
  bvec <- c(sum(y[young]), sum(y[old]))

  sol <- solve.QP(Dmat, dvec, t(Amat), bvec, meq = 2)
  alpha <- sol$solution
  fit <- as.vector(K %*% alpha)

  list(alpha = alpha, X_train = X, sigma = sigma,
       predict = fit, lambda = lambda, y_train = y)
}

predict_krrup <- function(model, X_new) {
  K_new <- gaussian_kernel(as.matrix(X_new), model$X_train, model$sigma)
  as.vector(K_new %*% model$alpha)
}
```

### Hyperparameter Tuning

``` r
tune_krr <- function(X, y, lambdas, sigmas, method = c("KRR", "KRR-UP")) {
  method <- match.arg(method)
  best_crit <- Inf
  best_par <- list(lambda = lambdas[1], sigma = sigmas[1])
  best_fit <- NULL

  for (sig in sigmas) {
    for (lam in lambdas) {
      tryCatch({
        if (method == "KRR") {
          fit   <- kernel_ridge_regression(X, y, lam, sig)
          resid <- fit$predict - y
          crit  <- sqrt(mean(resid^2))
        } else if (method == "KRR-UP") {
          fit   <- kernel_ridge_regression_eq(X, y, lam, sig)
          resid <- fit$predict - y
          crit  <- abs(cor(resid, y))
        }
        if (crit < best_crit) {
          best_crit <- crit
          best_par  <- list(lambda = lam, sigma = sig)
          best_fit  <- fit
        }
      }, error = function(e) NULL)
    }
  }

  list(fit = best_fit, lambda = best_par$lambda, sigma = best_par$sigma,
       crit = best_crit)
}

# Hyperparameter grid
lambdas <- c(0.001, 0.01, 0.1, 1, 5, 10)

# Sigma scaled around sqrt(p) as median heuristic baseline for high dimensions
p <- 200
sigmas <- c(1, 2, 5, 10, sqrt(p)) * sqrt(p)
```

------------------------------------------------------------------------

## Data Generation

``` r
n <- 500
p <- 200

X <- matrix(rnorm(n * p, mean = 0, sd = 1), ncol = p)

s <- 5 # sparsity
beta1 <- c( 2,  4, -3,  1.5,  2, rep(0, p - s))
beta0 <- c( 1,  2, -1,  0.5,  1, rep(0, p - s))

gamma <- c(0.5, -0.5, 0.8, -0.2, 0.5, rep(0, p - s))
p_t <- plogis(X %*% gamma)
T <- rbinom(n, 1, prob = p_t)

Y1 <- 2  + X %*% beta1 + rnorm(n, 0, 1)
Y0 <- -2 + X %*% beta0 + rnorm(n, 0, 1)
Y <- ifelse(T == 1, Y1, Y0)

cate_true <- Y1 - Y0
ate_true <- mean(Y1 - Y0)

df <- data.frame(Y = Y, T = T, X)

idx1 <- which(T == 1);  X_t1 <- X[idx1, ];  Y_t1 <- Y[idx1]
idx0 <- which(T == 0);  X_t0 <- X[idx0, ];  Y_t0 <- Y[idx0]
```

------------------------------------------------------------------------

## ATE Estimation via S-Learner

### Tuning

``` r
tune_krr_s <- tune_krr(cbind(X, T), Y, lambdas, sigmas, "KRR")
tune_krrup_s <- tune_krr(cbind(X, T), Y, lambdas, sigmas, "KRR-UP")
```

### Predictions

``` r
# MLR
mu1_hat_krr_s <- predict_krr(tune_krr_s$fit, cbind(X, 1))
mu0_hat_krr_s <- predict_krr(tune_krr_s$fit, cbind(X, 0))
cate_krr_s <- mu1_hat_krr_s - mu0_hat_krr_s
ate_krr_s <- mean(cate_krr_s)
rmse_cate_krr_s <- sqrt(mean((cate_krr_s - cate_true)^2))

# UMLR
mu1_hat_krrup_s <- predict_krrup(tune_krrup_s$fit, cbind(X, 1))
mu0_hat_krrup_s <- predict_krrup(tune_krrup_s$fit, cbind(X, 0))
cate_krrup_s <- mu1_hat_krrup_s - mu0_hat_krrup_s
ate_krrup_s <- mean(cate_krrup_s)
rmse_cate_krrup_s <- sqrt(mean((cate_krrup_s - cate_true)^2))
```

|               | True ATE |     MLR |    UMLR |
|:--------------|---------:|--------:|--------:|
| Estimated ATE |   4.1234 |  1.3025 |  3.7499 |
| ATE Bias      |        — | -2.8209 | -0.3735 |
| RMSE          |        — |  4.5820 |  3.6365 |

Table 1. ATE Estimation Results — S-Learner

------------------------------------------------------------------------

## ATE Estimation via T-Learner

### Tuning

``` r
tune_krr_t1 <- tune_krr(X_t1, Y_t1, lambdas, sigmas, "KRR")
tune_krr_t0 <- tune_krr(X_t0, Y_t0, lambdas, sigmas, "KRR")
tune_krrup_t1 <- tune_krr(X_t1, Y_t1, lambdas, sigmas, "KRR-UP")
tune_krrup_t0 <- tune_krr(X_t0, Y_t0, lambdas, sigmas, "KRR-UP")
```

### Predictions

``` r
# MLR
mu1_hat_krr_t <- predict_krr(tune_krr_t1$fit, X)
mu0_hat_krr_t <- predict_krr(tune_krr_t0$fit, X)
cate_krr_t <- mu1_hat_krr_t - mu0_hat_krr_t
ate_krr_t <- mean(cate_krr_t)
rmse_cate_krr_t <- sqrt(mean((cate_krr_t - cate_true)^2))

# UMLR
mu1_hat_krrup_t <- predict_krrup(tune_krrup_t1$fit, X)
mu0_hat_krrup_t <- predict_krrup(tune_krrup_t0$fit, X)
cate_krrup_t <- mu1_hat_krrup_t - mu0_hat_krrup_t
ate_krrup_t <- mean(cate_krrup_t)
rmse_cate_krrup_t <- sqrt(mean((cate_krrup_t - cate_true)^2))
```

|               | True ATE |     MLR |    UMLR |
|:--------------|---------:|--------:|--------:|
| Estimated ATE |   4.1234 |  3.4788 |  4.0964 |
| ATE Bias      |        — | -0.6446 | -0.0270 |
| RMSE          |        — |  3.1599 |  2.0527 |

Table 2. ATE Estimation Results — T-Learner

------------------------------------------------------------------------

## ATE Estimation via X-Learner

### Tuning

``` r
# Stage 1: Fit outcome models on each arm (same as T-Learner)

# Stage 2: Impute pseudo-outcomes
D1_krr <- Y_t1 - mu0_hat_krr_t[idx1]
D0_krr <- mu1_hat_krr_t[idx0] - Y_t0
tune_krr_d1 <- tune_krr(X_t1, D1_krr, lambdas, sigmas, "KRR")
tune_krr_d0 <- tune_krr(X_t0, D0_krr, lambdas, sigmas, "KRR")

D1_krrup <- Y_t1 - mu0_hat_krrup_t[idx1]
D0_krrup <- mu1_hat_krrup_t[idx0] - Y_t0
tune_krrup_d1 <- tune_krr(X_t1, D1_krrup, lambdas, sigmas, "KRR-UP")
tune_krrup_d0 <- tune_krr(X_t0, D0_krrup, lambdas, sigmas, "KRR-UP")
```

### Predictions

``` r
# Stage 3: Combine using propensity score
# Propensity Score
ps_lasso <- cv.glmnet(X, T, family = "binomial", alpha = 1, nfolds = 5)
e_hat_lasso <- as.vector(predict(ps_lasso, newx = X, s = "lambda.min", type = "response"))

# MLR
tau1_krr <- predict_krr(tune_krr_d1$fit, X)   
tau0_krr <- predict_krr(tune_krr_d0$fit, X)   
cate_krr_x <- e_hat_lasso * tau0_krr + (1 - e_hat_lasso) * tau1_krr 
ate_krr_x <- mean(cate_krr_x)
rmse_cate_krr_x <- sqrt(mean((cate_krr_x - cate_true)^2))

# UMLR
tau1_krrup <- predict_krrup(tune_krrup_d1$fit, X)
tau0_krrup <- predict_krrup(tune_krrup_d0$fit, X)
cate_krrup_x <- e_hat_lasso * tau0_krrup + (1 - e_hat_lasso) * tau1_krrup 
ate_krrup_x <- mean(cate_krrup_x)
rmse_cate_krrup_x <- sqrt(mean((cate_krrup_x - cate_true)^2))
```

|               | True ATE |     MLR |   UMLR |
|:--------------|---------:|--------:|-------:|
| Estimated ATE |   4.1234 |  3.7801 | 4.2229 |
| ATE Bias      |        — | -0.3433 | 0.0995 |
| RMSE          |        — |  2.5913 | 1.9503 |

Table 3. ATE Estimation Results — X-Learner

------------------------------------------------------------------------

## Doubly Robust ATE Estimation

``` r
aipw_score <- function(Y, T, mu1, mu0, e_hat) {
  psi <- (mu1 - mu0) + (T / e_hat) * (Y - mu1) - ((1 - T) / (1 - e_hat)) * (Y - mu0)
  return(psi)
}

# MLR-T-Learner + PS
cate_krr_aipw <- (mu1_hat_krr_t - mu0_hat_krr_t) + 
  (T / e_hat_lasso) * (Y - mu1_hat_krr_t) - ((1 - T) / (1 - e_hat_lasso)) * (Y - mu0_hat_krr_t)
ate_krr_aipw <- mean(cate_krr_aipw)
rmse_cate_krr_aipw <- sqrt(mean((cate_krr_aipw - cate_true)^2))

# UMLR-T-Learner + PS
cate_krrup_aipw <- (mu1_hat_krrup_t - mu0_hat_krrup_t) + 
  (T / e_hat_lasso) * (Y - mu1_hat_krrup_t) - ((1 - T) / (1 - e_hat_lasso)) * (Y - mu0_hat_krrup_t)
ate_krrup_aipw <- mean(cate_krrup_aipw)
rmse_cate_krrup_aipw <- sqrt(mean((cate_krrup_aipw - cate_true)^2))

# Double ML
# Initialize DoubleMLData 
x_cols <- grep("^X", colnames(df), value = TRUE)

dml_data <- DoubleMLData$new(
  data = df,
  y_col = "Y",
  d_cols = "T",
  x_cols = x_cols     
)

# Initialize learners
lasso = lrn("regr.cv_glmnet", nfolds = 5, s = "lambda.min")
lasso_class = lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min")

# Initialize DoubleMLPLR model
dml_fit <- DoubleMLPLR$new(
  data = dml_data,
  ml_l = lasso,
  ml_m = lasso_class,
  n_folds = 3
)

dml_fit$fit()
ate_dml <- as.numeric(dml_fit$coef)
se_dml <- as.numeric(dml_fit$se)
```

|               | True ATE |     MLR |    UMLR | Double ML |
|:--------------|---------:|--------:|--------:|----------:|
| Estimated ATE |   4.1234 |  3.4795 |  4.0925 |    3.8722 |
| ATE Bias      |        — | -0.6438 | -0.0308 |   -0.2512 |
| RMSE          |        — |  3.1648 |  2.0373 |         — |

Table 4. ATE Estimation Results — Doubly Robust
