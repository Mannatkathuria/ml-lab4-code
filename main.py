import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv("ic272_lab5_rent.csv")

features = [c for c in df.columns if c != "rent"]

X = df[features].to_numpy(float)
y = df["rent"].to_numpy(float)

idx = np.random.default_rng(0).permutation(len(y))

tr, te = idx[:40], idx[40:]

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

print("train:", X_train.shape, " test:", X_test.shape)


# ------------------------------------------------------------
# Lab 4 functions
# ------------------------------------------------------------

def design_matrix(X):
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X, y):
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ y)


# ============================================================
# TASK 1 — STANDARDISATION
# ============================================================

def standardise(X_train, X_test):
    # Calculate mean and standard deviation
    # separately for every feature/column
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)

    # Standardise BOTH using training mean and training std
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    return X_train_std, X_test_std, mu, sigma


# Standardise the data
X_train_std, X_test_std, mu, sigma = standardise(
    X_train, X_test
)

print("\nTASK 1")
print("Training std:")
print(X_train_std.std(axis=0))

print("\nTest std:")
print(X_test_std.std(axis=0))


# ============================================================
# TASK 2 — RIDGE REGRESSION
# ============================================================

def fit_ridge(X, y, lam):
    # Add intercept column
    A = design_matrix(X)

    # Identity matrix
    I = np.eye(A.shape[1])

    # Do NOT penalise intercept
    I[0, 0] = 0

    # Ridge solution:
    # theta = (A^T A + lambda I)^(-1) A^T y
    theta = np.linalg.solve(
        A.T @ A + lam * I,
        A.T @ y
    )

    return theta


print("\nTASK 2")

lambdas = [0, 1, 10, 100]

for lam in lambdas:
    theta = fit_ridge(X_train_std, y_train, lam)

    # Count coefficients that are effectively zero
    zero_count = np.sum(np.abs(theta) < 1e-8)

    print(f"\nlambda = {lam}")
    print("coefficients:")
    print(theta)
    print("number of exact zeros:", zero_count)


# Check lambda = 0 against normal equation
theta_ridge_0 = fit_ridge(X_train_std, y_train, 0)
theta_normal = fit_normal_equation(X_train_std, y_train)

print("\nCheck lambda = 0:")
print("Same as normal equation:",
      np.allclose(theta_ridge_0, theta_normal))


# ============================================================
# TASK 3 — K-FOLD CROSS VALIDATION
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def k_fold_cv(X, y, lam, k=5, seed=0):

    # Shuffle row indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))

    # Split indices into k folds
    folds = np.array_split(indices, k)

    errors = []

    for i in range(k):

        # Current fold = validation set
        val_idx = folds[i]

        # All other folds = training set
        train_folds = folds[:i] + folds[i+1:]
        train_idx = np.concatenate(train_folds)

        X_tr = X[train_idx]
        y_tr = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        # Fit ridge model
        theta = fit_ridge(X_tr, y_tr, lam)

        # Predict validation set
        A_val = design_matrix(X_val)
        y_pred = A_val @ theta

        # Calculate validation RMSE
        error = rmse(y_val, y_pred)

        errors.append(error)

    # Average error across all folds
    return np.mean(errors)


# ------------------------------------------------------------
# Evaluate different lambda values
# ------------------------------------------------------------

lambda_values = [0, 0.1, 0.3, 1, 3, 10, 30, 100]

cv_results = []

for lam in lambda_values:

    score = k_fold_cv(
        X_train_std,
        y_train,
        lam,
        k=5,
        seed=0
    )

    cv_results.append(score)

print("\nTASK 3")
print("\nLambda\tCV RMSE")
print("----------------")

for lam, score in zip(lambda_values, cv_results):
    print(f"{lam:<8} {score:.6f}")


# Find best lambda
best_index = np.argmin(cv_results)
best_lambda = lambda_values[best_index]

print("\nBest lambda:", best_lambda)


# ============================================================
# TEST SET — ONLY AFTER CHOOSING LAMBDA
# ============================================================

# lambda = 0
theta_0 = fit_ridge(
    X_train_std,
    y_train,
    0
)

y_test_pred_0 = design_matrix(X_test_std) @ theta_0

test_rmse_0 = rmse(
    y_test,
    y_test_pred_0
)


# Chosen lambda
theta_best = fit_ridge(
    X_train_std,
    y_train,
    best_lambda
)

y_test_pred_best = design_matrix(X_test_std) @ theta_best

test_rmse_best = rmse(
    y_test,
    y_test_pred_best
)


print("\nTest RMSE")
print("lambda = 0:      ", test_rmse_0)
print(f"lambda = {best_lambda}: ", test_rmse_best)