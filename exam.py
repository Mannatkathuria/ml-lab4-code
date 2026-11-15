"""IC-272 Open-Book Exam Template

Reusable NumPy / pandas / Matplotlib implementations for the supplied Labs 1-9
and homework. Keep this file beside your exam notebook and copy only the pieces
you need. The handouts prohibit scikit-learn / SciPy for several tasks, so this
template uses NumPy, pandas, and Matplotlib only.

Recommended order for a question:
1) load data -> 2) split -> 3) fit preprocessing on training data only
-> 4) fit/choose model using training/CV data -> 5) evaluate once on test data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# 1. NumPy foundations (Lab 1) and small general helpers
# ============================================================================

def column_stats(X):
    """Return column means, population stds, minima, maxima (four 1-D arrays)."""
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0)


def cap_above_and_count_rows(X, threshold):
    """Return a capped copy and the count above threshold in each row."""
    X = np.asarray(X)
    capped = np.minimum(X, threshold).copy()  # np.minimum makes a new array
    row_counts = (X > threshold).sum(axis=1)
    return capped, row_counts


def pairwise_distances(A, B):
    """Euclidean distances, shape (len(A), len(B)); broadcasting only."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]  # (n, m, d), all pair differences
    return np.sqrt(np.sum(diff * diff, axis=2))


def closest_indices(D):
    """Index of the closest column in each row of a distance matrix."""
    return np.argmin(D, axis=1)


def _safe_divide(numerator, denominator):
    """Division that returns zero where a metric denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    np.divide(numerator, denominator, out=out, where=(denominator != 0))
    return out


# ============================================================================
# 2. Confusion matrices and classification metrics (Labs 1-3, 6-8 homework)
# ============================================================================

def confusion_matrix(y_true, y_pred, labels=None):
    """Counts with true labels on rows and predicted labels on columns."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    lookup = {label: i for i, label in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in lookup or predicted not in lookup:
            raise ValueError("A label is missing from the labels argument")
        cm[lookup[actual], lookup[predicted]] += 1
    return cm


def accuracy_score_scratch(y_true, y_pred):
    """Fraction of predictions equal to the true labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1_from_cm(cm):
    """Per-class precision, recall, F1 and macro-F1 from a square confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1, float(np.mean(f1))


def precision_recall_f1(y_true, y_pred, positive_label=1):
    """Binary metrics for one positive label; zero denominators produce zero."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    precision = float(_safe_divide(tp, tp + fp))
    recall = float(_safe_divide(tp, tp + fn))
    f1 = float(_safe_divide(2 * precision * recall, precision + recall))
    return precision, recall, f1


def binary_metrics(y_true, y_pred, positive_label=1):
    """Return accuracy, precision, recall, F1 and the 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label)
    return accuracy_score_scratch(y_true, y_pred), p, r, f1, cm


# ============================================================================
# 3. Splits and pandas preprocessing (Labs 2, 4-8)
# ============================================================================

def train_test_split(X, y, test_size=0.2, seed=42):
    """Lab 2 convention: first int(n*test_size) shuffled rows are the test set."""
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_split_scratch(X, y, test_frac=0.2, seed=0):
    """Lab 4 named version of the same deterministic shuffled split."""
    return train_test_split(X, y, test_size=test_frac, seed=seed)


def fixed_count_split(X, y, n_test=100, seed=0):
    """Labs 6-8 convention: first n_test shuffled indices are the test rows."""
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    """Approximate class proportions in both splits; returns NumPy arrays."""
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = int(len(idx) * test_size)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot_encode(values, categories=None):
    """Hand-coded one-hot encoding; unknown values become all-zero rows."""
    values = np.asarray(values).ravel()
    if categories is None:
        categories = np.unique(values)
    categories = np.asarray(categories)
    encoded = np.column_stack([values == category for category in categories]).astype(int)
    return encoded, categories


def one_hot_encode_pair(train_values, test_values):
    """Use categories learned from train for both splits (prevents test leakage)."""
    train_encoded, categories = one_hot_encode(train_values)
    test_encoded, _ = one_hot_encode(test_values, categories=categories)
    return train_encoded, test_encoded, categories


def impute_median_train_test(X_train, X_test):
    """Learn each numeric median on train, then apply the same medians to both."""
    X_train = np.asarray(X_train, dtype=float).copy()
    X_test = np.asarray(X_test, dtype=float).copy()
    medians = np.nanmedian(X_train, axis=0)
    rows, cols = np.where(np.isnan(X_train))
    X_train[rows, cols] = medians[cols]
    rows, cols = np.where(np.isnan(X_test))
    X_test[rows, cols] = medians[cols]
    return X_train, X_test, medians


def impute_series_global_median(series):
    """Lab 2 comparison: return globally median-imputed Series and its median."""
    median = series.median()
    return series.fillna(median), median


def impute_series_by_group(df, value_column, group_column):
    """Lab 2 exploration only: median within each observed group."""
    group_medians = df.groupby(group_column)[value_column].transform("median")
    return df[value_column].fillna(group_medians), group_medians


def missing_rate_by_class(df, target_column, feature_columns):
    """Lab 3: missing fraction for each feature within each target class."""
    return df.groupby(target_column)[feature_columns].apply(lambda part: part.isna().mean())


def majority_class_predict(y_train, n_samples):
    """Baseline: repeat the most common training label for every query row."""
    labels, counts = np.unique(y_train, return_counts=True)
    return np.full(n_samples, labels[np.argmax(counts)])


# ============================================================================
# 4. Linear regression, standardisation, ridge, CV and lasso (Labs 4-5)
# ============================================================================

def fit_simple_lr(x, y):
    """Fit y = w*x+b by the one-feature least-squares formulas."""
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denom = np.sum(x_centered ** 2)
    w = 0.0 if denom == 0 else np.sum(x_centered * y_centered) / denom
    b = y.mean() - w * x.mean()
    return w, b


def design_matrix(X):
    """Prepend the intercept column of ones to an (n,d) feature matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X, y):
    """OLS using solve (more stable than explicitly constructing an inverse)."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    return np.linalg.solve(A.T @ A, A.T @ y)


def fit_gradient_descent(X, y, lr, n_iters):
    """Batch descent for J=mean squared error; return theta and post-step losses."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    losses = []
    for _ in range(n_iters):
        error = A @ theta - y
        gradient = (2 / len(y)) * (A.T @ error)
        theta -= lr * gradient
        losses.append(float(np.mean((A @ theta - y) ** 2)))
    return theta, losses


def predict_linear(X, theta):
    """Predict with an intercept-first coefficient vector."""
    return design_matrix(X) @ np.asarray(theta)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score_scratch(y_true, y_pred):
    """R-squared; return zero for a constant target to avoid division by zero."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


def standardise(X_train, X_test):
    """Train-only z-score scaling; constant columns use scale 1 and become zero."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma_safe, (X_test - mu) / sigma_safe, mu, sigma_safe


def unstandardise_coefficients(theta, mu, sigma):
    """Convert z-score-space linear coefficients back to original feature units."""
    theta = np.asarray(theta, dtype=float)
    raw_w = theta[1:] / np.asarray(sigma)
    raw_b = theta[0] - np.sum(raw_w * np.asarray(mu))
    return raw_w, raw_b


def fit_ridge(X, y, lam):
    """Closed-form ridge; intercept is unpenalised."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0  # the intercept is the overall level, not a feature weight
    return np.linalg.solve(A.T @ A + lam * penalty, A.T @ y)


def k_fold_cv(X, y, lam, k=5, seed=0):
    """Mean validation RMSE for ridge; each row is validation exactly once."""
    X, y = np.asarray(X), np.asarray(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        theta = fit_ridge(X[train_idx], y[train_idx], lam)
        scores.append(rmse(y[val_idx], predict_linear(X[val_idx], theta)))
    return float(np.mean(scores))


def ridge_cv_sweep(X, y, lambda_values, k=5, seed=0):
    """Table of (lambda, mean CV RMSE) for choosing ridge regularisation."""
    return np.asarray([(lam, k_fold_cv(X, y, lam, k=k, seed=seed))
                       for lam in lambda_values], dtype=float)


def ridge_coefficient_path(X, y, lambda_values):
    """Coefficient matrix, one row per lambda, for the ridge path plot."""
    return np.asarray([fit_ridge(X, y, lam) for lam in lambda_values])


def soft_threshold(v, t):
    """Vectorised L1 shrinkage: reduce magnitude by t and clamp at zero."""
    v = np.asarray(v)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)


def fit_lasso(X, y, lam, lr=0.05, n_iters=30000):
    """Gradient descent plus soft-thresholding; the intercept is not shrunk."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    for _ in range(n_iters):
        gradient = (2 / len(y)) * (A.T @ (A @ theta - y))
        theta -= lr * gradient
        theta[1:] = soft_threshold(theta[1:], lr * lam)  # keep theta[0] unpenalised
    return theta


# ============================================================================
# 5. Logistic regression, thresholds and ROC/AUC (Labs 3, 7-8)
# ============================================================================

def sigmoid(z):
    """Numerically stable sigmoid; accepts a scalar or an array."""
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, theta):
    """Binary logistic log-likelihood; X excludes the intercept column."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    p = sigmoid(A @ theta)
    eps = 1e-12  # prevents log(0) at extreme probabilities
    return float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def fit_logistic(X, y, lr=0.1, n_iters=2000):
    """Batch gradient ascent for binary logistic regression."""
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    log_likelihoods = []
    for _ in range(n_iters):
        p = sigmoid(A @ theta)
        theta += (lr / len(y)) * (A.T @ (y - p))  # plus sign maximises likelihood
        log_likelihoods.append(log_likelihood(X, y, theta))
    return theta, log_likelihoods


def predict_proba_logistic(X, theta):
    """Probability of class 1 from a fitted logistic coefficient vector."""
    return sigmoid(design_matrix(X) @ theta)


def predict_logistic(X, theta, threshold=0.5):
    """Turn logistic probabilities into 0/1 labels at the chosen cutoff."""
    return (predict_proba_logistic(X, theta) >= threshold).astype(int)


def threshold_metrics(y_true, probabilities, thresholds):
    """Return precision, recall and F1 arrays across a threshold sweep."""
    probabilities = np.asarray(probabilities)
    p_values, r_values, f_values = [], [], []
    for threshold in thresholds:
        labels = (probabilities >= threshold).astype(int)
        p, r, f1 = precision_recall_f1(y_true, labels)
        p_values.append(p); r_values.append(r); f_values.append(f1)
    return np.asarray(p_values), np.asarray(r_values), np.asarray(f_values)


def feature_threshold_sweep(y_true, feature, thresholds, greater_is_positive=True):
    """Lab 3 one-rule classifier sweep; uses strict > as the handout specifies."""
    feature, thresholds = np.asarray(feature), np.asarray(thresholds)
    rows = []
    for threshold in thresholds:
        pred = feature > threshold if greater_is_positive else feature < threshold
        p, r, f1 = precision_recall_f1(y_true, pred.astype(int))
        rows.append((p, r, f1))
    return np.asarray(rows).T  # precision, recall, F1 arrays


def roc_curve_scratch(y_true, p):
    """ROC points in increasing FPR order; tied scores share a threshold."""
    y_true, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    thresholds = np.r_[np.inf, np.unique(p)[::-1], -np.inf]
    tpr, fpr = [], []
    positives, negatives = np.sum(y_true == 1), np.sum(y_true == 0)
    for threshold in thresholds:
        pred = p >= threshold
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        tpr.append(float(_safe_divide(tp, positives)))
        fpr.append(float(_safe_divide(fp, negatives)))
    order = np.argsort(fpr, kind="stable")
    return np.asarray(fpr)[order], np.asarray(tpr)[order]


def auc_trapezoid(fpr, tpr):
    """Area under a ROC curve using trapezoids after sorting by FPR."""
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# ============================================================================
# 6. k-Nearest Neighbours and cross-validation (Lab 6 and homework)
# ============================================================================

def knn_predict(X_train, y_train, X_query, k):
    """Vectorised k-NN vote; no loop over query rows."""
    y_train = np.asarray(y_train)
    D = pairwise_distances(X_query, X_train)  # each query row against all training rows
    neighbour_idx = np.argsort(D, axis=1)[:, :k]
    neighbour_labels = y_train[neighbour_idx]
    classes = np.unique(y_train)
    class_codes = np.searchsorted(classes, neighbour_labels)
    votes = np.eye(len(classes), dtype=int)[class_codes].sum(axis=1)
    return classes[np.argmax(votes, axis=1)]  # sorted class order breaks ties consistently


def knn_predict_weighted(X_train, y_train, X_query, k, eps=1e-12):
    """Distance-weighted k-NN; closer neighbours cast stronger 1/(d+eps) votes."""
    y_train = np.asarray(y_train)
    D = pairwise_distances(X_query, X_train)
    idx = np.argsort(D, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)
    weights = 1 / (d + eps)
    labels = y_train[idx]
    classes = np.unique(y_train)
    votes = np.stack([(weights * (labels == c)).sum(axis=1) for c in classes], axis=1)
    return classes[np.argmax(votes, axis=1)]


def k_fold_cv_knn(X, y, k, folds=5, seed=0):
    """Mean validation accuracy; held-out rows are never passed as neighbours."""
    X, y = np.asarray(X), np.asarray(y)
    groups = np.array_split(np.random.default_rng(seed).permutation(len(y)), folds)
    accuracies = []
    for i in range(folds):
        val_idx = groups[i]
        train_idx = np.concatenate([groups[j] for j in range(folds) if j != i])
        pred = knn_predict(X[train_idx], y[train_idx], X[val_idx], k)
        accuracies.append(accuracy_score_scratch(y[val_idx], pred))
    return float(np.mean(accuracies))


def knn_k_sweep(X, y, k_values, folds=5, seed=0):
    """Return (k, CV accuracy) rows for a k-NN model-selection table."""
    return np.asarray([(k, k_fold_cv_knn(X, y, int(k), folds, seed)) for k in k_values])


def append_gaussian_noise(X_train, X_test, n_noise, seed=0):
    """Homework helper: append independent N(0,1) noise columns to train/test."""
    rng = np.random.default_rng(seed)
    train_noise = rng.normal(0, 1, size=(len(X_train), n_noise))
    test_noise = rng.normal(0, 1, size=(len(X_test), n_noise))
    return np.c_[X_train, train_noise], np.c_[X_test, test_noise]


def distance_concentration_ratio(D):
    """min/max distance: approaches 1 as nearest and farthest become alike."""
    D = np.asarray(D)
    return float(D.min() / D.max()) if D.max() != 0 else 0.0


# ============================================================================
# 7. Gaussian and categorical Naive Bayes (Lab 8 and homework)
# ============================================================================

def gaussian_pdf(x, mu, var):
    """Vectorised Gaussian probability density; tiny variance floor avoids /0."""
    x, mu, var = np.asarray(x), np.asarray(mu), np.maximum(np.asarray(var), 1e-12)
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def fit_gaussian_nb(X, y):
    """Fit two-class Gaussian NB; variances divide by class count (population)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    priors = np.zeros(2)
    means = np.zeros((2, X.shape[1]))
    variances = np.zeros((2, X.shape[1]))
    for c in (0, 1):
        Xc = X[y == c]
        if len(Xc):
            priors[c] = len(Xc) / len(y)
            means[c] = Xc.mean(axis=0)
            variances[c] = Xc.var(axis=0, ddof=0)  # divide by n_c, not n_c-1
    return priors, means, variances


def _gaussian_log_scores(X, priors, means, variances):
    """Internal: log prior plus summed feature-wise Gaussian log densities."""
    X = np.asarray(X, dtype=float)
    safe_var = np.maximum(np.asarray(variances), 1e-12)
    log_density = -0.5 * (np.log(2 * np.pi * safe_var)[None, :, :]
                          + (X[:, None, :] - means[None, :, :]) ** 2
                          / safe_var[None, :, :])
    return np.log(np.asarray(priors)[None, :] + 1e-300) + log_density.sum(axis=2)


def predict_proba_nb(X, priors, means, variances):
    """Return P(class=1|x), comparing Gaussian NB class scores in log space."""
    log_scores = _gaussian_log_scores(X, priors, means, variances)
    return sigmoid(log_scores[:, 1] - log_scores[:, 0])  # stable two-class softmax


def fit_categorical_nb(e, y, l=1):
    """Laplace-smoothed category probabilities; e must be encoded as 0..K-1."""
    e, y = np.asarray(e, dtype=int).ravel(), np.asarray(y, dtype=int).ravel()
    K = 4  # four employment categories in the Lab 8 handout
    if np.any((e < 0) | (e >= K)):
        raise ValueError("Encode the four employment categories as integers 0,1,2,3")
    probabilities = np.zeros((2, K), dtype=float)
    for c in (0, 1):
        mask = y == c
        counts = np.bincount(e[mask], minlength=K)
        probabilities[c] = (counts + l) / (mask.sum() + l * K)
    return probabilities


def predict_proba_nb_with_categorical(X, e, priors, means, variances,
                                       category_probabilities):
    """Combine Gaussian numeric features and categorical log-likelihoods."""
    e = np.asarray(e, dtype=int).ravel()
    scores = _gaussian_log_scores(X, priors, means, variances)
    probs = np.asarray(category_probabilities)
    scores[:, 0] += np.log(probs[0, e] + 1e-300)
    scores[:, 1] += np.log(probs[1, e] + 1e-300)
    return sigmoid(scores[:, 1] - scores[:, 0])


# ============================================================================
# 8. K-means, initialisation, outliers and image segmentation (Lab 9)
# ============================================================================

def random_init(X, K, seed):
    """Choose K distinct rows as initial centroids with a seeded generator."""
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return np.asarray(X)[idx].copy()


def assign_clusters(X, centroids):
    """Assign each row to its nearest centroid."""
    return np.argmin(pairwise_distances(X, centroids), axis=1)


def update_centroids(X, labels, centroids):
    """Replace each centroid by its assigned-point mean; keep empty clusters."""
    X, labels = np.asarray(X), np.asarray(labels)
    updated = np.asarray(centroids, dtype=float).copy()
    for c in range(len(updated)):
        members = X[labels == c]
        if len(members):
            updated[c] = members.mean(axis=0)
    return updated


def distortion(X, labels, centroids):
    """Sum squared distances from points to their assigned centroids."""
    residual = np.asarray(X) - np.asarray(centroids)[np.asarray(labels)]
    return float(np.sum(residual ** 2))


def kmeans(X, init_centroids, max_iter=100):
    """Alternate nearest-centroid assignment and mean update until convergence."""
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            labels = assign_clusters(X, centroids)
            return centroids, labels, distortion(X, labels, centroids), iteration
        centroids = new_centroids
    labels = assign_clusters(X, centroids)
    return centroids, labels, distortion(X, labels, centroids), max_iter


def best_of_kmeans(X, K, n_init=10):
    """Return the lowest-distortion result over seeds 0..n_init-1."""
    best = None
    for seed in range(n_init):
        result = kmeans(X, random_init(X, K, seed))
        if best is None or result[2] < best[2]:
            best = result
    return best


def kmeans_multistart(X, K, n_init=10):
    """Return every seeded K-means result, useful for inspecting local minima."""
    return [kmeans(X, random_init(X, K, seed)) for seed in range(n_init)]


def kmeans_pp_init(X, K, seed):
    """K-means++: sample each next center proportional to squared nearest distance."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(X)))]
    while len(chosen) < K:
        nearest_sq = np.min(pairwise_distances(X, X[chosen]) ** 2, axis=1)
        nearest_sq[chosen] = 0
        total = nearest_sq.sum()
        if total == 0:  # all remaining points coincide with a chosen point
            remaining = np.setdiff1d(np.arange(len(X)), chosen)
            idx = int(rng.choice(remaining))
        else:
            idx = int(rng.choice(len(X), p=nearest_sq / total))
        chosen.append(idx)
    return X[chosen].copy()


def kmeans_elbow_sweep(X, K_values, n_init=10):
    """Return K, best distortion, and iterations for each candidate cluster count."""
    rows = []
    for K in K_values:
        centroids, labels, J, iterations = best_of_kmeans(X, int(K), n_init=n_init)
        rows.append((K, J, iterations))
    return np.asarray(rows, dtype=float)


def segment_image_kmeans(image, K, seed=0, max_iter=100):
    """Cluster RGB pixels and return segmented RGB image plus K-means outputs."""
    image = np.asarray(image)
    rgb = image[:, :, :3]
    h, w, channels = rgb.shape
    pixels = rgb.reshape(-1, channels)
    result = kmeans(pixels, random_init(pixels, K, seed), max_iter=max_iter)
    centroids, labels, J, iterations = result
    segmented = centroids[labels].reshape(h, w, channels)
    return segmented, centroids, labels, J, iterations


# ============================================================================
# 9. Plot helpers for recurring lab / homework figure requirements
# ============================================================================

def plot_feature_histograms(X, feature_names, bins=20):
    """Lab 3: one histogram per feature, with mean and +/- one std lines."""
    X = np.asarray(X, dtype=float)
    means, stds, _, _ = column_stats(X)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for j, ax in enumerate(axes.ravel()):
        if j >= X.shape[1]:
            ax.axis("off"); continue
        ax.hist(X[:, j], bins=bins, alpha=0.8)
        ax.axvline(means[j], color="red", label="mean")
        ax.axvline(means[j] - stds[j], color="black", linestyle="--", label="mean +/- std")
        ax.axvline(means[j] + stds[j], color="black", linestyle="--")
        ax.set(title=feature_names[j], xlabel=feature_names[j], ylabel="count")
        ax.legend()
    fig.tight_layout()
    return fig, axes


def plot_distance_matrix(D):
    """Lab 3: show D as an image and mark each row's nearest column."""
    D = np.asarray(D)
    fig, ax = plt.subplots()
    image = ax.imshow(D, aspect="auto", interpolation="nearest")
    rows = np.arange(D.shape[0])
    ax.scatter(np.argmin(D, axis=1), rows, marker="x", color="red", s=12)
    ax.set(xlabel="B row / candidate", ylabel="A row / query", title="Pairwise distances")
    fig.colorbar(image, ax=ax, label="distance")
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(cm, labels=None):
    """Display counts with explicit true-row and predicted-column labels."""
    cm = np.asarray(cm)
    if labels is None:
        labels = np.arange(cm.shape[0])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    return fig, ax


def plot_missing_rates_by_class(df, target_column, feature_columns):
    """Lab 3 grouped bars: missing-value rate for each feature and class."""
    rates = missing_rate_by_class(df, target_column, feature_columns)
    fig, ax = plt.subplots()
    x = np.arange(len(feature_columns))
    width = 0.8 / max(1, len(rates))
    for i, (label, row) in enumerate(rates.iterrows()):
        ax.bar(x + (i - (len(rates) - 1) / 2) * width, row.to_numpy(),
               width=width, label=str(label))
    ax.set_xticks(x, labels=feature_columns)
    ax.set(xlabel="feature", ylabel="missing fraction", title="Missingness by class")
    ax.legend(title=target_column); fig.tight_layout()
    return fig, ax, rates


def plot_imputation_distributions(original, global_imputed, group_imputed,
                                  global_median, group_medians, column_name):
    """Lab 3 overlaid histograms; accepts original/global/group-imputed Series."""
    fig, ax = plt.subplots()
    ax.hist(original.dropna(), bins=30, alpha=0.5, label="observed only")
    ax.hist(global_imputed, bins=30, alpha=0.5, label="global median")
    ax.hist(group_imputed, bins=30, alpha=0.5, label="group median")
    ax.axvline(global_median, color="black", linestyle="--", label="global median")
    for label, median in group_medians.items():
        ax.axvline(median, linestyle=":", label=f"class {label} median")
    ax.set(xlabel=column_name, ylabel="count", title=f"Imputation effect: {column_name}")
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_scatter_by_class(x, y_value, labels, xlabel, ylabel, title):
    """Scatter plot with a different marker for each class."""
    x, y_value, labels = np.asarray(x), np.asarray(y_value), np.asarray(labels)
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(x[mask], y_value[mask], label=str(label), alpha=0.75,
                   marker="o" if label == np.unique(labels)[0] else "^")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(title="class"); fig.tight_layout()
    return fig, ax


def plot_regression_line(x, y, w, b, xlabel="x", ylabel="y", title="Fit"):
    """Scatter plus fitted one-feature regression line."""
    x, y = np.asarray(x), np.asarray(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="observations")
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, w * line_x + b, color="red", label="fit")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_residuals(y_pred, residuals):
    """Lab 4 residual plot with a zero reference line."""
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="predicted value", ylabel="residual (actual - predicted)", title="Residuals")
    fig.tight_layout()
    return fig, ax


def plot_curves(curves, xlabel, ylabel, title, log_x=False, log_y=False):
    """Plot a dict {label: (x,y)}; convenient for loss, CV, ridge, and k sweeps."""
    fig, ax = plt.subplots()
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="." if len(x) < 30 else None, label=str(label))
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_threshold_sweep(y_true, probabilities, thresholds):
    """Lab 3: plot precision, recall and F1 against thresholds."""
    p, r, f1 = threshold_metrics(y_true, probabilities, thresholds)
    return plot_curves({"precision": (thresholds, p), "recall": (thresholds, r),
                        "F1": (thresholds, f1)}, "threshold", "score",
                       "Metrics versus threshold"), (p, r, f1)


def plot_roc_curves(curves):
    """Plot {model_name: (fpr,tpr)} ROC curves and the random baseline."""
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="random baseline")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves")
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_2d_decision_boundary(X_train, y_train, predict_fn, resolution=200,
                              title="Decision regions", probabilities=False):
    """Plot a model's 2-D labels or probabilities over a mesh plus training rows."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    pad = 0.05 * (X_train.max(axis=0) - X_train.min(axis=0) + 1e-12)
    x0 = np.linspace(X_train[:, 0].min() - pad[0], X_train[:, 0].max() + pad[0], resolution)
    x1 = np.linspace(X_train[:, 1].min() - pad[1], X_train[:, 1].max() + pad[1], resolution)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_fn(grid)).reshape(xx.shape)
    fig, ax = plt.subplots()
    if probabilities:
        image = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.75)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black")
        fig.colorbar(image, ax=ax, label="P(class 1)")
    else:
        image = ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
        fig.colorbar(image, ax=ax, label="predicted class")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="black", s=18)
    ax.set(xlabel="feature 1", ylabel="feature 2", title=title)
    fig.tight_layout()
    return fig, ax


def plot_kmeans_result(X, centroids, labels, title="K-means clustering"):
    """2-D clusters with centroid markers."""
    X, centroids, labels = np.asarray(X), np.asarray(centroids), np.asarray(labels)
    fig, ax = plt.subplots()
    points = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=18)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", c="black", s=130, label="centroids")
    ax.set(xlabel="feature 1", ylabel="feature 2", title=title, aspect="equal")
    ax.legend(); fig.colorbar(points, ax=ax, label="cluster")
    fig.tight_layout()
    return fig, ax


# ============================================================================
# 10. Exam reminders / common question patterns
# ============================================================================
# - Always print shapes and labels; confusion matrix rows=true, columns=predicted.
# - Standardise/impute/category-encode using training rows only; reuse parameters.
# - In k-fold CV, held-out rows must not appear among the neighbours or fit rows.
# - For binary classification, compare with majority baseline and show F1/recall.
# - For regression, keep the intercept unpenalised; leave y in original units.
# - K-means is unsupervised: choose K with the elbow, and compare multiple starts.
# - Give each plot a title, labelled axes, and a legend when multiple series appear.
# - Save figures before plt.show(): fig.savefig("figure.png", dpi=150).
