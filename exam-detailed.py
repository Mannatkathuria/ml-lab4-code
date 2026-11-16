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
    """Return column means, population stds, minima, maxima (four 1-D arrays).

    Input:
        X: numeric matrix shaped (n_samples, n_features).
    Output:
        Tuple of four vectors, each shaped (n_features,): mean, population std, minimum, maximum.

    Example input:
        X=[[1, 2], [3, 4]]
    Example output:
        (means=[2, 3], std=[1, 1], minima=[1, 2], maxima=[3, 4])
    """
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0)


def cap_above_and_count_rows(X, threshold):
    """Return a capped copy and the count above threshold in each row.

    Input:
        X: numeric matrix shaped (n_samples, n_features); threshold: one number.
    Output:
        Tuple: capped copy shaped like X, and row counts shaped (n_samples,).

    Example input:
        X=[[1, 5], [4, 2]], threshold=3
    Example output:
        (capped=[[1, 3], [3, 2]], row_counts=[1, 1])
    """
    X = np.asarray(X)
    capped = np.minimum(X, threshold).copy()  # np.minimum makes a new array
    row_counts = (X > threshold).sum(axis=1)
    return capped, row_counts


def pairwise_distances(A, B):
    """Euclidean distances, shape (len(A), len(B)); broadcasting only.

    Input:
        A: matrix (n, d); B: matrix (m, d), with the same feature count d.
    Output:
        Distance matrix shaped (n, m); D[i, j] is the Euclidean distance from A[i] to B[j].

    Example input:
        A=[[0, 0], [3, 4]], B=[[0, 4]]
    Example output:
        D=[[4], [3]]
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]  # (n, m, d), all pair differences
    return np.sqrt(np.sum(diff * diff, axis=2))


def closest_indices(D):
    """Index of the closest column in each row of a distance matrix.

    Input:
        D: distance matrix shaped (n_queries, n_candidates).
    Output:
        Integer vector shaped (n_queries,), with the closest candidate column for each row.

    Example input:
        D=[[3, 1], [2, 5]]
    Example output:
        [1, 0]
    """
    return np.argmin(D, axis=1)


def _safe_divide(numerator, denominator):
    """Division that returns zero where a metric denominator is zero.

    Input:
        numerator and denominator: scalars or arrays whose shapes can broadcast together.
    Output:
        Array of broadcasted shape; denominator-zero positions contain 0.

    Example input:
        numerator=[4, 6], denominator=[2, 0]
    Example output:
        [2, 0]
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    np.divide(numerator, denominator, out=out, where=(denominator != 0))
    return out


# ============================================================================
# 2. Confusion matrices and classification metrics (Labs 1-3, 6-8 homework)
# ============================================================================

def confusion_matrix(y_true, y_pred, labels=None):
    """Counts with true labels on rows and predicted labels on columns.

    Input:
        y_true and y_pred: label vectors of equal length n; labels: optional ordered class list shaped (K,).
    Output:
        Integer confusion matrix shaped (K, K); rows are true classes and columns are predicted classes.

    Example input:
        y_true=[0, 1, 1], y_pred=[0, 0, 1], labels=[0, 1]
    Example output:
        [[1, 0], [1, 1]] (rows=true labels, columns=predicted labels)
    """
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
    """Fraction of predictions equal to the true labels.

    Input:
        y_true and y_pred: equal-length label vectors shaped (n,).
    Output:
        One scalar accuracy in [0, 1].

    Example input:
        y_true=[0, 1, 1], y_pred=[0, 0, 1]
    Example output:
        0.667
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1_from_cm(cm):
    """Per-class precision, recall, F1 and macro-F1 from a square confusion matrix.

    Input:
        cm: square confusion matrix shaped (K, K), true classes on rows.
    Output:
        Tuple of precision, recall, and F1 vectors shaped (K,), followed by scalar macro-F1.

    Example input:
        cm=[[1, 0], [1, 1]]
    Example output:
        (precision=[0.5, 1], recall=[1, 0.5], F1=[0.667, 0.667], macro_F1=0.667)
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1, float(np.mean(f1))


def precision_recall_f1(y_true, y_pred, positive_label=1):
    """Binary metrics for one positive label; zero denominators produce zero.

    Input:
        y_true and y_pred: equal-length binary label vectors shaped (n,); positive_label: scalar label.
    Output:
        Tuple of scalar precision, recall, and F1 for positive_label.

    Example input:
        y_true=[0, 1, 1], y_pred=[0, 0, 1], positive_label=1
    Example output:
        (precision=1, recall=0.5, F1=0.667)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    precision = float(_safe_divide(tp, tp + fp))
    recall = float(_safe_divide(tp, tp + fn))
    f1 = float(_safe_divide(2 * precision * recall, precision + recall))
    return precision, recall, f1


def binary_metrics(y_true, y_pred, positive_label=1):
    """Return accuracy, precision, recall, F1 and the 2x2 confusion matrix.

    Input:
        y_true and y_pred: equal-length vectors containing binary labels 0 and 1; positive_label: scalar.
    Output:
        Tuple: scalar accuracy, precision, recall, F1, and a 2-by-2 confusion matrix.

    Example input:
        y_true=[0, 1, 1], y_pred=[0, 0, 1]
    Example output:
        (accuracy=0.667, precision=1, recall=0.5, F1=0.667, cm=[[1, 0], [1, 1]])
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label)
    return accuracy_score_scratch(y_true, y_pred), p, r, f1, cm


# ============================================================================
# 3. Splits and pandas preprocessing (Labs 2, 4-8)
# ============================================================================

def train_test_split(X, y, test_size=0.2, seed=42):
    """Lab 2 convention: first int(n*test_size) shuffled rows are the test set.

    Input:
        X: feature matrix (n, d); y: target vector (n,); test_size: fraction; seed: integer.
    Output:
        X_train (n_train,d), X_test (n_test,d), y_train (n_train,), y_test (n_test,).

    Example input:
        X=[[1, 2], [3, 4], [5, 6], [7, 8]], y=[0, 1, 0, 1], test_size=0.5, seed=0
    Example output:
        (X_train=[[3, 4], [7, 8]], X_test=[[5, 6], [1, 2]], y_train=[1, 1], y_test=[0, 0])
    """
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_split_scratch(X, y, test_frac=0.2, seed=0):
    """Lab 4 named version of the same deterministic shuffled split.

    Input:
        X: feature matrix (n, d); y: target vector (n,); test_frac: fraction; seed: integer.
    Output:
        Same four-array order and shapes as train_test_split: X_train, X_test, y_train, y_test.

    Example input:
        X=[[1, 2], [3, 4], [5, 6], [7, 8]], y=[0, 1, 0, 1], test_frac=0.5, seed=0
    Example output:
        (X_train=[[3, 4], [7, 8]], X_test=[[5, 6], [1, 2]], y_train=[1, 1], y_test=[0, 0])
    """
    return train_test_split(X, y, test_size=test_frac, seed=seed)


def fixed_count_split(X, y, n_test=100, seed=0):
    """Labs 6-8 convention: first n_test shuffled indices are the test rows.

    Input:
        X: feature matrix (n, d); y: target vector (n,); n_test: test-row count; seed: integer.
    Output:
        X_train (n-n_test,d), X_test (n_test,d), y_train (n-n_test,), y_test (n_test,).

    Example input:
        X=[[1, 2], [3, 4], [5, 6], [7, 8]], y=[0, 1, 0, 1], n_test=2, seed=0
    Example output:
        (X_train=[[3, 4], [7, 8]], X_test=[[5, 6], [1, 2]], y_train=[1, 1], y_test=[0, 0])
    """
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    """Approximate class proportions in both splits; returns NumPy arrays.

    Input:
        X: feature matrix (n, d); y: class labels (n,); test_size: fraction; seed: integer.
    Output:
        Four arrays: X_train, X_test, y_train, y_test, with approximate class proportions preserved.

    Example input:
        X=[[1], [2], [3], [4]], y=[0, 0, 1, 1], test_size=0.5
    Example output:
        Four arrays: train/test feature rows and train/test labels; each split has about half class 0 and half class 1.
    """
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
    """Hand-coded one-hot encoding; unknown values become all-zero rows.

    Input:
        values: category vector (n,); categories: optional category list (K,).
    Output:
        Tuple: indicator matrix (n, K) and the category order (K,).

    Example input:
        values=["cat", "dog", "cat"], categories=["cat", "dog"]
    Example output:
        (encoded=[[1, 0], [0, 1], [1, 0]], categories=["cat", "dog"])
    """
    values = np.asarray(values).ravel()
    if categories is None:
        categories = np.unique(values)
    categories = np.asarray(categories)
    encoded = np.column_stack([values == category for category in categories]).astype(int)
    return encoded, categories


def one_hot_encode_pair(train_values, test_values):
    """Use categories learned from train for both splits (prevents test leakage).

    Input:
        train_values: category vector (n_train,); test_values: category vector (n_test,).
    Output:
        Tuple: train indicator matrix (n_train,K), test matrix (n_test,K), and categories (K,).

    Example input:
        train_values=["cat", "dog"], test_values=["dog", "cat"]
    Example output:
        (train=[[1, 0], [0, 1]], test=[[0, 1], [1, 0]], categories=["cat", "dog"])
    """
    train_encoded, categories = one_hot_encode(train_values)
    test_encoded, _ = one_hot_encode(test_values, categories=categories)
    return train_encoded, test_encoded, categories


def impute_median_train_test(X_train, X_test):
    """Learn each numeric median on train, then apply the same medians to both.

    Input:
        X_train: numeric matrix (n_train,d) with NaNs; X_test: numeric matrix (n_test,d) with NaNs.
    Output:
        Tuple: imputed train matrix, imputed test matrix, and train medians (d,).

    Example input:
        X_train=[[1, np.nan], [3, 4]], X_test=[[np.nan, 6]]
    Example output:
        (X_train=[[1, 4], [3, 4]], X_test=[[2, 6]], medians=[2, 4])
    """
    X_train = np.asarray(X_train, dtype=float).copy()
    X_test = np.asarray(X_test, dtype=float).copy()
    medians = np.nanmedian(X_train, axis=0)
    rows, cols = np.where(np.isnan(X_train))
    X_train[rows, cols] = medians[cols]
    rows, cols = np.where(np.isnan(X_test))
    X_test[rows, cols] = medians[cols]
    return X_train, X_test, medians


def impute_series_global_median(series):
    """Lab 2 comparison: return globally median-imputed Series and its median.

    Input:
        series: pandas Series of numeric values, possibly containing missing entries.
    Output:
        Tuple: imputed Series of the same length and the scalar global median.

    Example input:
        Series([1, np.nan, 3])
    Example output:
        (Series([1, 2, 3]), median=2)
    """
    median = series.median()
    return series.fillna(median), median


def impute_series_by_group(df, value_column, group_column):
    """Lab 2 exploration only: median within each observed group.

    Input:
        df: pandas DataFrame; value_column and group_column: column-name strings.
    Output:
        Tuple: group-median-imputed value Series and row-aligned group-median Series.

    Example input:
        df=pd.DataFrame({"group":["A","A","B","B"], "value":[1,np.nan,5,np.nan]})
    Example output:
        (imputed=[1, 1, 5, 5], row-aligned group medians=[1, 1, 5, 5])
    """
    group_medians = df.groupby(group_column)[value_column].transform("median")
    return df[value_column].fillna(group_medians), group_medians


def missing_rate_by_class(df, target_column, feature_columns):
    """Lab 3: missing fraction for each feature within each target class.

    Input:
        df: pandas DataFrame; target_column: class column name; feature_columns: list of feature names.
    Output:
        DataFrame indexed by class, with each feature column containing its missing-value fraction.

    Example input:
        df=pd.DataFrame({"class":[0,0,1], "x":[1,np.nan,3]}), target_column="class", feature_columns=["x"]
    Example output:
        DataFrame: class 0 -> x=0.5; class 1 -> x=0.0
    """
    return df.groupby(target_column)[feature_columns].apply(lambda part: part.isna().mean())


def majority_class_predict(y_train, n_samples):
    """Baseline: repeat the most common training label for every query row.

    Input:
        y_train: label vector (n_train,); n_samples: number of rows to predict.
    Output:
        Prediction vector shaped (n_samples,), filled with the most frequent training label.

    Example input:
        y_train=[0, 1, 1], n_samples=4
    Example output:
        [1, 1, 1, 1]
    """
    labels, counts = np.unique(y_train, return_counts=True)
    return np.full(n_samples, labels[np.argmax(counts)])


# ============================================================================
# 4. Linear regression, standardisation, ridge, CV and lasso (Labs 4-5)
# ============================================================================

def fit_simple_lr(x, y):
    """Fit y = w*x+b by the one-feature least-squares formulas.

    Input:
        x and y: numeric vectors of equal length n, each containing one feature/target.
    Output:
        Tuple of scalar slope w and intercept b for y_hat = w*x+b.

    Example input:
        x=[1, 2, 3], y=[3, 5, 7]
    Example output:
        (w=2, b=1)
    """
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denom = np.sum(x_centered ** 2)
    w = 0.0 if denom == 0 else np.sum(x_centered * y_centered) / denom
    b = y.mean() - w * x.mean()
    return w, b


def design_matrix(X):
    """Prepend the intercept column of ones to an (n,d) feature matrix.

    Input:
        X: feature matrix (n,d), or one feature vector (n,).
    Output:
        Matrix shaped (n,d+1), with an all-ones intercept column first.

    Example input:
        X=[[2, 4], [5, 6]]
    Example output:
        [[1, 2, 4], [1, 5, 6]]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.c_[np.ones(len(X)), X]


def fit_normal_equation(X, y):
    """OLS using solve (more stable than explicitly constructing an inverse).

    Input:
        X: feature matrix (n,d); y: numeric target vector (n,).
    Output:
        Coefficient vector theta shaped (d+1,), with intercept first.

    Example input:
        X=[[0], [1], [2]], y=[1, 3, 5]
    Example output:
        theta=[1, 2] (intercept first)
    """
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    return np.linalg.solve(A.T @ A, A.T @ y)


def fit_gradient_descent(X, y, lr, n_iters):
    """Batch descent for J=mean squared error; return theta and post-step losses.

    Input:
        X: feature matrix (n,d); y: target (n,); lr and n_iters: learning rate and step count.
    Output:
        Tuple: theta shaped (d+1,) and a list of n_iters post-update mean-squared losses.

    Example input:
        X=[[0], [1]], y=[1, 3], lr=0.1, n_iters=1
    Example output:
        (theta=[0.4, 0.3], losses=[2.825])
    """
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
    """Predict with an intercept-first coefficient vector.

    Input:
        X: feature matrix (n,d); theta: coefficients shaped (d+1,), intercept first.
    Output:
        Numeric prediction vector shaped (n,).

    Example input:
        X=[[0], [2]], theta=[1, 2]
    Example output:
        [1, 5]
    """
    return design_matrix(X) @ np.asarray(theta)


def rmse(y_true, y_pred):
    """Root mean squared error.

    Input:
        y_true and y_pred: equal-length numeric vectors shaped (n,).
    Output:
        One nonnegative scalar root mean squared error.

    Example input:
        y_true=[1, 3], y_pred=[2, 1]
    Example output:
        1.581
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score_scratch(y_true, y_pred):
    """R-squared; return zero for a constant target to avoid division by zero.

    Input:
        y_true and y_pred: equal-length numeric vectors shaped (n,).
    Output:
        One scalar R-squared score; this template returns 0 for a constant target.

    Example input:
        y_true=[1, 3], y_pred=[1, 2]
    Example output:
        0.5
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


def standardise(X_train, X_test):
    """Train-only z-score scaling; constant columns use scale 1 and become zero.

    Input:
        X_train: numeric matrix (n_train,d); X_test: numeric matrix (n_test,d).
    Output:
        Tuple: scaled train (n_train,d), scaled test (n_test,d), train means (d,), safe train stds (d,).

    Example input:
        X_train=[[1], [3]], X_test=[[5]]
    Example output:
        (Z_train=[[-1], [1]], Z_test=[[3]], mu=[2], sigma=[1])
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma_safe, (X_test - mu) / sigma_safe, mu, sigma_safe


def unstandardise_coefficients(theta, mu, sigma):
    """Convert z-score-space linear coefficients back to original feature units.

    Input:
        theta: scaled-model coefficients (d+1,); mu and sigma: training vectors (d,).
    Output:
        Tuple: raw-unit feature weights (d,) and raw-unit scalar intercept.

    Example input:
        theta=[1, 2], mu=[2], sigma=[2]
    Example output:
        (raw_w=[1], raw_b=-1)
    """
    theta = np.asarray(theta, dtype=float)
    raw_w = theta[1:] / np.asarray(sigma)
    raw_b = theta[0] - np.sum(raw_w * np.asarray(mu))
    return raw_w, raw_b


def fit_ridge(X, y, lam):
    """Closed-form ridge; intercept is unpenalised.

    Input:
        X: feature matrix (n,d); y: target vector (n,); lam: nonnegative penalty strength.
    Output:
        Ridge coefficient vector theta shaped (d+1,), with the intercept unpenalised.

    Example input:
        X=[[0], [1], [2]], y=[1, 3, 5], lam=0
    Example output:
        theta=[1, 2] (intercept first)
    """
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0  # the intercept is the overall level, not a feature weight
    return np.linalg.solve(A.T @ A + lam * penalty, A.T @ y)


def k_fold_cv(X, y, lam, k=5, seed=0):
    """Mean validation RMSE for ridge; each row is validation exactly once.

    Input:
        X: feature matrix (n,d); y: target (n,); lam: ridge penalty; k: fold count; seed: integer.
    Output:
        One scalar mean validation RMSE across k folds.

    Example input:
        X=[[0], [1], [2], [3], [4]], y=[1, 3, 5, 7, 9], lam=1, k=5, seed=0
    Example output:
        One scalar RMSE, for example 1.23; its exact value depends on the folds and data.
    """
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
    """Table of (lambda, mean CV RMSE) for choosing ridge regularisation.

    Input:
        X and y: training features (n,d) and targets (n,); lambda_values: vector (L,); k and seed.
    Output:
        Numeric table shaped (L,2): lambda in column 0 and mean CV RMSE in column 1.

    Example input:
        X=[[0], [1], [2], [3]], y=[1, 3, 5, 7], lambda_values=[0.1, 1.0], k=2, seed=0
    Example output:
        Example table [[0.1, 2.8], [1.0, 2.4]]: columns are lambda and CV RMSE.
    """
    return np.asarray([(lam, k_fold_cv(X, y, lam, k=k, seed=seed))
                       for lam in lambda_values], dtype=float)


def ridge_coefficient_path(X, y, lambda_values):
    """Coefficient matrix, one row per lambda, for the ridge path plot.

    Input:
        X and y: training data (n,d) and (n,); lambda_values: vector (L,).
    Output:
        Coefficient matrix shaped (L,d+1), one intercept-first row per lambda.

    Example input:
        lambda_values=[0, 1], X=[[0], [1], [2]], y=[1, 3, 5]
    Example output:
        Two coefficient rows, e.g. [[1, 2], [1.2, 1.5]]; one row for each lambda.
    """
    return np.asarray([fit_ridge(X, y, lam) for lam in lambda_values])


def soft_threshold(v, t):
    """Vectorised L1 shrinkage: reduce magnitude by t and clamp at zero.

    Input:
        v: scalar or numeric array of any shape; t: nonnegative shrinkage amount.
    Output:
        Array with the same shape as v, shrunk toward zero and clamped at zero.

    Example input:
        v=[-3, -0.5, 2], t=1
    Example output:
        [-2, 0, 1]
    """
    v = np.asarray(v)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)


def fit_lasso(X, y, lam, lr=0.05, n_iters=30000):
    """Gradient descent plus soft-thresholding; the intercept is not shrunk.

    Input:
        X: feature matrix (n,d); y: target (n,); lam, lr, n_iters: penalty, step size, iteration count.
    Output:
        Coefficient vector theta shaped (d+1,), with the intercept exempt from thresholding.

    Example input:
        X=[[0], [1], [2]], y=[1, 3, 5], lam=0.1, lr=0.01, n_iters=10
    Example output:
        theta=[intercept, feature_weight], for example theta=[0.5, 0.7] after a small number of steps.
    """
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
    """Numerically stable sigmoid; accepts a scalar or an array.

    Input:
        z: scalar or numeric array of any shape.
    Output:
        Probabilities with the same shape as z, each between 0 and 1.

    Example input:
        z=[0, 2]
    Example output:
        [0.5, 0.881]
    """
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, theta):
    """Binary logistic log-likelihood; X excludes the intercept column.

    Input:
        X: feature matrix (n,d), without intercept; y: binary vector (n,); theta: coefficients (d+1,).
    Output:
        One scalar total binary logistic log-likelihood.

    Example input:
        X=[[0], [1]], y=[0, 1], theta=[0, 1]
    Example output:
        -1.006 (one scalar total log-likelihood)
    """
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    p = sigmoid(A @ theta)
    eps = 1e-12  # prevents log(0) at extreme probabilities
    return float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def fit_logistic(X, y, lr=0.1, n_iters=2000):
    """Batch gradient ascent for binary logistic regression.

    Input:
        X: feature matrix (n,d); y: binary labels (n,); lr and n_iters: learning rate and steps.
    Output:
        Tuple: theta shaped (d+1,) and log-likelihood list of length n_iters.

    Example input:
        X=[[0], [1]], y=[0, 1], lr=0.1, n_iters=1
    Example output:
        (theta=[0, 0.025], one-item log-likelihood list about [-1.374])
    """
    A, y = design_matrix(X), np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    log_likelihoods = []
    for _ in range(n_iters):
        p = sigmoid(A @ theta)
        theta += (lr / len(y)) * (A.T @ (y - p))  # plus sign maximises likelihood
        log_likelihoods.append(log_likelihood(X, y, theta))
    return theta, log_likelihoods


def predict_proba_logistic(X, theta):
    """Probability of class 1 from a fitted logistic coefficient vector.

    Input:
        X: feature matrix (n,d); theta: fitted coefficients (d+1,).
    Output:
        Class-1 probability vector shaped (n,).

    Example input:
        X=[[0], [2]], theta=[0, 1]
    Example output:
        [0.5, 0.881]
    """
    return sigmoid(design_matrix(X) @ theta)


def predict_logistic(X, theta, threshold=0.5):
    """Turn logistic probabilities into 0/1 labels at the chosen cutoff.

    Input:
        X: feature matrix (n,d); theta: coefficients (d+1,); threshold: probability cutoff.
    Output:
        Integer 0/1 label vector shaped (n,).

    Example input:
        X=[[0], [2]], theta=[0, 1], threshold=0.5
    Example output:
        [1, 1] (the rule is probability >= threshold)
    """
    return (predict_proba_logistic(X, theta) >= threshold).astype(int)


def threshold_metrics(y_true, probabilities, thresholds):
    """Return precision, recall and F1 arrays across a threshold sweep.

    Input:
        y_true: binary labels (n,); probabilities: scores (n,); thresholds: cutoff vector (T,).
    Output:
        Tuple of precision, recall, and F1 vectors, each shaped (T,).

    Example input:
        y_true=[0, 1, 1], probabilities=[0.1, 0.6, 0.9], thresholds=[0.5, 0.8]
    Example output:
        (precision=[1, 1], recall=[1, 0.5], F1=[1, 0.667])
    """
    probabilities = np.asarray(probabilities)
    p_values, r_values, f_values = [], [], []
    for threshold in thresholds:
        labels = (probabilities >= threshold).astype(int)
        p, r, f1 = precision_recall_f1(y_true, labels)
        p_values.append(p); r_values.append(r); f_values.append(f1)
    return np.asarray(p_values), np.asarray(r_values), np.asarray(f_values)


def feature_threshold_sweep(y_true, feature, thresholds, greater_is_positive=True):
    """Lab 3 one-rule classifier sweep; uses strict > as the handout specifies.

    Input:
        y_true: binary labels (n,); feature: numeric vector (n,); thresholds: vector (T,); direction flag.
    Output:
        Tuple of precision, recall, and F1 vectors, each shaped (T,); prediction uses strict > or <.

    Example input:
        y_true=[0, 1, 1], feature=[1, 2, 3], thresholds=[1.5, 2.5]
    Example output:
        (precision=[1, 1], recall=[1, 0.5], F1=[1, 0.667])
    """
    feature, thresholds = np.asarray(feature), np.asarray(thresholds)
    rows = []
    for threshold in thresholds:
        pred = feature > threshold if greater_is_positive else feature < threshold
        p, r, f1 = precision_recall_f1(y_true, pred.astype(int))
        rows.append((p, r, f1))
    values = np.asarray(rows)
    return values[:, 0], values[:, 1], values[:, 2]  # precision, recall, F1 arrays


def roc_curve_scratch(y_true, p):
    """ROC points in increasing FPR order; tied scores share a threshold.

    Input:
        y_true: binary labels (n,); p: class-1 probabilities/scores (n,).
    Output:
        Tuple of FPR and TPR vectors, same length, ordered by increasing FPR and including endpoints.

    Example input:
        y_true=[0, 1], p=[0.2, 0.8]
    Example output:
        (FPR=[0, 0, 1, 1], TPR=[0, 1, 1, 1])
    """
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
    """Area under a ROC curve using trapezoids after sorting by FPR.

    Input:
        fpr and tpr: equal-length ROC coordinate vectors (T,).
    Output:
        One scalar trapezoidal area under the ROC curve.

    Example input:
        fpr=[0, 0, 1], tpr=[0, 1, 1]
    Example output:
        1.0
    """
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# ============================================================================
# 6. k-Nearest Neighbours and cross-validation (Lab 6 and homework)
# ============================================================================

def knn_predict(X_train, y_train, X_query, k):
    """Vectorised k-NN vote; no loop over query rows.

    Input:
        X_train: features (n_train,d); y_train: labels (n_train,); X_query: features (n_query,d); k: neighbor count.
    Output:
        Predicted class vector shaped (n_query,).

    Example input:
        X_train=[[0], [2], [5]], y_train=[0, 1, 1], X_query=[[0.2]], k=1
    Example output:
        [0]
    """
    y_train = np.asarray(y_train)
    D = pairwise_distances(X_query, X_train)  # each query row against all training rows
    neighbour_idx = np.argsort(D, axis=1)[:, :k]
    neighbour_labels = y_train[neighbour_idx]
    classes = np.unique(y_train)
    class_codes = np.searchsorted(classes, neighbour_labels)
    votes = np.eye(len(classes), dtype=int)[class_codes].sum(axis=1)
    return classes[np.argmax(votes, axis=1)]  # sorted class order breaks ties consistently


def knn_predict_weighted(X_train, y_train, X_query, k, eps=1e-12):
    """Distance-weighted k-NN; closer neighbours cast stronger 1/(d+eps) votes.

    Input:
        X_train (n_train,d), y_train (n_train,), X_query (n_query,d), k, and small distance epsilon.
    Output:
        Distance-weighted predicted class vector shaped (n_query,).

    Example input:
        X_train=[[0], [2], [5]], y_train=[0, 1, 1], X_query=[[0.1]], k=2
    Example output:
        [0] (the closer class-0 neighbor has the stronger inverse-distance vote).
    """
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
    """Mean validation accuracy; held-out rows are never passed as neighbours.

    Input:
        X: features (n,d); y: labels (n,); k: neighbors; folds: fold count; seed: integer.
    Output:
        One scalar mean validation accuracy; held-out rows are excluded from their own neighbor set.

    Example input:
        X=[[0], [1], [4], [5]], y=[0, 0, 1, 1], k=1, folds=2, seed=0
    Example output:
        One scalar accuracy, e.g. 0.75.
    """
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
    """Return (k, CV accuracy) rows for a k-NN model-selection table.

    Input:
        X and y: training data (n,d) and (n,); k_values: candidate neighbor counts (L,).
    Output:
        Numeric table shaped (L,2): k in column 0 and cross-validation accuracy in column 1.

    Example input:
        k_values=[1, 3]
    Example output:
        Example table [[1, 0.8], [3, 0.9]]: columns are k and CV accuracy; scores depend on the data.
    """
    return np.asarray([(k, k_fold_cv_knn(X, y, int(k), folds, seed)) for k in k_values])


def append_gaussian_noise(X_train, X_test, n_noise, seed=0):
    """Homework helper: append independent N(0,1) noise columns to train/test.

    Input:
        X_train (n_train,d), X_test (n_test,d), n_noise: added-column count; seed: integer.
    Output:
        Tuple of matrices shaped (n_train,d+n_noise) and (n_test,d+n_noise).

    Example input:
        X_train=[[1, 2]], X_test=[[3, 4]], n_noise=1, seed=0
    Example output:
        (two arrays shaped (1,3), for example [[1, 2, 0.126]] and [[3, 4, -0.132]])
    """
    rng = np.random.default_rng(seed)
    train_noise = rng.normal(0, 1, size=(len(X_train), n_noise))
    test_noise = rng.normal(0, 1, size=(len(X_test), n_noise))
    return np.c_[X_train, train_noise], np.c_[X_test, test_noise]


def distance_concentration_ratio(D):
    """min/max distance: approaches 1 as nearest and farthest become alike.

    Input:
        D: nonnegative distance matrix (n_queries,n_candidates).
    Output:
        One scalar min(D)/max(D), or 0 when every distance is zero.

    Example input:
        D=[[1, 4], [2, 3]]
    Example output:
        0.25 (minimum distance 1 divided by maximum distance 4)
    """
    D = np.asarray(D)
    return float(D.min() / D.max()) if D.max() != 0 else 0.0


# ============================================================================
# 7. Gaussian and categorical Naive Bayes (Lab 8 and homework)
# ============================================================================

def gaussian_pdf(x, mu, var):
    """Vectorised Gaussian probability density; tiny variance floor avoids /0.

    Input:
        x, mu, var: scalars or NumPy arrays with compatible broadcast shapes.
    Output:
        Gaussian density array with the common broadcast shape.

    Example input:
        x=0, mu=0, var=1
    Example output:
        0.399
    """
    x, mu, var = np.asarray(x), np.asarray(mu), np.maximum(np.asarray(var), 1e-12)
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def fit_gaussian_nb(X, y):
    """Fit two-class Gaussian NB; variances divide by class count (population).

    Input:
        X: numeric training matrix (n,d); y: binary class vector (n,) with labels 0 and 1.
    Output:
        Tuple: priors (2,), class means (2,d), and population variances (2,d).

    Example input:
        X=[[0], [1], [2], [3]], y=[0, 0, 1, 1]
    Example output:
        (priors=[0.5, 0.5], means=[[0.5], [2.5]], variances=[[0.25], [0.25]])
    """
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
    """Internal: log prior plus summed feature-wise Gaussian log densities.

    Input:
        X: numeric matrix (n,d); priors (2,); means and variances (2,d).
    Output:
        Log-score matrix shaped (n,2), one column per class.

    Example input:
        X=[[0], [3]], priors=[0.5, 0.5], means=[[0], [3]], variances=[[1], [1]]
    Example output:
        A 2-by-2 matrix of class log-scores, one row per input point.
    """
    X = np.asarray(X, dtype=float)
    safe_var = np.maximum(np.asarray(variances), 1e-12)
    log_density = -0.5 * (np.log(2 * np.pi * safe_var)[None, :, :]
                          + (X[:, None, :] - means[None, :, :]) ** 2
                          / safe_var[None, :, :])
    return np.log(np.asarray(priors)[None, :] + 1e-300) + log_density.sum(axis=2)


def predict_proba_nb(X, priors, means, variances):
    """Return P(class=1|x), comparing Gaussian NB class scores in log space.

    Input:
        X: numeric matrix (n,d); priors (2,); means and variances (2,d).
    Output:
        Gaussian Naive Bayes class-1 probability vector shaped (n,).

    Example input:
        X=[[0], [3]], priors=[0.5, 0.5], means=[[0], [3]], variances=[[1], [1]]
    Example output:
        Probabilities close to [0.011, 0.989], one per row.
    """
    log_scores = _gaussian_log_scores(X, priors, means, variances)
    return sigmoid(log_scores[:, 1] - log_scores[:, 0])  # stable two-class softmax


def fit_categorical_nb(e, y, l=1):
    """Laplace-smoothed category probabilities; e must be encoded as 0..K-1.

    Input:
        e: integer category vector (n,) encoded 0..3; y: binary vector (n,); l: Laplace smoothing value.
    Output:
        Probability table shaped (2,4), rows are classes and columns are categories.

    Example input:
        e=[0, 1, 1, 2], y=[0, 0, 1, 1], l=1
    Example output:
        [[0.333, 0.333, 0.167, 0.167], [0.167, 0.333, 0.333, 0.167]]
    """
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
    """Combine Gaussian numeric features and categorical log-likelihoods.

    Input:
        X: numeric matrix (n,d); e: encoded category vector (n,); NB parameters and category table (2,4).
    Output:
        Combined class-1 probability vector shaped (n,).

    Example input:
        X=[[0], [3]], e=[0, 3], priors=[0.5,0.5], means=[[0],[3]], variances=[[1],[1]], category_probabilities=[[0.5,0.2,0.2,0.1],[0.1,0.2,0.2,0.5]]
    Example output:
        Two combined class-1 probabilities, approximately [0.002, 0.998] for these class-aligned values.
    """
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
    """Choose K distinct rows as initial centroids with a seeded generator.

    Input:
        X: feature matrix (n,d); K: cluster count; seed: integer.
    Output:
        Initial centroid matrix shaped (K,d), selected from distinct rows of X.

    Example input:
        X=[[2, 3]], K=1, seed=0
    Example output:
        [[2, 3]]
    """
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return np.asarray(X)[idx].copy()


def assign_clusters(X, centroids):
    """Assign each row to its nearest centroid.

    Input:
        X: points (n,d); centroids: matrix (K,d).
    Output:
        Integer cluster-label vector shaped (n,), with values 0..K-1.

    Example input:
        X=[[0, 0], [1, 1], [9, 9]], centroids=[[0, 0], [10, 10]]
    Example output:
        [0, 0, 1]
    """
    return np.argmin(pairwise_distances(X, centroids), axis=1)


def update_centroids(X, labels, centroids):
    """Replace each centroid by its assigned-point mean; keep empty clusters.

    Input:
        X: points (n,d); labels: integer vector (n,); centroids: old matrix (K,d).
    Output:
        Updated centroid matrix shaped (K,d); empty clusters keep their old center.

    Example input:
        X=[[0, 0], [0, 2], [8, 8]], labels=[0, 0, 1], centroids=[[0, 0], [8, 8]]
    Example output:
        [[0, 1], [8, 8]]
    """
    X, labels = np.asarray(X), np.asarray(labels)
    updated = np.asarray(centroids, dtype=float).copy()
    for c in range(len(updated)):
        members = X[labels == c]
        if len(members):
            updated[c] = members.mean(axis=0)
    return updated


def distortion(X, labels, centroids):
    """Sum squared distances from points to their assigned centroids.

    Input:
        X: points (n,d); labels: cluster indices (n,); centroids: matrix (K,d).
    Output:
        One scalar sum of squared distances to assigned centroids.

    Example input:
        X=[[0, 0], [0, 2], [8, 8]], labels=[0, 0, 1], centroids=[[0, 1], [8, 8]]
    Example output:
        2.0
    """
    residual = np.asarray(X) - np.asarray(centroids)[np.asarray(labels)]
    return float(np.sum(residual ** 2))


def kmeans(X, init_centroids, max_iter=100):
    """Alternate nearest-centroid assignment and mean update until convergence.

    Input:
        X: numeric points (n,d); init_centroids: starting centers (K,d); max_iter: iteration limit.
    Output:
        Tuple: final centers (K,d), labels (n,), scalar distortion J, and iteration count.

    Example input:
        X=[[0, 0], [0, 2], [8, 8]], init_centroids=[[0, 0], [8, 8]]
    Example output:
        (centroids=[[0, 1], [8, 8]], labels=[0, 0, 1], J=2, iterations=2)
    """
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
    """Return the lowest-distortion result over seeds 0..n_init-1.

    Input:
        X: points (n,d); K: cluster count; n_init: number of seeded starts.
    Output:
        Lowest-J K-means result tuple: centers (K,d), labels (n,), J, iterations.

    Example input:
        X=[[0, 0], [8, 8]], K=2, n_init=3
    Example output:
        A tuple with centers [[0,0],[8,8]], labels containing one ID per row, J=0, and iterations=1; center order can vary.
    """
    best = None
    for seed in range(n_init):
        result = kmeans(X, random_init(X, K, seed))
        if best is None or result[2] < best[2]:
            best = result
    return best


def kmeans_multistart(X, K, n_init=10):
    """Return every seeded K-means result, useful for inspecting local minima.

    Input:
        X: points (n,d); K: cluster count; n_init: number of starts.
    Output:
        List of n_init K-means result tuples, one for each seed from 0 onward.

    Example input:
        X=[[0, 0], [8, 8]], K=2, n_init=2
    Example output:
        Two result tuples; for this two-point input both have J=0, and label IDs may be [0,1] or [1,0].
    """
    return [kmeans(X, random_init(X, K, seed)) for seed in range(n_init)]


def kmeans_pp_init(X, K, seed):
    """K-means++: sample each next center proportional to squared nearest distance.

    Input:
        X: points (n,d); K: cluster count; seed: integer.
    Output:
        K-means++ starting centroid matrix shaped (K,d), chosen from rows of X.

    Example input:
        X=[[2, 3]], K=1, seed=0
    Example output:
        [[2, 3]]
    """
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
    """Return K, best distortion, and iterations for each candidate cluster count.

    Input:
        X: points (n,d); K_values: candidate counts (L,); n_init: starts per K.
    Output:
        Numeric table shaped (L,3): K, best distortion, and iteration count.

    Example input:
        X=[[0, 0], [2, 0]], K_values=[1, 2]
    Example output:
        Rows [[1, 2, 2], [2, 0, 1]]: columns are K, best distortion, iterations.
    """
    rows = []
    for K in K_values:
        centroids, labels, J, iterations = best_of_kmeans(X, int(K), n_init=n_init)
        rows.append((K, J, iterations))
    return np.asarray(rows, dtype=float)


def segment_image_kmeans(image, K, seed=0, max_iter=100):
    """Cluster RGB pixels and return segmented RGB image plus K-means outputs.

    Input:
        image: RGB/RGBA array (height,width,channels); K: color clusters; seed and max_iter.
    Output:
        Tuple: segmented RGB image (height,width,3), centers (K,3), pixel labels, distortion, iterations.

    Example input:
        image=[[[1,0,0], [0,0,1]]], K=2
    Example output:
        Segmented image [[[1,0,0], [0,0,1]]], plus centers, two pixel labels, J, and iteration count.
    """
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
    """Lab 3: one histogram per feature, with mean and +/- one std lines.

    Input:
        X: numeric matrix (n,d), feature_names: d strings, bins: histogram bin count.
    Output:
        Tuple: Matplotlib Figure and 2-by-3 Axes array; unused panels are hidden.

    Example input:
        X=[[1, 2], [3, 4]], feature_names=["a", "b"]
    Example output:
        A Figure and 2-by-3 Axes; two panels show histograms and four are hidden.
    """
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
    """Lab 3: show D as an image and mark each row's nearest column.

    Input:
        D: distance matrix (n,m), query rows by candidate columns.
    Output:
        Tuple: Matplotlib Figure and Axes showing distances and each row minimum.

    Example input:
        D=[[1, 3], [4, 2]]
    Example output:
        Figure/Axes with a 2-by-2 heatmap and red nearest-column marks at (row0,col0), (row1,col1).
    """
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
    """Display counts with explicit true-row and predicted-column labels.

    Input:
        cm: square count matrix (K,K); labels: optional class labels (K,).
    Output:
        Tuple: Matplotlib Figure and Axes with counts annotated in cells.

    Example input:
        cm=[[2, 1], [0, 3]], labels=[0, 1]
    Example output:
        Figure/Axes with the four cell counts 2, 1, 0, 3 printed on the image.
    """
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
    """Lab 3 grouped bars: missing-value rate for each feature and class.

    Input:
        df: DataFrame; target_column and feature_columns: column names.
    Output:
        Tuple: Figure, Axes, and class-indexed missing-rate DataFrame.

    Example input:
        df=pd.DataFrame({"class":[0,0,1], "x":[1,np.nan,3]}), target_column="class", feature_columns=["x"]
    Example output:
        Figure/Axes with two class bars, plus rates table {class 0: 0.5, class 1: 0.0}.
    """
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
    """Lab 3 overlaid histograms; accepts original/global/group-imputed Series.

    Input:
        Three numeric Series, global median scalar, group_medians mapping, and column_name string.
    Output:
        Tuple: Matplotlib Figure and Axes with overlay histograms and median lines.

    Example input:
        original=[1, np.nan, 3], global_imputed=[1,2,3], group_imputed=[1,1,3], global_median=2, group_medians={"A":1,"B":3}, column_name="temp"
    Example output:
        Figure/Axes with three overlaid histograms and vertical median lines.
    """
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
    """Scatter plot with a different marker for each class.

    Input:
        x, y_value, labels: equal-length vectors (n,); axis labels and title: strings.
    Output:
        Tuple: Matplotlib Figure and Axes with a marker/color group per class.

    Example input:
        x=[1, 2, 3], y_value=[2, 1, 4], labels=[0, 1, 0]
    Example output:
        Figure/Axes with points (1,2),(3,4) for class 0 and (2,1) for class 1.
    """
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
    """Scatter plus fitted one-feature regression line.

    Input:
        x and y: numeric vectors (n,); w and b: scalar slope/intercept; optional labels/title.
    Output:
        Tuple: Matplotlib Figure and Axes showing observations and fitted line.

    Example input:
        x=[1, 2, 3], y=[3, 5, 7], w=2, b=1
    Example output:
        Figure/Axes with those three points and the line y=2x+1.
    """
    x, y = np.asarray(x), np.asarray(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="observations")
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, w * line_x + b, color="red", label="fit")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_residuals(y_pred, residuals):
    """Lab 4 residual plot with a zero reference line.

    Input:
        y_pred and residuals: numeric vectors of equal length n.
    Output:
        Tuple: Matplotlib Figure and Axes with residual scatter and zero line.

    Example input:
        y_pred=[2, 4], residuals=[1, -1]
    Example output:
        Figure/Axes with two points and a horizontal line at residual 0.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="predicted value", ylabel="residual (actual - predicted)", title="Residuals")
    fig.tight_layout()
    return fig, ax


def plot_curves(curves, xlabel, ylabel, title, log_x=False, log_y=False):
    """Plot a dict {label: (x,y)}; convenient for loss, CV, ridge, and k sweeps.

    Input:
        curves: mapping label -> (x,y) vectors; axis labels/title and optional log scales.
    Output:
        Tuple: Matplotlib Figure and Axes with one labeled line per mapping entry.

    Example input:
        curves={"A": ([0,1], [0,1]), "B": ([0,1], [1,0])}, xlabel="x", ylabel="y", title="Example"
    Example output:
        Figure/Axes with two labeled lines, A and B.
    """
    fig, ax = plt.subplots()
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="." if len(x) < 30 else None, label=str(label))
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_threshold_sweep(y_true, probabilities, thresholds):
    """Lab 3: plot precision, recall and F1 against thresholds.

    Input:
        y_true: labels (n,), probabilities (n,), thresholds (T,).
    Output:
        Tuple: ((Figure, Axes), (precision, recall, F1 arrays)), each metric array shaped (T,).

    Example input:
        y_true=[0,1,1], probabilities=[0.1,0.6,0.9], thresholds=[0.5,0.8]
    Example output:
        A Figure/Axes with metric lines and arrays: precision=[1,1], recall=[1,0.5], F1=[1,0.667].
    """
    p, r, f1 = threshold_metrics(y_true, probabilities, thresholds)
    return plot_curves({"precision": (thresholds, p), "recall": (thresholds, r),
                        "F1": (thresholds, f1)}, "threshold", "score",
                       "Metrics versus threshold"), (p, r, f1)


def plot_roc_curves(curves):
    """Plot {model_name: (fpr,tpr)} ROC curves and the random baseline.

    Input:
        curves: mapping model name -> (FPR,TPR) coordinate vectors.
    Output:
        Tuple: Matplotlib Figure and Axes with ROC lines and a diagonal baseline.

    Example input:
        curves={"model": ([0,0,1], [0,1,1])}
    Example output:
        Figure/Axes with the model ROC line and the dashed diagonal baseline.
    """
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="random baseline")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves")
    ax.legend(); fig.tight_layout()
    return fig, ax


def plot_2d_decision_boundary(X_train, y_train, predict_fn, resolution=200,
                              title="Decision regions", probabilities=False):
    """Plot a model's 2-D labels or probabilities over a mesh plus training rows.

    Input:
        X_train: two features (n,2); y_train: labels (n,); predict_fn: accepts a grid (m,2); resolution: grid side length.
    Output:
        Tuple: Matplotlib Figure and Axes showing predicted regions/probabilities and training points.

    Example input:
        X_train=[[0,0],[1,1]], y_train=[0,1], predict_fn(grid) returns 0/1 labels
    Example output:
        Figure/Axes with a colored mesh and the two training points overlaid.
    """
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
    """2-D clusters with centroid markers.

    Input:
        X: 2-D points (n,2); centroids (K,2); labels (n,); title: string.
    Output:
        Tuple: Matplotlib Figure and Axes with colored clusters and centroid markers.

    Example input:
        X=[[0,0],[0,2],[8,8]], centroids=[[0,1],[8,8]], labels=[0,0,1]
    Example output:
        Figure/Axes with two colored groups and X markers at both centroids.
    """
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
