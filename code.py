import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def column_stats(X):


    import numpy as np
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0)

def cap_above_and_count_rows(X, threshold):


    import numpy as np
    X = np.asarray(X)
    capped = np.minimum(X, threshold).copy()
    row_counts = (X > threshold).sum(axis=1)
    return capped, row_counts

def pairwise_distances(A, B):


    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def closest_indices(D):


    import numpy as np
    return np.argmin(D, axis=1)

def _safe_divide(numerator, denominator):


    import numpy as np
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=float)
    np.divide(numerator, denominator, out=out, where=(denominator != 0))
    return out

def confusion_matrix(y_true, y_pred, labels=None):


    import numpy as np
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


    import numpy as np
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def precision_recall_f1_from_cm(cm):


    import numpy as np
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    return precision, recall, f1, float(np.mean(f1))

def precision_recall_f1(y_true, y_pred, positive_label=1):


    import numpy as np
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def binary_metrics(y_true, y_pred, positive_label=1):


    import numpy as np
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("binary_metrics expects labels 0 and 1")
        cm[int(actual), int(predicted)] += 1
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(y_true == y_pred))
    return accuracy, precision, recall, f1, cm

def train_test_split(X, y, test_size=0.2, seed=42):


    import numpy as np
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def train_test_split_scratch(X, y, test_frac=0.2, seed=0):


    import numpy as np
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_frac)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def fixed_count_split(X, y, n_test=100, seed=0):


    import numpy as np
    X, y = np.asarray(X), np.asarray(y)
    order = np.random.default_rng(seed).permutation(len(y))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def stratified_train_test_split(X, y, test_size=0.2, seed=42):


    import numpy as np
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


    import numpy as np
    values = np.asarray(values).ravel()
    if categories is None:
        categories = np.unique(values)
    categories = np.asarray(categories)
    encoded = np.column_stack([values == category for category in categories]).astype(int)
    return encoded, categories

def one_hot_encode_pair(train_values, test_values):


    import numpy as np
    train_values = np.asarray(train_values).ravel()
    test_values = np.asarray(test_values).ravel()
    categories = np.unique(train_values)
    train_encoded = np.column_stack([train_values == category for category in categories]).astype(int)
    test_encoded = np.column_stack([test_values == category for category in categories]).astype(int)
    return train_encoded, test_encoded, categories

def impute_median_train_test(X_train, X_test):


    import numpy as np
    X_train = np.asarray(X_train, dtype=float).copy()
    X_test = np.asarray(X_test, dtype=float).copy()
    medians = np.nanmedian(X_train, axis=0)
    rows, cols = np.where(np.isnan(X_train))
    X_train[rows, cols] = medians[cols]
    rows, cols = np.where(np.isnan(X_test))
    X_test[rows, cols] = medians[cols]
    return X_train, X_test, medians

def impute_series_global_median(series):

    median = series.median()
    return series.fillna(median), median

def impute_series_by_group(df, value_column, group_column):

    group_medians = df.groupby(group_column)[value_column].transform("median")
    return df[value_column].fillna(group_medians), group_medians

def missing_rate_by_class(df, target_column, feature_columns):

    return df.groupby(target_column)[feature_columns].apply(lambda part: part.isna().mean())

def majority_class_predict(y_train, n_samples):


    import numpy as np
    labels, counts = np.unique(y_train, return_counts=True)
    return np.full(n_samples, labels[np.argmax(counts)])

def fit_simple_lr(x, y):


    import numpy as np
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denom = np.sum(x_centered ** 2)
    w = 0.0 if denom == 0 else np.sum(x_centered * y_centered) / denom
    b = y.mean() - w * x.mean()
    return w, b

def design_matrix(X):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.c_[np.ones(len(X)), X]

def fit_normal_equation(X, y):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    return np.linalg.solve(A.T @ A, A.T @ y)

def fit_gradient_descent(X, y, lr, n_iters):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    losses = []
    for _ in range(n_iters):
        error = A @ theta - y
        theta -= lr * (2 / len(y)) * (A.T @ error)
        losses.append(float(np.mean((A @ theta - y) ** 2)))
    return theta, losses

def predict_linear(X, theta):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    return A @ np.asarray(theta, dtype=float)

def rmse(y_true, y_pred):


    import numpy as np
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def r2_score_scratch(y_true, y_pred):


    import numpy as np
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

def standardise(X_train, X_test):


    import numpy as np
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma_safe, (X_test - mu) / sigma_safe, mu, sigma_safe

def unstandardise_coefficients(theta, mu, sigma):


    import numpy as np
    theta = np.asarray(theta, dtype=float)
    raw_w = theta[1:] / np.asarray(sigma)
    raw_b = theta[0] - np.sum(raw_w * np.asarray(mu))
    return raw_w, raw_b

def fit_ridge(X, y, lam):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(A.T @ A + lam * penalty, A.T @ y)

def k_fold_cv(X, y, lam, k=5, seed=0):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, X_val = X[train_idx], X[val_idx]
        A_train = np.column_stack([np.ones(len(X_train)), X_train])
        A_val = np.column_stack([np.ones(len(X_val)), X_val])
        penalty = np.eye(A_train.shape[1])
        penalty[0, 0] = 0
        theta = np.linalg.solve(A_train.T @ A_train + lam * penalty,
                                A_train.T @ y[train_idx])
        predictions = A_val @ theta
        scores.append(np.sqrt(np.mean((y[val_idx] - predictions) ** 2)))
    return float(np.mean(scores))

def ridge_cv_sweep(X, y, lambda_values, k=5, seed=0):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    rows = []
    for lam in lambda_values:
        scores = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            X_train, X_val = X[train_idx], X[val_idx]
            A_train = np.column_stack([np.ones(len(X_train)), X_train])
            A_val = np.column_stack([np.ones(len(X_val)), X_val])
            penalty = np.eye(A_train.shape[1])
            penalty[0, 0] = 0
            theta = np.linalg.solve(A_train.T @ A_train + lam * penalty,
                                    A_train.T @ y[train_idx])
            predictions = A_val @ theta
            scores.append(np.sqrt(np.mean((y[val_idx] - predictions) ** 2)))
        rows.append((lam, float(np.mean(scores))))
    return np.asarray(rows, dtype=float)

def ridge_coefficient_path(X, y, lambda_values):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(A.shape[1])
    penalty[0, 0] = 0
    return np.asarray([np.linalg.solve(A.T @ A + lam * penalty, A.T @ y)
                       for lam in lambda_values])

def soft_threshold(v, t):


    import numpy as np
    v = np.asarray(v)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)

def fit_lasso(X, y, lam, lr=0.05, n_iters=30000):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    for _ in range(n_iters):
        gradient = (2 / len(y)) * (A.T @ (A @ theta - y))
        theta -= lr * gradient
        weights = theta[1:]
        theta[1:] = np.sign(weights) * np.maximum(np.abs(weights) - lr * lam, 0)
    return theta

def sigmoid(z):


    import numpy as np
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    z = np.clip(A @ np.asarray(theta, dtype=float), -709, 709)
    p = 1 / (1 + np.exp(-z))
    eps = 1e-12
    return float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

def fit_logistic(X, y, lr=0.1, n_iters=2000):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    theta = np.zeros(A.shape[1])
    log_likelihoods = []
    eps = 1e-12
    for _ in range(n_iters):
        z = np.clip(A @ theta, -709, 709)
        probabilities = 1 / (1 + np.exp(-z))
        theta += (lr / len(y)) * (A.T @ (y - probabilities))
        z_after = np.clip(A @ theta, -709, 709)
        p_after = 1 / (1 + np.exp(-z_after))
        log_likelihoods.append(float(np.sum(y * np.log(p_after + eps)
                                             + (1 - y) * np.log(1 - p_after + eps))))
    return theta, log_likelihoods

def predict_proba_logistic(X, theta):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    z = np.clip(A @ np.asarray(theta, dtype=float), -709, 709)
    return 1 / (1 + np.exp(-z))

def predict_logistic(X, theta, threshold=0.5):


    import numpy as np
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    z = np.clip(A @ np.asarray(theta, dtype=float), -709, 709)
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= threshold).astype(int)

def threshold_metrics(y_true, probabilities, thresholds):


    import numpy as np
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()
    precision_values, recall_values, f1_values = [], [], []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tp = np.sum((y_true == 1) & predictions)
        fp = np.sum((y_true != 1) & predictions)
        fn = np.sum((y_true == 1) & ~predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return np.asarray(precision_values), np.asarray(recall_values), np.asarray(f1_values)

def feature_threshold_sweep(y_true, feature, thresholds, greater_is_positive=True):


    import numpy as np
    y_true, feature = np.asarray(y_true).ravel(), np.asarray(feature).ravel()
    rows = []
    for threshold in thresholds:
        predictions = feature > threshold if greater_is_positive else feature < threshold
        tp = np.sum((y_true == 1) & predictions)
        fp = np.sum((y_true != 1) & predictions)
        fn = np.sum((y_true == 1) & ~predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((precision, recall, f1))
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]

def roc_curve_scratch(y_true, p):


    import numpy as np
    y_true, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    thresholds = np.r_[np.inf, np.unique(p)[::-1], -np.inf]
    tpr, fpr = [], []
    positives, negatives = np.sum(y_true == 1), np.sum(y_true == 0)
    for threshold in thresholds:
        predictions = p >= threshold
        tp = np.sum(predictions & (y_true == 1))
        fp = np.sum(predictions & (y_true == 0))
        tpr.append(float(tp / positives) if positives else 0.0)
        fpr.append(float(fp / negatives) if negatives else 0.0)
    order = np.argsort(fpr, kind="stable")
    return np.asarray(fpr)[order], np.asarray(tpr)[order]

def auc_trapezoid(fpr, tpr):


    import numpy as np
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

def knn_predict(X_train, y_train, X_query, k):


    import numpy as np
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train)
    differences = X_query[:, None, :] - X_train[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_labels = y_train[neighbor_indices]
    classes = np.unique(y_train)
    class_codes = np.searchsorted(classes, neighbor_labels)
    votes = np.eye(len(classes), dtype=int)[class_codes].sum(axis=1)
    return classes[np.argmax(votes, axis=1)]

def knn_predict_weighted(X_train, y_train, X_query, k, eps=1e-12):


    import numpy as np
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train)
    differences = X_query[:, None, :] - X_train[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, indices, axis=1)
    weights = 1 / (nearest_distances + eps)
    neighbor_labels = y_train[indices]
    classes = np.unique(y_train)
    votes = np.stack([(weights * (neighbor_labels == label)).sum(axis=1)
                      for label in classes], axis=1)
    return classes[np.argmax(votes, axis=1)]

def k_fold_cv_knn(X, y, k, folds=5, seed=0):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim == 1:
        X = X[:, None]
    groups = np.array_split(np.random.default_rng(seed).permutation(len(y)), folds)
    accuracies = []
    for i in range(folds):
        validation_indices = groups[i]
        training_indices = np.concatenate([groups[j] for j in range(folds) if j != i])
        X_train, X_validation = X[training_indices], X[validation_indices]
        y_train, y_validation = y[training_indices], y[validation_indices]
        differences = X_validation[:, None, :] - X_train[None, :, :]
        distances = np.sqrt(np.sum(differences ** 2, axis=2))
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        neighbor_labels = y_train[neighbor_indices]
        classes = np.unique(y_train)
        class_codes = np.searchsorted(classes, neighbor_labels)
        votes = np.eye(len(classes), dtype=int)[class_codes].sum(axis=1)
        predictions = classes[np.argmax(votes, axis=1)]
        accuracies.append(np.mean(y_validation == predictions))
    return float(np.mean(accuracies))

def knn_k_sweep(X, y, k_values, folds=5, seed=0):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim == 1:
        X = X[:, None]
    groups = np.array_split(np.random.default_rng(seed).permutation(len(y)), folds)
    rows = []
    for candidate_k in k_values:
        neighbor_count = int(candidate_k)
        accuracies = []
        for i in range(folds):
            validation_indices = groups[i]
            training_indices = np.concatenate([groups[j] for j in range(folds) if j != i])
            X_train, X_validation = X[training_indices], X[validation_indices]
            y_train, y_validation = y[training_indices], y[validation_indices]
            differences = X_validation[:, None, :] - X_train[None, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=2))
            neighbor_indices = np.argsort(distances, axis=1)[:, :neighbor_count]
            neighbor_labels = y_train[neighbor_indices]
            classes = np.unique(y_train)
            class_codes = np.searchsorted(classes, neighbor_labels)
            votes = np.eye(len(classes), dtype=int)[class_codes].sum(axis=1)
            predictions = classes[np.argmax(votes, axis=1)]
            accuracies.append(np.mean(y_validation == predictions))
        rows.append((neighbor_count, float(np.mean(accuracies))))
    return np.asarray(rows, dtype=float)

def append_gaussian_noise(X_train, X_test, n_noise, seed=0):


    import numpy as np
    rng = np.random.default_rng(seed)
    train_noise = rng.normal(0, 1, size=(len(X_train), n_noise))
    test_noise = rng.normal(0, 1, size=(len(X_test), n_noise))
    return np.c_[X_train, train_noise], np.c_[X_test, test_noise]

def distance_concentration_ratio(D):


    import numpy as np
    D = np.asarray(D)
    return float(D.min() / D.max()) if D.max() != 0 else 0.0

def gaussian_pdf(x, mu, var):


    import numpy as np
    x, mu, var = np.asarray(x), np.asarray(mu), np.maximum(np.asarray(var), 1e-12)
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

def fit_gaussian_nb(X, y):


    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    priors = np.zeros(2)
    means = np.zeros((2, X.shape[1]))
    variances = np.zeros((2, X.shape[1]))
    for c in (0, 1):
        Xc = X[y == c]
        if len(Xc):
            priors[c] = len(Xc) / len(y)
            means[c] = Xc.mean(axis=0)
            variances[c] = Xc.var(axis=0, ddof=0)
    return priors, means, variances

def _gaussian_log_scores(X, priors, means, variances):


    import numpy as np
    X = np.asarray(X, dtype=float)
    safe_var = np.maximum(np.asarray(variances), 1e-12)
    log_density = -0.5 * (np.log(2 * np.pi * safe_var)[None, :, :]
                          + (X[:, None, :] - means[None, :, :]) ** 2
                          / safe_var[None, :, :])
    return np.log(np.asarray(priors)[None, :] + 1e-300) + log_density.sum(axis=2)

def predict_proba_nb(X, priors, means, variances):


    import numpy as np
    X = np.asarray(X, dtype=float)
    priors = np.asarray(priors, dtype=float)
    means = np.asarray(means, dtype=float)
    safe_variances = np.maximum(np.asarray(variances, dtype=float), 1e-12)
    log_density = -0.5 * (np.log(2 * np.pi * safe_variances)[None, :, :]
                          + (X[:, None, :] - means[None, :, :]) ** 2
                          / safe_variances[None, :, :])
    log_scores = np.log(priors[None, :] + 1e-300) + log_density.sum(axis=2)
    difference = np.clip(log_scores[:, 1] - log_scores[:, 0], -709, 709)
    return 1 / (1 + np.exp(-difference))

def fit_categorical_nb(e, y, l=1):


    import numpy as np
    e, y = np.asarray(e, dtype=int).ravel(), np.asarray(y, dtype=int).ravel()
    K = 4
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


    import numpy as np
    X = np.asarray(X, dtype=float)
    e = np.asarray(e, dtype=int).ravel()
    priors = np.asarray(priors, dtype=float)
    means = np.asarray(means, dtype=float)
    safe_variances = np.maximum(np.asarray(variances, dtype=float), 1e-12)
    log_density = -0.5 * (np.log(2 * np.pi * safe_variances)[None, :, :]
                          + (X[:, None, :] - means[None, :, :]) ** 2
                          / safe_variances[None, :, :])
    log_scores = np.log(priors[None, :] + 1e-300) + log_density.sum(axis=2)
    category_probabilities = np.asarray(category_probabilities, dtype=float)
    log_scores[:, 0] += np.log(category_probabilities[0, e] + 1e-300)
    log_scores[:, 1] += np.log(category_probabilities[1, e] + 1e-300)
    difference = np.clip(log_scores[:, 1] - log_scores[:, 0], -709, 709)
    return 1 / (1 + np.exp(-difference))

def random_init(X, K, seed):


    import numpy as np
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return np.asarray(X)[idx].copy()

def assign_clusters(X, centroids):


    import numpy as np
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    differences = X[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, centroids):


    import numpy as np
    X, labels = np.asarray(X), np.asarray(labels)
    updated = np.asarray(centroids, dtype=float).copy()
    for c in range(len(updated)):
        members = X[labels == c]
        if len(members):
            updated[c] = members.mean(axis=0)
    return updated

def distortion(X, labels, centroids):


    import numpy as np
    residual = np.asarray(X) - np.asarray(centroids)[np.asarray(labels)]
    return float(np.sum(residual ** 2))

def kmeans(X, init_centroids, max_iter=100):


    import numpy as np
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    for iteration in range(1, max_iter + 1):
        differences = X[:, None, :] - centroids[None, :, :]
        distances = np.sqrt(np.sum(differences ** 2, axis=2))
        labels = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for cluster in range(len(centroids)):
            members = X[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        converged = np.allclose(updated, centroids)
        centroids = updated
        if converged:
            differences = X[:, None, :] - centroids[None, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=2))
            labels = np.argmin(distances, axis=1)
            residuals = X - centroids[labels]
            return centroids, labels, float(np.sum(residuals ** 2)), iteration
    differences = X[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    residuals = X - centroids[labels]
    return centroids, labels, float(np.sum(residuals ** 2)), max_iter

def best_of_kmeans(X, K, n_init=10):


    import numpy as np
    X = np.asarray(X, dtype=float)
    best = None
    for seed in range(n_init):
        indices = np.random.default_rng(seed).choice(len(X), K, replace=False)
        centroids = X[indices].copy()
        for iteration in range(1, 101):
            differences = X[:, None, :] - centroids[None, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=2))
            labels = np.argmin(distances, axis=1)
            updated = centroids.copy()
            for cluster in range(K):
                members = X[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
            converged = np.allclose(updated, centroids)
            centroids = updated
            if converged:
                break
        differences = X[:, None, :] - centroids[None, :, :]
        labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
        residuals = X - centroids[labels]
        result = (centroids, labels, float(np.sum(residuals ** 2)), iteration)
        if best is None or result[2] < best[2]:
            best = result
    return best

def kmeans_multistart(X, K, n_init=10):


    import numpy as np
    X = np.asarray(X, dtype=float)
    results = []
    for seed in range(n_init):
        indices = np.random.default_rng(seed).choice(len(X), K, replace=False)
        centroids = X[indices].copy()
        for iteration in range(1, 101):
            differences = X[:, None, :] - centroids[None, :, :]
            labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
            updated = centroids.copy()
            for cluster in range(K):
                members = X[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
            converged = np.allclose(updated, centroids)
            centroids = updated
            if converged:
                break
        differences = X[:, None, :] - centroids[None, :, :]
        labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
        residuals = X - centroids[labels]
        results.append((centroids, labels, float(np.sum(residuals ** 2)), iteration))
    return results

def kmeans_pp_init(X, K, seed):


    import numpy as np
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(X)))]
    while len(chosen) < K:
        differences = X[:, None, :] - X[np.asarray(chosen)][None, :, :]
        nearest_squared = np.min(np.sum(differences ** 2, axis=2), axis=1)
        nearest_squared[chosen] = 0
        total = nearest_squared.sum()
        if total == 0:
            remaining = np.setdiff1d(np.arange(len(X)), chosen)
            next_index = int(rng.choice(remaining))
        else:
            next_index = int(rng.choice(len(X), p=nearest_squared / total))
        chosen.append(next_index)
    return X[chosen].copy()

def kmeans_elbow_sweep(X, K_values, n_init=10):


    import numpy as np
    X = np.asarray(X, dtype=float)
    rows = []
    for candidate in K_values:
        K = int(candidate)
        best_distortion = None
        best_iterations = None
        for seed in range(n_init):
            indices = np.random.default_rng(seed).choice(len(X), K, replace=False)
            centroids = X[indices].copy()
            for iteration in range(1, 101):
                differences = X[:, None, :] - centroids[None, :, :]
                labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
                updated = centroids.copy()
                for cluster in range(K):
                    members = X[labels == cluster]
                    if len(members):
                        updated[cluster] = members.mean(axis=0)
                converged = np.allclose(updated, centroids)
                centroids = updated
                if converged:
                    break
            differences = X[:, None, :] - centroids[None, :, :]
            labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
            residuals = X - centroids[labels]
            score = float(np.sum(residuals ** 2))
            if best_distortion is None or score < best_distortion:
                best_distortion = score
                best_iterations = iteration
        rows.append((candidate, best_distortion, best_iterations))
    return np.asarray(rows, dtype=float)

def segment_image_kmeans(image, K, seed=0, max_iter=100):


    import numpy as np
    image = np.asarray(image)
    rgb = image[:, :, :3]
    height, width, channels = rgb.shape
    pixels = rgb.reshape(-1, channels).astype(float)
    indices = np.random.default_rng(seed).choice(len(pixels), K, replace=False)
    centroids = pixels[indices].copy()
    for iteration in range(1, max_iter + 1):
        differences = pixels[:, None, :] - centroids[None, :, :]
        labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
        updated = centroids.copy()
        for cluster in range(K):
            members = pixels[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        converged = np.allclose(updated, centroids)
        centroids = updated
        if converged:
            break
    differences = pixels[:, None, :] - centroids[None, :, :]
    labels = np.argmin(np.sum(differences ** 2, axis=2), axis=1)
    residuals = pixels - centroids[labels]
    total_distortion = float(np.sum(residuals ** 2))
    segmented = centroids[labels].reshape(height, width, channels)
    return segmented, centroids, labels, total_distortion, iteration

def plot_feature_histograms(X, feature_names, bins=20):




    import numpy as np
    import matplotlib.pyplot as plt
    X = np.asarray(X, dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for j, ax in enumerate(axes.ravel()):
        if j >= X.shape[1]:
            ax.axis("off")
            continue
        ax.hist(X[:, j], bins=bins, alpha=0.8)
        ax.axvline(means[j], color="red", label="mean")
        ax.axvline(means[j] - stds[j], color="black", linestyle="--", label="mean +/- std")
        ax.axvline(means[j] + stds[j], color="black", linestyle="--")
        ax.set(title=feature_names[j], xlabel=feature_names[j], ylabel="count")
        ax.legend()
    fig.tight_layout()
    return fig, axes

def plot_distance_matrix(D):



    import numpy as np
    import matplotlib.pyplot as plt
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



    import numpy as np
    import matplotlib.pyplot as plt
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




    import numpy as np
    import matplotlib.pyplot as plt
    rates = df.groupby(target_column)[feature_columns].apply(lambda part: part.isna().mean())
    fig, ax = plt.subplots()
    x = np.arange(len(feature_columns))
    width = 0.8 / max(1, len(rates))
    for i, (label, row) in enumerate(rates.iterrows()):
        ax.bar(x + (i - (len(rates) - 1) / 2) * width, row.to_numpy(),
               width=width, label=str(label))
    ax.set_xticks(x, labels=feature_columns)
    ax.set(xlabel="feature", ylabel="missing fraction", title="Missingness by class")
    ax.legend(title=target_column)
    fig.tight_layout()
    return fig, ax, rates

def plot_imputation_distributions(original, global_imputed, group_imputed,
                                  global_median, group_medians, column_name):


    import matplotlib.pyplot as plt
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



    import numpy as np
    import matplotlib.pyplot as plt
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



    import numpy as np
    import matplotlib.pyplot as plt
    x, y = np.asarray(x), np.asarray(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="observations")
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, w * line_x + b, color="red", label="fit")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax

def plot_residuals(y_pred, residuals):


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="predicted value", ylabel="residual (actual - predicted)", title="Residuals")
    fig.tight_layout()
    return fig, ax

def plot_curves(curves, xlabel, ylabel, title, log_x=False, log_y=False):


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="." if len(x) < 30 else None, label=str(label))
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(); fig.tight_layout()
    return fig, ax

def plot_threshold_sweep(y_true, probabilities, thresholds):




    import numpy as np
    import matplotlib.pyplot as plt
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities).ravel()
    thresholds = np.asarray(thresholds)
    precision_values, recall_values, f1_values = [], [], []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tp = np.sum((y_true == 1) & predictions)
        fp = np.sum((y_true != 1) & predictions)
        fn = np.sum((y_true == 1) & ~predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    precision_values = np.asarray(precision_values)
    recall_values = np.asarray(recall_values)
    f1_values = np.asarray(f1_values)
    fig, ax = plt.subplots()
    ax.plot(thresholds, precision_values, marker=".", label="precision")
    ax.plot(thresholds, recall_values, marker=".", label="recall")
    ax.plot(thresholds, f1_values, marker=".", label="F1")
    ax.set(xlabel="threshold", ylabel="score", title="Metrics versus threshold")
    ax.legend()
    fig.tight_layout()
    return (fig, ax), (precision_values, recall_values, f1_values)

def plot_roc_curves(curves):


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", label="random baseline")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves")
    ax.legend(); fig.tight_layout()
    return fig, ax

def plot_2d_decision_boundary(X_train, y_train, predict_fn, resolution=200,
                              title="Decision regions", probabilities=False):



    import numpy as np
    import matplotlib.pyplot as plt
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



    import numpy as np
    import matplotlib.pyplot as plt
    X, centroids, labels = np.asarray(X), np.asarray(centroids), np.asarray(labels)
    fig, ax = plt.subplots()
    points = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=18)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", c="black", s=130, label="centroids")
    ax.set(xlabel="feature 1", ylabel="feature 2", title=title, aspect="equal")
    ax.legend(); fig.colorbar(points, ax=ax, label="cluster")
    fig.tight_layout()
    return fig, ax