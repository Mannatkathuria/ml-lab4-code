
def permutation_importance(X, y, predict_fn, score_fn, n_repeats=5, seed=0):
    """
    Model-agnostic permutation importance on a validation/test split.
    Importance = baseline score - score after shuffling one feature.
    For loss functions where lower is better, pass score_fn = negative loss.
    """
    X, y = check_xy(X, y)
    rng = np.random.default_rng(seed)
    baseline = score_fn(y, predict_fn(X))
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            importances[j, r] = baseline - score_fn(y, predict_fn(shuffled))
    return importances.mean(axis=1), importances.std(axis=1)


def learning_curve_scores(X, y, fit_predict_fn, train_sizes, score_fn,
                          repeats=3, seed=0):
    """
    Repeated random-subset learning curve.
    fit_predict_fn(X_small, y_small, X_eval) returns predictions.
    score_fn(y_eval, predictions) returns a score (higher is better).
    """
    X, y = check_xy(X, y)
    rng = np.random.default_rng(seed)
    rows = []
    for size in train_sizes:
        size = int(size)
        if size < 1 or size >= len(y):
            raise ValueError("Each train size must be in [1, n_samples-1]")
        scores = []
        for _ in range(repeats):
            order = rng.permutation(len(y))
            train_idx, eval_idx = order[:size], order[size:]
            pred = fit_predict_fn(X[train_idx], y[train_idx], X[eval_idx])
            scores.append(score_fn(y[eval_idx], pred))
        rows.append((size, np.mean(scores), np.std(scores)))
    return np.asarray(rows, dtype=float)


# ============================================================================
# 7. Plotting helpers
# ============================================================================

def plot_learning_curve(train_sizes, train_scores, val_scores):
    """Plot training and validation score against training-set size."""
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores, marker="o", label="training")
    ax.plot(train_sizes, val_scores, marker="o", label="validation")
    ax.set(xlabel="number of training examples", ylabel="score",
           title="Learning curve")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_pca_2d(Z, y=None, title="PCA projection"):
    """Plot the first two PCA coordinates, optionally coloured by class."""
    Z = np.asarray(Z)
    if Z.shape[1] < 2:
        raise ValueError("Need at least two PCA components to make a 2-D plot")
    fig, ax = plt.subplots()
    if y is None:
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.75)
    else:
        y = np.asarray(y)
        for label in np.unique(y):
            mask = y == label
            ax.scatter(Z[mask, 0], Z[mask, 1], label=str(label), alpha=0.75)
        ax.legend(title="class")
    ax.set(xlabel="PC 1", ylabel="PC 2", title=title)
    fig.tight_layout()
    return fig, ax


def plot_feature_importance(importances, feature_names=None, title="Permutation importance"):
    """Horizontal bar plot of feature importances, sorted from low to high."""
    importances = np.asarray(importances, dtype=float)
    if feature_names is None:
        feature_names = [f"feature {i}" for i in range(len(importances))]
    order = np.argsort(importances)
    fig, ax = plt.subplots()
    ax.barh(np.asarray(feature_names)[order], importances[order])
    ax.set(xlabel="score decrease after shuffling", title=title)
    fig.tight_layout()
    return fig, ax


# ============================================================================
# 8. Quick reminders for lab work
# ============================================================================

# DATA LEAKAGE:
#   Wrong: fit scaler/imputer/PCA on the full dataset, then split.
#   Right: split first; fit preprocessing on X_train; transform val/test with it.
#
# MODEL SELECTION:
#   Use validation data or cross-validation to choose hyperparameters.
#   Do not repeatedly tune against the test set.
#
# SCALING:
#   Distance-based models (k-NN, K-means) and gradient-based optimisation are
#   often sensitive to feature scale. Tree-based models generally are not.
#
# CLASS IMBALANCE:
#   Accuracy can be misleading. Inspect class counts, per-class recall, F1,
#   balanced accuracy, and the confusion matrix.
#
# REGRESSION:
#   Inspect residuals. A pattern in residuals can indicate non-linearity,
#   heteroscedasticity, omitted variables, or a poor model.
#
# PCA:
#   PCA is unsupervised. It finds directions of high feature variance, not
#   necessarily directions that best separate target classes.
#
# RANDOMNESS:
#   Use np.random.default_rng(seed) for reproducible random operations.
#
# SHAPES:
#   X should usually be (n_samples, n_features); y should be (n_samples,).
#   A single feature should be X.reshape(-1, 1), not a 1-D X array.
#
# NUMERICAL ISSUES:
#   Use np.linalg.solve(A, b), not np.linalg.inv(A) @ b.

