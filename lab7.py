import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ic272_lab7_loans.csv")

features = ["income_k", "utilisation"]
X = df[features].to_numpy(float)
y = df["defaulted"].to_numpy(int)

idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]

X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]

print("train:", X_train.shape, " test:", X_test.shape)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1 / (1 + np.exp(-z))



def design_matrix(X):
    return np.c_[np.ones(X.shape[0]), X]


def log_likelihood(X, y, theta):
    A = design_matrix(X)
    h = sigmoid(A @ theta)
    eps = 1e-12

    ll = np.sum(
        y * np.log(h + eps)
        + (1 - y) * np.log(1 - h + eps)
    )

    return ll



mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

X_train_std = (X_train - mu) / sigma
X_test_std = (X_test - mu) / sigma

theta_zero = np.zeros(X_train_std.shape[1] + 1)

ll_zero = log_likelihood(
    X_train_std, y_train, theta_zero
)

print("Total log-likelihood:", ll_zero)
print("Per-example log-likelihood:", ll_zero / len(y_train))







def fit_logistic(X, y, lr=0.1, n_iters=2000):
    A = design_matrix(X)
    m, n = A.shape

    theta = np.zeros(n)
    history = []

    for i in range(n_iters):
        h = sigmoid(A @ theta)

        gradient = (A.T @ (y - h)) / m
        theta = theta + lr * gradient

        history.append(log_likelihood(X, y, theta))

    return theta, history


theta, history = fit_logistic(
    X_train_std,
    y_train,
    lr=0.1,
    n_iters=2000
)

print("Theta:", theta)
print("Initial log-likelihood:", history[0])
print("Final log-likelihood:", history[-1])

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history) + 1), history)
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.title("Log-likelihood during Gradient Ascent")
plt.grid(True)
plt.show()






A_test = design_matrix(X_test_std)

prob_default = sigmoid(A_test @ theta)

print("First 10 default probabilities:")
print(prob_default[:10].round(4))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def evaluate_threshold(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall": recall(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }


results = pd.DataFrame([
    evaluate_threshold(y_test, prob_default, 0.5),
    evaluate_threshold(y_test, prob_default, 0.3)
])

print(results.round(4).to_string(index=False))


baseline_pred = np.zeros_like(y_test)

print("Baseline accuracy:", accuracy(y_test, baseline_pred))

print("\nThreshold comparison:")
for threshold in [0.5, 0.3]:
    y_pred = (prob_default >= threshold).astype(int)

    print(f"\nThreshold = {threshold}")
    print("Accuracy :", round(accuracy(y_test, y_pred), 4))
    print("Precision:", round(precision(y_test, y_pred), 4))
    print("Recall   :", round(recall(y_test, y_pred), 4))
    print("F1       :", round(f1_score(y_test, y_pred), 4))
