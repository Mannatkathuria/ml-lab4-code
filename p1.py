import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ic272_lab4_rent.csv")

def train_test_split_scratch(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)


    p = rng.permutation(len(X))
    spl = int(test_frac * len(p))

    x = np.array(X)
    y = np.array(y)

    x_test = x[p[:spl]]
    x_train = x[p[spl:]]

    y_test = y[p[:spl]]
    y_train = y[p[spl:]]

    return (x_train, x_test, y_train, y_test)

def fit_simple_lr(x, y):

    xm = x.mean()
    ym = y.mean()

    w = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    b = ym - w * xm

    print(w, b)
    return (w, b)




def design_matrix(x):

    y = np.hstack((np.ones((x.shape[0], 1)), x))
    return y

def fit_normal_equation(x, y):

    a = design_matrix(x)

    # Solve (a.T @ a)θ = a.T @ y instead of computing the inverse.
    # np.linalg.solve is faster and numerically more stable.

    theta = np.linalg.solve(a.T @ a, a.T @ y)

    return theta




def fit_gradient_descent(X, y, lr, n_iters):
    A = design_matrix(X)

    n = A.shape[0]
    theta = np.zeros(A.shape[1])

    losses = []

    for _ in range(n_iters):
        error = A @ theta - y

        loss = np.sum(error ** 2) / n
        losses.append(loss)

        grad = (2 / n) * (A.T @ error)

        theta = theta - lr * grad

    return theta, losses




def rmse(y_true, y_pred):

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_scratch(y_true, y_pred):

    ym = y_true.mean()

    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - ym) ** 2))










def ques1(df):

    print(df.min())
    print(df.max())

    x_train, x_test, y_train, y_test = train_test_split_scratch(df['area'], df['rent'])

    w, b = fit_simple_lr(x_train, y_train)
    x = np.linspace(df['area'].min(), df['area'].max(), 1000)

    plt.scatter(x_test, y_test, label="Test data")
    plt.plot(x, b + w * x, label="Fitted line")
    plt.xlabel('Area')
    plt.ylabel('Rent')

    plt.show()

    return



def ques2(df):
    X = df.drop(columns="rent")
    y = df["rent"]

    x_train, x_test, y_train, y_test = train_test_split_scratch(X, y)

    theta = fit_normal_equation(x_train, y_train)
    labels = ["intercept"] + list(X.columns)

    theta_table = pd.DataFrame({
        "Feature": labels,
        "Theta": theta.flatten()
    })

    theta_table["Theta"] = theta_table["Theta"].round(3)

    print(theta_table)
    return



def ques3(df):
    X = df.drop(columns="rent")
    y = df["rent"]

    x_train, x_test, y_train, y_test = train_test_split_scratch(X, y)

    theta_gd, losses = fit_gradient_descent(x_train, y_train, 0.005, 20000)
    theta_ln = fit_normal_equation(x_train, y_train)

    print("Normal Equation theta:")
    print(theta_ln)

    print("\nGradient Descent theta:")
    print(theta_gd)

    print("\nLargest absolute difference:")
    print(np.max(np.abs(theta_gd - theta_ln)))

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Loss")
    plt.show()

    l_rates = [0.001, 0.005, 0.01]

    for lr in l_rates:
        theta, losses = fit_gradient_descent(
            x_train,
            y_train,
            lr,
            20000
        )

        plt.plot(losses, label=f"lr = {lr}")

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent for Different Learning Rates")
    plt.legend()
    plt.show()

    return


    
def ques4(df):

    X = df.drop(columns="rent")
    y = df["rent"]

    x_train, x_test, y_train, y_test = train_test_split_scratch(X, y)

    theta, losses = fit_gradient_descent(x_train, y_train, 0.005, 20000)

    A_train = design_matrix(x_train)
    A_test = design_matrix(x_test)

    y_train_pred = A_train @ theta
    y_test_pred = A_test @ theta

    print(f"Test RMSE: {rmse(y_test, y_test_pred):.4f}")
    print(f"Test R²:   {r2_score_scratch(y_test, y_test_pred):.4f}")

    print(f"Train R²:  {r2_score_scratch(y_train, y_train_pred):.4f}")

    y_dif = y_test - y_test_pred
    plt.scatter(y_test_pred, y_dif)
    plt.axhline(0)
    plt.xlabel("Predicted Rent")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted Values")
    plt.show()

    return