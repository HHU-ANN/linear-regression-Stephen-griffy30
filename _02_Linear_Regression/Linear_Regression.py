import os

try:
    import numpy as np
    from scipy.linalg import inv
    from scipy.optimize import minimize_scalar
except ImportError as e:
    os.system("sudo pip3 install numpy scipy")
    import numpy as np
    from scipy.linalg import inv
    from scipy.optimize import minimize_scalar

def ridge(data, alpha):
    X, y = data[:, :-1], data[:, -1]
    X_T = X.T
    n, p = X.shape
    I = np.identity(p)
    beta = np.linalg.inv(X_T @ X + alpha * I) @ X_T @ y
    return X @ beta
    
def lasso(data, alpha):
    X, y = data[:, :-1], data[:, -1]
    n, p = X.shape
    beta = np.zeros(p)
    w = np.ones(p)
    X_T = X.T
    converged = False
    while not converged:
        for j in range(p):
            beta_except_j = np.delete(beta, j)
            X_except_j = np.delete(X, j, 1)
            y_hat = X_except_j @ beta_except_j
            r_j = X[:, j] @ (y - y_hat)
            z_j = X[:, j] @ X_except_j
            soft_t = np.abs(r_j) - alpha / 2
            if soft_t < 0:
                beta[j] = 0
            else:
                if r_j < 0:
                    beta[j] = - soft_t / z_j
                else:
                    beta[j] = soft_t / z_j
        if np.linalg.norm(beta - w) < 1e-6:
            converged = True
        w = beta.copy()
    return X @ beta

def objective(beta):
        return np.sum((y - np.dot(x, beta))**2) + alpha*np.sum(np.abs(beta))

    beta_init = np.zeros(p)
    beta_init[0] = np.mean(y)  # Intercept is just the mean of y
    res = minimize_scalar(objective)
    beta_hat = res.x
    return np.dot(data[:, :-1], beta_hat)

def main():
    path = './data/exp02/'
    X, y = read_data(path)
    data = np.concatenate((X, y[:, np.newaxis]), axis=1)
    alpha = 0.1
    ridge_result = ridge(data, alpha)
    lasso_result = lasso(data, alpha)
    print("Ridge Result:", ridge_result)
    print("Lasso Result:", lasso_result)

