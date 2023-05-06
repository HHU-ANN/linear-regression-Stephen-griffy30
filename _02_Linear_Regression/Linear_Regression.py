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
    x, y = data[:, :-1], data[:, -1]
    n, p = x.shape
    xtx = np.dot(x.T, x)
    ridge_term = alpha * np.identity(p)
    ridge_term[0, 0] = 0 # Don't regularize the intercept
    xty = np.dot(x.T, y)
    beta = np.dot(inv(xtx + ridge_term), xty)
    return np.dot(data[:, :-1], beta)

def lasso(data, alpha):
    x, y = data[:, :-1], data[:, -1]
    n, p = x.shape
    xtx = np.dot(x.T, x)
    xty = np.dot(x.T, y)

    def objective(beta):
        return np.sum((y - np.dot(x, beta))**2) + alpha*np.sum(np.abs(beta))

    beta_init = np.zeros(p)
    beta_init[0] = np.mean(y)  # Intercept is just the mean of y
    res = minimize_scalar(objective)
    beta_hat = res.x
    return np.dot(data[:, :-1], beta_hat)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    data = np.hstack((x, y.reshape(-1, 1)))
    return data

