import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    w = np.dot(np.linalg.inv(np.dot(x.T, x) + 0.5 * np.eye(6)), np.dot(x.T, y))
    return np.dot(w, data)


def lasso(data):
    x, y = read_data()
    w = np.array([1, 1, 1, 1, 1, 1])

    def objective(beta):
        return np.sum((y - np.dot(x, beta))**2) + 0.5*np.sum(np.abs(beta))

    beta_init = np.zeros(6)
    beta_init[0] = np.mean(y)  # Intercept is just the mean of y
    res = minimize_scalar(objective)
    beta_hat = res.x
    return np.dot(data, beta_hat)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
