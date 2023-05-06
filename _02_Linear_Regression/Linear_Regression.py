# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    # 假设 'data' 是一个 n x m 的 numpy 数组
    X = data[:-1]
    y = data[-1]
    XtX = np.dot(X.T, X)
    lambd = 0.1
    I = np.eye(XtX.shape[0])
    w = np.dot(np.linalg.inv(XtX + lambd*I), np.dot(X.T, y))
    return w
    
def lasso(data):
    # 假设 'data' 是一个 n x m 的 numpy 数组
    X = data.reshape((-1, 6))[:, :-1]
    y = data.reshape((-1, 6))[:, -1]
    n, m = X.shape
    # 实现 Lasso 回归
    w = np.zeros(m)
    alpha = 0.1
    n_iterations = 1000
    for iteration in range(n_iterations):
        y_pred = X.dot(w)
        gradients = 2 / n * X.T.dot(y_pred - y) + alpha * np.sign(w)
        w = w - 0.01 * gradients
    return X.dot(w)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
