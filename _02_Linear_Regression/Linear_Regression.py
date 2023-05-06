# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
   # 假设 'data' 是一个 n x m 的 numpy 数组
    X = data[:, :-1]
    y = data[:, -1]
    XtX = np.dot(X.T, X)
    XtXpI = XtX + np.eye(X.shape[1])
    XtXpI_inv = np.linalg.inv(XtXpI)
    XtY = np.dot(X.T, y)
    w = np.dot(XtXpI_inv, XtY)
    return w
    
def lasso(data, alpha=1.0, max_iter=1000, tol=1e-4):
    # 假设 'data' 是一个 n x m 的 numpy 数组
    X = data[:, :-1]
    y = data[:, -1]
    w = np.zeros((X.shape[1], 1))
    converged = False
    iter_count = 0
    while not converged and iter_count < max_iter:
        w_old = w.copy()
        for j in range(X.shape[1]):
            a_j = 2 * np.sum(X[:, j]**2)
            c_j = 2 * np.dot(X[:, j], y - np.dot(X, w) + w[j]*X[:, j])
            if c_j < -alpha:
                w[j] = (c_j + alpha) / a_j
            elif c_j > alpha:
                w[j] = (c_j - alpha) / a_j
            else:
                w[j] = 0
        if np.linalg.norm(w - w_old) < tol:
            converged = True
        iter_count += 1
    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
