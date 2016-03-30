from random import gauss
import numpy as np
from numpy import linalg as la


#### LSO
def lso(X, Y):
    x = np.array(X)
    y = np.array(Y)

    h = None
    alpha = 0
    beta = eta = np.dot(x[:, 0], x[:, 0])
    gamma = np.dot(x[:, 0], y)
    v = gamma / beta
    params = np.array([v])
    H_inv = np.array([[1 / beta]])
    y_estimated = x[:, 0] * params[0]
    rss = np.linalg.norm(y - y_estimated) ** 2
    yield params, rss
    for s in range(1, x.shape[1]):
        h = x[:, s].dot(x[:, :s])
        alpha = H_inv.dot(h)
        eta = np.dot(x[:, s], x[:, s])
        beta = eta - h.dot(alpha)
        gamma = np.dot(x[:, s], y)
        v = (gamma - h.dot(params)) / beta
        params = np.r_[params - v * alpha, v]
        H_top_left = H_inv + np.tensordot(alpha, alpha, axes=0) / beta
        H_anti_diag = -alpha / beta
        H_inv = np.r_['0,2,1', np.r_['1,2,0', H_top_left, H_anti_diag], np.r_[H_anti_diag, 1 / beta]]
        rss -= v ** 2 * beta
        yield params, rss


def make_full_info(alpha, beta, gamma, v, rss, params):
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'v': v, 'rss': rss, 'params': params}


def lso_full_info(X, Y):
    x = np.array(X)
    y = np.array(Y)

    h = None
    alpha = 0
    beta = eta = np.dot(x[:, 0], x[:, 0])
    gamma = np.dot(x[:, 0], y)
    v = gamma / beta
    params = np.array([v])
    H_inv = np.array([[1 / beta]])
    y_estimated = x[:, 0] * params[0]
    rss = np.linalg.norm(y - y_estimated) ** 2
    yield make_full_info(alpha, beta, gamma, v, rss, params)
    for s in range(1, x.shape[1]):
        h = x[:, s].dot(x[:, :s])
        alpha = H_inv.dot(h)
        eta = np.dot(x[:, s], x[:, s])
        beta = eta - h.dot(alpha)
        gamma = np.dot(x[:, s], y)
        v = (gamma - h.dot(params)) / beta
        params = np.r_[params - v * alpha, v]
        H_top_left = H_inv + np.tensordot(alpha, alpha, axes=0) / beta
        H_anti_diag = -alpha / beta
        H_inv = np.r_['0,2,1', np.r_['1,2,0', H_top_left, H_anti_diag], np.r_[H_anti_diag, 1 / beta]]
        rss -= v ** 2 * beta
        yield make_full_info(alpha, beta, gamma, v, rss, params)
