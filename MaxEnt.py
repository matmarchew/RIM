import numpy as np
import scipy.optimize as sc

def maxEnt(X, Y, K, D, N, alfa):
    return np.reshape(sc.minimize(maxEntFunction, np.random.normal(loc=0, scale=1, size=K*D), method='L-BFGS-B', args=(X, Y, K, D, N, alfa), jac=True).x, (K, D))

def maxEntFunction(lambdas_, X, Y, K, D, N, alfa):
    lambdas = lambdas_.reshape((K, D))
    jac = np.zeros((K, D))
    value = 0
    for i in range(N):
        value_, jac_ = maxEntFunctionForX(X[i], Y[i], lambdas, K, D)
        value += np.log(value_)
        jac += jac_
    r = np.sum(lambdas[:, :1]**2)
    value -= alfa * r
    jac[:, :1] -= 2 * lambdas[:, :1] * alfa
    return -value, -jac.flatten()

def maxEntFunctionForX(x, y, lambdas, K, D):
    jac = np.zeros((K, D))
    den, lambdaDotX = posterioriProbability(x, lambdas, K)
    cou = lambdaDotX[y]

    for i in range(K):
        for j in range(D):
            if i == j:
                jac[i][j] = (x[j] * (den - cou)) / den
            else:
                jac[i][j] = -(x[j] * lambdaDotX[i]) / den

    return cou / den, jac

def posterioriProbability(x, lambdas, K):
    den = 0
    lambdaDotX = []
    for i in range(K):
        tmp = np.exp(lambdas[i].dot(x))
        den += tmp
        lambdaDotX.append(tmp)
    return den, lambdaDotX
