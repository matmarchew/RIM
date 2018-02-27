import numpy as np
import scipy.optimize as sc
from MaxEnt import posterioriProbability

def RIM(X, Y, K, D, N, alfa, lambdas):
    return np.reshape(sc.minimize(RIMFunction, lambdas, method='L-BFGS-B', args=(X, Y, K, D, N, alfa), jac=True).x, (K, D))

def RIMFunction(lambdas_, X, Y, K, D, N, alfa):
    lambdas = lambdas_.reshape((K, D))
    jac = np.zeros((K, D))
    value = 0
    proMatrix = createProbabilityMatrix(X, K, N, lambdas)
    meanColumnProMatrix = proMatrix.mean(0)
    sum = createSumArray(K, N, proMatrix, meanColumnProMatrix)
    for i in range(N):
        value_, jac_ = RIMFunctionForX(X[Y[i]], i, Y[i], K, D, proMatrix, meanColumnProMatrix, sum)
        value += np.log(value_)
        jac += jac_
    r = np.sum(lambdas[:, :1]**2)
    value -= alfa * r
    jac[:, :1] -= 2 * lambdas[:, :1] * alfa
    jac /= N
    return -value, -jac.flatten()

def RIMFunctionForX(var, n, k, K, D, proMatrix, meanProMatrixColumn, sum):
    result = proMatrix[n][k] * np.log(proMatrix[n][k])
    jac = np.zeros((K, D))
    for k in range(K):
        result -= proMatrix[n][k] * np.log(meanProMatrixColumn[k])

    for i in range(K):
        for j in range(D):
            jac[i][j] = var[j] * proMatrix[n][i] * (np.log(proMatrix[n][i] / meanProMatrixColumn[i]) - sum[n])

    return result, jac

def createProbabilityMatrix(X, K, N, lambdas):
    matrix = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            den, tmp = posterioriProbability(X[i], lambdas, K)
            matrix[i][j] = tmp[j] / den
    return matrix

def createSumArray(K, N, proMatrix, meanColumnProMatrix):
    t = []
    for i in range(N):
        tmp = 0
        for j in range(K):
            tmp += proMatrix[i][j] * np.log(proMatrix[i][j] / meanColumnProMatrix[j])
        t.append(tmp)
    return t

