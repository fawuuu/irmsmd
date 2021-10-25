##### Code for "Implicit Regularization in Matrix Sensing via Mirror Descent" #####

import numpy as np
from scipy import linalg
from scipy.stats import ortho_group 
import cvxpy as cp
import random
import itertools

# Define the function to generate data (square symmetric case)
# n: dimension of the matrix X*
# r: rank of the matrix X*
# m: number of observations
# completion: False: matrix sensing with random Gaussian sensing matrices
#             True: matrix completion
# psd: False: matrix sensing with symmetric indefinite X*
#      True: matrix sensing with symmetric positive semidefinite X*
def gen_data(n, r, m, completion = False, psd = True):
    if psd:
        # Generate X* = UU^T
        U = np.random.randn(n, r)
        X = np.matmul(U, np.transpose(U))
        X /= np.linalg.norm(X, ord = "nuc")
    else:
        # Generate X* = UU^T - VV^T
        U, V = np.random.randn(n, r), np.random.randn(n, r)
        X = np.matmul(U, np.transpose(U)) - np.matmul(V, np.transpose(V))
        X /= np.linalg.norm(X, ord = "nuc")

    if completion:
        # Generate sensing matrices corresponding to observing random entries
        indices = np.random.choice(int(n * (n + 1) / 2), size = m, replace = False)
        coords = list(itertools.combinations_with_replacement(range(n), 2))
        A = np.zeros((m, n, n))
        for i in range(m):
            A[i][coords[indices[i]][0]][coords[indices[i]][1]] = 1
    else:
        # Generate random Gaussian sensing matrices
        A = np.random.randn(m, n, n)    
    
    # Symmetrize sensing matrices
    A = (A + np.transpose(A, (0, 2, 1))) / 2

    y = np.einsum('kij,ij->k', A, X)

    return X, A, y

# Nuclear norm minimization
# A: sensing matrices
# y: observations
# eps: specify convergence criterion for optimization
# psd: specify if optimization variable is positive semidefinite
def cvx_opt(A, y, eps = 1.e-5, psd = True):
    n = A.shape[1]
    x = cp.Variable(shape=(n,n), PSD = psd)

    objective = cp.Minimize(cp.norm(x, 'nuc'))

    constraints = []
    for i in range(A.shape[0]):
        constraints.append(cp.abs(cp.sum(cp.multiply(x, A[i])) - y[i]) <= eps)

    problem = cp.Problem(objective, constraints)
    
    return problem.solve(solver=cp.CVXOPT, verbose=True, use_indirect=False), x.value

# Gradient descent with factorized parametrization
# A: sensing matrices
# y: observations
# alpha: initialization size
# step: step size
# iter: number of iterations
# psd: True: use parametrization X = UU^T
#      False: use parametrization X = UU^T - VV^T
def gd(A, y, alpha = 0.001, step = 0.125, iter = 100, psd = True):
    m, n = A.shape[0], A.shape[1]
    
    X = np.zeros((iter, n, n))

    if psd:
        # Run gradient descent on U, where X = UU^T
        U = np.sqrt(alpha) * np.identity(n)

        for t in range(iter):
            X_cur = np.matmul(U, np.transpose(U))
            X[t] = X_cur
            residual = np.einsum('kij,ij->k', A, X_cur) - y
            grad = np.einsum('ijk, i -> jk', A, residual) / m
            U = U - 2 * step * np.matmul(grad, U)
    
    else:
        # Run gradient descent on (U,V), where X = UU^T - VV^T 
        U, V = np.sqrt(alpha) * np.identity(n), np.sqrt(alpha) * np.identity(n)

        for t in range(iter):
            X_cur = np.matmul(U, np.transpose(U)) - np.matmul(V, np.transpose(V))
            X[t] = X_cur
            residual = np.einsum('kij,ij->k', A, X_cur) - y
            grad = np.einsum('ijk, i -> jk', A, residual) / m
            U = U - 2 * step * np.matmul(grad, U)
            V = V + 2 * step * np.matmul(grad, V)

    return X

# Mirror descent
# A: sensing matrices
# y: observations
# alpha: initialization size for spectral entropy, or mirror map
#        parameter for spectral hypentropy
# step: step size
# iter: number of iterations
# psd: True: use parametrization X = UU^T
#      False: use parametrization X = UU^T - VV^T
def md(A, y, alpha = 0.001, step = 1, iter = 100, psd = True):
    m, n =  A.shape[0], A.shape[1]

    X = np.zeros((iter, n, n))

    if psd:
        # Run mirror descent with spectral entropy mirror map
        X_cur = alpha * np.identity(n)      # X_t
        X_phi = linalg.logm(X_cur)          # \nabla\Phi(X_t)

        for t in range(iter):
            X[t] = X_cur
            residual = np.einsum('kij,ij->k', A, X_cur) - y
            grad = np.einsum('ijk, i -> jk', A, residual) / m

            X_phi = X_phi - step * grad
            X_cur = linalg.expm(X_phi)

    else:
        # Run mirror descent with spectral hypentropy mirror map
        X_cur = np.zeros((n, n))            # X_t
        X_phi = np.zeros((n, n))            # \nabla\Phi(X_t)

        for t in range(iter):
            X[t] = X_cur
            residual = np.einsum('kij,ij->k', A, X_cur) - y
            grad = np.einsum('ijk, i -> jk', A, residual) / m

            X_phi = X_phi - step * grad
            X_cur = (alpha / 2) * (linalg.expm(X_phi) - linalg.expm(-X_phi))
        
    return X

# Compute the effective rank of a matrix X
def effrank(X):
    n = min(X.shape[0], X.shape[1])
    u, sigma, vh = np.linalg.svd(X)
    h = 0 

    for j in range(n):
        # For numerical reasons ignore very small singular values (since 0 * log 0 = 0)
        if sigma[j] > 1.e-10:
            h += sigma[j]/sum(abs(sigma)) * np.log(sigma[j]/sum(abs(sigma)))
        
    return np.exp(-h)