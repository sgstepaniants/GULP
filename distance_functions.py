import sys
import numpy as np

# # CCA

def cca_decomp(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    assert A.shape[0] <= A.shape[1]
    assert B.shape[0] <= B.shape[1]
    
    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)
    
    evals_a = (evals_a + np.abs(evals_a)) / 2
    inv_a = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_a])
    
    evals_b = (evals_b + np.abs(evals_b)) / 2
    inv_b = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_b])

    cov_ab = A @ B.T

    temp = (
        (evecs_a @ np.diag(inv_a) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = np.linalg.svd(temp)
    except:
        u, s, vh = np.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ np.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ np.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b


def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho * rho) / len(rho)


def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho) / len(rho)


def pwcca_dist(A, rho, transformed_a):
    """Computes projection weighted CCA distance between A and B given the correlation
    coefficients rho and the transformed matrices after running CCA
    :param A: np.array of size a x n where a is the number of neurons and n is the dataset size
    :param B: np.array of size b x n where b is the number of neurons and n is the dataset size
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    :param transformed_a: canonical vectors for A returned by cca_decomp(A,B)
    :param transformed_b: canonical vectors for B returned by cca_decomp(A,B)
    :return: PWCCA distance
    """
    in_prod = transformed_a.T @ A.T
    weights = np.sum(np.abs(in_prod), axis=1)
    weights = weights / np.sum(weights)
    dim = min(len(weights), len(rho))
    return 1 - np.dot(weights[:dim], rho[:dim])


# # CKA


def lin_cka_dist(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    """
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(
        B @ B.T, ord="fro"
    )
    return 1 - similarity / normalization


def lin_cka_prime_dist(A, B):
    """
    Computes Linear CKA prime distance bewteen representations A and B
    The version here is suited to a, b >> n
    """
    if A.shape[0] > A.shape[1]:
        At_A = A.T @ A  # O(n * n * a)
        Bt_B = B.T @ B  # O(n * n * a)
        numerator = np.sum((At_A - Bt_B) ** 2)
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return numerator / denominator
    else:
        similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return 1 - 2 * similarity / denominator


# # Procrustes


def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    """
    n = A.shape[1]
    A_sq_frob = np.sum(A ** 2) / n
    B_sq_frob = np.sum(B ** 2) / n
    nuc = np.linalg.norm(A @ B.T, ord="nuc") / n  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc


# # Our predictor distances

def predictor_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k <= n
    assert l <= n
    
    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a])
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b])
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])
    
    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))
    
    cov_ab = A @ B.T / n
    T3 = np.trace(
        (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
        @ cov_ab.T
        @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    )
    
    return T1 + T2 - 2*T3

def squared_procrustes(A, B):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k < n
    assert l < n

    cov_a = A @ A.T / (n-1)
    cov_b = B @ B.T / (n-1)
    cov_ab = A @ B.T / (n-1)
    
    T1 = np.trace(cov_a @ cov_a)
    T2 = np.trace(cov_b @ cov_b)
    T3 = np.trace(cov_ab @ cov_ab.T)
    
    return T1 + T2 - 2*T3
