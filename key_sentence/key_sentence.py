import numpy as np
from scipy import linalg

def rank_k_A(A, k):
    """
    Rank-k Approximation

    ## Parameters
    A: mxn Matrix
    k: Rank, k < min(m,n)
    """

    U,S,V = np.linalg.svd(A)
    C = U[:, :k]
    D = np.diag(S[:k])@V[:,:k].T
    return C@D, C, D

def key_sentence(A, k, top=3):
    """
    Key sentences

    ## Parameters
    A: mxn Matrix
    k: Rank, k < min(m,n)
    top: Top n sentences.
    """

    # Approximation of A ≈ C*D
    _, _, D = rank_k_A(A, k)
    
    # QR pivoting of matrix D
    Q, RS, P = linalg.qr(D, pivoting=True)
    
    return P[:top]