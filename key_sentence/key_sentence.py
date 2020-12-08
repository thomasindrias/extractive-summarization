import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds

def rank_k_A(A, k):
    """
    Rank-k Approximation

    ## Parameters
    A: mxn Matrix
    k: Rank, k < min(m,n)
    """

    C,S,V = svds(A, k=k, which='LM')
    # C = U[:, :k]
    # D should have shape = k * n
    # C should have shape = m * k
    D = np.diag(S)@V
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
    _, C, D = rank_k_A(A, k)
    
    # QR pivoting of matrix D
    Q, RS, P = linalg.qr(D, pivoting=True)
    
    # This seems to be in the wrong order maybe?
    return P[:top], C