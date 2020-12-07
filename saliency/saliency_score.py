import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def saliency_score(A):
    # u, s, vh = np.linalg.svd(A, 1)
    u, s, vt = svds(A, k=1, which='LM')

    # u = u[0]

    u = abs(u)

    return(u, np.argsort(-u.T), np.argsort(vt))