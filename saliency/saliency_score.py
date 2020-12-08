import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def saliency_score(A):
    # u, s, vh = np.linalg.svd(A, 1)
    u, s, vt = svds(A, k=1, which='LM')

    # sort u and vt
    u = np.array(abs(u)).flatten()
    vt = np.array(abs(vt)).flatten()

    return(u, np.argsort(-u), np.argsort(-vt))