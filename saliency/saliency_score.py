import numpy as np
import matplotlib.pyplot as plt

def saliency_score(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    max_values = []

    for index in range(1,u.shape[1]):
        col = u[index]
        value = max(col)
        max_values.append(value)

    plt.plot(max_values)
    plt.savefig('./plot.png')