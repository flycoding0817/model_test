import numpy as np


def gaussian_kernel(x1, x2, l=1.0, sigma_f=1.0):
    """Easy to understand but inefficient."""
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)


def gaussian_kernel_vectorization(x1, x2, l=1.0, sigma_f=1.0):
    """More efficient approach."""
    dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    print("dist_matrix: {}".format(dist_matrix))
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)


x = np.array([1, 2, 3]).reshape(-1, 1)
y = np.array([4, 5, 6]).reshape(-1, 1)
print("x:{}".format(x))
print("协方差矩阵： {}".format(gaussian_kernel_vectorization(x, y, l=500, sigma_f=10)))

