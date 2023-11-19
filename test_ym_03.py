import numpy as np
import matplotlib.pyplot as plt
from kernels import RBF, ConstantKernel as C
from test_ym_02 import GaussianProcessRegressor
from mpl_toolkits.mplot3d import Axes3D  # 实现数据可视化3D


#创建数据集
test = np.array([[2004, 98.31]])
data = np.array([
    [2001, 100.83, 410], [2005, 90.9, 500], [2007, 130.03, 550], [2003, 147.5, 560], [2006, 149, 580]])

kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
# alpha 会作用在协方差矩阵的对角线上
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
# print(data[:, :-1])
reg.fit(data[:, :-1], data[:, -1])
print("predict: {}".format(reg.predict([[2009, 131]])))

