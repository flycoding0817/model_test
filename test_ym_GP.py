import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# boston数据
# data, target = load_boston(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=33, test_size=0.25)

# sklearn 回归数据集
dataset = make_regression(n_samples=3000, n_features=10, n_informative=10, n_targets=1, bias=0.0,
                effective_rank = None, tail_strength=0.5, noise=10, shuffle=True, coef=False, random_state=None)
data = dataset[0]
target = dataset[1]
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=33, test_size=0.25)

print("y_test: {}".format(y_test))
ss_x, ss_y = StandardScaler(), StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape([-1,1])).reshape(-1)
y_test = ss_y.transform(y_test.reshape([-1,1])).reshape(-1)

# kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
kernel = C(0.1, (0.001, 100)) * RBF(5, (1e-4, 60))
# reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
reg.fit(x_train, y_train)
mu, cov = reg.predict(x_test, return_cov=True)


# 标准化还原
mu = ss_y.inverse_transform(mu)
y_test = ss_y.inverse_transform(y_test)
print("mu_inverse: {}".format(mu))


EVS = explained_variance_score(y_test, mu)
print("EVS: {}".format(EVS))

# 均方误差 MSE
MSE = mean_squared_error(y_test, mu)
print("MSE: {}".format(MSE))

# 画图
plt.figure()
plt.plot(np.arange(0, 750, 1), y_test, c="red", label="real")
plt.plot(np.arange(0, 750, 1), mu, c="blue", label="pred")

plt.show()














