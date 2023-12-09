import numpy as np
from sklearn import ensemble
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = make_regression(n_samples=3000, n_features=10, n_informative=10, n_targets=1, bias=0.0,
                effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
data = dataset[0]
target = dataset[1]
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=33, test_size=0.25)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
gbdt = ensemble.GradientBoostingRegressor(**params)
gbdt.fit(x_train, y_train)

score = gbdt.score(x_test, y_test)
print("score: {}".format(score))

# predict
y_pred = gbdt.predict(x_test)
print("y_pred: {}".format(y_pred))

EVS = explained_variance_score(y_test, y_pred)
print("EVS: {}".format(EVS))

MSE = mean_squared_error(y_test, y_pred)
print("MSE: {}".format(MSE))

# 画图
plt.figure()
plt.plot(np.arange(0, 750, 1), y_test, c="red", label="real")
plt.plot(np.arange(0, 750, 1), y_pred, c="blue", label="pred")
plt.show()

