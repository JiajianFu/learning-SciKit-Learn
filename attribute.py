from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :])) #前四个预测值
print(model.coef_)
print(model.intercept_)
print(model.get_params())
print(model.score(data_X, data_y)) # R^2 coefficient of determination