import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

T = 100
x_axis = np.linspace(0, 2*np.pi, T)
y_axis = np.sin(x_axis)

N = 30
idx = np.random.choice(T, size=N, replace=False)
xtrain = x_axis[idx].reshape(N, 1)
ytrain = y_axis[idx]

model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
prediction = model.predict(x_axis.reshape(T, 1))
print("score for 1 tree:", model.score(x_axis.reshape(T, 1), y_axis))

plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()


class BaggedTreeRegressor:
    def __init__(self, B):
        self.B = B

    def fit(self, x, y):
        n = len(x)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(n, size=n, replace=True)
            xb = x[idx]
            yb = y[idx]

            model = DecisionTreeRegressor()
            model.fit(xb, yb)
            self.models.append(model)

    def predict(self, x):
        predictions = np.zeros(len(x))
        for model in self.models:
            predictions += model.predict(x)
        return predictions / self.B

    def score(self, x, y):
        d1 = y - self.predict(x)
        d2 = y - y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)


model = BaggedTreeRegressor(200)
model.fit(xtrain, ytrain)
print("Score for bagged Tree:", model.score(x_axis.reshape(T, 1), y_axis))
prediction = model.predict(x_axis.reshape(T, 1))

plt.plot(x_axis, prediction)
plt.plot(x_axis, y_axis)
plt.show()