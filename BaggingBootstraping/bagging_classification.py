import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def plot_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


n = 500
d = 2
x = np.random.randn(n, d)

sep = 2
x[:125] += np.array([sep, sep])
x[125:250] += np.array([-sep, -sep])
x[250:375] += np.array([sep, -sep])
x[375:] += np.array([-sep, sep])
y = np.array([0]*125 + [0]*125 + [1]*125 + [1]*125)

plt.scatter(x[:, 0], x[:, 1], s=100, c=y, alpha=0.5)
plt.show()

model = DecisionTreeClassifier()
model.fit(x, y)
print('Score for 1 tree {}'.format(model.score(x, y)))

plt.scatter(x[:, 0], x[:, 1], s=100, c=y, alpha=0.5)
plot_decision_boundary(x, model)
plt.show()


class BaggedTreeClassifier:
    def __init__(self, B):
        self.B = B

    def fit(self, x, y):
        n = len(x)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(n, size=n, replace=True)
            xb = x[idx]
            yb = y[idx]

            model = DecisionTreeClassifier(max_depth=2)
            model.fit(xb, yb)
            self.models.append(model)

    def predict(self, x):
        predictions = np.zeros(len(x))
        for model in self.models:
            predictions += model.predict(x)
        return np.round(predictions / self.B)

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(y == p)


model = BaggedTreeClassifier(200)
model.fit(x, y)

print('Score for Bagged model', model.score(x, y))

plt.scatter(x[:, 0], x[:, 1], s=100, c=y, alpha=0.5)
plot_decision_boundary(x, model)
plt.show()

