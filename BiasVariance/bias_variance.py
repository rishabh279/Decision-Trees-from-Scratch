import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

NUM_DATASETS = 50
NOISE_VARIANCE = 0.5
MAX_POLY = 12
N = 25
Ntrain = int(0.9*N)

np.random.seed(2)


def make_poly(x, d):
  n = len(x)
  X = np.empty((n, d+1))
  for d in range(d+1):
    X[:, d] = x**d
    if d > 1:
      X[:, d] = (X[:, d] - X[:, d].mean()) / X[:, d].std()
  return X


def f(x):
  return np.sin(x)


x_axis = np.linspace(-np.pi, np.pi, 100)
y_axis = f(x_axis)

x = np.linspace(-np.pi, np.pi, N)
np.random.shuffle(x)
f_x = f(x)

x_poly = make_poly(x, MAX_POLY)


train_scores = np.zeros((NUM_DATASETS, MAX_POLY))
test_scores = np.zeros((NUM_DATASETS, MAX_POLY))

train_predictions = np.zeros((Ntrain, NUM_DATASETS, MAX_POLY))
prediction_curves = np.zeros((100, NUM_DATASETS, MAX_POLY))

model = LinearRegression()

for k in range(NUM_DATASETS):
  y = f_x + np.random.randn(N) * NOISE_VARIANCE

  xtrain = x_poly[:Ntrain]
  ytrain = y[:Ntrain]

  xtest = x_poly[Ntrain:]
  ytest = y[Ntrain:]

  for d in range(MAX_POLY):
    model.fit(xtrain[:, :d+2], ytrain)
    predictions = model.predict(x_poly[:, :d+2])
    predictions_shape = predictions.shape

    # x_axis_poly = make_poly(x_axis, d+1)
    # prediction_axis = model.predict(x_axis_poly)
    #
    # prediction_curves[:, k, d] = prediction_axis

    train_prediction = predictions[:Ntrain]
    train_prediction_shape = train_prediction.shape
    test_prediction = predictions[Ntrain:]

    train_predictions[:, k, d] = train_prediction

    train_score = mse(train_prediction, ytrain)
    train_score_shape = train_score.shape
    test_score = mse(test_prediction, ytest)

    train_scores[k, d] = train_score
    test_scores[k, d] = test_score


avg_train_prediction = np.zeros((Ntrain, MAX_POLY))
squared_bias = np.zeros(MAX_POLY)
f_xtrain = f_x[:Ntrain]

for d in range(MAX_POLY):
  for i in range(Ntrain):
    avg_train_prediction[i, d] = train_predictions[i, :, d].mean()
  squared_bias[d] = ((avg_train_prediction[:, d] - f_xtrain)**2).mean()


variances = np.zeros((Ntrain, MAX_POLY))
for d in range(MAX_POLY):
  for i in range(Ntrain):
    delta = train_predictions[i, :, d] - avg_train_prediction[i, d]
    variances[i, d] = delta.dot(delta) / len(delta)

variance = variances.mean(axis=0)

degrees = np.arange(MAX_POLY) + 1
best_degree = np.argmin(test_scores.mean(axis=0)) + 1
plt.plot(degrees, squared_bias, label='squared bias')
plt.plot(degrees, variance, label='variance')
plt.plot(degrees, test_scores.mean(axis=0), label='test scores')
plt.plot(degrees, squared_bias + variance, label='squared bias + variance')
plt.axvline(x=best_degree, linestyle='--', label='best complexity')
plt.legend()
plt.show()

# train score vs test score
plt.plot(degrees, train_scores.mean(axis=0), label='train scores')
plt.plot(degrees, test_scores.mean(axis=0), label='test scores')
plt.axvline(x=best_degree, linestyle='--', label='best complexity')
plt.legend()
plt.show()