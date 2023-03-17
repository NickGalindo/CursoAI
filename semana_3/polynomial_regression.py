from datetime import datetime

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

seed = int(datetime.now().timestamp())
np.random.seed(seed)

# Generate random data for the following function
# y=x+x^2-0.3x^3
x = 2*np.random.normal(0, 1, 20) + 1
y = x + (x**2) - 0.3*(x**3)
y = y + (np.random.normal(0, 0.5, 20)%1.5)

x, y = zip(*sorted(zip(x, y), key= lambda k: k[0]))
x = np.array(x)
y = np.array(y)

plt.scatter(x, y, s=10)
plt.show()

m_x = x.reshape(20, 1)
m_y = y.reshape(20, 1)

# Try to predict with simple linear regression
model = LinearRegression()
model.fit(m_x, m_y)
pred = model.predict(m_x)

pred = pred.reshape(20)

plt.scatter(x, y, s=10)
plt.plot(x, pred, color="red")
plt.show()

RMSE = np.sqrt(mean_squared_error(y, pred)) # How bad it is
R2 = r2_score(y, pred) # How good it is

print(f"RMSE: {RMSE}\tR2: {R2}")


# Predict with polynomial regression
# Transform features into second degree
poly_feature = PolynomialFeatures(degree=2)
m_x_poly = poly_feature.fit_transform(m_x)

model.fit(m_x_poly, m_y)
pred = model.predict(m_x_poly)

pred = pred.reshape(-1)

plt.scatter(x, y, s=10)
plt.plot(x, pred, color = "red")
plt.show()

RMSE = np.sqrt(mean_squared_error(y, pred)) # How bad it is
R2 = r2_score(y, pred) # How good it is

print(f"RMSE: {RMSE}\tR2: {R2}")

# Transform into third degree
poly_feature = PolynomialFeatures(degree=3)
m_x_poly = poly_feature.fit_transform(m_x)

model.fit(m_x_poly, m_y)
pred = model.predict(m_x_poly)

pred = pred.reshape(-1)

plt.scatter(x, y, s=10)
plt.plot(x, pred, color = "green")
plt.show()

RMSE = np.sqrt(mean_squared_error(y, pred)) # How bad it is
R2 = r2_score(y, pred) # How good it is

print(f"RMSE: {RMSE}\tR2: {R2}")

# Transform into twenty degrees
poly_feature = PolynomialFeatures(degree=20)
m_x_poly = poly_feature.fit_transform(m_x)

model.fit(m_x_poly, m_y)
pred = model.predict(m_x_poly)

pred = pred.reshape(-1)

plt.scatter(x, y, s=10)
plt.plot(x, pred, color = "green")
plt.show()

RMSE = np.sqrt(mean_squared_error(y, pred)) # How bad it is
R2 = r2_score(y, pred) # How good it is

print(f"RMSE: {RMSE}\tR2: {R2}")
