import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Make a function for multiple linear trainings
def poly_regression(x: np.ndarray, y: np.ndarray, k_test: np.ndarray, degree: int=1):
    model = LinearRegression()
    poly_feature = PolynomialFeatures(degree=degree)

    x_poly: np.ndarray = np.zeros(1)
    k_poly: np.ndarray = np.zeros(1)
    if degree > 1:
        x_poly = poly_feature.fit_transform(x)
        k_poly = poly_feature.fit_transform(k_test)
    else:
        x_poly = x
        k_poly = k_test


    model.fit(x_poly, y)

    pred = model.predict(k_poly)
    train_pred = model.predict(x_poly).reshape(-1)

    RMSE = np.sqrt(mean_squared_error(price, train_pred))
    R2 = r2_score(price, train_pred)
    
    print(f"RMSE: {RMSE}\tR2: {R2}")

    x = x.reshape(-1)
    y = y.reshape(-1)
    k_test = k_test.reshape(-1)
    pred = pred.reshape(-1)

    plt.scatter(x, y, s=10, color="red")
    plt.plot(k_test, pred, color="blue")
    plt.show()


#Load csv
df = pd.read_csv("Diamonds Prices2022.csv")
print(df.describe())

# Grab relevant columns
df = df[["price", "x", "y", "z"]]
df.dropna()
print(df.describe())

# Calculate volume
df["volume"] = df["x"]*df["y"]*df["z"]
df = df[["price", "volume"]]
print(df.describe())

df = df[df["volume"] <= 500]
df = df[df["volume"] >= 5]


# Convert into training data
price = df["price"].to_numpy()
volume = df["volume"].to_numpy()

volume, price = zip(*sorted(zip(volume, price), key= lambda k: k[0]))
volume = np.array(volume)
price = np.array(price)

plt.scatter(volume, price, s=10, color="red")
plt.show()

# Reshaoe for training
m_price = price.reshape(-1, 1)
m_volume = volume.reshape(-1, 1)
test_range = np.linspace(np.min(volume), np.max(volume), 100)
test_range = test_range.reshape(-1, 1)

poly_regression(m_volume, m_price, test_range, degree=1)
poly_regression(m_volume, m_price, test_range, degree=2)
poly_regression(m_volume, m_price, test_range, degree=3)
