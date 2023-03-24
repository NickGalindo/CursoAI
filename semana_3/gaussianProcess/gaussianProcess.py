from typing import Callable, Tuple
from scipy.optimize import linear_sum_assignment, OptimizeResult
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from skopt import gp_minimize

def gaussianProcessFit(x_train: np.ndarray, y_train: np.ndarray, rbf_theta: float=1.0, rbf_lengthscale: float=(1.0), n_restarts_optimizer: int=10):
    x_train = x_train.reshape(-1, 1)
    kernel = rbf_theta * RBF(length_scale=rbf_lengthscale)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(x_train.reshape(-1, 1), y_train)

    return gp

def gaussianProcessPlot(gp: GaussianProcessRegressor, x_data: np.ndarray, confidence: float=0.95):
    x_data = x_data.reshape(-1, 1)
    mean_prediction, std_prediction = gp.predict(x_data, return_std=True)

    conf_interval_coef = norm.ppf(1-(1-confidence)/2)

    plt.plot(x_data, mean_prediction, label="GP Mean Prediction", color="goldenrod")
    plt.fill_between(
        x_data.ravel(),
        mean_prediction - conf_interval_coef * std_prediction,
        mean_prediction + conf_interval_coef * std_prediction,
        alpha=0.5,
        label=f"{confidence}% confidence interval",
        color="goldenrod"
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression")


def gaussianProcessBayesianOptimize(f: Callable, range: Tuple[float, float]): # One dimensional bayesian optimization of function using gaussian process 
    res = gp_minimize(
        f,
        [range],
        acq_func="gp_hedge",
        n_calls=50,
        n_random_starts=15,
    )

    return res

def plotBayesianOptimize(res: OptimizeResult, d_range: Tuple[float, float]):
    predict_data_x = np.linspace(start=d_range[0], stop=d_range[1], num=10000)
    gaussianProcessPlot(res.models[-1], predict_data_x, confidence=0.999)

    return

if __name__ == "__main__":
    x = np.linspace(start=0, stop=10, num=10000)
    y = np.squeeze(x*x*2+3*x+4)

    rng = np.random.RandomState(1)
    train_indices = rng.choice(np.arange(y.size), size=3, replace=False)
    x_train, y_train = x[train_indices], y[train_indices]
    
    plt.plot(x, y, label=r"$sin(x)", linestyle="dotted")
    plt.scatter(x_train, y_train)
    
    gp = gaussianProcessFit(x_train, y_train)
    gaussianProcessPlot(gp, x, confidence=0.03)
    plt.show()

