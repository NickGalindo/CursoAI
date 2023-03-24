from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt import gp_minimize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gpr_samples(gpr_model, n_samples):
    x = np.linspace(0, 15, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    #y_samples = gpr_model.sample_y(X, n_samples)

    #for idx, single_prior in enumerate(y_samples.T):
    #    ax.plot(
    #        x,
    #        single_prior,
    #        linestyle="--",
    #        alpha=0.7,
    #        label=f"Sampled function #{idx + 1}",
    #    )
    plt.plot(x, y_mean, color="orange", label="Mean")
    plt.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="orange",
        label=r"$\pm$ 1 std. dev.",
    )


x = np.random.uniform(0, 15, 5)
x = np.sort(x)
y = np.sin(x/(5/np.pi))*5

kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(x.reshape(-1, 1), y)

plt.scatter(x, y, c="black")
plot_gpr_samples(gp, 10)
plt.show()



def f(x):
    x = np.array(x)
    return np.sum(np.sin(x/(5/np.pi))*5)

gp = gp_minimize(
    f,
    [(0, 15)],
    acq_func="EI",
    n_calls=5,
    n_random_starts=5
)

print(gp.x_iters)
print(gp.func_vals)

plt.scatter(np.array(gp.x_iters).reshape(-1, 1), gp.func_vals)
plot_gpr_samples(gp.models[-1], 10)
plt.show()
