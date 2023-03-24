import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal

def plot_gpr_samples(gpr_model, n_samples):
    x = np.linspace(0, 10, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    #for idx, single_prior in enumerate(y_samples.T):
    #    ax.plot(
    #        x,
    #        single_prior,
    #        linestyle="--",
    #        alpha=0.7,
    #        label=f"Sampled function #{idx + 1}",
    #    )
    plt.plot(x, y_mean, color="black", label="Mean")
    plt.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )


#x = np.array([8.85912347, 0.66212503, 1.46659491, 7.21834476, 6.14693627])
#x = np.sort(x)
#y = np.sin(x/(5/np.pi))*5


#kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
#gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

#gpr.fit(x.reshape(-1, 1), y)
#plot_gpr_samples(gpr, n_samples=5, ax=plt)

#plt.scatter(x, y, c="r")
#plt.show()


def calc_mu_cov_conditional_multivariate(mu, cov, x):
    mu_x = mu[:len(x)]
    mu_y = mu[len(x):]

    cov_xx = cov[:len(x), :len(x)]
    cov_xy = cov[:len(x), len(x):]
    cov_yy = cov[len(x):, len(x):]

    mu_y_given_x = mu_y + cov_xy.dot(np.linalg.inv(cov_xx)).dot(x - mu_x)
    cov_y_given_x = cov_yy - cov_xy.dot(np.linalg.inv(cov_xx)).dot(cov_xy.T)

    return mu_y_given_x, cov_y_given_x

def sample_conditional_gaussian(mu, cov, initial_terms, num_samples):
    # Extract the dimensions of the distribution
    num_vars = len(mu)
    num_initial = len(initial_terms)
    num_samples = max(num_samples, 1)

    # Split the mean vector into the initial and remaining terms
    mu_initial = mu[:num_initial]
    mu_remaining = mu[num_initial:]

    # Split the covariance matrix into four parts
    cov_initial_initial = cov[:num_initial, :num_initial]
    cov_initial_remaining = cov[:num_initial, num_initial:]
    cov_remaining_initial = cov[num_initial:, :num_initial]
    cov_remaining_remaining = cov[num_initial:, num_initial:]

    # Compute the conditional mean and covariance
    conditional_mean = mu_remaining + cov_remaining_initial @ np.linalg.solve(cov_initial_initial, initial_terms - mu_initial)
    conditional_cov = cov_remaining_remaining - cov_remaining_initial @ np.linalg.solve(cov_initial_initial, cov_initial_remaining)

    # Generate samples from the conditional distribution
    samples = np.random.multivariate_normal(conditional_mean, conditional_cov, size=num_samples)

    return samples



n = 50
#mu = np.array([0, 0, 0, 0, 0])
#cov = np.array([
#    [1, 0.9, 0.8, 0.7, 0.6],
#    [0.9, 1, 0.9, 0.8, 0.7],
#    [0.8, 0.9, 1, 0.9, 0.8],
#    [0.7, 0.8, 0.9, 1, 0.9],
#    [0.6, 0.7, 0.8, 0.9, 1]
#])

mu = [0 for i in range(n)]
cov = [
    [1-(0.002*abs(i-j)) for j in range(n)] for i in range(n)
]
mu = np.array(mu)
cov = np.array(cov)
print(cov)

#samples = np.random.multivariate_normal(mu, cov, size=1)
#x_samples = [i[0] for i in samples]
#y_samples = [i[1] for i in samples]

#x = np.linspace(-3,3,500)
#y = np.linspace(-3,3,500)
#X,Y = np.meshgrid(x,y)
#pos = np.array([X.flatten(),Y.flatten()]).T
#rv = multivariate_normal(mu, cov)
#plt.contour(X, Y, rv.pdf(pos).reshape(500,500))


x_samples = []
y_samples = sample_conditional_gaussian(mu, cov, x_samples, 1)
y_samples = y_samples.reshape(-1)

#plt.scatter(x_samples, y_samples, c="r")
#plt.show()

plt.plot([i+1 for i in range(n)], y_samples, c="black")
plt.scatter([i+1 for i in range(n)], y_samples, s=5, c="black")
#plt.scatter([1, 2, 3], x_samples, c="r")

#plt.plot([1, 2], [x_samples[0], y_samples[0]], c="black")
#plt.scatter([1], [x_samples[0]], c="r")
#plt.scatter([2], [y_samples[0]], c="black")
#plt.margins(x=0.01, tight=True)
plt.xticks([i for i in range(n)])
plt.ylim(-3, 3)
plt.show()


import seaborn as sn

sn.heatmap(cov, fmt="g")
plt.show()
