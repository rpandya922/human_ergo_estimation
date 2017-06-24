from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gamma
import probability_estimation as pe
#######################################################
# CONSTANTS/FUNCTIONS
DOF = 7
true = np.array([0, 0, 0, 0, 0, 0, 0])
# true = np.array([0, 0])
sample_precision = np.identity(DOF)
true_weights = np.array([1, 1, 1, 1, 1, 1, 1])
def plot_helper(feasible, mean, true):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feasible[:,0], feasible[:,1], feasible[:,2], label='feasible')
    ax.scatter(mean[0], mean[1], mean[2], marker='^', c='r', s=500, label='estimated')
    ax.scatter(true[0], true[1], true[2], c='r', s=500, label='true')
    plt.legend(loc='upper left')
def plot_helper2(feasible, mean, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((-3.14, 3.14))
    ax.scatter(feasible, np.zeros(len(feasible)), label='feasible')
    ax.scatter(mean, 0, c='r', s=200, marker='^', label='estimated')
    ax.scatter(true, 0, c='r', s=200, label='true')
    plt.legend(loc='upper left')
def plot_data(data, mean):
    feasible_full = []
    for (theta, feasible) in data:
        feasible_full.extend(feasible)
    feasible = np.array(feasible_full)
    # plot_helper(feasible[:,:3], mean[:3], true[:3])
    # plot_helper(feasible[:,3:6], mean[3:6], true[3:6])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(feasible[:,6], np.zeros(len(feasible[:,6])), label='feasible')
    # ax.scatter(mean[6], 0, c='r', s=200, marker='^', label='estimated')
    # ax.scatter(true[6], 0, c='r', s=200, label='true')
    # plt.legend(loc='upper left')
    # for i in range(DOF):
    #     plot_helper2(feasible[:,i], mean[i], true[i])
    # plt.show()
    for (theta, feasible) in data:
        for i in range(DOF):
            plot_helper2(feasible[:,i], mean[i], true[i])
        plt.show()
def estimate_mean(data):
    mean = np.array([0]*DOF)
    precision = np.identity(DOF)
    for (theta, feasible) in data:
        mean = np.dot(np.linalg.inv(precision + sample_precision), \
        np.dot(precision, mean) + np.dot(sample_precision, theta))
        precision += sample_precision
    return mean
def plot_gamma(alpha, beta):
    scale = 1 / beta
    x = np.linspace(0, 10, 2000)
    y = gamma.pdf(x, a=alpha, scale=scale)
    plt.plot(x, y, label="a: " + str(alpha) + ", beta: " + str(beta))
    plt.title(str((alpha - 1) / beta))
    plt.legend(loc='upper left')
    plt.show()
#######################################################
data = np.load("./random_training_data.npy")
# mean = estimate_mean(data)
# plot_data(data, mean)
alphas = np.ones(DOF)
betas = np.ones(DOF)
# alpha = 1
# beta = 1
# feasible = np.random.normal(0, 3, size=(1000, 5))
# variance = 0
# for row in feasible:
#     variance += np.sum(np.square(row))
#     alpha += len(row) / 2
#     beta += np.sum(np.square(row)) / 2
# variance /= 5 * len(feasible)
# print variance
# plot_gamma(alpha, beta)
variance = 0
for (thetas, feasible) in data:
    print thetas
    variance += np.sum(np.square(thetas))
    alphas += len(thetas) / 2
    betas += np.sum(np.square(thetas - true), axis=0) / 2
variance /= len(thetas) * len(data)
print variance
for i in range(DOF):
    plot_gamma(alphas[i], betas[i])