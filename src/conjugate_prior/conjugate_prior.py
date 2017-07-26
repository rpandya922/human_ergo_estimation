from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gamma
from scipy.stats import norm
import probability_estimation as pe
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import time
#######################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 0.5
true = np.array([2, 2, 2, 2, 0, 0, 0])
# true = np.array([0])
sample_precision = np.identity(DOF)
true_weights = np.array([0.5, 0.25, 0.75, 2, 1, 1, 1])
# true_weights = np.array([1])
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def plot_helper(feasible, mean, true):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feasible[:,0], feasible[:,1], feasible[:,2], s=1, label='feasible')
    ax.scatter(mean[0], mean[1], mean[2], marker='^', c='r', s=500, label='estimated')
    ax.scatter(true[0], true[1], true[2], c='r', s=500, label='true')
    plt.legend(loc='upper left')
def plot_helper2(feasible, mean, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((-5, 5))
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
    for i in range(DOF):
        plot_helper2(feasible[:,i], mean[i], true[i])
    plt.show()
    # for (theta, feasible) in data:
    #     for i in range(DOF):
    #         plot_helper2(feasible[:,i], mean[i], true[i])
    #     plt.show()
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
    x = np.linspace(0, 15, 2000)
    y = gamma.pdf(x, a=alpha, scale=scale)
    plt.plot(x, y, label="a: " + str(alpha) + ", beta: " + str(beta))
    plt.title(str((alpha - 1) / beta))
    plt.legend(loc='upper left')
    plt.show()
    return (alpha - 1) / beta
def estimate_weights(data):
    alphas = np.ones(DOF)
    betas = np.ones(DOF)
    for (thetas, feasible) in data:
        alphas += len(thetas) / 2.0
        betas += np.sum(np.square(thetas - true), axis=0) / 2.0
    w = []
    for i in range(DOF):
        w.append((alphas[i] - 1) / betas[i])
    return w
def plot_distributions(mu, tau, alpha, beta):
    stddev = np.sqrt(1 / tau)
    scale = 1 / beta
    x = np.linspace(-5, 5, 2000)
    y_gamma = gamma.pdf(x, a=alpha, scale=scale)
    y_normal = norm.pdf(x, loc=mu, scale=stddev)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, y_gamma, label="a: " + str(alpha) + ", beta: " + str(beta))
    plt.title(str((alpha - 1) / beta))
    plt.legend(loc='upper left')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, y_normal, label="mean: " + str(mu) + ", stddev: " + str(stddev))
    plt.legend(loc='upper left')

    plt.show()
def plot_helper3(ax, chosen, feasible, mean, weight, true_mean, true_weight):
    stddev = 1 / np.sqrt(weight)
    true_stddev = 1 / np.sqrt(true_weight)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax.set_xlim((-5, 5))
    x = np.linspace(-5, 5, 2000)
    y = norm.pdf(x, loc=mean, scale=stddev)
    y_true = norm.pdf(x, loc=true_mean, scale=true_stddev)
    ax.plot(x, y, label='calculated')
    ax.plot(x, y_true, label='true')
    ax.hist(feasible, normed=True, label='feasible')
    ax.hist(chosen, normed=True, fc=(0.3, 0.2, 1, 0.5), label='chosen')
    ax.legend(loc='upper left')
def plot_data2(data, mean, weights):
    feasible_full = []
    chosen_full = []
    fig = plt.figure()
    subplots = [fig.add_subplot(4, 2, i+1) for i in range(DOF)]
    for (thetas, feasible) in data:
        feasible_full.extend(feasible)
        chosen_full.extend(thetas)
    feasible = np.array(feasible_full)
    chosen = np.array(chosen_full)

    for i in range(DOF):
        plot_helper3(subplots[i], chosen[:,i], feasible[:,i], mean[i], weights[i], true[i], true_weights[i])
    plt.show()
def update_plot(fig, ax, row, col, feasible, chosen, mean, weight):
    stddev = 1 / np.sqrt(weight)
    true_mean = true[col]
    true_stddev = 1 / np.sqrt(true_weights[col])
    # ax.clear()

    if col == 0:
        ylabel = ax.set_ylabel('iteration ' + str(row))
    ax.set_xlim((-5, 5))
    ax.set_ylim((0, 1.5))
    x = np.linspace(-5, 5, 2000)
    y = norm.pdf(x, loc=mean, scale=stddev)
    y_true = norm.pdf(x, loc=true_mean, scale=true_stddev)
    ax.plot(x, y, label='calculated')
    ax.plot(x, y_true, label='true')
    ax.hist(feasible, normed=True, label='feasible')
    ax.hist(chosen, normed=True, fc=(0.3, 0.2, 1, 0.5), label='chosen')
    if row % 7 == 0 and col == 0:
        ax.legend(loc='upper left')
def create_updating_plots(data):
    feasible_full = []
    chosen_full = []
    for (thetas, feasible) in data:
        feasible_full.extend(feasible)
        chosen_full.extend(thetas)
    feasible = np.array(feasible_full)
    chosen = np.array(chosen_full)
    lines = []
    labels = []
    fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(12,6), sharex=True, sharey=True)
    for i in range(DOF):
        mean = means[i]
        stddev = 1 / np.sqrt((alphas[i] - 1) / betas[i])
        true_stddev = 1 / np.sqrt(true_weights[i])
        true_mean = true[i]

        ax = axes[0][i]
        ax.set_xlim((-5, 5))
        ax.set_ylim((0, 1.5))
        ax.set_xlabel('joint ' + str(i+1))
        ax.xaxis.set_label_position('top')
        if i == 0:
            ax.set_ylabel('iteration 0')
        x = np.linspace(-5, 5, 2000)
        y = norm.pdf(x, loc=mean, scale=stddev)
        y_true = norm.pdf(x, loc=true_mean, scale=true_stddev)
        ax.plot(x, y, label='initial')
        ax.plot(x, y_true, label='true')
        ax.hist(feasible[:,i], normed=True, label='feasible')
        ax.hist(chosen[:,i], normed=True, fc=(0.3, 0.2, 1, 0.5), label='chosen')
        if i == 0:
            ax.legend(loc='upper left', prop={'size': 6})
    fig.canvas.draw()
    plt.show(block=False)
    return fig, axes, feasible, chosen, lines, labels
#######################################################
data = np.load("./random_training_data_k5.npy")

means = np.zeros(DOF)
taus = np.ones(DOF) * 0.001
alphas = np.ones(DOF) * 2
betas = np.ones(DOF) / 2

fig, axes, full_feasible, full_chosen, lines, labels = create_updating_plots(data)
time.sleep(2)
j = 1

for (thetas, feasible) in data:
    K = len(thetas)
    mean_theta = np.mean(thetas, axis=0)
    new_means = (np.multiply(taus, means) + (K * mean_theta)) / (taus + K)
    new_taus = taus + K
    new_alphas = alphas + (K / 2.0)
    new_betas = betas + (np.sum(np.square(thetas - mean_theta), axis=0) / 2) + \
    np.multiply((K * taus) / (taus + K), np.square(mean_theta - means) / 2.0)

    means = new_means
    taus = new_taus
    alphas = new_alphas
    betas = new_betas
    for i in range(DOF):
        update_plot(fig, axes[j%7][i], j, i, full_feasible[:,i], full_chosen[:,i], means[i], (alphas[i] - 1) / betas[i])
    fig.canvas.draw()
    plt.show(block=False)
    time.sleep(1)
    print j
    j += 1
    if j == 7:
        plt.show()
w = []
theta_star = []
for i in range(DOF):
    print "Mean: " + str(means[i])
    print "Weight: " + str((alphas[i] - 1) / betas[i])
    print
    w.append((alphas[i] - 1) / betas[i])
    theta_star.append(means[i])
    # plot_distributions(means[i], taus[i], alphas[i], betas[i])
plot_data2(data, theta_star, w)
# likelihood1 = 0
# likelihood2 = 0
# lam1 = np.hstack((w, theta_star))
# lam2 = np.hstack((true_weights, true))
# for (thetas, feasible) in data:
#     denom1 = pe.prob_stable2_denom(feasible, lam1, cost, ALPHA)
#     denom2 = pe.prob_stable2_denom(feasible, lam2, cost, ALPHA)
#     for theta in thetas:
#         likelihood1 += pe.prob_stable2_num(theta, lam1, cost, ALPHA) - denom1
#         likelihood2 += pe.prob_stable2_num(theta, lam2, cost, ALPHA) - denom2
# print "Computed weights negative log-likelihood: " + str(-likelihood1)
# print "Real weights negative log-likelihood: " + str(-likelihood2)

# for i in range(DOF):
#     plot_distributions(means[i], taus[i], alphas[i], betas[i])