from __future__ import division
import numpy as np
import scipy
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gamma
from scipy.stats import norm
import probability_estimation as pe
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
import time
from random import shuffle
#######################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 0.5
true = np.array([2, 2, 2, 2, 0, 0, 0])
# true = np.array([0])
sample_precision = np.identity(DOF)
true_weights = np.array([0.5, 0.25, 0.75, 2, 1, 1, 1])
# true_weights = np.array([1])
ENT_CONST = 0.5 * np.log(2 * np.pi) + 1
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
def update(means, taus, alphas, betas, thetas):
    K = len(thetas)
    mean_theta = np.mean(thetas, axis=0)
    new_means = (np.multiply(taus, means) + (K * mean_theta)) / (taus + K)
    new_taus = taus + K
    new_alphas = alphas + (K / 2.0)
    new_betas = betas + (np.sum(np.square(thetas - mean_theta), axis=0) / 2.0) + \
    np.multiply((K * taus) / (taus + K), np.square(mean_theta - means) / 2.0)
    return new_means, new_taus, new_alphas, new_betas
def entropy_normal(means, taus):
    return 0.5 * np.log(2 * np.pi * np.exp(1) * 1 / taus)
def entropy_gamma(alphas, betas):
    return np.sum(alphas - np.log(betas) + scipy.special.gammaln(alphas) + ((1 - alphas) * scipy.special.digamma(alphas)))
def entropy_multivariate_normal(means, taus):
    return 0.5 * np.log(np.power(2 * np.pi * np.exp(1), len(means)) * np.linalg.det(np.diag(1 / taus)))
def entropy(means, taus, alphas, betas):
    return entropy_multivariate_normal(means, taus) + entropy_gamma(alphas, betas)
def info_gain(means, taus, alphas, betas, thetas):
    return entropy(means, taus, alphas, betas) - entropy(*update(means, taus, alphas, betas, thetas))
def blend(color, alpha, base=[255,255,255]):
    return [int(round((alpha * color[i]) + ((1 - alpha) * base[i]))) for i in range(3)]
def to_hex(color):
    return '#' + ''.join(["%02x" % e for e in color])
#######################################################
data = np.load("./full_sim_k_training_data_0.npy")
# shuffle(data)

means = np.zeros(DOF)
taus = np.ones(DOF) * 0.001
alphas = np.ones(DOF) * 2
betas = np.ones(DOF) / 2

means, taus, alphas, betas = update(means, taus, alphas, betas, [data[0][1][0]])

full_feasible = []
full_feasible.append(data[0][1][0])
for (thetas, feasible) in data:
    idxs = np.random.choice(len(feasible), size=100)
    full_feasible.extend(feasible[idxs])
full_feasible = np.array(full_feasible)
print len(full_feasible)
infos = []
for theta in full_feasible:
    infos.append(info_gain(means, taus, alphas, betas, [theta]))
print np.amin(infos)
print np.amax(infos)
infos = (np.array(infos) - np.amin(infos))/ np.amax(infos)
colors = [to_hex(blend([120, 51, 255], i)) for i in infos]

model = PCA(n_components=2)
new_data = model.fit_transform(full_feasible)
point = new_data[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(point[0], point[1], s=200, c='r', zorder=1)
ax.scatter(new_data[:,0], new_data[:,1], lw=0, s=5, c=colors, zorder=0)
plt.show()

# point = data[0][1][0]
# fig = plt.figure()
# ax = fig.add_subplot(221, projection='3d')
# ax.scatter(point[0], point[1], point[2], s=2000, c='r')
# ax.scatter(full_feasible[:,0], full_feasible[:,1], full_feasible[:,2], s=5, c=colors)
# ax2 = fig.add_subplot(222, projection='3d')
# ax2.scatter(point[3], point[4], point[5], s=2000, c='r')
# ax2.scatter(full_feasible[:,3], full_feasible[:,4], full_feasible[:,5], s=5, c=colors)
# ax3 = fig.add_subplot(223)
# ax3.scatter(point[6], 0, s=200, c='r')
# ax3.scatter(full_feasible[:,6], np.zeros(len(full_feasible)), s=10, c=colors)
# plt.show()
