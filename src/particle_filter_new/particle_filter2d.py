from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 2
NUM_PARTICLES = 500
box_size = 0.5
ALPHA_I = 0.5
ALPHA_O = 0.1
TRUE_MEAN = [0, 0]
def cost(theta, theta_star, w):
    return np.dot(theta - theta_star, np.dot(w, theta - theta_star))
def prior(vec):
    return 1
def create_feasible_set(upper_left, lower_right, chosen):
    feas = []
    x = upper_left[:]
    while x[1] >= lower_right[1]:
        while x[0] <= lower_right[0]:
            feas.append(x[:])
            x[0] += box_size
        x[0] = upper_left[0]
        x[1] -= box_size
    feas.append(chosen)
    return np.unique(feas, axis=0)
def create_feasible_set_ellipse(x0, y0, a, b, chosen):
    rand = np.random.uniform(-10, 10, size=(2000, 2))
    feas = []
    vals = np.sum(((rand - np.array([x0, y0])) / np.array([a, b])) ** 2, axis=1)
    for i, v in enumerate(vals):
        if v <= 1:
            feas.append(rand[i])
    feas.append(chosen)
    return np.array(feas)
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-10, 10, (DOF))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
dist = SetWeightsParticleDistribution(particles, weights, cost, w=np.array([[0.5, 0], [0, 0.5]]), ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20))
if __name__ == '__main__':
    pool = mp.Pool(6)
    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes = np.ndarray.flatten(np.array(axes))

    hist_fig, hist_axes = plt.subplots(nrows=3, ncols=3)
    hist_axes = np.ndarray.flatten(np.array(hist_axes))
    hist_ax = hist_axes[0]
    particles = np.array(dist.particles)
    data_means = particles.T
    kernel = kde(data_means)

    xx, yy = np.mgrid[-12:12:100j, -12:12:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = hist_ax.contourf(xx, yy, f, cmap='Greens')
    cset = hist_ax.contour(xx, yy, f, colors='k')
    hist_ax.clabel(cset, inline=1, fontsize=10)
    plt.pause(0.2)
    data = [([-1,0], create_feasible_set_ellipse(-3, 0, 2, 7, [-1,0])), \
            ([6,-6], create_feasible_set([6,-6], [10,-9], [6,-6])), \
            ([-9,-3], create_feasible_set_ellipse(-9.5, -7, 1, 6, [-9,-3])), \
            ([0,7], create_feasible_set([-5,10], [3,7], [0,7])), \
            ([0,0], create_feasible_set_ellipse(0, 2, 5, 3, [0,0]))]
    for i in range(1, 6):
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        print
        expected_infos = [sample[1] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        (theta, feasible) = pooled[max_idx][0]
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)
        for j in range(len(data)):
            t, f = data[j]
            d = SetWeightsParticleDistribution(dist.particles, dist.weights, dist.cost, dist.w, dist.ALPHA_I, dist.ALPHA_O)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))
        ax = axes[i]
        hist_ax = hist_axes[i]

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        particles = np.array(dist.particles)
        data_means = particles.T
        kernel = kde(data_means)

        xx, yy = np.mgrid[-12:12:100j, -12:12:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cfset = hist_ax.contourf(xx, yy, f, cmap='Greens')
        cset = hist_ax.contour(xx, yy, f, colors='k')
        hist_ax.clabel(cset, inline=1, fontsize=10)
        hist_ax.scatter(feasible[:,0], feasible[:,1], c='C0')
        hist_ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        hist_ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

        ax.bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
        ax.bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
        ax.bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')
        plt.pause(0.2)
    plt.show()
