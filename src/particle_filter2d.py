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
NUM_PARTICLES = 1000
box_size = 0.5
ALPHA_I = 0.5
ALPHA_O = 0.5
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

hist_fig, hist_axes = plt.subplots(nrows=2, ncols=3)#, sharex=True, sharey=True)
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

data = [([-1,0], create_feasible_set([-7,6], [-1,-6], [-1,0])), \
        ([6,-6], create_feasible_set([6,-6], [10,-9], [6,-6])), \
        ([-9,-3], create_feasible_set([-10,-3], [-9,-10], [-9,-3])), \
        ([0,7], create_feasible_set([-5,10], [3,7], [0,7])), \
        ([0,0], create_feasible_set([-2,2], [2,-1], [0,0]))]
for i in range(1, len(data)+1):
    (theta, feasible) = data[i-1]
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
    mins, maxes = np.amin(feasible, axis=0), np.amax(feasible, axis=0)
    width, height = maxes - mins
    hist_ax.add_patch(patches.Rectangle(np.amin(feasible, axis=0), width=width, height=height, color='C0'))
    hist_ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
    hist_ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

    plt.pause(0.2)
plt.show()
